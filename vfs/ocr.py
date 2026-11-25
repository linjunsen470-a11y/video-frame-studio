"""CPU OCR, content-addressed caching and conservative overlay classification.

Coordinates always refer to the supplied, unscaled BGR frame. OCR detects text;
it does not establish whether that text belongs to the photographed scene.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class OCRUnavailable(RuntimeError):
    """Detection was not performed; callers must not report a clean image."""


def _valid_image(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("OCR expects a nonempty HxWx3 uint8 BGR image")
    if not image.shape[0] or not image.shape[1]:
        raise ValueError("Empty image")


def _normalise_regions(regions: list, shape: tuple) -> list[dict]:
    height, width = shape[:2]
    normalised = []
    for item in regions:
        try:
            if isinstance(item, dict):
                polygon, text, score = item["polygon"], item.get("text", ""), item.get("confidence", 1.0)
            else:
                polygon, text, score = item[:3]
            points = np.asarray(polygon, dtype=np.float64)
            score = float(score)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
                continue
            if not np.isfinite(points).all() or not np.isfinite(score):
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            if abs(cv2.contourArea(points.astype(np.float32))) < 2:
                continue
            result = {"polygon": points.round(2).tolist(), "text": str(text),
                      "confidence": float(np.clip(score, 0, 1))}
            if isinstance(item, dict):
                for key in ("is_overlay", "manual", "reason"):
                    if key in item:
                        result[key] = item[key]
            normalised.append(result)
        except (ValueError, TypeError, KeyError, IndexError):
            continue
    return normalised


class OCRDetector:
    """Lazily initialise RapidOCR once, caching each image/config/version result.

    ``backend=mock`` accepts explicit ``mock_regions`` and is visibly reported
    through ``is_mock``. It is intended for tests only. ``backend=off`` raises.
    The legacy packaged ONNX backend and current RapidOCR API are supported.
    """

    def __init__(self, cache_dir: Path, config: dict):
        self.config = dict(config)
        self.backend = str(config.get("backend", "rapidocr"))
        if self.backend not in {"rapidocr", "mock", "off"}:
            raise ValueError(f"Unknown OCR backend: {self.backend}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.engine: Any = None
        self.api = ""
        self.is_mock = self.backend == "mock"
        self.stats = {"calls": 0, "cache_hits": 0, "mock": self.is_mock}
        self.package_versions = {}
        for name in ("rapidocr-onnxruntime", "rapidocr", "onnxruntime"):
            try:
                self.package_versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass

    def _initialise(self) -> None:
        if self.engine is not None:
            return
        threads = max(1, int(self.config.get("threads", 2)))
        try:
            from rapidocr import RapidOCR
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as error:
                raise OCRUnavailable("RapidOCR is missing; run setup.sh or install the OCR requirements") from error
            self.api = "legacy"
            self.engine = RapidOCR(
                intra_op_num_threads=threads, inter_op_num_threads=1,
                det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=False,
                det_limit_side_len=int(self.config.get("max_side", 1280)),
            )
        else:
            self.api = "modern"
            # Require the small models shipped in the wheel. Explicit paths
            # prevent the upstream engine from starting an implicit download.
            import rapidocr
            model_dir = Path(rapidocr.__file__).parent / "models"
            model_params = {}
            for role in ("det", "cls", "rec"):
                matches = list(model_dir.glob(f"*{role}*.onnx"))
                if len(matches) != 1:
                    raise OCRUnavailable(f"Expected one bundled {role} ONNX model; reinstall the pinned rapidocr wheel")
                model_params[f"{role.title()}.model_path"] = str(matches[0])
            self.engine = RapidOCR(params={
                **model_params,
                "Global.log_level": "error",
                "Global.max_side_len": int(self.config.get("max_side", 1280)),
                "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                "EngineConfig.onnxruntime.use_cuda": False,
                "EngineConfig.onnxruntime.use_dml": False,
                "EngineConfig.onnxruntime.use_cann": False,
                "EngineConfig.onnxruntime.use_coreml": False,
                "Det.limit_side_len": int(self.config.get("max_side", 1280)),
                "Det.limit_type": "max",
            })

    def detect(self, image: np.ndarray) -> list[dict]:
        _valid_image(image)
        if self.backend == "off":
            raise OCRUnavailable("OCR disabled: subtitle absence cannot be verified")
        digest = hashlib.sha256()
        digest.update(b"vfs-ocr-v3\0")
        digest.update(json.dumps({"config": self.config, "versions": self.package_versions},
                                 sort_keys=True, ensure_ascii=False).encode())
        digest.update(str(image.shape).encode())
        digest.update(np.ascontiguousarray(image).tobytes())
        destination = self.cache_dir / (digest.hexdigest() + ".json")
        if destination.exists():
            try:
                payload = json.loads(destination.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    self.stats["cache_hits"] += 1
                    return _normalise_regions(payload, image.shape)
            except (OSError, ValueError):
                pass
        if self.is_mock:
            regions = self.config.get("mock_regions", [])
        else:
            try:
                self._initialise()
                result = self.engine(image)
                if self.api == "legacy":
                    regions = result[0] or []
                else:
                    if not all(hasattr(result, name) for name in ("boxes", "txts", "scores")):
                        raise OCRUnavailable("Unrecognised RapidOCR response schema")
                    boxes = getattr(result, "boxes", None)
                    texts = getattr(result, "txts", None)
                    scores = getattr(result, "scores", None)
                    if boxes is None:
                        if texts is not None or scores is not None:
                            raise OCRUnavailable("Inconsistent empty RapidOCR response")
                        regions = []
                    else:
                        if texts is None or scores is None or len(boxes) != len(texts) or len(boxes) != len(scores):
                            raise OCRUnavailable("Unrecognised RapidOCR response schema")
                        regions = list(zip(boxes, texts, scores))
            except OCRUnavailable:
                raise
            except Exception as error:
                raise OCRUnavailable(f"RapidOCR failed: {type(error).__name__}: {error}") from error
        self.stats["calls"] += 1
        regions = _normalise_regions(regions, image.shape)
        threshold = float(self.config.get("confidence", 0.50))
        regions = [region for region in regions if region["confidence"] >= threshold]
        temporary = destination.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(regions, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, destination)
        return regions


def _bounds(region: dict) -> tuple[float, float, float, float]:
    points = np.asarray(region["polygon"], dtype=np.float64)
    return float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())


def _overlap(a: tuple, b: tuple) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    minimum_area = min(max(1, (a[2]-a[0])*(a[3]-a[1])), max(1, (b[2]-b[0])*(b[3]-b[1])))
    return intersection / minimum_area


def _same_line(a: dict, b: dict) -> bool:
    first, second = _bounds(a), _bounds(b)
    height = max(first[3]-first[1], second[3]-second[1], 1)
    return (_overlap(first, second) >= 0.65
            and abs((first[1]+first[3])-(second[1]+second[3])) <= 0.35*height
            and min(first[3]-first[1], second[3]-second[1]) / height > 0.65)


def select_regions(regions: list[dict], image_shape: tuple, config: dict,
                   temporal_regions: list | None = None) -> dict[str, list[dict]]:
    """Select overlay candidates, preserving uncertain scene text by default.

    ``auto`` uses same-position changing text from neighbouring frames as
    overlay evidence; a static shop sign alone is not sufficient. ``subtitle``
    explicitly authorises horizontal text in the configured bottom band.
    ``all`` authorises all OCR text, subject to explicit keep words/regions.
    ``manual`` selects ``manual_regions`` (rectangles or polygons). Protected
    regions use absolute ``[x1,y1,x2,y2]`` coordinates, not percentages.
    """
    height, width = image_shape[:2]
    mode = config.get("mode", "auto")
    if mode not in {"auto", "subtitle", "all", "manual"}:
        raise ValueError(f"Unknown text selection mode: {mode}")
    groups: dict[str, list[dict]] = {"selected": [], "protected": [], "uncertain": []}
    keep_words = [str(word).casefold() for word in config.get("keep_words", [])]
    protect = config.get("protect_regions", [])
    candidates = _normalise_regions(regions, image_shape)
    if mode == "manual":
        candidates = []
        for item in config.get("manual_regions", []):
            if isinstance(item, dict):
                polygon = item.get("polygon")
            elif len(item) == 4 and all(isinstance(value, (int, float)) for value in item):
                x1, y1, x2, y2 = item
                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                polygon = item
            candidates.extend(_normalise_regions([{"polygon": polygon, "text": "", "confidence": 1.0, "manual": True}], image_shape))
    neighbours = []
    for frame in temporal_regions or []:
        neighbours.append(frame.get("regions", []) if isinstance(frame, dict) else frame)
    for region in candidates:
        bounds = _bounds(region)
        text = region["text"].casefold()
        reason = ""
        if any(word and word in text for word in keep_words):
            reason = "explicit_keep_word"
        elif any(_overlap(bounds, tuple(rect)) > 0 for rect in protect):
            reason = "explicit_protected_region"
        if reason:
            groups["protected"].append({**region, "reason": reason})
            continue
        if mode == "manual" or mode == "all":
            groups["selected"].append({**region, "reason": f"explicit_{mode}"})
            continue
        box_width, box_height = bounds[2]-bounds[0], max(bounds[3]-bounds[1], 1)
        lower_line = (bounds[1] >= float(config.get("bottom_start", 0.65))*height
                      and box_width >= float(config.get("min_line_width", 0.12))*width
                      and box_width/box_height >= 2.0
                      and box_height <= 0.15*height)
        changed = any(_same_line(region, other) and other.get("text", "").casefold() != text
                      and bool(other.get("text", "").strip())
                      for frame in neighbours for other in frame)
        if region.get("is_overlay") is True:
            reason = "explicit_overlay_label"
        elif changed:
            reason = "same_screen_position_changed_text"
        elif lower_line and (mode == "subtitle" or config.get("auto_allow_bottom", False)):
            reason = "authorised_bottom_subtitle_band"
        if reason:
            groups["selected"].append({**region, "reason": reason})
        else:
            groups["uncertain"].append({**region, "reason": "overlay_or_scene_text_ambiguous"})
    return groups


def reviewed_overlay_regions(regions, image_shape, config, reviewed):
    """Convert visually confirmed source-pixel boxes to the normal mask input."""
    height, width = image_shape[:2]
    boxes = [entry["box"] for entry in reviewed]
    if any(not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height)
           for x0, y0, x1, y1 in boxes):
        raise ValueError("Reviewed overlay box is outside the source frame")
    groups = select_regions(regions, image_shape, {**config, "mode": "auto"})
    manual = select_regions([], image_shape, {**config, "mode": "manual", "manual_regions": boxes})
    if len(manual["selected"]) != len(reviewed) or any(
            _overlap(_bounds(protected), tuple(box)) > 0
            for protected in groups["protected"] for box in boxes):
        raise ValueError("Reviewed overlay overlaps protected text or has an invalid box")
    groups["selected"] = [
        {**region, "reason": f"reviewed_{entry['kind']}", "text": entry.get("text", "")}
        for region, entry in zip(manual["selected"], reviewed, strict=True)
    ]
    # The operator reviewed the whole frame and supplied all target overlays.
    groups["uncertain"] = []
    return groups
