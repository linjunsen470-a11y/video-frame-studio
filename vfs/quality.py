"""Deterministic selection and fail-closed PNG quality gates."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .masks import polygon_mask
from .util import read_image, resize_long


def metrics(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(resize_long(image, 640), cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark, bright = float(np.mean(gray < 9)), float(np.mean(gray > 246))
    contrast = float(gray.std())
    # Only a cheap technical pre-rank. Composition/meaning belongs to the second stage.
    score = 45 * min(1.0, np.log1p(sharpness) / 7) + 30 * min(1, contrast / 65)
    score += 25 * (1 - max(dark, bright))
    return {"sharpness": round(sharpness, 3), "dark_fraction": round(dark, 4),
            "bright_fraction": round(bright, 4), "gray_std": round(contrast, 3),
            "technical_score": round(float(score), 3)}


def acceptable(metrics_: dict, config: dict) -> bool:
    return (metrics_["sharpness"] >= config["min_sharpness"]
            and metrics_["dark_fraction"] <= config["max_dark_fraction"]
            and metrics_["bright_fraction"] <= config["max_bright_fraction"]
            and metrics_["gray_std"] >= config["min_gray_std"])


def phash(image: np.ndarray) -> str:
    """64 bit DCT perceptual hash; no ImageHash/scipy runtime dependency."""
    gray = cv2.cvtColor(cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    low = cv2.dct(gray.astype(np.float32))[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"


def hash_distance(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def check_image(original: np.ndarray, edited: np.ndarray, mask: np.ndarray,
                selected: list[dict], detector, config: dict,
                prior_hashes: list[str] | None = None,
                duplicate_distance: int = 5,
                protected_regions: list[dict] | None = None) -> dict:
    result = {"passed": False, "issues": [], "warnings": []}
    if edited.shape != original.shape or mask.shape != original.shape[:2]:
        result["issues"].append("dimension_mismatch")
        return result
    if edited.dtype != np.uint8 or original.dtype != np.uint8 or mask.dtype != np.uint8:
        result["issues"].append("unsupported_pixel_format")
        return result
    support = mask > 0
    outside = ~support
    mismatch = np.any(original != edited, axis=2) & outside
    n_out = int(np.count_nonzero(outside))
    result.update({"dimensions": [int(edited.shape[1]), int(edited.shape[0])],
                   "outside_pixels": n_out,
                   "outside_changed_pixels": int(mismatch.sum()),
                   "outside_pixel_equality": 1.0 if not mismatch.any() else
                       1 - float(mismatch.sum()) / max(1, n_out),
                   "mask_fraction": round(float(support.mean()), 6)})
    if mismatch.any():
        result["issues"].append("outside_mask_pixels_changed")
    if not n_out or support.mean() > config["max_mask_fraction"]:
        result["issues"].append("mask_too_large")
    before, after = metrics(original), metrics(edited)
    result["metrics"] = after
    if after["sharpness"] < config["min_sharpness"]:
        result["issues"].append("insufficient_sharpness")
    # Subtitles themselves contribute edges; this is deliberately a loose gate.
    if after["sharpness"] < before["sharpness"] * config["min_sharpness_ratio"]:
        result["issues"].append("sharpness_loss")
    result["phash"] = phash(edited)
    if any(hash_distance(result["phash"], other) <= duplicate_distance
           for other in (prior_hashes or [])):
        result["issues"].append("duplicate_output")
    if np.any(support):
        boundary = (cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0) ^ support
        gray_a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_b = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY).astype(np.float32)
        edge_a = np.abs(cv2.Laplacian(gray_a, cv2.CV_32F))
        edge_b = np.abs(cv2.Laplacian(gray_b, cv2.CV_32F))
        seam = float(np.percentile(np.maximum(0, edge_b[boundary] - edge_a[boundary]), 95)) if boundary.any() else 0
        result["seam_excess_p95"] = round(seam, 3)
        if seam > config["max_seam_excess"]:
            result["issues"].append("visible_boundary")
    try:
        remaining = detector.detect(edited)
        target = polygon_mask(original.shape, selected, 5) > 0
        residuals = []
        for region in remaining:
            region_mask = polygon_mask(original.shape, [region]) > 0
            overlap = float(np.count_nonzero(region_mask & target)) / max(1, int(region_mask.sum()))
            if overlap > config["residual_overlap"]:
                residuals.append({"text": region["text"], "polygon": region["polygon"],
                                  "confidence": region["confidence"], "overlap": round(overlap, 3)})
        result["residual_text"] = residuals
        if residuals:
            result["issues"].append("subtitle_residual")
        # Whole-frame OCR may miss a single truncated character. Look for
        # unchanged high-contrast edges just beyond detected text. Use these
        # edges to locate an enlarged OCR check, since scene edges can look
        # identical to subtitle strokes in a grayscale image.
        gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
        near_dark = cv2.dilate((gray < 70).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        near_bright = cv2.dilate((gray > 190).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        styled = (((gray > 160) & near_dark) | ((gray < 70) & near_bright)) & ~support
        if protected_regions:
            styled[polygon_mask(original.shape, protected_regions, 4) > 0] = False
        guard = np.zeros(original.shape[:2], np.uint8)
        for region in selected:
            points = np.asarray(region["polygon"], float)
            x0, y0 = np.floor(points.min(axis=0)).astype(int)
            x1, y1 = np.ceil(points.max(axis=0)).astype(int)
            height = max(1, y1 - y0)
            px, py = round(height * .9), max(3, round(height * .12))
            guard[max(0, y0-py):min(gray.shape[0], y1+py+1),
                  max(0, x0-px):min(gray.shape[1], x1+px+1)] = 255
        suspicious = styled & (guard > 0)
        result["persistent_stroke_pixels"] = int(suspicious.sum())
        if suspicious.sum() > config.get("max_persistent_stroke_pixels", 16):
            ys, xs = np.where(suspicious)
            x0, x1 = max(0, int(xs.min())-16), min(gray.shape[1], int(xs.max())+17)
            y0, y1 = max(0, int(ys.min())-16), min(gray.shape[0], int(ys.max())+17)
            local = cv2.resize(edited[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            local_edges = cv2.resize(suspicious[y0:y1, x0:x1].astype(np.uint8),
                                     None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) > 0
            focused = [region for region in detector.detect(local)
                       if region["text"].strip()
                       and np.any(local_edges & (polygon_mask(local.shape, [region], 2) > 0))]
            result["focused_residual_ocr"] = focused
            result["issues" if focused else "warnings"].append("possible_stroke_residual")
        result["ocr_checked"] = True
    except Exception as exc:
        result["ocr_checked"] = False
        result["issues"].append("ocr_unavailable")
        result["warnings"].append(type(exc).__name__)
    result["passed"] = not result["issues"]
    return result


def check_saved_png(original_path: Path, output_path: Path, mask_path: Path,
                    selected: list[dict], detector, config: dict, **kwargs) -> dict:
    """The gate uses losslessly reloaded files, not an in-memory pre-export assertion."""
    return check_image(read_image(original_path), read_image(output_path),
                       read_image(mask_path, gray=True), selected, detector, config, **kwargs)
