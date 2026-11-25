"""Fixed processing policy: the agent supplies inputs, not ad-hoc decisions."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from . import __version__
from .config import semantic_config
from .gemini import GeminiGateway, GeminiError, BudgetExceeded, InvalidResponse
from .masks import build_mask
from .ocr import OCRDetector, reviewed_overlay_regions, select_regions
from .quality import check_saved_png, polygon_mask
from .repair import composite_local
from .report import write_report
from .util import (atomic_json, read_json, file_sha256, digest, read_image, resize_long,
                   write_png, output_lock, safe_output_path)
from .video import resolve_input, scan_candidates


RECOVERABLE = {"api_budget_exhausted", "gemini_unavailable"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preview(original, final, mask, path: Path, result_label="RESULT"):
    overlay = original.copy()
    marked = mask > 0
    overlay[marked] = (original[marked] * .45 + np.array([30, 40, 240]) * .55).astype(np.uint8)
    panels = [resize_long(im, 480).copy() for im in (original, overlay, final)]
    labels = ("ORIGINAL", "EDIT MASK", result_label)
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (240, 240, 240), 1, cv2.LINE_AA)
    write_png(path, np.hstack(panels))


def _protected_mask(shape: tuple, groups: dict, cfg: dict):
    result = polygon_mask(shape, groups["protected"], 2)
    for x1, y1, x2, y2 in cfg["ocr"].get("protect_regions", []):
        cv2.rectangle(result, (max(0, int(x1)), max(0, int(y1))),
                      (min(shape[1]-1, int(x2)), min(shape[0]-1, int(y2))), 255, -1)
    return result > 0


def _candidate(candidate: dict, output: Path, cfg: dict, detector: OCRDetector,
               gateway: GeminiGateway, existing_hashes: list[str], previous: dict | None = None) -> dict:
    original_path = safe_output_path(output, candidate["path"])
    original = read_image(original_path)
    item = {k: candidate[k] for k in ("id", "time", "frame_index", "scene", "metrics")}
    item.update({"status": "review", "attempts": list((previous or {}).get("attempts", [])), "warnings": []})
    regions = detector.detect(original)
    reviewed = cfg["ocr"].get("reviewed_frames", {}).get(candidate["id"])
    if reviewed is None:
        temporal_regions = [
            detector.detect(read_image(safe_output_path(output, donor["path"])))
            for donor in candidate["neighbors"] if donor["scene"] == candidate["scene"]
        ]
        groups = select_regions(regions, original.shape, cfg["ocr"], temporal_regions)
    else:
        groups = reviewed_overlay_regions(regions, original.shape, cfg["ocr"], reviewed)
    item["text_regions"] = groups
    if groups["uncertain"] and cfg["qc"]["require_no_uncertain_text"]:
        item["issues"] = ["uncertain_text_origin"]
        _review_preview(original, groups, output, item, cfg)
        return item
    protection = _protected_mask(original.shape, groups, cfg)
    selected = groups["selected"]
    # A network failure did not produce an image attempt. Existing failed image
    # attempts count toward the same three-attempt cap across process restarts.
    completed_attempts = sum("qc" in a for a in item["attempts"])
    for attempt in range(completed_attempts, cfg["run"]["max_attempts"]):
        trial = {"attempt": attempt + 1}
        mask_config = {**cfg["mask"], "protect_regions": cfg["ocr"].get("protect_regions", [])}
        mask, mask_meta = build_mask(original, selected, mask_config, attempt)
        mask[protection] = 0
        trial["mask"] = mask_meta
        method = "unchanged"
        try:
            if mask_meta["area_fraction"] > cfg["mask"]["max_area_fraction"]:
                item["issues"] = ["mask_too_large"]
                break
            if reviewed == []:
                proposal = original
            else:
                if not np.any(mask):
                    item["issues"] = ["selected_text_has_empty_mask" if selected else "no_target_text"]
                    break
                if mask_meta.get("edge_search_inconclusive", 0):
                    trial["qc"] = {"passed": False, "issues": ["mask_edge_uncertain"]}
                    item["attempts"].append(trial)
                    item["qc"] = trial["qc"]
                    continue
                method = "gemini_full_frame"
                proposal = gateway.edit_frame(original, mask, attempt)
                trial["gemini_edit"] = gateway.last_edit_info
            final, final_mask = composite_local(original, proposal, mask, cfg["mask"]["feather_px"])
            # Protection applies to the entire feather support, not just the initial mask.
            final[protection] = original[protection]
            final_mask[protection] = 0
            if selected and np.array_equal(final, original):
                item["issues"] = ["no_pixels_changed"]
                break
            stage = output / ".cache/qc"
            staged_image, staged_mask = stage / f"{item['id']}.png", stage / f"{item['id']}_mask.png"
            write_png(staged_image, final)
            write_png(staged_mask, final_mask)
            qc = check_saved_png(original_path, staged_image, staged_mask, selected, detector,
                                 cfg["qc"], prior_hashes=existing_hashes,
                                 duplicate_distance=cfg["selection"]["phash_distance"],
                                 protected_regions=groups["protected"])
            trial.update({"method": method, "qc": qc})
            item["attempts"].append(trial)
            item.update({"method": method, "qc": qc})
            if qc["passed"]:
                paths = {"image": f"images/{item['id']}.png",
                         "original": f"originals/{item['id']}.png",
                         "mask": f"masks/{item['id']}.png"}
                for folder in ("images", "originals", "masks", "previews"):
                    (output / folder).mkdir(exist_ok=True)
                # The validated PNG bytes are moved; no later lossy encoding occurs.
                staged_image.replace(output / paths["image"])
                staged_mask.replace(output / paths["mask"])
                shutil.copyfile(original_path, output / paths["original"])
                item.update({"status": "passed", "files": paths,
                             "file_sha256": {k: file_sha256(output / v) for k, v in paths.items()}})
                if cfg["run"]["previews"]:
                    preview = f"previews/{item['id']}.png"
                    _preview(original, final, final_mask, output / preview)
                    item["files"]["preview"] = preview
                return item
            if cfg["run"]["previews"]:
                preview = f"previews/{item['id']}_attempt_{attempt+1}.png"
                _preview(original, final, final_mask, output / preview)
                trial["preview"] = preview
            staged_image.unlink(missing_ok=True)
            staged_mask.unlink(missing_ok=True)
            if "duplicate_output" in qc["issues"] or "outside_mask_pixels_changed" in qc["issues"]:
                break
        except InvalidResponse as exc:
            trial.update(method=method, qc={"passed": False, "issues": ["proposal_alignment_failed"]},
                         detail=str(exc))
            proposal = getattr(gateway, "last_unaligned_proposal", None)
            if cfg["run"]["previews"] and proposal is not None:
                preview = f"previews/{item['id']}_attempt_{attempt+1}_unaligned.png"
                _preview(original, proposal, mask, output / preview, "UNALIGNED")
                trial["preview"] = preview
            item["attempts"].append(trial)
            item.update(method=method, qc=trial["qc"])
        except GeminiError as exc:
            trial.update({"method": method, "error": type(exc).__name__})
            item["attempts"].append(trial)
            item["issues"] = ["api_budget_exhausted" if isinstance(exc, BudgetExceeded) else "gemini_unavailable"]
            break
    if item.get("qc") and not item.get("issues"):
        item["status"] = "rejected"
    _review_preview(original, groups, output, item, cfg)
    return item


def _review_preview(original, groups, output, item, cfg):
    if not cfg["run"]["previews"]:
        return
    preview = resize_long(original.copy(), 960)
    scale = preview.shape[1] / original.shape[1]
    for region in groups["selected"] + groups["uncertain"]:
        cv2.polylines(preview, [np.rint(np.asarray(region["polygon"]) * scale).astype(np.int32)],
                      True, (0, 80, 255), 2)
    path = f"previews/{item['id']}_review.png"
    write_png(output / path, preview)
    item["review_preview"] = path


def quality_only(output: Path, config: dict | None = None) -> dict:
    source_report = read_json(output / "report.json")
    cfg = config or source_report["config"]
    detector = OCRDetector(output / ".cache/ocr", cfg["ocr"])
    result = copy.deepcopy(source_report)
    result["audit_at"] = _now()
    hashes = []
    known = set()
    for item in result["items"]:
        if item["status"] != "passed":
            continue
        files = {key: safe_output_path(output, value) for key, value in item["files"].items()}
        known.add(files["image"].resolve())
        try:
            qc = check_saved_png(files["original"], files["image"], files["mask"],
                       item["text_regions"]["selected"], detector, cfg["qc"], prior_hashes=hashes,
                       duplicate_distance=cfg["selection"]["phash_distance"],
                       protected_regions=item["text_regions"].get("protected", []))
            for key, expected in item.get("file_sha256", {}).items():
                if file_sha256(files[key]) != expected:
                    qc["issues"].append(f"manifest_hash_mismatch:{key}")
            # A deterministic audit does not rerun or fabricate prior semantic reviews.
            if "visual_review" in item.get("qc", {}):
                qc["visual_review"] = item["qc"]["visual_review"]
            qc["passed"] = not qc["issues"]
        except (OSError, ValueError) as exc:
            qc = {"passed": False, "issues": ["missing_or_invalid_file"], "error": type(exc).__name__}
        item["qc"] = qc
        if qc["passed"]:
            hashes.append(qc["phash"])
        else:
            item["status"] = "rejected"
    unexpected = [p.name for p in (output / "images").glob("*") if p.resolve() not in known]
    if unexpected:
        result.setdefault("warnings", []).append("Unmanifested files in images/: " + ", ".join(unexpected))
    audited_items = [item for item in result["items"] if item.get("files", {}).get("image")]
    result["status"] = "qc_passed" if audited_items and not unexpected and all(
        item["status"] == "passed" for item in result["items"]
        if item.get("files", {}).get("image")) else "qc_failed"
    if not audited_items:
        result.setdefault("warnings", []).append("No exported images are available to audit.")
    result["qc_only"] = True
    result["ocr"] = detector.stats
    write_report(output, result, "qc_report")
    return result


def _validate_output_paths(output: Path) -> None:
    for name in (".cache", "images", "originals", "masks", "previews", ".checkpoint.json", "selection"):
        child = output / name
        if child.is_symlink() or (child.is_dir() and any(p.is_symlink() for p in child.rglob("*"))):
            raise ValueError("Symlinks are not allowed inside managed output paths")


def prepare_selection(input_value: str, output: Path, cfg: dict) -> dict:
    """Decode and publish a chronological candidate index without opening a gateway."""
    output = output.resolve()
    with output_lock(output):
        _validate_output_paths(output)
        for name in (".checkpoint.json", "report.json", "images", "selection"):
            if (output / name).exists():
                raise ValueError("Choose a fresh output directory to prepare candidates")
        input_path = resolve_input(input_value, output / ".cache/download", cfg["video"])
        if input_path.is_relative_to(output / ".cache/frames"):
            raise ValueError("Input cannot be inside the candidate cache")
        source_hash = file_sha256(input_path)
        cv2.setNumThreads(int(cfg["run"]["threads"]))
        video_cfg = {**cfg["video"], "threads": cfg["run"]["threads"]}
        scan = scan_candidates(input_path, output, video_cfg, cfg["selection"],
                               cfg["run"]["count"])
        candidates = sorted(scan["candidates"], key=lambda c: c["time"])
        if not candidates:
            raise ValueError("No candidates passed the technical filters")
        index = {"source_sha256": source_hash, "candidates": [],
                 "scan": {key: value for key, value in scan.items() if key != "candidates"}}
        for offset in range(0, len(candidates), 12):
            batch = candidates[offset:offset+12]
            canvas = np.zeros((240 * ((len(batch)+2)//3), 960, 3), dtype=np.uint8)
            for i, candidate in enumerate(batch):
                path = safe_output_path(output, candidate["path"])
                image = read_image(path)
                scale = min(320 / image.shape[1], 210 / image.shape[0], 1.)
                thumb = cv2.resize(image, (max(1, round(image.shape[1]*scale)),
                                           max(1, round(image.shape[0]*scale))), interpolation=cv2.INTER_AREA)
                x, y = (i % 3)*320, (i // 3)*240
                canvas[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb
                label = f"{candidate['id']}  {candidate['time']:.3f}s"
                cv2.putText(canvas, label, (x+5, y+232), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (235, 235, 235), 1, cv2.LINE_AA)
                index["candidates"].append({**candidate, "sha256": file_sha256(path)})
            write_png(output / "selection" / f"sheet-{offset//12+1:03d}.png", canvas)
        atomic_json(output / "selection/index.json", index)
        return index


def _selected_scan(output: Path, source_hash: str, selected: list[str]) -> dict:
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate --select-frame IDs")
    index = read_json(output / "selection/index.json")
    if index["source_sha256"] != source_hash:
        raise ValueError("Prepared candidates belong to a different input")
    by_id = {candidate["id"]: candidate for candidate in index["candidates"]}
    if not set(selected) <= by_id.keys():
        raise ValueError("Every --select-frame ID must appear in selection/index.json")
    candidates = [by_id[key] for key in selected]
    for candidate in candidates:
        if file_sha256(safe_output_path(output, candidate["path"])) != candidate["sha256"]:
            raise ValueError("Prepared candidate pixels changed; prepare a fresh output directory")
    return {**index.get("scan", {}), "candidates": candidates, "selection": "operator_selected"}


def run(input_value: str, output: Path, cfg: dict, resume: bool = False,
        selected: list[str] | None = None) -> dict:
    output = output.resolve()
    started = time.monotonic()
    if not selected:
        raise ValueError("Editing requires explicit --select-frame IDs from a prepared candidate index")
    cfg = copy.deepcopy(cfg)
    cfg["run"]["count"] = len(selected)
    reviewed = cfg["ocr"].get("reviewed_frames", {})
    if reviewed and set(reviewed) != set(selected):
        raise ValueError("Reviewed overlay configuration must match the selected frame IDs")
    with output_lock(output):
        _validate_output_paths(output)
        cv2.setNumThreads(int(cfg["run"]["threads"]))
        checkpoint = output / ".checkpoint.json"
        if checkpoint.exists() and not resume:
            raise ValueError("Output already has a run. Use --resume or choose a new output directory")
        if not checkpoint.exists() and any((output / p).exists() for p in ("report.json", "images")):
            raise ValueError("Output contains files without a checkpoint; choose a fresh output directory")
        config_signature = {"version": __version__, "config": semantic_config(cfg),
                            "selected_frames": selected}
        cfg_key = digest(config_signature)
        input_path = resolve_input(input_value, output / ".cache/download", cfg["video"])
        if input_path.is_relative_to(output / ".cache/frames"):
            raise ValueError("Input cannot be inside the candidate cache")
        source_hash = file_sha256(input_path)
        scan = _selected_scan(output, source_hash, selected)
        if checkpoint.exists():
            state = read_json(checkpoint)
            if state["source_sha256"] != source_hash or state["config_hash"] != cfg_key:
                raise ValueError("Resume input/config changed; use a new output directory")
            if state["status"] == "complete":
                audited = quality_only(output)
                if audited["status"] != "qc_passed":
                    raise ValueError("Saved output failed resume audit; inspect qc_report.json")
                result = read_json(output / "report.json")
                result["resumed_complete"] = True
                return result
        else:
            state = {"schema": 1, "version": __version__, "source_sha256": source_hash,
                     "config_hash": cfg_key, "started_at": _now(), "status": "running",
                     "items": {}}
            atomic_json(checkpoint, state)
        for folder in ("images", "originals", "masks", "previews"):
            (output / folder).mkdir(exist_ok=True)
        gateway = GeminiGateway(cfg["gemini"], output / ".cache/gemini")
        detector = OCRDetector(output / ".cache/ocr", cfg["ocr"])
        try:
            hashes = []
            for old in state["items"].values():
                if old["status"] == "passed":
                    for key, expected in old.get("file_sha256", {}).items():
                        if file_sha256(safe_output_path(output, old["files"][key])) != expected:
                            raise ValueError("Checkpoint output files changed; run --qc-only and use a fresh output")
                    hashes.append(old["qc"]["phash"])
            for candidate in scan["candidates"]:
                previous_item = state["items"].get(candidate["id"])
                if previous_item and not (RECOVERABLE & set(previous_item.get("issues", []))):
                    continue
                print(f"process: {candidate['id']} at {candidate['time']:.2f}s; {len(hashes)}/{cfg['run']['count']} passed", flush=True)
                item = _candidate(candidate, output, cfg, detector, gateway, hashes, previous_item)
                item["ranking"] = {"source": "operator_selected"}
                state["items"][candidate["id"]] = item
                if item["status"] == "passed":
                    hashes.append(item["qc"]["phash"])
                atomic_json(checkpoint, state)
            mode = "mock_gemini" if gateway.mock else "gemini"
            if detector.is_mock:
                mode += "+mock_ocr"
            report = {"version": __version__, "status": "complete" if len(hashes) >= cfg["run"]["count"] else "partial",
                      "mode": mode, "requested_count": cfg["run"]["count"],
                      "source_sha256": source_hash, "started_at": state["started_at"], "finished_at": _now(),
                      "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
                      "config": cfg, "scan": {k: v for k, v in scan.items() if k != "candidates"},
                      "api": gateway.stats, "ocr": detector.stats,
                      "items": list(state["items"].values()),
                      "warnings": [*scan.get("warnings", []),
                                   "Review repaired regions visually before delivery; automated checks are local."]}
            if gateway.mock or detector.is_mock:
                report["warnings"].append("Mock enabled: these results do not validate the live provider.")
            if report["status"] == "partial":
                report["warnings"].append("Quality gates did not yield the requested count; no failed images were promoted.")
            recoverable = report["status"] == "partial" and any(
                RECOVERABLE & set(i.get("issues", [])) for i in report["items"])
            state["status"] = "resumable" if recoverable else "complete"
            report["resumable"] = recoverable
            write_report(output, report)
            atomic_json(checkpoint, state)
            return report
        finally:
            gateway.close()
