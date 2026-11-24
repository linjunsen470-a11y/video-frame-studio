"""Fixed processing policy: the agent supplies inputs, not ad-hoc decisions."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from . import __version__
from .config import semantic_config
from .gemini import GeminiGateway, GeminiError, BudgetExceeded
from .masks import build_mask
from .ocr import OCRDetector, OCRUnavailable, select_regions
from .quality import (background_complexity, check_saved_png, polygon_mask, phash)
from .repair import conventional_inpaint, temporal_restore, onnx_inpaint, composite_local
from .report import write_report
from .util import (atomic_json, read_json, file_sha256, digest, read_image, resize_long,
                   write_png, output_lock, safe_output_path)
from .video import resolve_input, scan_candidates


RECOVERABLE = {"api_budget_exhausted", "gemini_unavailable", "semantic_classification_unavailable"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preview(original, final, mask, path: Path):
    overlay = original.copy()
    marked = mask > 0
    overlay[marked] = (original[marked] * .45 + np.array([30, 40, 240]) * .55).astype(np.uint8)
    panels = [resize_long(im, 480).copy() for im in (original, overlay, final)]
    labels = ("ORIGINAL", "FINAL MASK", "RESULT")
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


def _classify_uncertain(image, groups: dict, gateway: GeminiGateway, warnings: list[str]):
    if not groups["uncertain"] or not gateway.enabled:
        return False
    try:
        answers = []
        for offset in range(0, len(groups["uncertain"]), 24):
            chunk = gateway.classify_regions(image, groups["uncertain"][offset:offset+24])
            if isinstance(chunk, dict):
                chunk = chunk.get("regions", [])
            answers += [{**a, "index": a["index"] + offset} for a in chunk]
        if isinstance(answers, dict):
            answers = answers.get("regions", [])
        classified = {int(x["index"]): x for x in answers}
        unresolved = []
        for index, region in enumerate(groups["uncertain"]):
            item = classified.get(index, {})
            category = item.get("category") if item.get("confidence", 0) >= .90 else "uncertain"
            if category == "overlay":
                groups["selected"].append({**region, "reason": "gemini_semantic_overlay"})
            elif category == "scene":
                groups["protected"].append({**region, "reason": "gemini_semantic_scene_text"})
            else:
                unresolved.append(region)
        groups["uncertain"] = unresolved
        return False
    except GeminiError as exc:
        warnings.append(f"Semantic text classification unavailable: {type(exc).__name__}")
        return True


def _candidate(candidate: dict, output: Path, cfg: dict, detector: OCRDetector,
               gateway: GeminiGateway, existing_hashes: list[str], previous: dict | None = None) -> dict:
    original_path = safe_output_path(output, candidate["path"])
    original = read_image(original_path)
    item = {k: candidate[k] for k in ("id", "time", "frame_index", "scene", "metrics")}
    item.update({"status": "review", "attempts": list((previous or {}).get("attempts", [])), "warnings": []})
    regions = detector.detect(original)
    neighbors, temporal_regions = [], []
    for donor in candidate["neighbors"]:
        if donor["scene"] != candidate["scene"]:
            continue
        donor_image = read_image(safe_output_path(output, donor["path"]))
        donor_regions = detector.detect(donor_image)
        temporal_regions.append(donor_regions)
        # Mask every donor OCR polygon, including real text: no text may be imported.
        donor_mask = polygon_mask(donor_image.shape, donor_regions, 6)
        neighbors.append({"image": donor_image, "mask": donor_mask, "same_scene": True})
    groups = select_regions(regions, original.shape, cfg["ocr"], temporal_regions)
    classification_failed = _classify_uncertain(original, groups, gateway, item["warnings"])
    item["text_regions"] = groups
    if groups["uncertain"] and cfg["qc"]["require_no_uncertain_text"]:
        item["issues"] = ["uncertain_text_origin"] + (["semantic_classification_unavailable"] if classification_failed else [])
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
        complexity = background_complexity(original, mask)
        trial.update({"mask": mask_meta, "background": complexity})
        simple = (complexity["edge_density"] <= cfg["repair"]["simple_edge_density"]
                  and complexity["gray_std"] <= cfg["repair"]["simple_gray_std"])
        generated, proposal, method = False, None, "unchanged"
        try:
            if mask_meta["area_fraction"] > cfg["mask"]["max_area_fraction"]:
                item["issues"] = ["mask_too_large"]
                break
            if not np.any(mask):
                if selected:
                    item["issues"] = ["selected_text_has_empty_mask"]
                    break
                proposal = original.copy()
            elif simple and attempt == 0:
                method = "level1_telea"
                proposal = conventional_inpaint(original, mask, cfg["repair"]["radius"])
            else:
                if cfg["repair"]["temporal"] and neighbors:
                    proposal_temporal, evidence = temporal_restore(original, mask, neighbors, cfg["repair"])
                    trial["temporal"] = evidence
                    if evidence.get("success"):
                        proposal, method = proposal_temporal, "level2_temporal"
                if proposal is None and cfg["repair"]["onnx_model"]:
                    try:
                        proposal = onnx_inpaint(original, mask, cfg["repair"]["onnx_model"])
                        generated, method = True, "level3_onnx"
                    except (ValueError, RuntimeError, OSError) as exc:
                        trial["onnx_error"] = type(exc).__name__
                if proposal is None and simple and attempt < 2:
                    method = "level1_navier_stokes"
                    proposal = conventional_inpaint(original, mask,
                                   cfg["repair"]["radius"] + attempt, "ns")
                if (proposal is None and cfg["repair"]["gemini_fallback"] and gateway.enabled):
                    proposal = gateway.edit_crop(original, mask, attempt)
                    generated, method = True, "level4_gemini"
                    trial["gemini_crops"] = gateway.last_edit_info
                if proposal is None and (simple or cfg["repair"]["allow_complex_cv"]):
                    method = "level1_telea_retry"
                    proposal = conventional_inpaint(original, mask, cfg["repair"]["radius"] + attempt)
                if proposal is None:
                    item["issues"] = ["complex_background_needs_model_or_review"]
                    break
            final, final_mask = composite_local(original, proposal, mask, cfg["mask"]["feather_px"])
            # Protection applies to the entire feather support, not just the initial mask.
            final[protection] = original[protection]
            final_mask[protection] = 0
            stage = output / ".cache/qc"
            staged_image, staged_mask = stage / f"{item['id']}.png", stage / f"{item['id']}_mask.png"
            write_png(staged_image, final)
            write_png(staged_mask, final_mask)
            qc = check_saved_png(original_path, staged_image, staged_mask, selected, detector,
                                 cfg["qc"], prior_hashes=existing_hashes,
                                 duplicate_distance=cfg["selection"]["phash_distance"],
                                 protected_regions=groups["protected"])
            if mask_meta.get("edge_search_inconclusive", 0):
                qc["issues"].append("mask_edge_uncertain")
            if generated and cfg["qc"]["review_ai_edits"]:
                if gateway.enabled:
                    visual = gateway.review_edit(original, final, final_mask)
                    qc["visual_review"] = visual
                    if not visual["passed"]:
                        qc["issues"].append("visual_review_failed")
                else:
                    qc["issues"].append("visual_review_unavailable")
            qc["passed"] = not qc["issues"]
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
            staged_image.unlink(missing_ok=True)
            staged_mask.unlink(missing_ok=True)
            if "duplicate_output" in qc["issues"] or "outside_mask_pixels_changed" in qc["issues"]:
                break
        except GeminiError as exc:
            trial.update({"method": method, "error": type(exc).__name__})
            item["attempts"].append(trial)
            item["issues"] = ["api_budget_exhausted" if isinstance(exc, BudgetExceeded) else "gemini_unavailable"]
            break
    if item.get("qc") and not item.get("issues"):
        item["status"] = "rejected"
    if cfg["run"]["previews"]:
        folder = output / "previews"
        folder.mkdir(exist_ok=True)
        # Review preview includes OCR bounds; it never enters images/.
        preview = resize_long(original.copy(), 960)
        scale = preview.shape[1] / original.shape[1]
        for region in selected + groups["uncertain"]:
            cv2.polylines(preview, [np.rint(np.asarray(region["polygon"]) * scale).astype(np.int32)],
                          True, (0, 80, 255), 2)
        write_png(folder / f"{item['id']}_review.png", preview)
    return item


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


def run(input_value: str, output: Path, cfg: dict, resume: bool = False) -> dict:
    output = output.resolve()
    started = time.monotonic()
    with output_lock(output):
        # A user-supplied output tree must not redirect managed writes elsewhere.
        for name in (".cache", "images", "originals", "masks", "previews", ".checkpoint.json"):
            child = output / name
            if child.is_symlink() or (child.is_dir() and any(p.is_symlink() for p in child.rglob("*"))):
                raise ValueError("Symlinks are not allowed inside managed output paths")
        cv2.setNumThreads(int(cfg["run"]["threads"]))
        checkpoint = output / ".checkpoint.json"
        if checkpoint.exists() and not resume:
            raise ValueError("Output already has a run. Use --resume or choose a new output directory")
        if not checkpoint.exists() and any((output / p).exists() for p in ("report.json", "images")):
            raise ValueError("Output contains files without a checkpoint; choose a fresh output directory")
        cfg_key = digest({"version": __version__, "config": semantic_config(cfg)})
        input_path = resolve_input(input_value, output / ".cache/download", cfg["video"])
        if input_path.is_relative_to(output / ".cache/frames"):
            raise ValueError("Input cannot be inside the candidate cache")
        source_hash = file_sha256(input_path)
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
                     "items": {}, "ranking": {}, "warnings": []}
            atomic_json(checkpoint, state)
        for folder in ("images", "originals", "masks", "previews"):
            (output / folder).mkdir(exist_ok=True)
        gateway = GeminiGateway(cfg["gemini"], output / ".cache/gemini")
        detector = OCRDetector(output / ".cache/ocr", cfg["ocr"])
        try:
            video_cfg = {**cfg["video"], "threads": cfg["run"]["threads"]}
            scan = scan_candidates(input_path, output, video_cfg, cfg["selection"],
                                   cfg["run"]["count"], source_hash, resume)
            candidates = scan["candidates"]
            if gateway.enabled:
                batch_size = cfg["gemini"]["rank_batch_size"]
                unranked = [c for c in candidates if c["id"] not in state["ranking"]]
                for start in range(0, len(unranked), batch_size):
                    batch = unranked[start:start + batch_size]
                    payload = [{"id": c["id"], "image": resize_long(read_image(output / c["path"]), 768)}
                               for c in batch]
                    try:
                        state["ranking"].update(gateway.rank_frames(payload))
                        atomic_json(checkpoint, state)
                    except GeminiError as exc:
                        state["warnings"].append(f"Semantic ranking incomplete ({type(exc).__name__}); technical order used where missing")
                        break
            weight = cfg["selection"]["semantic_weight"]
            for candidate in candidates:
                technical = candidate["metrics"]["technical_score"]
                semantic = state["ranking"].get(candidate["id"], {}).get("score", technical)
                candidate["rank_score"] = (1 - weight) * technical + weight * semantic
            candidates.sort(key=lambda c: (-c["rank_score"], c["time"]))
            hashes = []
            for old in state["items"].values():
                if old["status"] == "passed":
                    for key, expected in old.get("file_sha256", {}).items():
                        if file_sha256(safe_output_path(output, old["files"][key])) != expected:
                            raise ValueError("Checkpoint output files changed; run --qc-only and use a fresh output")
                    hashes.append(old["qc"]["phash"])
            for candidate in candidates:
                if len(hashes) >= cfg["run"]["count"]:
                    break
                previous_item = state["items"].get(candidate["id"])
                if previous_item and not (RECOVERABLE & set(previous_item.get("issues", []))):
                    continue
                print(f"process: {candidate['id']} at {candidate['time']:.2f}s; {len(hashes)}/{cfg['run']['count']} passed", flush=True)
                try:
                    item = _candidate(candidate, output, cfg, detector, gateway, hashes, previous_item)
                except OCRUnavailable:
                    # No OCR = no claim of no subtitles. Keep remaining work resumable.
                    raise
                item["ranking"] = state["ranking"].get(candidate["id"], {"source": "technical_only"})
                state["items"][candidate["id"]] = item
                if item["status"] == "passed":
                    hashes.append(item["qc"]["phash"])
                atomic_json(checkpoint, state)
            mode = "mock_gemini" if gateway.mock else ("gemini" if gateway.enabled else "offline_deterministic")
            if detector.is_mock:
                mode += "+mock_ocr"
            report = {"version": __version__, "status": "complete" if len(hashes) >= cfg["run"]["count"] else "partial",
                      "mode": mode, "requested_count": cfg["run"]["count"],
                      "source_sha256": source_hash, "started_at": state["started_at"], "finished_at": _now(),
                      "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
                      "config": cfg, "scan": {k: v for k, v in scan.items() if k != "candidates"},
                      "ranking": state["ranking"], "api": gateway.stats, "ocr": detector.stats,
                      "items": list(state["items"].values()),
                      "warnings": list(dict.fromkeys(state["warnings"] + scan["warnings"]))}
            if mode.startswith("mock") or detector.is_mock:
                report["warnings"].append("Mock was enabled: semantic scores/generative quality are NOT real API validation.")
            if not gateway.enabled:
                report["warnings"].append("Offline mode: composition, identity and visual naturalness were not assessed by Gemini.")
            if report["status"] == "partial":
                report["warnings"].append("Quality gates did not yield the requested count; no failed images were promoted.")
            recoverable = report["status"] == "partial" and any(
                RECOVERABLE & set(i.get("issues", [])) for i in report["items"])
            state["status"] = "resumable" if recoverable else "complete"
            report["resumable"] = recoverable
            write_report(output, report)
            atomic_json(checkpoint, state)
            if not cfg["run"]["keep_intermediates"] and not recoverable:
                for dirname in ("frames", "qc", "download"):
                    shutil.rmtree(output / ".cache" / dirname, ignore_errors=True)
            return report
        finally:
            gateway.close()
