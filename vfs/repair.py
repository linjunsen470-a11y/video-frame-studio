"""Bounded CPU repair and pixel-exact compositing; no implicit model downloads."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def _validate(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HxWx3 uint8 BGR image")
    if mask.shape != image.shape[:2] or mask.ndim != 2:
        raise ValueError("Mask dimensions must equal image dimensions")
    return (mask > 0).astype(np.uint8)*255


def conventional_inpaint(image: np.ndarray, mask: np.ndarray, radius: float = 3,
                         method: str = "telea") -> np.ndarray:
    binary = _validate(image, mask)
    if method not in {"telea", "ns"}:
        raise ValueError("method must be telea or ns")
    if not np.any(binary):
        return image.copy()
    return cv2.inpaint(image, binary, max(1, float(radius)),
                       cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS)


def composite_local(original: np.ndarray, edited: np.ndarray, mask: np.ndarray,
                    feather_px: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Blend *only* in the returned binary support mask.

    The original is copied first, and assignment occurs only within support.
    The entire core mask has alpha=1, so original glyphs cannot bleed back in.
    Finite-kernel feather pixels are part of final_mask (including image edges).
    A generated whole-frame result is therefore never exported directly.
    """
    binary = _validate(original, mask)
    if edited.shape != original.shape or edited.dtype != np.uint8:
        raise ValueError("Edited image shape and dtype must equal original")
    radius = max(0, min(64, int(feather_px)))
    if radius:
        alpha = cv2.GaussianBlur(binary.astype(np.float32)/255.0,
                                 (2*radius+1, 2*radius+1), 0,
                                 borderType=cv2.BORDER_CONSTANT)
        alpha[binary > 0] = 1.0
    else:
        alpha = binary.astype(np.float32)/255.0
    support = alpha > 0
    output = original.copy()
    selected_alpha = alpha[support, None]
    output[support] = np.clip(np.rint(original[support].astype(np.float32)*(1-selected_alpha)
                                    + edited[support].astype(np.float32)*selected_alpha), 0, 255).astype(np.uint8)
    return output, support.astype(np.uint8)*255


def _register(reference: np.ndarray, donor: np.ndarray, excluded: np.ndarray,
              max_shift_fraction: float) -> tuple[np.ndarray | None, str]:
    h, w = reference.shape
    usable = (excluded == 0).astype(np.uint8)*255
    # Static backgrounds may contain too few corner features to register.
    difference = cv2.absdiff(reference, donor)[usable > 0]
    if difference.size and float(np.mean(difference)) <= 3.0:
        return np.eye(2, 3, dtype=np.float32), "static"
    points = cv2.goodFeaturesToTrack(reference, maxCorners=240, qualityLevel=0.015,
                                    minDistance=8, mask=usable, blockSize=5)
    if points is None or len(points) < 12:
        return None, "insufficient_registration_features"
    moved, status, _ = cv2.calcOpticalFlowPyrLK(reference, donor, points, None,
                                              winSize=(21, 21), maxLevel=3)
    if moved is None or status is None:
        return None, "optical_flow_failed"
    keep = status.ravel() > 0
    if np.count_nonzero(keep) < 10:
        return None, "insufficient_tracks"
    matrix, inliers = cv2.estimateAffinePartial2D(moved[keep], points[keep],
                                                 method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if matrix is None or inliers is None or float(inliers.mean()) < 0.65:
        return None, "registration_outliers"
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    rotation = abs(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
    if not (0.95 <= scale <= 1.05) or rotation > 5:
        return None, "excessive_transform"
    if abs(matrix[0, 2]) > w*max_shift_fraction or abs(matrix[1, 2]) > h*max_shift_fraction:
        return None, "excessive_motion"
    return matrix.astype(np.float32), "affine"


def temporal_restore(image: np.ndarray, mask: np.ndarray, neighbors: list[dict],
                     config: dict) -> tuple[np.ndarray, dict]:
    """Recover only when registered, OCR-clean donors cover the entire mask.

    Every donor must supply a text mask; a missing mask is not evidence of a
    clean frame. Donor text is dilated and warped with nearest interpolation.
    Global and local-ring photometric checks reject cuts and bad alignment.
    A partially recovered result is never returned as a successful repair.
    """
    binary = _validate(image, mask)
    target = binary > 0
    total = int(target.sum())
    meta = {"success": False, "coverage": 0.0, "accepted_donors": 0, "rejected": []}
    if not total:
        return image.copy(), {**meta, "success": True, "coverage": 1.0}
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ring = cv2.dilate(binary, np.ones((19, 19), np.uint8)) > 0
    ring &= ~target
    output = image.copy()
    covered = np.zeros_like(target)
    best_error = np.full(image.shape[:2], np.inf, np.float32)
    for index, neighbor in enumerate(neighbors[:max(1, int(config.get("max_neighbors", 6)))]):
        donor = neighbor.get("image")
        donor_mask = neighbor.get("mask")
        reason = None
        if not isinstance(donor, np.ndarray) or donor.shape != image.shape:
            reason = "shape_mismatch"
        elif not isinstance(donor_mask, np.ndarray) or donor_mask.shape != binary.shape:
            reason = "missing_verified_donor_mask"
        elif neighbor.get("same_scene") is False:
            reason = "different_scene"
        if reason:
            meta["rejected"].append({"index": index, "reason": reason})
            continue
        clean_donor_mask = cv2.dilate((donor_mask > 0).astype(np.uint8)*255,
                                      np.ones((7, 7), np.uint8))
        exclusion = cv2.bitwise_or(binary, clean_donor_mask)
        donor_gray = cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY)
        matrix, registration = _register(gray, donor_gray, exclusion,
                                          float(config.get("max_shift_fraction", 0.15)))
        if matrix is None:
            meta["rejected"].append({"index": index, "reason": registration})
            continue
        warped = cv2.warpAffine(donor, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        warped_mask = cv2.warpAffine(clean_donor_mask, matrix, (width, height),
                                     flags=cv2.INTER_NEAREST, borderValue=255)
        valid = cv2.warpAffine(np.full((height, width), 255, np.uint8), matrix,
                               (width, height), flags=cv2.INTER_NEAREST, borderValue=0) > 0
        valid &= warped_mask == 0
        difference = np.mean(cv2.absdiff(image, warped), axis=2)
        comparison = valid & ~target
        local = comparison & ring
        if int(comparison.sum()) < image.shape[0]*image.shape[1]*0.35 or local.sum() < 12:
            reason = "insufficient_clean_comparison"
        else:
            global_error = float(np.mean(difference[comparison]))
            local_error = float(np.mean(difference[local]))
            if global_error > float(config.get("max_global_error", 12.0)):
                reason = "scene_or_alignment_mismatch"
            elif local_error > float(config.get("max_ring_error", 10.0)):
                reason = "local_motion_mismatch"
        if reason:
            meta["rejected"].append({"index": index, "reason": reason})
            continue
        usable = valid & target & (local_error < best_error)
        output[usable] = warped[usable]
        best_error[usable] = local_error
        covered |= usable
        meta["accepted_donors"] += 1
    meta["coverage"] = float(covered[target].sum()/total)
    meta["success"] = bool(np.all(covered[target]))
    if not meta["success"]:
        meta["reason"] = "no_complete_verified_background"
        return image.copy(), meta
    return output, meta


@lru_cache(maxsize=1)
def _onnx_session(path: str, modified_ns: int):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def onnx_inpaint(image: np.ndarray, mask: np.ndarray, model_path: str | Path) -> np.ndarray:
    """Opt-in LaMa-compatible ONNX adapter; never downloads model weights.

    Contract: float32 inputs ``image`` RGB NCHW [0,1], ``mask`` N1HW [0,1],
    output RGB NCHW [0,1]. Spatial dimensions may be fixed or dynamic; fixed
    models get a contextual crop resized to that resolution. Supply an export
    with this exact contract and verify its licence separately.
    """
    binary = _validate(image, mask)
    if not np.any(binary):
        return image.copy()
    path = Path(model_path).expanduser().resolve(strict=True)
    try:
        session = _onnx_session(str(path), path.stat().st_mtime_ns)
    except Exception as error:
        raise RuntimeError(f"Unable to load ONNX inpainting model: {type(error).__name__}") from error
    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {"image", "mask"}:
        raise ValueError("ONNX model must have exactly image and mask inputs")
    if any(item.type != "tensor(float)" for item in inputs.values()):
        raise ValueError("ONNX image/mask inputs must use float32")
    ys, xs = np.where(binary > 0)
    padding = max(32, int(max(xs.max()-xs.min(), ys.max()-ys.min())*0.25))
    x1, x2 = max(0, xs.min()-padding), min(image.shape[1], xs.max()+padding+1)
    y1, y2 = max(0, ys.min()-padding), min(image.shape[0], ys.max()+padding+1)
    crop, local_mask = image[y1:y2, x1:x2], binary[y1:y2, x1:x2]
    shape = inputs["image"].shape
    if len(shape) != 4 or shape[1] != 3:
        raise ValueError("ONNX image input must be NCHW RGB")
    target_h = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else ((crop.shape[0]+7)//8)*8
    target_w = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else ((crop.shape[1]+7)//8)*8
    if target_h*target_w > 2048*2048:
        raise ValueError("ONNX crop exceeds CPU memory safeguard (4 megapixels)")
    resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None]/255.0
    mask_tensor = (cv2.resize(local_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)[None, None]
    try:
        result = np.asarray(session.run(None, {"image": tensor, "mask": mask_tensor})[0])
    except Exception as error:
        raise RuntimeError(f"ONNX inpainting inference failed: {type(error).__name__}") from error
    if result.shape != tensor.shape or not np.isfinite(result).all():
        raise ValueError("Invalid ONNX output shape or non-finite values")
    if float(result.min()) < -0.05 or float(result.max()) > 1.05:
        raise ValueError("ONNX output must be RGB float32 in [0,1]")
    rgb = np.clip(np.rint(result[0].transpose(1, 2, 0)*255), 0, 255).astype(np.uint8)
    repaired = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    output = image.copy()
    output[y1:y2, x1:x2] = repaired
    return output
