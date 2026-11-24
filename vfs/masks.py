"""Fine text masks, with bounded growth for antialiasing, outline and shadow."""
from __future__ import annotations

import cv2
import numpy as np


def polygon_mask(shape: tuple, regions: list[dict], padding: int = 0) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for region in regions:
        points = np.asarray(region["polygon"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            continue
        points[:, 0] = np.clip(points[:, 0], 0, shape[1]-1)
        points[:, 1] = np.clip(points[:, 1], 0, shape[0]-1)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    if padding:
        radius = max(0, int(padding))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
    return mask


def _smooth_background_strokes(crop: np.ndarray, polygon: np.ndarray,
                               threshold: int) -> np.ndarray | None:
    """Estimate a smooth background from the clean exterior, not from glyphs.

    Combining bright/dark hats alone can join an entire outlined subtitle into
    one component. A robust affine colour plane preserves letter gaps on flat
    or gently graded backgrounds. It is used only when border residuals agree.
    """
    height, width = crop.shape[:2]
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    design = np.stack((np.ones_like(xx), xx, yy), axis=-1)
    border = np.zeros((height, width), bool)
    border[:2] = border[-2:] = True
    border[:, :2] = border[:, -2:] = True
    border &= polygon == 0
    if np.count_nonzero(border) < 20:
        return None
    locations = design[border]
    values = crop[border].astype(np.float64)
    use = np.ones(len(values), bool)
    for _ in range(3):
        plane = np.linalg.lstsq(locations[use], values[use], rcond=None)[0]
        errors = np.max(np.abs(locations @ plane-values), axis=1)
        use = errors <= max(6.0, float(np.median(errors))*3)
        if np.count_nonzero(use) < len(values)*0.75:
            return None
    if float(np.percentile(errors, 90)) > 8.0:
        return None
    predicted = design @ plane
    residual = np.max(np.abs(crop.astype(np.float32)-predicted), axis=2)
    return ((residual >= min(10, threshold)) & (polygon > 0)).astype(np.uint8)*255


def _stroke_mask(crop: np.ndarray, polygon: np.ndarray, config: dict,
                 attempt: int) -> tuple[np.ndarray, bool, str]:
    threshold = max(8, int(config.get("contrast_threshold", 22))-attempt*4)
    smooth = _smooth_background_strokes(crop, polygon, threshold)
    if smooth is not None and np.count_nonzero(smooth) >= 4:
        return smooth, False, "smooth_background_residual"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Morphological bright/dark hats retain thin strokes of either polarity.
    kernel_size = max(5, min(41, int(crop.shape[0]*0.55) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Per-channel contrast also finds saturated glyphs with similar luminance.
    median = cv2.medianBlur(crop, 5)
    colour_residual = np.max(cv2.absdiff(crop, median), axis=2)
    seed = (((bright >= threshold) | (dark >= threshold) |
             (colour_residual >= threshold)) & (polygon > 0)).astype(np.uint8)*255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    filtered = np.zeros_like(seed)
    area = max(1, np.count_nonzero(polygon))
    for label in range(1, count):
        size = stats[label, cv2.CC_STAT_AREA]
        if 2 <= size <= area*0.75:
            filtered[labels == label] = 255
    # Fill enclosed glyph cores; leave inter-letter spaces unselected.
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) <= area*0.75:
            cv2.drawContours(filtered, [contour], -1, 255, cv2.FILLED)
    coverage = np.count_nonzero(filtered) / area
    fallback = bool(coverage < float(config.get("min_stroke_coverage", 0.015)))
    if fallback:
        # An OCR-positive low contrast region is not silently left untouched.
        filtered = polygon.copy()
    return filtered, fallback, "morphological_strokes"


def build_mask(image: np.ndarray, regions: list[dict], config: dict,
               attempt: int = 0) -> tuple[np.ndarray, dict]:
    """Generate per-glyph masks inside dilated OCR polygons.

    The output is binary uint8. A few extra pixels cover antialiasing/outline
    and shadow. Retry growth is bounded; it never grows to the whole image.
    Ambiguous fine segmentation falls back to the selected OCR polygon and is
    explicitly recorded in metadata. Manual regions are filled by design.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HxWx3 uint8 BGR image")
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    fallback_count = 0
    inconclusive_count = 0
    details = []
    attempt = min(max(0, int(attempt)), 2)
    for region in regions:
        anchor = polygon_mask(image.shape, [region])
        ys, xs = np.where(anchor > 0)
        if not len(xs):
            continue
        glyph_height = int(ys.max()-ys.min()+1)
        # OCR sometimes recognises a line but truncates its first/last glyph.
        # Search up to one glyph beside it; retain actual connected strokes,
        # never blindly fill this expanded search rectangle.
        horizontal = int(xs.max()-xs.min()+1) >= glyph_height*1.5
        search_px = (int(np.ceil(glyph_height*min(0.8, max(0.0, float(config.get("edge_search_ratio", 0.8))))))
                     if horizontal and not region.get("manual") else 0)
        polygon = cv2.dilate(anchor, np.ones((1, search_px*2+1), np.uint8)) if search_px else anchor
        ys, xs = np.where(polygon > 0)
        stroke_grow = max(1, int(config.get("dilate_px", 2))) + attempt
        shadow = max(0, int(config.get("shadow_px", 2)))
        radius = min(max(0, int(config.get("max_padding", 12))), stroke_grow + shadow)
        pad = radius + 2
        x1, x2 = max(0, int(xs.min())-pad), min(width, int(xs.max())+pad+1)
        y1, y2 = max(0, int(ys.min())-pad), min(height, int(ys.max())+pad+1)
        local_polygon = polygon[y1:y2, x1:x2]
        local_anchor = anchor[y1:y2, x1:x2]
        if region.get("manual") or config.get("type", "fine") == "polygon":
            local = local_anchor.copy()
            fallback = False
            strategy = "explicit_polygon"
        else:
            local, fallback, strategy = _stroke_mask(image[y1:y2, x1:x2], local_polygon, config, attempt)
            if search_px and not fallback:
                join = max(2, int(round(glyph_height*0.15)))
                joined = cv2.dilate(local, np.ones((3, 2*join+1), np.uint8))
                _, labels = cv2.connectedComponents(joined, 8)
                anchored_labels = np.unique(labels[(local_anchor > 0) & (local > 0)])
                anchored_labels = anchored_labels[anchored_labels > 0]
                local[~np.isin(labels, anchored_labels)] = 0
            elif fallback:
                # Preserve the bounded original OCR region and expose the
                # inability to assess adjacent truncated glyphs to QC.
                local = local_anchor.copy()
                inconclusive_count += int(search_px > 0)
        if radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
            local = cv2.dilate(local, kernel)
            allowed = cv2.dilate(local_polygon, kernel)
            local = cv2.bitwise_and(local, allowed)
        mask[y1:y2, x1:x2] |= local
        fallback_count += int(fallback)
        details.append({"bbox": [x1, y1, x2, y2], "glyph_height": glyph_height,
                        "padding": radius, "polygon_fallback": fallback,
                        "edge_search_px": search_px, "segmentation": strategy})
    # Protect absolute rectangular regions even if mask dilation approached one.
    for x1, y1, x2, y2 in config.get("protect_regions", []):
        mask[max(0, int(y1)):min(height, int(y2)+1), max(0, int(x1)):min(width, int(x2)+1)] = 0
    return mask, {"pixels": int(np.count_nonzero(mask)),
                  "area_fraction": float(np.count_nonzero(mask)/mask.size),
                  "polygon_fallbacks": fallback_count, "regions": details,
                  "edge_search_inconclusive": inconclusive_count,
                  "attempt": attempt, "algorithm": "bounded_strokes_v2"}
