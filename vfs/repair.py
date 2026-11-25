"""Composite a validated edit through a bounded local mask."""
import cv2
import numpy as np


def _validate(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HxWx3 uint8 BGR image")
    if mask.shape != image.shape[:2] or mask.ndim != 2:
        raise ValueError("Mask dimensions must equal image dimensions")
    return (mask > 0).astype(np.uint8)*255


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
