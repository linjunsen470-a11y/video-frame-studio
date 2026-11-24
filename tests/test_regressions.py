"""Regressions found by visual inspection and audit, not implementation mirrors."""
from pathlib import Path
import copy

import cv2
import numpy as np
import pytest

from vfs.config import load_config, validate_config
from vfs.quality import check_image


class EmptyOCR:
    def detect(self, image):
        return []


def test_lone_character_missed_by_ocr_is_not_approved():
    image = np.full((150, 400, 3), (140, 160, 180), np.uint8)
    cv2.putText(image, "T TEXT", (70, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(image, "T TEXT", (70, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (250, 250, 250), 2, cv2.LINE_AA)
    # OCR starts too far right and misses the first isolated letter.
    selected = [{"polygon": [[112, 70], [230, 70], [230, 112], [112, 112]],
                 "text": "TEXT", "confidence": .99}]
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask, (110, 68), (233, 115), 255, -1)
    edited = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    config = {**load_config()["qc"], "max_mask_fraction": .5,
              "min_sharpness_ratio": 0, "min_sharpness": 0}
    qc = check_image(image, edited, mask, selected, EmptyOCR(), config)
    assert not qc["passed"]
    assert "possible_stroke_residual" in qc["issues"]
    assert qc["persistent_stroke_pixels"] > 16
    assert qc["outside_changed_pixels"] == 0


@pytest.mark.parametrize("section,key,value", [
    ("run", "count", 2.5), ("video", "max_cache_mb", float("nan")),
    ("video", "max_input_mb", float("inf")), ("video", "candidates", -1),
    ("qc", "residual_overlap", 1.5), ("mask", "feather_px", -4),
    ("selection", "phash_distance", 99), ("gemini", "rank_batch_size", 0),
])
def test_bad_limits_fail_before_processing(section, key, value):
    config = copy.deepcopy(load_config())
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)
