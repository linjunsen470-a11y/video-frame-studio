"""Safety and actual pixel behaviour, independently generated fixtures."""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from vfs.masks import build_mask, polygon_mask
from vfs.ocr import OCRDetector, OCRUnavailable, select_regions
from vfs.repair import composite_local


def region(text="SUBTITLE", x=80, y=270, width=400, height=35):
    return {"polygon": [[x, y], [x+width, y], [x+width, y+height], [x, y+height]],
            "text": text, "confidence": 0.99}


def text_fixture(background=(100, 110, 120), foreground=(255, 255, 255), outline=True):
    clean = np.full((240, 640, 3), background, np.uint8)
    image = clean.copy()
    if outline:
        cv2.putText(image, "Amazing moment", (46, 187), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (18, 18, 18), 6, cv2.LINE_AA)
        cv2.putText(image, "Amazing moment", (50, 190), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (25, 25, 25), 4, cv2.LINE_AA)
    cv2.putText(image, "Amazing moment", (46, 187), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, foreground, 2, cv2.LINE_AA)
    return clean, image, region("Amazing moment", 43, 160, 280, 38)


@pytest.mark.parametrize("background,foreground,outline", [
    ((100, 110, 120), (255, 255, 255), True),
    ((240, 240, 240), (15, 15, 15), False),
    ((80, 80, 80), (0, 220, 255), True),
])
def test_fine_mask_covers_strokes_outline_shadow(background, foreground, outline):
    clean, image, box = text_fixture(background, foreground, outline)
    mask, info = build_mask(image, [box], {})
    json.dumps(info)
    actual_text = np.max(cv2.absdiff(clean, image), axis=2) > 3
    coverage = np.count_nonzero((mask > 0) & actual_text) / np.count_nonzero(actual_text)
    assert coverage >= 0.995
    assert info["pixels"] < np.count_nonzero(polygon_mask(image.shape, [box], 4)) * 0.95
    assert np.all(mask[:130] == 0)
    assert set(np.unique(mask)).issubset({0, 255})
    assert info["polygon_fallbacks"] == 0


@pytest.mark.parametrize("feather", [0, 1, 3, 7])
def test_composite_exact_outside_support_after_lossless_png(tmp_path, feather):
    random = np.random.default_rng(17)
    original = random.integers(0, 256, (90, 120, 3), dtype=np.uint8)
    malicious_whole_frame_edit = 255-original
    mask = np.zeros((90, 120), np.uint8)
    mask[0:8, 0:8] = 255
    mask[45:55, 75:92] = 255
    output, final = composite_local(original, malicious_whole_frame_edit, mask, feather)
    cv2.imwrite(str(tmp_path / "output.png"), output)
    loaded = cv2.imread(str(tmp_path / "output.png"))
    assert np.array_equal(loaded[final == 0], original[final == 0])
    assert np.array_equal(loaded[mask > 0], malicious_whole_frame_edit[mask > 0])
    assert np.all(final[mask > 0] == 255)
    assert np.count_nonzero(final) <= 2*(17+2*feather)*(17+2*feather)


def test_auto_preserves_static_brand_text_and_tracks_dynamic_overlay():
    brand = region("REAL PRODUCT", 90, 70, 200, 40)
    subtitle = region("New scene today", 80, 285, 410, 32)
    previous = region("Another phrase", 85, 285, 400, 32)
    selection = select_regions([brand, subtitle], (360, 640, 3), {}, [[brand, previous]])
    assert [r["text"] for r in selection["selected"]] == ["New scene today"]
    assert [r["text"] for r in selection["uncertain"]] == ["REAL PRODUCT"]
    static = select_regions([subtitle], (360, 640, 3), {}, [[subtitle]])
    assert not static["selected"]
    assert static["uncertain"]


def test_explicit_protection_overrides_all_and_manual_modes():
    first = region("BRAND", 10, 10, 100, 30)
    second = region("TITLE", 200, 10, 100, 30)
    third = region("CAPTION", 100, 200, 200, 30)
    selection = select_regions([first, second, third], (360, 640, 3), {
        "mode": "all", "keep_words": ["brand"], "protect_regions": [[195, 0, 310, 50]]})
    assert len(selection["protected"]) == 2
    assert selection["selected"][0]["text"] == "CAPTION"
    manual = select_regions([], (360, 640, 3), {
        "mode": "manual", "manual_regions": [[90, 200, 300, 240]], "protect_regions": [[0, 0, 50, 50]]})
    assert len(manual["selected"]) == 1
    assert manual["selected"][0]["manual"] is True


def test_mask_respects_protected_pixel_region():
    _, image, box = text_fixture()
    mask, _ = build_mask(image, [box], {"protect_regions": [[50, 140, 140, 210]]})
    assert np.count_nonzero(mask[140:211, 50:141]) == 0


def test_ocr_mock_is_explicit_cached_and_config_separated(tmp_path):
    _, image, box = text_fixture()
    detector = OCRDetector(tmp_path, {"backend": "mock", "mock_regions": [box]})
    assert detector.is_mock
    assert detector.detect(image)[0]["text"] == "Amazing moment"
    assert detector.detect(image)[0]["text"] == "Amazing moment"
    assert detector.stats["calls"] == 1
    assert detector.stats["cache_hits"] == 1
    changed = OCRDetector(tmp_path, {"backend": "mock", "mock_regions": []})
    assert changed.detect(image) == []
    with pytest.raises(OCRUnavailable, match="disabled"):
        OCRDetector(tmp_path, {"backend": "off"}).detect(image)


def test_unknown_ocr_schema_cannot_mean_no_text(tmp_path):
    detector = OCRDetector(tmp_path, {"backend": "rapidocr"})
    detector.api = "modern"
    detector.engine = lambda image: object()
    with pytest.raises(OCRUnavailable, match="schema"):
        detector.detect(np.full((100, 100, 3), 180, np.uint8))


def test_real_rapidocr_english_cpu_and_cache(tmp_path):
    pytest.importorskip("rapidocr")
    image = np.full((240, 640, 3), 210, np.uint8)
    cv2.putText(image, "HIGH ENERGY SCENE", (45, 160), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 2, cv2.LINE_AA)
    detector = OCRDetector(tmp_path, {"backend": "rapidocr", "threads": 2, "max_side": 960})
    results = detector.detect(image)
    assert results and any("ENERGY" in r["text"].upper() for r in results)
    assert detector.api == "modern" and not detector.is_mock
    polygon = np.asarray(results[0]["polygon"])
    assert polygon[:, 1].min() >= 110 and polygon[:, 1].max() <= 180
    detector.detect(image)
    assert detector.stats["cache_hits"] == 1


def test_real_chinese_bilingual_mask_repair_with_brand_protection(tmp_path):
    pytest.importorskip("rapidocr")
    from PIL import Image, ImageDraw, ImageFont
    fonts = [Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
             Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    available = next((path for path in fonts if path.exists()), None)
    if available is None:
        pytest.skip("Noto CJK font unavailable for bilingual regression")
    font = ImageFont.truetype(str(available), 34)
    small = ImageFont.truetype(str(available), 26)
    clean_pil = Image.new("RGB", (960, 540), (100, 120, 140))
    draw = ImageDraw.Draw(clean_pil)
    draw.rectangle((60, 50, 310, 220), fill=(180, 170, 160))
    draw.text((80, 100), "BRAND 2026", font=small, fill="black")
    test_pil = clean_pil.copy()
    draw = ImageDraw.Draw(test_pil)
    draw.text((240, 400), "高能瞬间 精彩呈现", font=font, fill="white", stroke_width=3, stroke_fill="black")
    draw.text((285, 455), "Amazing moments", font=small, fill="white", stroke_width=2, stroke_fill="black")
    image = np.asarray(test_pil)[:, :, ::-1].copy()
    clean = np.asarray(clean_pil)[:, :, ::-1].copy()
    detector = OCRDetector(tmp_path, {"threads": 2, "max_side": 1280})
    regions = detector.detect(image)
    assert any("高能" in r["text"] for r in regions)
    assert any("Amazing" in r["text"] for r in regions)
    selection = select_regions(regions, image.shape, {"mode": "subtitle", "keep_words": ["BRAND"]})
    assert len(selection["selected"]) == 2
    assert len(selection["protected"]) == 1
    mask, meta = build_mask(image, selection["selected"], {})
    actual = np.any(image != clean, axis=2)
    assert np.all(mask[actual] > 0)
    output, support = composite_local(image, clean, mask)
    assert np.array_equal(output[support == 0], image[support == 0])
    final_text = detector.detect(output)
    assert len(final_text) == 1 and "BRAND" in final_text[0]["text"]


@pytest.mark.parametrize("cut_side", ["left", "right"])
def test_truncated_ocr_bbox_recovers_missing_chinese_glyph(cut_side):
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    if not font_path.exists():
        pytest.skip("Noto CJK font unavailable for truncated glyph regression")
    font = ImageFont.truetype(str(font_path), 28)
    base = Image.new("RGB", (640, 360), (104, 146, 166))
    clean = np.asarray(base)[:, :, ::-1].copy()
    draw = ImageDraw.Draw(base)
    text, xy = "今日新品上市", (240, 260)
    draw.text((242, 263), text, font=font, fill=(5, 5, 5), stroke_width=3, stroke_fill=(5, 5, 5))
    draw.text(xy, text, font=font, fill="white", stroke_width=2, stroke_fill="black")
    image = np.asarray(base)[:, :, ::-1].copy()
    actual = np.any(image != clean, axis=2)
    yy, xx = np.where(actual)
    x1, x2, y1, y2 = int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())
    if cut_side == "left":
        x1 += 23
    else:
        x2 -= 23
    truncated = {"polygon": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], "text": "日新品上市", "confidence": .99}
    mask, meta = build_mask(image, [truncated], {"dilate_px": 2, "shadow_px": 3})
    assert np.all(mask[actual] > 0)
    assert meta["polygon_fallbacks"] == 0
    assert meta["edge_search_inconclusive"] == 0
    assert np.all(mask[:200] == 0)


def test_reviewed_ad_overlay_and_subtitle_preserve_scene_text():
    from vfs.ocr import reviewed_overlay_regions

    frame_shape = (360, 640, 3)
    scene_text = {"polygon": [[420, 30], [560, 30], [560, 58], [420, 58]],
                  "text": "BRAND", "confidence": .99}
    reviewed = [
        {"kind": "ad_label", "box": [30, 20, 80, 44], "text": "广告"},
        {"kind": "subtitle", "box": [220, 280, 412, 312], "text": "员工餐"},
    ]
    config = {"keep_words": ["BRAND"]}
    groups = reviewed_overlay_regions([scene_text], frame_shape, config, reviewed)
    assert [r["reason"] for r in groups["selected"]] == ["reviewed_ad_label", "reviewed_subtitle"]
    assert groups["protected"][0]["text"] == "BRAND"
    mask, _ = build_mask(np.zeros(frame_shape, np.uint8), groups["selected"], {})
    assert mask[30, 55] == mask[300, 316] == 255
    assert mask[44, 490] == 0
