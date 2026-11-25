"""Network-free coverage of the downstream asset-to-gallery workflow."""

import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw
import pytest

from catering_visual.__main__ import main as gallery_main
from catering_visual.design import digest
from catering_visual.presets import PRESETS, STYLES, rectangles
from catering_visual.typography import coverage, select_font, system_font
from vfs import runtime


@pytest.fixture
def main(monkeypatch, tmp_path):
    if sys.platform != "linux":
        pytest.skip("gallery processing requires Linux")
    workspace = tmp_path / "workspace"
    for name in ("library-files", "scratch", "shared"):
        (workspace / name).mkdir(parents=True)
    monkeypatch.setattr(runtime, "CLOUD_WORKSPACE", workspace)
    return gallery_main


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    for i in range(4):
        image = Image.new("RGB", (630, 480), (70 + 25*i, 112, 70))
        draw = ImageDraw.Draw(image)
        draw.ellipse((20 + 80*i, 80, 250 + 80*i, 305), fill=(220, 145, 60 + 30*i))
        image.save(root / f"p{i}.jpg")
    return root


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_cli_plan_render_repeat_revise_and_audit(assets, tmp_path, main):
    inventory = tmp_path / "inventory.json"
    assert main(["analyze", str(assets), "--out", str(inventory)]) == 0
    assert len(read(inventory)["assets"]) == 4
    original = digest(assets / "p0.jpg")
    plan = tmp_path / "design.json"
    out = tmp_path / "gallery"
    render = ["render", str(plan), "--out", str(out)]
    inspect = ["inspect", str(out), "--plan", str(plan)]
    assert main(["plan", str(assets), "--out", str(plan), "--page", "food-hero", "p0.jpg",
                 "--page", "ingredient-mosaic", "p1.jpg", "p2.jpg", "p3.jpg"]) == 0
    assert main(render) == 0
    assert main(inspect) == 0
    assert (out / "preview.html").read_text().count("<img ") == 2
    assert read(out / "gallery-report.json")["warnings"] == []
    hero, mosaic = out / "pages/page-01.png", out / "pages/page-02.png"
    before = (digest(hero), digest(mosaic))
    assert main(render) == 0
    assert before == (digest(hero), digest(mosaic))

    updated = read(plan)
    updated["pages"][1]["style"] = "corporate-clean"
    updated["pages"][1]["photos"][0]["focal"] = [0.1, 0.8]
    assert not updated["pages"][0].get("style")
    write(plan, updated)
    assert main(inspect) == 2
    assert main([*render, "--page", "page-02"]) == 0
    assert main(inspect) == 0
    assert digest(hero) == before[0]
    assert digest(mosaic) != before[1]
    assert digest(assets / "p0.jpg") == original

    shutil.copyfile(assets / "p3.jpg", assets / "p0.jpg")
    assert main(inspect) == 2
    hero.write_bytes(b"broken")
    assert main(["inspect", str(out)]) == 2


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_keep_photo_slots_in_bounds_with_gutters(preset):
    boxes = rectangles(PRESETS[preset], 51, 140, 978, 1250, 18)
    assert all(51 <= l < r <= 1029 and 140 <= t < b <= 1390 for l, t, r, b in boxes)
    for i, (l, t, r, b) in enumerate(boxes):
        assert all(r <= x or u <= l or b <= y or v <= t for x, y, u, v in boxes[i+1:])


def test_missing_asset_and_duplicate_photo_are_reported(assets, tmp_path, main):
    plan = tmp_path / "design.json"
    out = tmp_path / "gallery"
    assert main(["plan", str(assets), "--out", str(plan), "--page", "food-grid-4",
                 "p0.jpg", "p0.jpg", "p1.jpg", "p2.jpg"]) == 0
    command = ["render", str(plan), "--out", str(out)]
    assert main(command) == 0
    assert read(out / "gallery-report.json")["warnings"] == ["The same source photo appears multiple times"]
    (assets / "p2.jpg").unlink()
    assert main(command) == 2
    assert read(out / "gallery-report.json")["pages"][0]["status"] == "failed"
    assert "<img " not in (out / "preview.html").read_text()


@pytest.mark.parametrize("style", STYLES)
def test_chinese_text_and_overflow(assets, tmp_path, style, main):
    if not shutil.which("fc-match"):
        pytest.skip("fontconfig not installed")
    path, font_index = system_font("Noto Sans CJK SC:lang=zh-cn")
    if not any(left <= ord("食") <= right for left, right in coverage(path, font_index)):
        pytest.skip("Simplified Chinese font not installed")
    # Prefer a custom font, falling back to system CJK when glyphs are unavailable.
    latin, _ = system_font("DejaVu Sans")
    shutil.copyfile(latin, tmp_path / "latin.ttf")
    chosen = select_font("食堂", {"regular": {"path": "latin.ttf"}}, tmp_path)
    assert Path(chosen[0]).name != "latin.ttf"
    plan = tmp_path / "design.json"
    out = tmp_path / "gallery"
    value = {"version": 1, "asset_root": str(assets), "style": style,
             "pages": [{"id": "cover", "layout": "single-hero", "label": "现场实拍",
                        "headline": "走进食堂", "caption": "公开照片版式示例",
                        "photos": [{"src": "p0.jpg", "label": "供餐现场"}]}]}
    write(plan, value)
    command = ["render", str(plan), "--out", str(out)]
    assert main(command) == 0
    with Image.open(out / "pages/cover.png") as image:
        assert image.size == (1080, 1440)
    value["pages"][0]["headline"] *= 70
    write(plan, value)
    assert main(command) == 2
    assert "Text overflow" in read(out / "gallery-report.json")["pages"][0]["error"]
