"""Network-free coverage for the two photo gallery delivery forms."""
import json

import numpy as np
from PIL import Image, ImageDraw
import pytest

from catering_visual.__main__ import main as gallery_main
from catering_visual.render import MAX_BYTES, encode_delivery, render_page
from vfs import runtime


@pytest.fixture(autouse=True)
def cloud_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    for name in ("library-files", "scratch", "shared"):
        (workspace / name).mkdir(parents=True)
    monkeypatch.setattr(runtime, "CLOUD_WORKSPACE", workspace)


@pytest.fixture
def photos(tmp_path):
    for name, size, color in (
        ("portrait.png", (360, 640), "#f5a049"),
        ("top.png", (640, 360), "#ca671a"),
        ("bottom.png", (640, 360), "#0c6539"),
    ):
        im = Image.new("RGB", size, color)
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, 30, 30), fill="black")
        draw.rectangle((size[0]-30, size[1]-30, size[0], size[1]), fill="white")
        if name == "portrait.png":
            draw.ellipse((90, 200, 270, 440), fill="#643314")
        im.save(tmp_path / name)
    return tmp_path


def test_plan_render_inspect_and_copy(photos):
    copy = photos / "copy.json"
    copy.write_text(json.dumps({"title": "员工餐开盒", "body": "米饭和配菜分格摆放。", "tags": ["员工餐"]}))
    plan, out = photos / "plan.json", photos / "gallery"
    assert gallery_main(["plan", str(photos), "--out", str(plan), "--copy", str(copy),
                         "--page", "portrait.png", "--page", "top.png", "bottom.png"]) == 0
    assert gallery_main(["render", str(plan), "--out", str(out)]) == 0
    audit = ["inspect", str(out), "--plan", str(plan)]
    assert gallery_main(audit) == 0
    for path in (out / "pages").glob("*.jpg"):
        with Image.open(path) as image:
            assert image.size == (900, 1200)
        assert path.stat().st_size <= MAX_BYTES
    post = out / "post.md"
    expected = "# 员工餐开盒\n\n米饭和配菜分格摆放。\n\n#员工餐\n"
    assert post.read_text() == expected
    post.write_text("edited")
    assert gallery_main(audit) != 0
    post.write_text(expected)
    (out/"pages/page-02.jpg").write_bytes(b"broken")
    assert gallery_main(audit) == 2


def test_pair_preserves_both_extents_and_order(photos):
    canvas, _, boxes = render_page({"photos": ["top.png", "bottom.png"]}, photos)
    first, second = boxes
    assert second[1] - first[3] == 12
    assert first[1] > 0 and second[3] < 1200
    for left, top, right, bottom in boxes:
        assert canvas.getpixel((left+5, top+5)) == (0, 0, 0)
        assert canvas.getpixel((right-5, bottom-5)) == (255, 255, 255)
    assert canvas.getpixel((450, first[1]+100)) == (202, 103, 26)
    assert canvas.getpixel((450, second[1]+100)) == (12, 101, 57)
    with pytest.raises(ValueError, match="second landscape frame"):
        render_page({"photos": ["top.png"]}, photos)
    with pytest.raises(ValueError, match="must be horizontal"):
        render_page({"photos": ["top.png", "portrait.png"]}, photos)


def test_portrait_can_keep_full_frame_and_compression_can_reduce_size(photos):
    _, _, [crop_box] = render_page({"photos": ["portrait.png"]}, photos)
    canvas, _, [box] = render_page({"photos": ["portrait.png"], "fit": "contain"}, photos)
    assert crop_box == (0, 0, 900, 1200)
    assert box[0] > 0 and box[1] == 0 and box[3] == 1200
    assert canvas.getpixel((box[0]+5, 5)) == (0, 0, 0)
    noise = Image.fromarray(np.random.default_rng(2).integers(0, 256, (1200, 900, 3), dtype=np.uint8))
    data, (width, height), _ = encode_delivery(noise)
    assert width < 900 and width * 4 == height * 3
    assert len(data) <= MAX_BYTES
