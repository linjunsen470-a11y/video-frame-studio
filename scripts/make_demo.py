#!/usr/bin/env python3
"""Create a small, entirely synthetic lossless test video and exact pixel truth.

No downloaded footage or bundled font is used. Noto CJK is recommended; on a
machine without a CJK font, supply --font. The generator aborts rather than
silently replacing Chinese characters with missing-glyph boxes.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)
CHINESE = ("今日新品上市", "清晰细节展示", "自然色彩记录", "精选精彩瞬间")
ENGLISH = ("NEW PRODUCT TODAY", "LOOK AT THE DETAILS", "NATURAL COLOR", "A MOMENT TO KEEP")
PALETTES = ((120, 166, 190), (172, 136, 104), (110, 162, 122), (152, 132, 180))


@lru_cache(maxsize=16)
def load_font(path: str, size: int):
    return ImageFont.truetype(path, size)


def find_font(supplied: str | None) -> str:
    if supplied:
        path = Path(supplied)
        if not path.is_file():
            raise ValueError(f"Font does not exist: {path}")
        return str(path)
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise ValueError("A Chinese-capable font is required; install fonts-noto-cjk or use --font PATH")


def bbox_polygon(box: tuple[int, int, int, int] | list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def render_frame(index: int, fps: int, width: int, height: int, font_path: str):
    """Return original, captioned, subtitle mask, region metadata (PIL images)."""
    scale = width / 640.0
    scene = (index // (fps * 2)) % len(PALETTES)
    tick = int((index / fps) * 2)
    time_seconds = index / fps
    yy, xx = np.mgrid[:height, :width]
    base = np.zeros((height, width, 3), dtype=np.float32)
    for channel, color in enumerate(PALETTES[scene]):
        base[:, :, channel] = color + 14 * xx / width - 12 * yy / height
    # Small visual texture in the upper image; caption backgrounds stay smooth.
    texture = (np.sin(xx / 19.0) * np.cos(yy / 27.0) * 3)[:, :, None]
    base = np.clip(base + texture, 0, 255).astype(np.uint8)
    clean = Image.fromarray(base)
    draw = ImageDraw.Draw(clean)
    font_brand = load_font(font_path, max(14, round(24 * scale)))
    font_sign = load_font(font_path, max(13, round(21 * scale)))
    protected = []
    # Scene text: deliberately authentic product/sign lettering to be retained.
    sign_box = (round(25 * scale), round(22 * scale), round(175 * scale), round(64 * scale))
    draw.rounded_rectangle(sign_box, radius=round(7 * scale), fill=(232, 227, 208), outline=(48, 61, 67), width=2)
    draw.text((round(38 * scale), round(29 * scale)), "真实招牌 SHOP", font=font_sign, fill=(30, 42, 48))
    protected.append({"text": "真实招牌 SHOP", "bbox": list(sign_box), "kind": "scene_sign"})

    # Four different still-life layouts with a slowly moving geometric subject.
    jar_x = round((262 + 30 * math.sin(time_seconds * 1.7)) * scale)
    jar_y = round((91 + 5 * math.cos(time_seconds * 1.1)) * scale)
    jar_w, jar_h = round(125 * scale), round(128 * scale)
    draw.ellipse((jar_x - 16, jar_y + jar_h - 2, jar_x + jar_w + 19, jar_y + jar_h + 17), fill=(69, 77, 74))
    draw.rounded_rectangle((jar_x, jar_y, jar_x + jar_w, jar_y + jar_h), radius=round(20 * scale), fill=(239, 213, 162), outline=(52, 67, 58), width=3)
    draw.rounded_rectangle((jar_x + 3, jar_y - 11, jar_x + jar_w - 3, jar_y + 14), radius=8, fill=(55, 65, 62))
    draw.rectangle((jar_x + 7, jar_y + 40, jar_x + jar_w - 7, jar_y + 104), fill=(245, 239, 209))
    draw.text((jar_x + round(13 * scale), jar_y + round(43 * scale)), "BRAND", font=font_brand, fill=(40, 56, 48))
    draw.text((jar_x + round(36 * scale), jar_y + round(76 * scale)), "原味", font=font_sign, fill=(40, 56, 48))
    protected.append({"text": "BRAND 原味", "bbox": [jar_x + 7, jar_y + 40, jar_x + jar_w - 7, jar_y + 104], "kind": "product_label"})
    for i in range(4):
        cx = round((65 + i * 36 + 13 * scene) * scale)
        cy = round((145 + 15 * math.sin(i + time_seconds)) * scale)
        draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=(210 - 20 * i, 68 + 27 * i, 53 + 20 * scene), outline=(40, 50, 45), width=2)
        draw.line((cx, cy + 13, cx - 9, round(229 * scale)), fill=(55, 94, 65), width=3)
    draw.polygon([(round(465 * scale), round(204 * scale)), (round(518 * scale), round(79 * scale)), (round(572 * scale), round(204 * scale))], fill=(216, 191, 109), outline=(56, 62, 57), width=3)
    draw.line((0, round(245 * scale), width, round(245 * scale)), fill=(94, 104, 97), width=2)

    # Render captions at 2x resolution so masks include anti-aliasing and shadow.
    supersample = 2
    overlay = Image.new("RGBA", (width * supersample, height * supersample), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    regions = []
    caption_lines = [
        (CHINESE[tick % len(CHINESE)], round(265 * scale), round(27 * scale), (255, 250, 238, 255)),
        (ENGLISH[tick % len(ENGLISH)], round(309 * scale), round(20 * scale), (255, 224, 65, 255)),
    ]
    # Non-fixed-position marketing overlay in alternate scenes.
    if scene % 2 == 1:
        caption_lines.append(("限时上新", round((76 + (tick % 3) * 9) * scale), round(23 * scale), (255, 122, 68, 255)))

    for line_no, (text, top, size, color) in enumerate(caption_lines):
        font = load_font(font_path, max(12, size) * supersample)
        text_box = odraw.textbbox((0, 0), text, font=font, stroke_width=2 * supersample)
        line_width = text_box[2] - text_box[0]
        if line_no == 2:
            left = round((410 + 6 * (tick % 3)) * scale) * supersample
        else:
            left = (width * supersample - line_width) // 2 + round(4 * math.sin(tick) * scale) * supersample
        anchor = (left - text_box[0], top * supersample - text_box[1])
        layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.text((anchor[0] + 3 * supersample, anchor[1] + 3 * supersample), text, font=font, fill=(14, 18, 21, 225), stroke_width=3 * supersample, stroke_fill=(14, 18, 21, 210))
        ld.text(anchor, text, font=font, fill=color, stroke_width=2 * supersample, stroke_fill=(10, 13, 15, 255))
        overlay = Image.alpha_composite(overlay, layer)
        down_layer = layer.resize((width, height), Image.Resampling.LANCZOS)
        alpha = np.asarray(down_layer.getchannel("A"))
        ys, xs = np.where(alpha > 0)
        if len(xs):
            box = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
            regions.append({"text": text, "bbox": box, "polygon": bbox_polygon(box), "kind": "marketing" if line_no == 2 else "subtitle", "confidence": 1.0})

    overlay = overlay.resize((width, height), Image.Resampling.LANCZOS)
    captioned = Image.alpha_composite(clean.convert("RGBA"), overlay).convert("RGB")
    alpha = np.asarray(overlay.getchannel("A"))
    mask = Image.fromarray(np.where(alpha > 0, 255, 0).astype(np.uint8))
    # Exact changed-pixel mask can be narrower than alpha when colors coincide.
    changed = Image.fromarray(np.where(np.any(np.asarray(clean) != np.asarray(captioned), axis=2), 255, 0).astype(np.uint8))
    return clean, captioned, mask, changed, regions, protected, scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output lossless .mkv video")
    parser.add_argument("--truth-dir", type=Path, help="Optional per-frame clean PNGs, exact subtitle masks and coordinates")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--font")
    args = parser.parse_args()
    if not 1 <= args.duration <= 30 or not 1 <= args.fps <= 30:
        parser.error("Demo duration must be 1..30 seconds and fps 1..30")
    if (args.width, args.height) not in {(640, 360), (1280, 720)}:
        parser.error("Use 640x360 or 1280x720 for this fixture")
    if args.output.suffix.lower() != ".mkv":
        parser.error("Use .mkv: this fixture uses lossless FFV1")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("ffmpeg is required")
    try:
        font_path = find_font(args.font)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    if args.truth_dir:
        args.truth_dir.mkdir(parents=True, exist_ok=True)
        for folder in ("clean", "masks", "changed", "samples"):
            (args.truth_dir / folder).mkdir(exist_ok=True)
    count = round(args.duration * args.fps)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{args.width}x{args.height}", "-framerate", str(args.fps), "-i", "pipe:0", "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", str(args.output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    metadata = {"fixture": "video-frame-studio synthetic demo", "footage_source": "generated geometric artwork; no external footage", "width": args.width, "height": args.height, "fps": args.fps, "frames": count, "duration": count / args.fps, "codec": "FFV1 bgr0 lossless", "frame_index_base": 0, "coordinates": "pixels, x/y; bbox right and bottom exclusive", "records": []}
    try:
        for index in range(count):
            clean, captioned, mask, changed, regions, protected, scene = render_frame(index, args.fps, args.width, args.height, font_path)
            process.stdin.write(captioned.tobytes())
            record = {"frame_index": index, "timestamp": index / args.fps, "scene": scene, "remove": regions, "protect": protected}
            if args.truth_dir:
                stem = f"frame_{index:06d}.png"
                clean.save(args.truth_dir / "clean" / stem)
                mask.save(args.truth_dir / "masks" / stem)
                changed.save(args.truth_dir / "changed" / stem)
                record.update({"clean": f"clean/{stem}", "mask": f"masks/{stem}", "changed": f"changed/{stem}"})
                if index % args.fps == args.fps // 2:
                    captioned.save(args.truth_dir / "samples" / stem)
            metadata["records"].append(record)
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait(timeout=60)
        if returncode:
            raise RuntimeError(f"ffmpeg failed ({returncode}): {stderr[:1000]}")
    except BaseException:
        process.kill()
        process.wait()
        args.output.unlink(missing_ok=True)
        raise
    finally:
        if process.stderr:
            process.stderr.close()
    if args.truth_dir:
        (args.truth_dir / "ground_truth.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "frames": count, "resolution": [args.width, args.height], "duration": count / args.fps, "truth_dir": str(args.truth_dir) if args.truth_dir else None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
