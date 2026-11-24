"""Read local assets and validate the editable gallery design."""

from hashlib import file_digest
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageOps

from .presets import LAYOUTS, PRESETS, STYLES

FORMATS = {"JPEG", "PNG", "WEBP"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PIXELS = 32_000_000
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")
HEX = re.compile(r"#[0-9a-fA-F]{6}\Z")


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def asset_path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Photo and font paths must be relative to their root")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path outside its root: {name}")
    if not path.is_file():
        raise ValueError(f"Missing file: {name}")
    return path


def photo_image(path):
    with Image.open(path) as image:
        if (image.format not in FORMATS or image.width * image.height > MAX_PIXELS
                or getattr(image, "is_animated", False)):
            raise ValueError(f"Expected a still JPEG, PNG or WebP under {MAX_PIXELS} pixels: {path.name}")
        return ImageOps.exif_transpose(image).convert("RGBA")


def inventory(root):
    if not root.is_dir():
        raise ValueError(f"Asset directory does not exist: {root}")
    assets, errors = [], []
    for candidate in sorted(root.rglob("*")):
        if candidate.suffix.lower() not in EXTENSIONS or not candidate.is_file():
            continue
        name = candidate.relative_to(root).as_posix()
        try:
            path = asset_path(root, name)
            image = photo_image(path)
            assets.append({"src": name, "width": image.width, "height": image.height,
                           "sha256": digest(path), "category": "unknown"})
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            errors.append({"src": name, "error": str(exc)})
    return {"assets": assets, "errors": errors}


def require_keys(value, keys, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    extra = sorted(value.keys() - set(keys.split()))
    if extra:
        raise ValueError(f"Unknown {name} fields: {', '.join(extra)}")


def number(value, low, high, name):
    if type(value) not in (float, int) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be a number between {low} and {high}")


def color(value, name):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError(f"{name} must be an RGB color such as #F5E9D9")


def text_value(value, field):
    if not isinstance(value, str) or len(value) > 300 or any(ord(c) < 32 and c != "\n" for c in value):
        raise ValueError(f"{field} must be text of up to 300 characters")


def validate_theme(theme, name):
    require_keys(theme, "background foreground accent line margin gap radius border shadow headline_size", name)
    for key, value in theme.items():
        if key in {"background", "foreground", "accent", "line"}:
            color(value, key)
        else:
            bounds = (24, 120) if key == "headline_size" else (0, 128)
            number(value, *bounds, key)


def validate_plan(plan):
    require_keys(plan, "version asset_root canvas style theme fonts pages", "plan")
    if type(plan.get("version")) is not int or plan["version"] != 1:
        raise ValueError("Expected plan version 1")
    style = plan.get("style", "real-photo")
    if not isinstance(style, str) or style not in STYLES:
        raise ValueError(f"Unknown style: {style}")
    if not isinstance(plan.get("asset_root", "."), str) or not plan.get("asset_root", "."):
        raise ValueError("asset_root must be a directory path")
    canvas = plan.get("canvas", {})
    require_keys(canvas, "width height", "canvas")
    for coordinate, default in (("width", 1080), ("height", 1440)):
        value = canvas.get(coordinate, default)
        if type(value) is not int or not 320 <= value <= 4096:
            raise ValueError(f"canvas.{coordinate} must be an integer between 320 and 4096")
    validate_theme(plan.get("theme", {}), "theme")
    fonts = plan.get("fonts", {})
    require_keys(fonts, "regular bold", "fonts")
    for role, entry in fonts.items():
        require_keys(entry, "path index", f"fonts.{role}")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError(f"fonts.{role}.path is required")
        index = entry.get("index", 0)
        if type(index) is not int or not 0 <= index <= 64:
            raise ValueError(f"fonts.{role}.index must be an integer between 0 and 64")
    pages = plan.get("pages")
    if not isinstance(pages, list) or not 1 <= len(pages) <= 100:
        raise ValueError("Plan must contain 1–100 pages")
    ids = set()
    for page in pages:
        require_keys(page, "id preset layout style theme label headline caption photos", "page")
        page_id = page.get("id")
        if not isinstance(page_id, str) or not IDENTIFIER.fullmatch(page_id) or page_id.casefold() in ids:
            raise ValueError(f"Invalid or duplicate page ID: {page_id}")
        ids.add(page_id.casefold())
        preset = page.get("preset")
        if preset is not None and (not isinstance(preset, str) or preset not in PRESETS):
            raise ValueError(f"Unknown preset: {preset}")
        layout = page.get("layout", PRESETS.get(preset))
        if not isinstance(layout, str) or layout not in LAYOUTS:
            raise ValueError(f"{page_id}: provide a known layout or preset")
        if preset and layout != PRESETS[preset]:
            raise ValueError(f"{page_id}: preset and layout disagree")
        page_style = page.get("style", style)
        if not isinstance(page_style, str) or page_style not in STYLES:
            raise ValueError(f"{page_id}: unknown style {page_style}")
        validate_theme(page.get("theme", {}), f"{page_id}.theme")
        for key in ("label", "headline", "caption"):
            if key in page:
                text_value(page[key], key)
        photos = page.get("photos")
        count = len(LAYOUTS[layout][1])
        if not isinstance(photos, list) or len(photos) != count:
            raise ValueError(f"{page_id}: {layout} requires {count} photos")
        for photo in photos:
            require_keys(photo, "src fit focal label adjustments", "photo")
            if not isinstance(photo.get("src"), str) or not photo["src"]:
                raise ValueError("photo.src is required")
            if photo.get("fit", "cover") not in ("cover", "contain"):
                raise ValueError("photo.fit must be cover or contain")
            focal = photo.get("focal", [0.5, 0.5])
            if not isinstance(focal, list) or len(focal) != 2:
                raise ValueError("photo.focal must be [x, y]")
            for value in focal:
                number(value, 0, 1, "photo.focal value")
            if "label" in photo:
                text_value(photo["label"], "photo.label")
            adjustments = photo.get("adjustments", {})
            require_keys(adjustments, "brightness contrast saturation", "photo.adjustments")
            for key, value in adjustments.items():
                number(value, 0.85, 1.15, key)
    return plan


def load_plan(path):
    with path.open(encoding="utf-8") as handle:
        return validate_plan(json.load(handle))


def page_layout(page):
    return page["layout"] if "layout" in page else PRESETS[page["preset"]]
