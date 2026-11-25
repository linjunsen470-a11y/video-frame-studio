"""Read local assets and validate the editable gallery design."""

from hashlib import file_digest
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageOps

FORMATS = {"JPEG", "PNG", "WEBP"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PIXELS = 32_000_000
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")


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


def validate_copy(copy):
    require_keys(copy, "title body tags", "copy")
    for key in ("title", "body"):
        text_value(copy.get(key), key)
        if not copy[key].strip():
            raise ValueError(f"copy.{key} is required")
    tags = copy.get("tags")
    if not isinstance(tags, list) or not 1 <= len(tags) <= 8:
        raise ValueError("copy.tags requires 1–8 topics")
    for tag in tags:
        if not isinstance(tag, str) or not re.fullmatch(r"[^#\s]{1,24}", tag):
            raise ValueError("Provide topic names without # or spaces")


def validate_plan(plan):
    if not isinstance(plan, dict) or type(plan.get("version")) is not int or plan["version"] != 2:
        raise ValueError("Expected a version 2 Xiaohongshu plan. Create it with the current plan command.")
    require_keys(plan, "version asset_root copy pages", "plan")
    if not isinstance(plan.get("asset_root"), str) or not plan["asset_root"]:
        raise ValueError("asset_root must be a directory path")
    validate_copy(plan.get("copy"))
    pages = plan.get("pages")
    if not isinstance(pages, list) or not 1 <= len(pages) <= 100:
        raise ValueError("Plan must contain 1–100 pages")
    ids = set()
    for page in pages:
        require_keys(page, "id photos fit focal", "page")
        page_id = page.get("id")
        if not isinstance(page_id, str) or not IDENTIFIER.fullmatch(page_id) or page_id.casefold() in ids:
            raise ValueError(f"Invalid or duplicate page ID: {page_id}")
        ids.add(page_id.casefold())
        photos = page.get("photos")
        if (not isinstance(photos, list) or len(photos) not in (1, 2)
                or any(not isinstance(photo, str) or not photo for photo in photos)):
            raise ValueError(f"{page_id}: provide one portrait or two landscape photos")
        if len(photos) == 2 and ("fit" in page or "focal" in page or photos[0] == photos[1]):
            raise ValueError(f"{page_id}: landscape pages require two distinct, unmodified frames")
        if page.get("fit", "cover") not in ("cover", "contain"):
            raise ValueError("fit must be cover or contain")
        focal = page.get("focal", [0.5, 0.5])
        if not isinstance(focal, list) or len(focal) != 2:
            raise ValueError("focal must be [x, y]")
        for value in focal:
            number(value, 0, 1, "focal value")
    return plan


def load_plan(path):
    with path.open(encoding="utf-8") as handle:
        return validate_plan(json.load(handle))
