"""Render bounded text and photos into separate regions of an opaque PNG."""

from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
from html import escape
import json
import os
import tempfile

from PIL import Image, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .design import asset_path, digest, page_layout, photo_image
from .presets import STYLES
from .presets import rectangles
from .typography import fitted_text, select_font


def plan_size(plan):
    canvas = plan.get("canvas", {})
    return canvas.get("width", 1080), canvas.get("height", 1440)


def page_signature(plan, page):
    shared = {key: value for key, value in plan.items() if key != "pages"}
    payload = json.dumps([shared, page], sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def draw_text(draw, content, xy, width, height, size, scale, color, fonts, base,
              *, bold=False, serif=False, lines=1):
    if not content.strip():
        return 0
    spec = select_font(content, fonts, base, bold=bold, serif=serif)
    value, font, spacing, box = fitted_text(
        draw, content, width, height, max(8, round(size * scale)),
        max(7, round(size * scale * .72)), spec, lines,
    )
    draw.multiline_text((xy[0] - box[0], xy[1] - box[1]), value, font=font, fill=color, spacing=spacing)
    return box[3] - box[1]


def draw_photo(canvas, image, box, photo, style, scale):
    left, top, right, bottom = box
    size = right-left, bottom-top
    for method, enhancer in (("brightness", ImageEnhance.Brightness),
                             ("contrast", ImageEnhance.Contrast),
                             ("saturation", ImageEnhance.Color)):
        if method in photo.get("adjustments", {}):
            image = enhancer(image).enhance(photo["adjustments"][method])
    slot = Image.new("RGBA", size, style["background"])
    fit = photo.get("fit", "cover")
    if fit == "cover":
        fitted = ImageOps.fit(image, size, Image.Resampling.LANCZOS,
                              centering=tuple(photo.get("focal", [0.5, 0.5])))
    else:
        fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    slot.alpha_composite(fitted, ((size[0]-fitted.width)//2, (size[1]-fitted.height)//2))

    radius = max(0, round(style["radius"] * scale))
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0]-1, size[1]-1), radius, fill=255)
    shadow = round(style["shadow"] * scale)
    if shadow:
        pad = shadow * 3
        alpha = Image.new("L", (size[0]+2*pad, size[1]+2*pad))
        alpha.paste(mask.point(lambda value: round(value*.23)), (pad, pad))
        alpha = alpha.filter(ImageFilter.GaussianBlur(shadow))
        canvas.paste("#142D29", (left-pad, top-pad+shadow, right+pad, bottom+pad+shadow), alpha)
    canvas.paste(slot.convert("RGB"), (left, top), mask)
    border = round(style["border"] * scale)
    if border:
        ImageDraw.Draw(canvas).rounded_rectangle((left, top, right-1, bottom-1), radius,
                                                 outline=style["line"], width=border)


def contrast(first, second):
    def luminance(color):
        values = [channel / 255 for channel in ImageColor.getrgb(color)]
        linear = (v / 12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in values)
        return sum(value * weight for value, weight in zip(linear, (.2126, .7152, .0722), strict=True))

    low, high = sorted((luminance(first), luminance(second)))
    return (high+.05) / (low+.05)


def render_page(plan, page, base, root):
    width, height = plan_size(plan)
    scale = min(width / 1080, height / 1440)
    style_name = page.get("style", plan.get("style", "real-photo"))
    style = {**STYLES[style_name], **plan.get("theme", {}), **page.get("theme", {})}
    canvas = Image.new("RGB", (width, height), style["background"])
    draw = ImageDraw.Draw(canvas)
    fonts = plan.get("fonts", {})
    margin = round(style["margin"] * scale)
    gap = round(style["gap"] * scale)
    usable_width = width - margin * 2
    top = margin
    label = page.get("label", "")
    headline = page.get("headline", "")
    caption = page.get("caption", "")
    if label.strip():
        top += draw_text(draw, label, (margin, top), usable_width, round(46*scale),
                         28, scale, style["accent"], fonts, base, bold=True) + round(18*scale)
    if headline.strip():
        top += draw_text(draw, headline, (margin, top), usable_width, round(215*scale),
                         style["headline_size"], scale, style["foreground"], fonts, base,
                         bold=True, serif=style["serif"], lines=2) + round(24*scale)
    if style_name == "editorial" and (label.strip() or headline.strip()):
        draw.line((margin, top, width-margin-1, top), fill=style["accent"], width=max(1, round(3*scale)))
        top += round(16*scale)
    footer = height - margin
    if caption.strip():
        caption_height = round(100*scale)
        footer -= caption_height + round(12*scale)
        draw_text(draw, caption, (margin, footer+round(20*scale)), usable_width, caption_height,
                  30, scale, style["foreground"], fonts, base, lines=2)

    assets, warnings = [], []
    if (headline.strip() or caption.strip()) and contrast(style["foreground"], style["background"]) < 4.5:
        warnings.append("Low text contrast: foreground")
    if (label.strip() or any(photo.get("label", "").strip() for photo in page["photos"])) and contrast(
            style["accent"], style["background"]) < 4.5:
        warnings.append("Low text contrast: accent")
    boxes = rectangles(page_layout(page), margin, top, usable_width, footer-top, gap)
    # Paint all shadows/photos first so labels remain above adjacent shadows.
    labels = []
    for photo, (left, y, right, bottom) in zip(page["photos"], boxes, strict=True):
        slot_label = photo.get("label", "").strip()
        photo_bottom = bottom - (round(45*scale) if slot_label else 0)
        if right-left < 32 or photo_bottom-y < 32:
            raise ValueError("Photo area is too small after spacing and typography")
        path = asset_path(root, photo["src"])
        source = photo_image(path)
        if not source.getchannel("A").getbbox():
            raise ValueError(f"Fully transparent photo: {photo['src']}")
        cropped = (right-left, photo_bottom-y)
        image_scale = (max if photo.get("fit", "cover") == "cover" else min)(
            cropped[0]/source.width, cropped[1]/source.height)
        if image_scale > 1.25:
            warnings.append(f"Low source resolution for slot: {photo['src']}")
        draw_photo(canvas, source, (left, y, right, photo_bottom), photo, style, scale)
        assets.append({"src": photo["src"], "sha256": digest(path), "box": [left, y, right, photo_bottom]})
        if slot_label:
            labels.append((slot_label, left, photo_bottom, right, bottom))
    for label, left, y, right, bottom in labels:
        padding = round(8*scale)
        draw_text(draw, label, (left, y+padding), right-left, bottom-y-padding,
                  26, scale, style["accent"], fonts, base, bold=True)
    if all(low == high for low, high in canvas.getextrema()):
        raise ValueError("Rendered page is blank")
    return canvas, assets, warnings


def atomic_output(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        staging = path.parent / os.path.basename(handle.name)
        try:
            write(handle)
            handle.flush()
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)


def write_json(path, data):
    raw = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()
    atomic_output(path, lambda handle: handle.write(raw))


@contextmanager
def render_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).is_symlink() for name in ("pages", "gallery-report.json", "preview.html", ".gallery.lock")):
        raise ValueError("Managed gallery output paths may not be symlinks")
    with (output / ".gallery.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def write_preview(path, rows):
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>图集预览</title>',
        '<style>body{font:16px system-ui;margin:2rem;background:#f5f2eb;color:#23302a}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:1.5rem}'
        'figure{margin:0}img{width:100%;height:auto;display:block}figcaption{margin:.5rem 0}</style>',
        '<h1>图集预览</h1><main>',
    ]
    for row in rows:
        label = escape(row["id"])
        if row["status"] == "rendered":
            src = escape(row["image"], quote=True)
            parts.append(f'<figure><a href="{src}"><img src="{src}" alt="{label}" loading="lazy"></a>'
                         f'<figcaption>{label}</figcaption></figure>')
        else:
            detail = escape(row.get("error", row["status"]))
            parts.append(f'<p>{label} · {detail}</p>')
    parts.append("</main></html>")
    payload = "\n".join(parts).encode()
    atomic_output(path, lambda handle: handle.write(payload))


def render_gallery(plan, plan_path, output, page_ids):
    base = plan_path.parent
    root = (base / plan.get("asset_root", ".")).resolve()
    all_ids = {page["id"] for page in plan["pages"]}
    selected = set(page_ids) if page_ids else all_ids
    if not selected <= all_ids:
        raise ValueError(f"Unknown page IDs: {', '.join(sorted(selected - all_ids))}")
    report_path = output / "gallery-report.json"
    preview_path = output / "preview.html"
    with render_lock(output):
        managed = {report_path.resolve(), preview_path.resolve(),
                   *((output / f"pages/{page_id}.png").resolve() for page_id in all_ids)}
        sources = {(root / photo["src"]).resolve() for page in plan["pages"] for photo in page["photos"]}
        if plan_path in managed or managed & sources:
            raise ValueError("Managed outputs would overwrite the design or a source photo")
        previous = {}
        if page_ids and report_path.exists():
            previous = {row["id"]: row for row in json.loads(report_path.read_text())["pages"]}
        report = {"version": 1, "pages": [], "warnings": []}
        duplicates = Counter()
        for page in plan["pages"]:
            page_id = page["id"]
            signature = page_signature(plan, page)
            if page_id not in selected:
                prior = previous.get(page_id, {})
                row = (prior if prior.get("signature") == signature else
                       {"id": page_id, "status": "stale", "warnings": ["Page needs rendering"]})
            else:
                row = {"id": page_id, "signature": signature, "style": page.get("style", plan.get("style", "real-photo")),
                       "size": list(plan_size(plan))}
                try:
                    page_image, sources, warnings = render_page(plan, page, base, root)
                    filename = f"pages/{page_id}.png"
                    target = output / filename
                    atomic_output(target, lambda handle: page_image.save(handle, format="PNG", compress_level=6))
                    row.update(status="rendered", image=filename, assets=sources,
                               sha256=digest(target), warnings=warnings)
                except (ValueError, OSError, Image.DecompressionBombError) as exc:
                    row.update(status="failed", error=str(exc))
            report["pages"].append(row)
            duplicates.update(asset["sha256"] for asset in row.get("assets", []))
        if any(count > 1 for count in duplicates.values()):
            report["warnings"].append("The same source photo appears multiple times")
        themes = {(page.get("style", plan.get("style", "real-photo")),
                   json.dumps(page.get("theme", {}), sort_keys=True)) for page in plan["pages"]}
        if len(themes) > 1:
            report["warnings"].append("Pages use more than one style or theme")
        write_json(report_path, report)
        write_preview(preview_path, report["pages"])
    return report


def inspect_gallery(output, plan=None, plan_path=None):
    report = json.loads((output / "gallery-report.json").read_text())
    expected = {page["id"]: page for page in plan["pages"]} if plan else {}
    root = (plan_path.parent / plan.get("asset_root", ".")).resolve() if plan else None
    findings = []
    for row in report["pages"]:
        try:
            if row["status"] != "rendered":
                raise ValueError(row.get("error", row["status"]))
            if plan:
                page = expected.pop(row["id"], None)
                if page is None or row["signature"] != page_signature(plan, page):
                    raise ValueError("Design has changed since rendering")
                for source in row["assets"]:
                    if digest(asset_path(root, source["src"])) != source["sha256"]:
                        raise ValueError(f"Source changed: {source['src']}")
            path = asset_path(output, row["image"])
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "RGB" or list(image.size) != row["size"]:
                    raise ValueError("Expected an opaque PNG of the planned size")
                if all(low == high for low, high in image.getextrema()):
                    raise ValueError("Blank PNG")
            if digest(path) != row["sha256"]:
                raise ValueError("PNG contents changed since rendering")
            findings.append({"id": row["id"], "status": "passed"})
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            findings.append({"id": row["id"], "status": "failed", "error": str(exc)})
    checked = {row["id"] for row in findings}
    findings.extend({"id": page_id, "status": "failed", "error": "Page was not rendered"}
                    for page_id in expected if page_id not in checked)
    return {"pages": findings, "warnings": report["warnings"]}
