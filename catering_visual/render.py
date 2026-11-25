"""Render the two Xiaohongshu page forms and verify delivery files."""
from collections import Counter
from contextlib import contextmanager
from html import escape
import fcntl
import hashlib
import io
import json
import os
import tempfile

from PIL import Image, ImageOps

from .design import asset_path, digest, photo_image


SIZE = (900, 1200)
MAX_BYTES = 300_000
GAP = 12
BACKGROUND = "#f5f5f2"


def page_signature(plan, page):
    payload = json.dumps([plan["version"], plan["asset_root"], page], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _opaque(image):
    background = Image.new("RGB", image.size, BACKGROUND)
    background.paste(image, mask=image.getchannel("A"))
    return background


def load_page_assets(page, root):
    photos, sources = [], []
    for name in page["photos"]:
        path = asset_path(root, name)
        image = photo_image(path)
        if not image.getchannel("A").getbbox():
            raise ValueError(f"Fully transparent photo: {name}")
        photos.append(_opaque(image))
        sources.append({"src": name, "sha256": digest(path)})
    if len(photos) == 1:
        if photos[0].width > photos[0].height:
            raise ValueError("Landscape frame needs a second landscape frame")
    else:
        if sources[0]["sha256"] == sources[1]["sha256"]:
            raise ValueError("Landscape pair requires two distinct frames")
        if any(photo.width <= photo.height for photo in photos):
            raise ValueError("Both frames of a landscape pair must be horizontal")
    return photos, sources


def render_page(page, root):
    photos, sources = load_page_assets(page, root)
    width, height = SIZE
    canvas = Image.new("RGB", SIZE, BACKGROUND)
    if len(photos) == 1:
        photo = photos[0]
        if page.get("fit", "cover") == "cover":
            canvas = ImageOps.fit(photo, SIZE, Image.Resampling.LANCZOS,
                                  centering=tuple(page.get("focal", [0.5, 0.5])))
            boxes = [(0, 0, width, height)]
        else:
            fitted = ImageOps.contain(photo, SIZE, Image.Resampling.LANCZOS)
            x, y = (width-fitted.width)//2, (height-fitted.height)//2
            canvas.paste(fitted, (x, y))
            boxes = [(x, y, x+fitted.width, y+fitted.height)]
    else:
        ratios = [photo.height / photo.width for photo in photos]
        slot_width = max(1, min(width, int((height-GAP) / sum(ratios))))
        sizes = [(slot_width, max(1, int(slot_width*ratio))) for ratio in ratios]
        x = (width-slot_width)//2
        y = (height-GAP-sum(h for _, h in sizes))//2
        boxes = []
        for photo, size in zip(photos, sizes, strict=True):
            fitted = photo.resize(size, Image.Resampling.LANCZOS)
            canvas.paste(fitted, (x, y))
            boxes.append((x, y, x+size[0], y+size[1]))
            y += size[1] + GAP
    if all(low == high for low, high in canvas.getextrema()):
        raise ValueError("Rendered page is blank")
    return canvas, sources, boxes


def encode_delivery(canvas):
    for width in (900, 810, 720, 630, 600):
        size = (width, width*4//3)
        fitted = canvas if size == canvas.size else canvas.resize(size, Image.Resampling.LANCZOS)
        for quality, subsampling in ((90, 0), (90, 2), (85, 2), (80, 2), (75, 2)):
            buffer = io.BytesIO()
            try:
                fitted.save(buffer, format="JPEG", quality=quality, optimize=True,
                            progressive=True, subsampling=subsampling)
            except OSError:
                if subsampling != 0:
                    raise
                # Some encoders cannot buffer high-detail 4:4:4 output.
                continue
            data = buffer.getvalue()
            if len(data) <= MAX_BYTES:
                return data, size, quality
    return data, size, quality


def copy_markdown(copy):
    topics = " ".join(f"#{tag}" for tag in copy["tags"])
    return f"# {copy['title'].strip()}\n\n{copy['body'].strip()}\n\n{topics}\n".encode()


def copy_warnings(copy):
    warnings = []
    title_len = len(copy["title"].strip())
    if title_len > 20:
        warnings.append(f"Title exceeds Xiaohongshu 20-char limit ({title_len} chars)")
    body_len = len("".join(copy["body"].split()))
    if body_len > 150:
        warnings.append(f"Body exceeds 150-char guideline ({body_len} chars)")
    return warnings


def atomic_output(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        staging = path.parent / os.path.basename(handle.name)
        try:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)


def write_json(path, data):
    raw = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()
    atomic_output(path, lambda handle: handle.write(raw))


@contextmanager
def render_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).is_symlink() for name in ("pages", "gallery-report.json", "preview.html", "post.md", ".gallery.lock")):
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
    root = (plan_path.parent / plan["asset_root"]).resolve()
    ids = {page["id"] for page in plan["pages"]}
    active = set(page_ids) if page_ids else ids
    if not active <= ids:
        raise ValueError(f"Unknown page IDs: {', '.join(sorted(active-ids))}")
    report_path = output / "gallery-report.json"
    post = output / "post.md"
    preview = output / "preview.html"
    with render_lock(output):
        managed = {report_path.resolve(), post.resolve(), preview.resolve(),
                   *((output / f"pages/{page_id}.jpg").resolve() for page_id in ids)}
        sources = {(root / name).resolve() for page in plan["pages"] for name in page["photos"]}
        if plan_path in managed or managed & sources:
            raise ValueError("Managed outputs would overwrite the design or a source")
        previous = {}
        if page_ids and report_path.exists():
            prior = json.loads(report_path.read_text(encoding="utf-8"))
            if prior["version"] == 2:
                previous = {row["id"]: row for row in prior["pages"]}
        rows = []
        for page in plan["pages"]:
            page_id = page["id"]
            signature = page_signature(plan, page)
            if page_id not in active:
                prior = previous.get(page_id, {})
                rows.append(prior if prior.get("signature") == signature else
                            {"id": page_id, "status": "stale"})
                continue
            row = {"id": page_id, "signature": signature}
            try:
                canvas, assets, boxes = render_page(page, root)
                data, size, quality = encode_delivery(canvas)
                filename = f"pages/{page_id}.jpg"
                target = output / filename
                if target.is_symlink():
                    raise ValueError("Managed page may not be a symlink")
                atomic_output(target, lambda handle: handle.write(data))
                for asset, box in zip(assets, boxes, strict=True):
                    asset["box"] = [round(coordinate * size[0] / SIZE[0]) for coordinate in box]
                row.update(status="rendered", image=filename, size=list(size), bytes=len(data),
                           jpeg_quality=quality, assets=assets, sha256=digest(target))
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                row.update(status="failed", error=str(exc))
            rows.append(row)
        duplicates = Counter(asset["sha256"] for row in rows for asset in row.get("assets", []))
        warnings = [f"{row['id']} exceeds {MAX_BYTES} bytes ({row['bytes']})"
                    for row in rows if row.get("bytes", 0) > MAX_BYTES]
        if any(count > 1 for count in duplicates.values()):
            warnings.append("The same source photo appears multiple times")
        warnings.extend(copy_warnings(plan["copy"]))
        text = copy_markdown(plan["copy"])
        atomic_output(post, lambda handle: handle.write(text))
        report = {"version": 2, "pages": rows, "warnings": warnings,
                  "post": {"path": "post.md", "sha256": digest(post)}}
        write_json(report_path, report)
        write_preview(preview, rows)
    return report


def inspect_gallery(output, plan=None, plan_path=None):
    report = json.loads((output / "gallery-report.json").read_text(encoding="utf-8"))
    if report.get("version") != 2:
        raise ValueError("Expected a version 2 Xiaohongshu gallery report")
    expected = {page["id"]: page for page in plan["pages"]} if plan else {}
    root = (plan_path.parent / plan["asset_root"]).resolve() if plan else None
    post = asset_path(output, report["post"]["path"])
    if digest(post) != report["post"]["sha256"] or (plan and post.read_bytes() != copy_markdown(plan["copy"])):
        raise ValueError("Post copy changed since rendering")
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
                if (image.format != "JPEG" or image.mode != "RGB" or list(image.size) != row["size"]
                        or image.width * 4 != image.height * 3):
                    raise ValueError("Expected an RGB JPEG at the planned 3:4 size")
                if all(low == high for low, high in image.getextrema()):
                    raise ValueError("Blank JPEG")
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError("JPEG contents changed since rendering")
            findings.append({"id": row["id"], "status": "passed", "bytes": row["bytes"]})
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            findings.append({"id": row["id"], "status": "failed", "error": str(exc)})
    checked = {row["id"] for row in findings}
    findings.extend({"id": page_id, "status": "failed", "error": "Page was not rendered"}
                    for page_id in expected if page_id not in checked)
    return {"pages": findings, "warnings": report["warnings"]}
