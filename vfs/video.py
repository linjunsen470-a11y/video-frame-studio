"""One sequential decode, bounded candidate groups and same-shot donor frames."""
from __future__ import annotations

from collections import deque
import hashlib
import ipaddress
import json
import math
import re
from pathlib import Path
import shutil
import socket
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

import cv2
from scenedetect.detectors import ContentDetector

from .quality import metrics, acceptable, phash, hash_distance
from .util import atomic_json, read_json, resize_long, write_png


def validate_public_url(url: str) -> None:
    p = urlsplit(url)
    if p.scheme not in {"https", "http"} or not p.hostname or p.username or p.password:
        raise ValueError("Use a public HTTP(S) video URL without embedded credentials")
    addresses = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Local/private/reserved URL destinations are not allowed")


class _PublicRedirect(HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def resolve_input(value: str, cache: Path, config: dict) -> Path:
    if not value.startswith(("https://", "http://")):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Input is not a readable local file")
        if path.stat().st_size > config["max_input_mb"] * 2**20:
            raise ValueError("Input exceeds video.max_input_mb")
        return path
    validate_public_url(value)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (hashlib.sha256(value.encode()).hexdigest() + ".video")
    if target.exists():
        return target
    partial = target.with_suffix(".partial")
    started = time.monotonic()
    try:
        request = Request(value, headers={"User-Agent": "VideoFrameStudio/0.1"})
        with build_opener(_PublicRedirect()).open(request,
                timeout=config["download_timeout_seconds"]) as response, partial.open("wb") as f:
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("URL is a webpage; use a direct media download URL")
            size = 0
            while data := response.read(1024 * 1024):
                size += len(data)
                if size > config["max_input_mb"] * 2**20:
                    raise ValueError("Download exceeds video.max_input_mb")
                if time.monotonic() - started > config["download_total_seconds"]:
                    raise ValueError("Download exceeded its total time budget")
                f.write(data)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def probe(path: Path, config: dict) -> dict:
    if not shutil.which("ffprobe"):
        raise ValueError("ffprobe is missing; run setup.sh or install ffmpeg")
    process = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-select_streams", "v:0",
                              "-show_streams", "-show_format", "-of", "json", str(path)],
                             capture_output=True, text=True, timeout=30, check=False)
    if process.returncode:
        raise ValueError("ffprobe could not decode this video container")
    data = json.loads(process.stdout)
    if any(fmt in data.get("format", {}).get("format_name", "").split(",")
           for fmt in ("hls", "dash", "concat")):
        raise ValueError("Streaming playlists are unsupported; download a standalone video file first")
    if not data.get("streams"):
        raise ValueError("Input contains no video stream")
    stream = data["streams"][0]
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
    if not math.isfinite(duration) or not 0 < duration <= config["max_duration_seconds"]:
        raise ValueError("Video duration is missing/invalid or exceeds max_duration_seconds")
    width, height = int(stream["width"]), int(stream["height"])
    if width <= 0 or height <= 0 or width * height > config["max_pixels"]:
        raise ValueError("Video dimensions exceed max_pixels")
    fmt = stream.get("pix_fmt", "unknown")
    # OpenCV's BGR8 conversion is an explicit limitation; never claim HDR preservation.
    raw_bits = int(stream.get("bits_per_raw_sample") or 0)
    if (stream.get("color_transfer") in {"smpte2084", "arib-std-b67"} or raw_bits > 8
            or re.search(r"(?:p|gray|gbrap?|ya)(?:9|10|12|14|16)(?:le|be)?$", fmt)
            or any(token in fmt for token in ("48", "64", "f32", "f16", "p010", "p016", "p210", "p216"))):
        raise ValueError("HDR / >8-bit video is unsupported: convert to SDR8 explicitly first")
    return {"width": width, "height": height, "duration": duration,
            "codec": stream.get("codec_name"), "pixel_format": fmt,
            "frame_rate": stream.get("avg_frame_rate"),
            "color_space": stream.get("color_space", "unspecified")}


def scan_candidates(path: Path, output: Path, video_config: dict, selection: dict,
                    count: int, source_hash: str, resume: bool = False) -> dict:
    cache = output / ".cache/frames"
    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / "index.json"
    if resume and index_path.exists():
        saved = read_json(index_path)
        if saved.get("source_sha256") == source_hash and all(
                (output / c["path"]).is_file() for c in saved["candidates"]):
            saved["reused_scan"] = True
            return saved
    # A failed first scan is repeated, not appended to an unaccounted disk cache.
    shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    info = probe(path, video_config)
    duration = info["duration"]
    target = int(video_config["candidates"]) or min(video_config["max_candidates"],
                      max(count * 3, math.ceil(duration / 4)))
    target = max(1, min(target, video_config["max_candidates"]))
    step = max(video_config["sample_seconds"], duration / video_config["max_scan_samples"])
    detector = ContentDetector(threshold=video_config["scene_threshold"],
                               min_scene_len=max(1, round(video_config["min_scene_seconds"] / step)))
    # Separate short files for candidate groups keep peak memory independent of duration.
    neighbors_n = min(2, max(0, int(video_config["neighbors"])))
    previous = deque(maxlen=max(1, neighbors_n))
    candidates, pending = {}, []
    cache_bytes, peak_cache_bytes = 0, 0
    budget = video_config["max_cache_mb"] * 2**20

    def save(group: dict, name: str, image, timestamp: float, frame_index: int, scene: int):
        nonlocal cache_bytes, peak_cache_bytes
        dest = cache / group["id"] / name
        write_png(dest, image)
        size = dest.stat().st_size
        cache_bytes += size
        group["cache_bytes"] += size
        peak_cache_bytes = max(cache_bytes, peak_cache_bytes)
        if cache_bytes > budget:
            raise ValueError("Candidate cache limit reached: reduce --candidates or increase video.max_cache_mb")
        return {"path": str(dest.relative_to(output)), "time": round(timestamp, 6),
                "frame_index": frame_index, "scene": scene}

    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_N_THREADS, int(video_config.get("threads", 2))])
    if not cap.isOpened():
        raise ValueError("OpenCV/FFmpeg could not open video")
    # Preserve coded orientation; report this clearly rather than silently transform pixels.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Unsupported video: no usable frame rate")
    next_sample, last_time, frame_index, sample_index, scene = 0., -1., -1, 0, 0
    scan_issues = []
    try:
        while cap.grab():
            frame_index += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(timestamp) or (frame_index > 0 and timestamp <= last_time):
                timestamp = frame_index / fps
                if "timestamp_fallback_to_fps" not in scan_issues:
                    scan_issues.append("timestamp_fallback_to_fps")
            last_time = timestamp
            if timestamp + 1e-6 < next_sample:
                continue
            ok, image = cap.retrieve()
            if not ok or image is None:
                raise ValueError("Video decode failed mid-stream")
            if image.shape[:2] != (info["height"], info["width"]):
                raise ValueError("Decoded dimensions changed or differ from stream dimensions")
            next_sample = timestamp + step
            if detector.process_frame(sample_index, resize_long(image, video_config["analysis_size"])):
                scene += 1
                previous.clear()
                pending.clear()
            for candidate in pending[:]:
                donor = save(candidate, f"next{candidate['future_count']}.png", image,
                             timestamp, frame_index, scene)
                candidate["neighbors"].append(donor)
                candidate["future_count"] += 1
                if candidate["future_count"] >= neighbors_n:
                    pending.remove(candidate)
            met = metrics(image)
            bucket = min(target - 1, int(timestamp / duration * target))
            current = candidates.get(bucket)
            if acceptable(met, selection) and (current is None or
                    met["technical_score"] > current["metrics"]["technical_score"]):
                if current:
                    if current in pending:
                        pending.remove(current)
                    shutil.rmtree(cache / current["id"], ignore_errors=False)
                    cache_bytes -= current["cache_bytes"]
                candidate = {"id": f"f{frame_index:08d}", "time": round(timestamp, 6),
                             "frame_index": frame_index, "scene": scene, "metrics": met,
                             "phash": phash(image), "neighbors": [], "cache_bytes": 0,
                             "future_count": 0}
                candidate["path"] = save(candidate, "original.png", image, timestamp,
                                         frame_index, scene)["path"]
                for i, prev in enumerate(previous):
                    donor = save(candidate, f"prev{i}.png", prev["image"], prev["time"],
                                 prev["frame_index"], prev["scene"])
                    candidate["neighbors"].append(donor)
                candidates[bucket] = candidate
                if neighbors_n:
                    pending.append(candidate)
            previous.append({"image": image, "time": timestamp,
                             "frame_index": frame_index, "scene": scene})
            sample_index += 1
            if sample_index % 40 == 0:
                print(f"scan: {timestamp:.1f}/{duration:.1f}s, {len(candidates)} candidates", flush=True)
        if last_time < duration - max(2 / fps, 0.25):
            raise ValueError("Video ended substantially before its advertised duration; possible corruption")
    finally:
        cap.release()
    ordered, seen = [], []
    for candidate in sorted(candidates.values(), key=lambda c: c["metrics"]["technical_score"], reverse=True):
        if any(hash_distance(candidate["phash"], other) <= selection["phash_distance"] for other in seen):
            shutil.rmtree(cache / candidate["id"])
            continue
        seen.append(candidate["phash"])
        candidate.pop("future_count", None)
        ordered.append(candidate)
    result = {"source_sha256": source_hash, "video": info, "candidates": ordered,
              "decoded_frames": frame_index + 1, "sampled_frames": sample_index,
              "sample_interval_seconds": step, "scenes": scene + 1,
              "peak_cache_bytes": peak_cache_bytes, "reused_scan": False,
              "warnings": scan_issues,
              "pixel_baseline": "OpenCV FFmpeg BGR uint8 decoded frame, coded orientation"}
    atomic_json(index_path, result)
    return result
