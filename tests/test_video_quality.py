"""Independent deterministic checks of input boundaries and image contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import socket
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vfs.config import load_config
from vfs.quality import check_image, check_saved_png, phash
from vfs.util import file_sha256, safe_output_path, write_png
from vfs import video


class EmptyDetector:
    """Explicit algorithm-test stub, not a claim of real OCR coverage."""

    def detect(self, image):
        return []


class UnavailableDetector:
    def detect(self, image):
        raise RuntimeError("test OCR unavailable")


@pytest.fixture
def image_pair():
    image = np.random.default_rng(7).integers(30, 220, (80, 120, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[24:45, 40:71] = 255
    edited = image.copy()
    edited[mask > 0] = [120, 130, 140]
    return image, edited, mask


@pytest.fixture
def qc_config():
    config = load_config()["qc"]
    # These tests isolate invariants rather than tune an aesthetic threshold.
    config.update(min_sharpness=0, min_sharpness_ratio=0, max_seam_excess=10000)
    return config


def test_png_round_trip_has_exact_outside_pixels(tmp_path, image_pair, qc_config):
    original, edited, mask = image_pair
    paths = [tmp_path / name for name in ("original.png", "edited.png", "mask.png")]
    for path, array in zip(paths, (original, edited, mask)):
        write_png(path, array)
    result = check_saved_png(*paths, [], EmptyDetector(), qc_config)
    assert result["passed"]
    assert result["outside_pixel_equality"] == 1.0
    assert result["outside_changed_pixels"] == 0
    changed = cv2.imread(str(paths[1]))
    changed[0, 0, 0] ^= 1
    write_png(paths[1], changed)
    result = check_saved_png(*paths, [], EmptyDetector(), qc_config)
    assert not result["passed"]
    assert result["outside_changed_pixels"] == 1
    assert "outside_mask_pixels_changed" in result["issues"]


def test_no_ocr_never_passes(image_pair, qc_config):
    result = check_image(*image_pair, [], UnavailableDetector(), qc_config)
    assert not result["passed"]
    assert result["ocr_checked"] is False
    assert "ocr_unavailable" in result["issues"]


def test_wrong_dimensions_never_pass(image_pair, qc_config):
    original, edited, mask = image_pair
    result = check_image(original, edited[:-1], mask, [], EmptyDetector(), qc_config)
    assert not result["passed"]
    assert "dimension_mismatch" in result["issues"]


def test_duplicate_final_frame_rejected(image_pair, qc_config):
    original, edited, mask = image_pair
    result = check_image(original, edited, mask, [], EmptyDetector(), qc_config,
                         prior_hashes=[phash(edited)], duplicate_distance=0)
    assert not result["passed"]
    assert "duplicate_output" in result["issues"]


def test_residual_ocr_fails_only_selected_text_area(image_pair, qc_config):
    # This test isolates OCR overlap on an intentionally noisy image; the
    # independently tested stroke guard correctly treats such noise as uncertain.
    qc_config = {**qc_config, "max_persistent_stroke_pixels": 100000}
    original, _, mask = image_pair
    selected = [{"polygon": [[35, 20], [75, 20], [75, 50], [35, 50]],
                 "text": "subtitle", "confidence": 0.99}]
    scene_text = {"polygon": [[1, 1], [20, 1], [20, 10], [1, 10]],
                  "text": "BRAND", "confidence": 0.99}

    class Detector:
        def __init__(self, regions):
            self.regions = regions

        def detect(self, image):
            return self.regions

    with_subtitle = check_image(original, original.copy(), mask, selected,
                               Detector(selected + [scene_text]), qc_config)
    assert not with_subtitle["passed"]
    assert len(with_subtitle["residual_text"]) == 1
    with_scene_only = check_image(original, original.copy(), mask, selected,
                                 Detector([scene_text]), qc_config)
    assert with_scene_only["passed"]


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/a.mp4", "https://u:p@example.com/a.mp4",
    "http://127.0.0.1/a.mp4", "http://10.1.2.3/a.mp4", "http://[::1]/a.mp4",
    "http://169.254.169.254/latest/meta-data/",
])
def test_private_and_non_http_input_urls_rejected(url, monkeypatch):
    hostname = video.urlsplit(url).hostname or ""
    address = hostname if hostname in {"127.0.0.1", "10.1.2.3", "::1", "169.254.169.254"} else "93.184.216.34"
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM,
         6, "", (address, 80)),
    ])
    with pytest.raises(ValueError):
        video.validate_public_url(url)


def test_mixed_public_private_dns_answers_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
    ])
    with pytest.raises(ValueError):
        video.validate_public_url("https://example.com/video.mp4")


def test_redirect_to_private_destination_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),
    ])
    handler = video._PublicRedirect()
    with pytest.raises(ValueError):
        handler.redirect_request(None, None, 302, "Found", {}, "http://localhost/private")


def test_report_paths_cannot_escape_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "link").symlink_to(outside, target_is_directory=True)
    for path in ("../outside/a.png", str(outside / "a.png"), "link/a.png"):
        with pytest.raises(ValueError):
            safe_output_path(output, path)
    assert safe_output_path(output, "images/a.png") == output / "images/a.png"


def _fake_probe(monkeypatch, stream, format_=None):
    monkeypatch.setattr(video.shutil, "which", lambda name: "/usr/bin/" + name)
    payload = {"streams": [stream], "format": format_ or {"duration": "1", "format_name": "matroska,webm"}}
    monkeypatch.setattr(video.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(payload)))


@pytest.mark.parametrize("fields", [
    {"color_transfer": "smpte2084"}, {"color_transfer": "arib-std-b67"},
    {"pix_fmt": "yuv420p10le"}, {"pix_fmt": "gbrp12le"},
    {"pix_fmt": "yuv420p9le"}, {"pix_fmt": "gbrp14le"}, {"pix_fmt": "rgb48le"},
])
def test_hdr_and_high_bit_depth_are_explicitly_rejected(monkeypatch, fields):
    stream = {"width": 100, "height": 80, "pix_fmt": "yuv420p", **fields}
    _fake_probe(monkeypatch, stream)
    with pytest.raises(ValueError, match="HDR|8-bit|SDR"):
        video.probe(Path("fixture.mkv"), load_config()["video"])


@pytest.mark.parametrize("width,height,duration", [(0, 80, "1"), (100000, 100000, "1"), (100, 80, "nan"), (100, 80, "-1")])
def test_invalid_stream_metadata_rejected(monkeypatch, width, height, duration):
    _fake_probe(monkeypatch, {"width": width, "height": height, "pix_fmt": "bgr0", "duration": duration})
    with pytest.raises(ValueError):
        video.probe(Path("fixture.mkv"), load_config()["video"])


def test_lossless_fixture_one_decode_pass_and_reusable_scan(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required for actual decode test")
    width, height, fps = 96, 64, 12
    frames = []
    for index in range(12):
        scene = index // 4
        frame = np.full((height, width, 3), ((70, 140, 170), (170, 110, 75), (95, 175, 100))[scene], np.uint8)
        cv2.rectangle(frame, (8 + scene * 12, 10), (36 + scene * 12, 42), (220, 210, 180), -1)
        cv2.circle(frame, (20 + index * 4, 45), 7 + scene, (25, 40, 65), 2)
        frames.append(frame)
    input_file = tmp_path / "sample.mkv"
    process = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
        "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "ffv1",
        "-pix_fmt", "bgr0", str(input_file),
    ], input=b"".join(frame.tobytes() for frame in frames), capture_output=True, timeout=20)
    assert process.returncode == 0, process.stderr.decode(errors="replace")
    config = load_config()
    video_cfg = copy.deepcopy(config["video"])
    video_cfg.update(candidates=6, sample_seconds=0.10, neighbors=1,
                     min_scene_seconds=0.10, scene_threshold=12.0, threads=1)
    selection = copy.deepcopy(config["selection"])
    selection.update(min_sharpness=0, min_gray_std=0, phash_distance=0)
    output = tmp_path / "output"
    output.mkdir()
    capture = cv2.VideoCapture
    opens = []

    def counted_capture(*args, **kwargs):
        opens.append(str(args[0]))
        return capture(*args, **kwargs)

    monkeypatch.setattr(cv2, "VideoCapture", counted_capture)
    result = video.scan_candidates(input_file, output, video_cfg, selection, 3)
    assert len(opens) == 1
    assert result["decoded_frames"] == len(frames)
    assert result["candidates"]
    for candidate in result["candidates"]:
        actual = cv2.imread(str(output / candidate["path"]))
        assert np.array_equal(actual, frames[candidate["frame_index"]])
        assert all(donor["scene"] == candidate["scene"] for donor in candidate["neighbors"])
