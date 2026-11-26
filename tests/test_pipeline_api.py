"""Integration gates around OCR, API proposals, protection and resumable work.

Synthetic frames and fake OCR isolate orchestration from model accuracy. The
Gemini mock adapter still exercises its real caching/budget/request flow.
"""
from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import pipeline as cli
from vfs.config import load_config
from vfs.gemini import GeminiGateway
from vfs.ocr import OCRDetector, OCRUnavailable
from vfs import pipeline
from vfs.quality import metrics
from vfs.util import read_image, read_json, write_png


@pytest.fixture
def config(monkeypatch):
    monkeypatch.delenv("GEMINI_IMAGE_MODEL", raising=False)
    cfg = load_config()
    cfg["run"].update(count=1, previews=False)
    cfg["ocr"].update(mode="all", backend="mock")
    cfg["mask"]["type"] = "polygon"  # mask extraction quality has separate real-image tests
    # This test suite isolates orchestration/pixel protection, not quality tuning.
    cfg["qc"].update(min_sharpness=0, min_sharpness_ratio=0, max_seam_excess=999)
    cfg["gemini"].update(mock=True, calls_per_minute=1e9)
    return cfg


def synthetic(seed=1):
    image = np.random.default_rng(seed).integers(60, 190, (180, 280, 3), dtype=np.uint8)
    cv2.putText(image, "TEST", (55, 134), cv2.FONT_HERSHEY_SIMPLEX, .65,
                (245, 245, 245), 2, cv2.LINE_AA)
    return image


def region(x1=50, y1=115, x2=113, y2=140, text="TEST"):
    return {"polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "text": text, "confidence": .99}


def candidate(output, image, name="frame001", index=1):
    path = f".cache/frames/{name}/original.png"
    write_png(output / path, image)
    return {"id": name, "path": path, "time": float(index), "frame_index": index,
            "scene": 0, "neighbors": [], "metrics": metrics(image)}


class ReferenceDetector:
    """Explicit fake: recognize only the known original test images."""
    is_mock = True

    def __init__(self, originals, regions=None, fail_on_output=False):
        self.originals = originals
        self.regions = regions or [region()]
        self.fail_on_output = fail_on_output
        self.stats = {"calls": 0, "cache_hits": 0, "mock": True}

    def detect(self, image):
        self.stats["calls"] += 1
        if any(np.array_equal(image, original) for original in self.originals):
            return deepcopy(self.regions)
        if self.fail_on_output:
            raise OCRUnavailable("simulated incomplete OCR")
        return []


class WholeFrameProposal:
    """A deliberately overreaching API proposal must remain confined by masks."""
    mock = True
    last_edit_info = {"mock": True, "proposal_only": True}

    def edit_frame(self, image, mask, attempt=0):
        return 255 - image



def test_whole_frame_mock_proposal_cannot_escape_mask_or_protection(tmp_path, config, monkeypatch):
    original = synthetic()
    item = candidate(tmp_path, original)
    # Beyond the OCR box, but inside its dilation/feather support. This exercises
    # the final protection restore, not just selection-time region filtering.
    config["ocr"]["protect_regions"] = [[118, 110, 122, 147]]
    config["mask"]["type"] = "polygon"
    actual_composite = pipeline.composite_local
    before_protection = []
    def observe_support(*args):
        edited, support = actual_composite(*args)
        before_protection.append(support.copy())
        return edited, support
    monkeypatch.setattr(pipeline, "composite_local", observe_support)
    detector = ReferenceDetector([original])
    result = pipeline._candidate(item, tmp_path, config, detector, WholeFrameProposal(), [])
    assert result["status"] == "passed", result
    edited = read_image(tmp_path / result["files"]["image"])
    mask = read_image(tmp_path / result["files"]["mask"], gray=True)
    assert np.any(mask)
    assert before_protection[0][110:148, 118:123].any(), "fixture must exercise feather restoration"
    assert np.array_equal(edited[mask == 0], original[mask == 0])
    assert not mask[110:148, 118:123].any()
    assert np.array_equal(edited[110:148, 118:123], original[110:148, 118:123])
    assert result["qc"]["outside_changed_pixels"] == 0
    assert result["qc"]["outside_pixel_equality"] == 1.0


def test_disabled_ocr_never_promotes_an_image(tmp_path, config):
    original = synthetic()
    item = candidate(tmp_path, original)
    detector = OCRDetector(tmp_path / "ocr", {"backend": "off"})
    with pytest.raises(OCRUnavailable):
        pipeline._candidate(item, tmp_path, config, detector, WholeFrameProposal(), [])
    assert not list((tmp_path / "images").glob("*.png"))


def test_output_ocr_failure_blocks_export(tmp_path, config):
    original = synthetic()
    item = candidate(tmp_path, original)
    detector = ReferenceDetector([original], fail_on_output=True)
    result = pipeline._candidate(item, tmp_path, config, detector, WholeFrameProposal(), [])
    assert result["status"] != "passed"
    assert result["qc"]["ocr_checked"] is False
    assert "ocr_unavailable" in result["qc"]["issues"]
    assert len(result["attempts"]) == config["run"]["max_attempts"]
    assert not list((tmp_path / "images").glob("*.png"))


@pytest.mark.parametrize("bad_result", [
    SimpleNamespace(boxes=None, txts=["VISIBLE TEXT"], scores=[.99]),
    SimpleNamespace(unexpected="not an OCR response"),
])
def test_incomplete_ocr_response_is_not_a_clean_frame(tmp_path, config, bad_result):
    original = synthetic()
    item = candidate(tmp_path, original)
    detector = OCRDetector(tmp_path / "ocr", {"backend": "rapidocr"})
    detector.api = "modern"
    detector.engine = lambda image: bad_result
    gateway = GeminiGateway(config["gemini"], tmp_path / "api")
    with pytest.raises(OCRUnavailable):
        pipeline._candidate(item, tmp_path, config, detector, gateway, [])
    assert not list((tmp_path / "images").glob("*.png"))


def test_repair_budget_failure_does_not_promote_output(tmp_path, config):
    original = synthetic()
    item = candidate(tmp_path, original)
    detector = ReferenceDetector([original])
    gateway = GeminiGateway({**config["gemini"], "max_calls": 0}, tmp_path / "api")
    result = pipeline._candidate(item, tmp_path, config, detector, gateway, [])
    assert result["status"] == "review"
    assert result["issues"] == ["api_budget_exhausted"]
    assert not list((tmp_path / "images").glob("*.png"))
    assert gateway.stats["api_calls"] == 0


def prepare_run(monkeypatch, tmp_path, images):
    source = tmp_path / "fake-local-input.mp4"
    source.write_bytes(b"fixture input identity; decoding replaced by known PNG candidates")
    output = tmp_path / "result"
    scans = []
    def fake_scan(path, destination, video_cfg, selection, count):
        scans.append(path)
        rows = [candidate(destination, image, f"frame{i:03d}", i) for i, image in enumerate(images, 1)]
        return {"candidates": rows, "warnings": [], "video": {"duration": len(images)},
                "decoded_frames": len(images), "sampled_frames": len(images)}
    monkeypatch.setattr(pipeline, "scan_candidates", fake_scan)
    monkeypatch.setattr(pipeline, "resolve_input", lambda *args: source)
    monkeypatch.setattr(pipeline, "OCRDetector", lambda *args: ReferenceDetector(images))
    return source, output, scans


def test_completed_resume_audits_without_new_api_or_decode(tmp_path, config, monkeypatch):
    source, output, scans = prepare_run(monkeypatch, tmp_path, [synthetic()])
    selected = ["frame001"]
    pipeline.prepare_selection(str(source), output, config)
    first = pipeline.run(str(source), output, config, selected=selected)
    assert first["status"] == "complete"
    assert first["mode"] == "mock_gemini+mock_ocr"
    assert first["api"]["mock_calls"] == 1
    assert first["api"]["api_calls"] == 0
    manifest = {p.name: p.read_bytes() for p in (output / "images").glob("*.png")}
    resumed = pipeline.run(str(source), output, config, resume=True, selected=selected)
    assert resumed["resumed_complete"] is True
    assert resumed["api"]["mock_calls"] == first["api"]["mock_calls"]
    assert len(scans) == 1
    assert {p.name: p.read_bytes() for p in (output / "images").glob("*.png")} == manifest


def test_qc_only_reports_partial_delivery_and_preserves_resume(tmp_path, config, monkeypatch, capsys):
    first = synthetic(1)
    source, output, _ = prepare_run(monkeypatch, tmp_path, [first, synthetic(2)])
    monkeypatch.setattr(pipeline, "OCRDetector", lambda *args: ReferenceDetector([first]))
    monkeypatch.setattr(cli, "cloud_startup_allowed", lambda: True)
    selected = ["frame001", "frame002"]
    pipeline.prepare_selection(str(source), output, config)
    report = pipeline.run(str(source), output, config, selected=selected)
    assert report["status"] == "partial"
    original_report = (output / "report.json").read_bytes()

    assert cli.main(["--output", str(output), "--qc-only"]) == 0
    assert "qc_passed | PNG 复核通过 1/2 张 | 原任务 partial" in capsys.readouterr().out
    assert (output / "report.json").read_bytes() == original_report
    resumed = pipeline.run(str(source), output, config, resume=True, selected=selected)
    assert resumed["status"] == "partial"


def test_interrupted_resume_reuses_passed_candidates(tmp_path, config, monkeypatch):
    config["run"]["count"] = 2
    source, output, scans = prepare_run(monkeypatch, tmp_path, [synthetic(1), synthetic(2)])
    selected = ["frame001", "frame002"]
    pipeline.prepare_selection(str(source), output, config)
    actual = pipeline._candidate
    calls = []
    def interrupt_second(row, *args):
        calls.append(row["id"])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return actual(row, *args)
    monkeypatch.setattr(pipeline, "_candidate", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run(str(source), output, config, selected=selected)
    saved = read_json(output / ".checkpoint.json")
    assert len(saved["items"]) == 1
    already_passed = next(iter(saved["items"]))
    resumed_calls = []
    def track(row, *args):
        resumed_calls.append(row["id"])
        return actual(row, *args)
    monkeypatch.setattr(pipeline, "_candidate", track)
    report = pipeline.run(str(source), output, config, resume=True, selected=selected)
    assert report["status"] == "complete", report
    assert already_passed not in resumed_calls
    assert len(resumed_calls) == 1
    assert report["api"]["mock_calls"] == 2


def test_budget_shortfall_can_resume_after_raising_budget(tmp_path, config, monkeypatch):
    source, output, scans = prepare_run(monkeypatch, tmp_path, [synthetic()])
    selected = ["frame001"]
    pipeline.prepare_selection(str(source), output, config)
    config["gemini"]["max_calls"] = 0
    first = pipeline.run(str(source), output, config, selected=selected)
    assert first["status"] == "partial"
    assert first["items"][0]["issues"] == ["api_budget_exhausted"]
    config["gemini"]["max_calls"] = 5
    resumed = pipeline.run(str(source), output, config, resume=True, selected=selected)
    assert resumed["status"] == "complete"
    assert resumed["items"][0]["status"] == "passed"
    assert resumed["api"]["mock_calls"] == 1

def test_selected_workflow_prepares_locally_and_sends_only_explicit_edits(tmp_path, config, monkeypatch):
    source, output, scans = prepare_run(monkeypatch, tmp_path, [synthetic(), synthetic(2)])
    with monkeypatch.context() as isolated:
        def unexpected_gateway(*args):
            pytest.fail("local preparation must not create a Gemini gateway")

        isolated.setattr(pipeline, "GeminiGateway", unexpected_gateway)
        index = pipeline.prepare_selection(str(source), output, config)
    assert [c["id"] for c in index["candidates"]] == ["frame001", "frame002"]
    assert (output / "selection/sheet-001.png").is_file()
    with pytest.raises(ValueError, match="must appear"):
        pipeline.run(str(source), output, config, selected=["missing"])
    assert not (output / ".checkpoint.json").exists()

    report = pipeline.run(str(source), output, config, selected=["frame002"])
    assert report["status"] == "complete"
    assert report["requested_count"] == 1
    assert report["api"]["mock_calls"] == 1
    assert report["api"]["api_calls"] == 0
    assert len(scans) == 1
    item, = report["items"]
    assert item["id"] == "frame002"
    assert item["ranking"]["source"] == "operator_selected"
    assert item["method"] == "gemini_full_frame"
    assert item["qc"]["outside_changed_pixels"] == 0
    assert list((output / "images").glob("*.png")) == [output / "images/frame002.png"]


def test_selected_workflow_does_not_export_frames_without_target_text(tmp_path, config, monkeypatch):
    source, output, _ = prepare_run(monkeypatch, tmp_path, [synthetic()])
    monkeypatch.setattr(pipeline, "OCRDetector", lambda *args: ReferenceDetector([]))
    pipeline.prepare_selection(str(source), output, config)
    report = pipeline.run(str(source), output, config, selected=["frame001"])
    assert report["status"] == "partial"
    assert report["api"]["mock_calls"] == 0
    assert report["items"][0]["issues"] == ["no_target_text"]
    assert not list((output / "images").glob("*.png"))



def test_reviewed_clean_frame_uses_no_provider_call(tmp_path, config, monkeypatch):
    source, output, _ = prepare_run(monkeypatch, tmp_path, [synthetic()])
    monkeypatch.setattr(pipeline, "OCRDetector", lambda *args: ReferenceDetector([]))
    pipeline.prepare_selection(str(source), output, config)
    config["ocr"]["reviewed_frames"] = {"frame001": []}
    report = pipeline.run(str(source), output, config, selected=["frame001"])
    assert report["status"] == "complete"
    assert report["api"]["mock_calls"] == 0
    assert report["items"][0]["method"] == "unchanged"
    assert report["items"][0]["qc"]["outside_changed_pixels"] == 0
