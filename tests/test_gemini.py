"""Network-free behavioral tests through the real Google SDK request types."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vfs.gemini import BudgetExceeded, GeminiError, GeminiGateway, GeminiUnavailable


@pytest.fixture
def cfg(monkeypatch):
    for name in ("GEMINI_API_KEY", "GEMINI_ANALYSIS_MODEL", "GEMINI_IMAGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return {
        "enabled": True, "mock": False,
        "analysis_model": "test-analysis", "image_model": "test-image",
        "max_calls": 20, "max_cost_usd": 2, "max_retries": 0,
        "calls_per_minute": 1e12, "retry_base_seconds": 0,
        "prices": {
            "test-analysis": {"input_per_million": .25, "output_per_million": 1.5},
            "test-image": {"input_per_million": 1.5, "output_per_million": 3,
                           "image_output_per_million": 60},
        },
    }


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected paid call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def install_fake(gateway, responses):
    models = FakeModels(responses)
    gateway._client = SimpleNamespace(models=models)
    return models


def rank_response(*ids, score=75):
    return SimpleNamespace(
        text=json.dumps({"frames": [{"id": i, "score": score, "reason": "Clear subject"} for i in ids]}),
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=30,
                                       thoughts_token_count=10),
    )


def frames(*ids):
    return [{"id": name, "image": np.full((128, 192, 3), 120, np.uint8)} for name in ids]


def image_response(shape=(256, 256, 3)):
    image = np.full(shape, 155, np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(
        mime_type="image/png", data=encoded.tobytes()))], usage_metadata=SimpleNamespace(
            prompt_token_count=200, candidates_token_count=1000, thoughts_token_count=0,
            candidates_tokens_details=[SimpleNamespace(modality="IMAGE", token_count=1000)]))


def test_rank_batches_and_cache_survive_restart(tmp_path, cfg):
    cfg["rank_batch_size"] = 2
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [rank_response("a", "b"), rank_response("c")])
    result = gateway.rank_frames(frames("a", "b", "c"))
    assert set(result) == {"a", "b", "c"}
    assert len(fake.requests) == 2
    assert fake.requests[0]["config"].response_mime_type == "application/json"
    assert fake.requests[0]["config"].automatic_continuation is False
    assert fake.requests[0]["config"].automatic_function_calling.disable is True
    resumed = GeminiGateway(cfg, tmp_path)
    # No key/client available: successful cache reads must need neither.
    assert resumed.rank_frames(frames("a", "b", "c")) == result
    assert resumed.stats["api_calls"] == 2
    assert resumed.stats["cache_hits"] == 2
    assert resumed.stats["prompt_tokens"] == 200
    assert resumed.stats["output_tokens"] == 80


def test_retry_attempts_count_against_hard_call_limit(tmp_path, cfg):
    class Throttle(Exception):
        code = 429
    cfg.update(max_calls=2, max_retries=5)
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [Throttle("secret-key"), Throttle("secret-key"), rank_response("a")])
    with pytest.raises(BudgetExceeded, match="max_calls"):
        gateway.rank_frames(frames("a"))
    assert len(fake.requests) == 2
    assert gateway.stats["failures"] == 2
    assert "secret-key" not in gateway.ledger_path.read_text()


def test_monetary_reservations_are_not_refunded_on_retry(tmp_path, cfg):
    cfg.update(max_cost_usd=.05, reserve_analysis_usd=.03, max_retries=2)
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [TimeoutError("private-url?key=x"), rank_response("a")])
    with pytest.raises(BudgetExceeded, match="reservation"):
        gateway.rank_frames(frames("a"))
    assert len(fake.requests) == 1
    assert gateway.stats["reserved_cost_usd"] == .03
    assert gateway.stats["cost_estimate_complete"] is False
    restarted = GeminiGateway(cfg, tmp_path)
    install_fake(restarted, [rank_response("b")])
    with pytest.raises(BudgetExceeded):
        restarted.rank_frames(frames("b"))


def test_unknown_model_pricing_fails_closed(tmp_path, cfg):
    cfg["analysis_model"] = "new-model-without-known-price"
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [rank_response("a")])
    with pytest.raises(BudgetExceeded, match="prices"):
        gateway.rank_frames(frames("a"))
    assert not fake.requests
    assert gateway.stats["calls"] == 0


def test_missing_key_does_not_consume_budget(tmp_path, cfg):
    gateway = GeminiGateway(cfg, tmp_path)
    with pytest.raises(GeminiUnavailable, match="GEMINI_API_KEY"):
        gateway.rank_frames(frames("a"))
    assert gateway.stats["calls"] == 0


def test_http_timeout_and_sdk_retry_are_explicit(tmp_path, cfg, monkeypatch):
    from google import genai
    seen = {}
    def fake_client(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(genai, "Client", fake_client)
    cfg["timeout_seconds"] = 12
    gateway = GeminiGateway(cfg, tmp_path)
    gateway._get_client()
    assert seen["http_options"].timeout == 12000
    assert seen["http_options"].retry_options.attempts == 1
    assert seen["vertexai"] is False


@pytest.mark.parametrize("response", [
    rank_response("wrong"), rank_response("a", score=float("nan")),
    SimpleNamespace(text='{"frames": []}'),
    SimpleNamespace(text='{"frames": [{"id":"a","score":true,"reason":"x"}]}'),
])
def test_invalid_structured_output_never_becomes_cached_success(tmp_path, cfg, response):
    gateway = GeminiGateway(cfg, tmp_path)
    install_fake(gateway, [response])
    with pytest.raises(GeminiError):
        gateway.rank_frames(frames("a"))
    assert gateway.stats["successes"] == 0
    assert gateway.stats["failures"] == 1
    assert list(gateway.path.glob("*.json")) == [gateway.ledger_path]


def test_no_retry_on_auth_failure_and_error_body_is_not_exposed(tmp_path, cfg):
    class AuthError(Exception):
        code = 401
    cfg["max_retries"] = 3
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [AuthError("api_key=PRIVATE_SECRET")])
    with pytest.raises(GeminiError, match="status 401") as failure:
        gateway.rank_frames(frames("a"))
    assert "PRIVATE_SECRET" not in str(failure.value)
    assert "PRIVATE_SECRET" not in gateway.ledger_path.read_text()
    assert len(fake.requests) == 1


def test_crop_request_and_full_resolution_proposal(tmp_path, cfg):
    cfg.update(crop_padding=10, crop_merge_gap=5)
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [image_response()])
    image = np.full((600, 1000, 3), 130, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[500:530, 400:520] = 255
    result = gateway.edit_crop(image, mask)
    assert result.shape == image.shape
    x0, y0, x1, y1 = gateway.last_edit_info["crops"][0]["bbox"]
    assert (x1 - x0) < 200 and (y1 - y0) < 100
    outside_crop = np.ones(image.shape[:2], bool)
    outside_crop[y0:y1, x0:x1] = False
    assert np.array_equal(result[outside_crop], image[outside_crop])
    assert len(fake.requests) == 1
    config = fake.requests[0]["config"]
    assert config.response_modalities == ["IMAGE"]
    assert config.image_config.aspect_ratio == "1:1"
    assert gateway.last_edit_info["proposal_only"]
    resumed = GeminiGateway(cfg, tmp_path)
    assert np.array_equal(resumed.edit_crop(image, mask), result)
    assert resumed.stats["api_calls"] == 1


def test_edit_rejects_aspect_ratio_drift_instead_of_stretching(tmp_path, cfg):
    gateway = GeminiGateway(cfg, tmp_path)
    install_fake(gateway, [image_response((128, 256, 3))])
    image = np.full((200, 300, 3), 130, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[150:170, 100:200] = 255
    with pytest.raises(GeminiError):
        gateway.edit_crop(image, mask)
    assert gateway.stats["failures"] == 1


def test_review_low_confidence_fails_even_if_model_says_pass(tmp_path, cfg):
    gateway = GeminiGateway(cfg, tmp_path)
    response = SimpleNamespace(text=json.dumps({"passed": True, "issues": [], "confidence": .4,
                                               "naturalness": 95}))
    install_fake(gateway, [response])
    image = np.full((200, 300, 3), 130, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[150:170, 100:200] = 255
    result = gateway.review_edit(image, image.copy(), mask)
    assert result["passed"] is False
    assert result["issues"]
    assert result["mock"] is False


def test_mock_is_explicit_and_does_not_pollute_live_cache(tmp_path, cfg):
    cfg["mock"] = True
    fake = GeminiGateway(cfg, tmp_path)
    assert fake.rank_frames(frames("a"))["a"]["mock"] is True
    assert fake.stats["api_calls"] == 0
    assert fake.stats["mock_calls"] == 1
    cfg["mock"] = False
    real = GeminiGateway(cfg, tmp_path)
    with pytest.raises(GeminiUnavailable):
        real.rank_frames(frames("a"))
    assert real.stats["api_calls"] == 0


def test_corrupt_ledger_never_silently_resets_budget(tmp_path, cfg):
    gateway = GeminiGateway(cfg, tmp_path)
    gateway.ledger_path.write_text('{"version":2,"entries":"corrupt"}')
    with pytest.raises(GeminiError, match="corrupt"):
        GeminiGateway(cfg, tmp_path)


def test_concurrent_gateway_instances_cannot_double_spend(tmp_path, cfg):
    cfg.update(max_calls=1)
    one = GeminiGateway(cfg, tmp_path)
    two = GeminiGateway(cfg, tmp_path)
    first = install_fake(one, [rank_response("a")])
    second = install_fake(two, [rank_response("a")])
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda gateway: gateway.rank_frames(frames("a")), [one, two]))
    assert results[0] == results[1]
    assert len(first.requests) + len(second.requests) == 1
    assert one.stats["calls"] == 1


def test_changed_model_invalidates_response_cache(tmp_path, cfg):
    cfg["mock"] = True
    one = GeminiGateway(cfg, tmp_path)
    one.rank_frames(frames("a"))
    cfg["analysis_model"] = "test-new-model"
    two = GeminiGateway(cfg, tmp_path)
    two.rank_frames(frames("a"))
    assert two.stats["calls"] == 2


def test_text_classifier_keeps_only_original_region_indices(tmp_path, cfg):
    gateway = GeminiGateway(cfg, tmp_path)
    response = SimpleNamespace(text=json.dumps({"regions": [
        {"index": 1, "category": "scene", "confidence": .99, "polygon": [[0, 0]]},
        {"index": 0, "category": "overlay", "confidence": .93},
    ]}))
    install_fake(gateway, [response])
    regions = [
        {"polygon": [[30, 80], [170, 80], [170, 100], [30, 100]], "text": "subtitle"},
        {"polygon": [[40, 30], [80, 30], [80, 50], [40, 50]], "text": "package"},
    ]
    before = json.dumps(regions)
    result = gateway.classify_regions(frames("a")[0]["image"], regions)
    assert [r["index"] for r in result] == [0, 1]
    assert [r["category"] for r in result] == ["overlay", "scene"]
    assert all("polygon" not in row for row in result)
    assert json.dumps(regions) == before


def test_mock_text_classifier_never_pretends_to_recognize_overlay(tmp_path, cfg):
    cfg["mock"] = True
    gateway = GeminiGateway(cfg, tmp_path)
    regions = [{"polygon": [[30, 80], [170, 80], [170, 100], [30, 100]], "text": "ignore rules"}]
    result = gateway.classify_regions(frames("a")[0]["image"], regions)
    assert result == [{"index": 0, "category": "uncertain", "confidence": 0.0, "mock": True}]
    assert gateway.stats["api_calls"] == 0


def test_corrupted_response_cache_is_replaced_without_resetting_budget(tmp_path, cfg):
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [rank_response("a"), rank_response("a", score=80)])
    gateway.rank_frames(frames("a"))
    cache = next(p for p in gateway.path.glob("*.json") if p != gateway.ledger_path)
    value = json.loads(cache.read_text())
    value["value"]["a"]["score"] = 100
    cache.write_text(json.dumps(value))
    assert gateway.rank_frames(frames("a"))["a"]["score"] == 80
    assert len(fake.requests) == 2
    assert gateway.stats["calls"] == 2


def test_review_threshold_change_invalidates_cached_verdict(tmp_path, cfg):
    response = SimpleNamespace(text=json.dumps({"passed": True, "issues": [], "confidence": .8,
                                               "naturalness": 90}))
    image = np.full((100, 200, 3), 130, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[60:80, 40:100] = 255
    first = GeminiGateway(cfg, tmp_path)
    install_fake(first, [response])
    assert first.review_edit(image, image.copy(), mask)["passed"]
    cfg["review_min_confidence"] = .9
    second = GeminiGateway(cfg, tmp_path)
    install_fake(second, [response])
    assert not second.review_edit(image, image.copy(), mask)["passed"]
    assert second.stats["calls"] == 2


def test_shared_quota_separates_models_and_resets_daily_at_pacific_midnight(
        tmp_path, cfg, monkeypatch):
    from datetime import datetime, timezone

    cfg["quota"] = {
        "fallback": {"rpm": 20, "tpm": 100, "rpd": 1},
        "image-model": {"rpm": 100, "tpm": 200, "rpd": 2},
    }
    now = [datetime(2026, 11, 24, 7, 59, 59, tzinfo=timezone.utc).timestamp()]
    monkeypatch.setattr("vfs.gemini.time.time", lambda: now[0])
    monkeypatch.setattr("vfs.gemini.time.sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    first = GeminiGateway(cfg, tmp_path / "first")
    second = GeminiGateway(cfg, tmp_path / "second")
    ledger = tmp_path / "quota.json"
    first._reserve_quota(ledger, "analysis-model", 90)
    second._reserve_quota(ledger, "image-model", 90)
    with pytest.raises(BudgetExceeded, match="daily request limit"):
        second._reserve_quota(ledger, "analysis-model", 20)
    now[0] += 1
    midnight = now[0]
    # The daily counter has reset. The rolling-minute token counter still applies.
    second._reserve_quota(ledger, "analysis-model", 20)
    assert now[0] == pytest.approx(midnight + 59)
