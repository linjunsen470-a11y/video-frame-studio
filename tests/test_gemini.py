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
    for name in ("GEMINI_API_KEY", "GEMINI_IMAGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return {
        "mock": False, "image_model": "test-image",
        "max_calls": 20, "max_cost_usd": 2, "max_retries": 0,
        "calls_per_minute": 1e12, "retry_base_seconds": 0,
        "prices": {
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


def image_response(shape=(256, 256, 3), image=None):
    if image is None:
        image = np.full(shape, 155, np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(
        mime_type="image/png", data=encoded.tobytes()))], usage_metadata=SimpleNamespace(
            prompt_token_count=200, candidates_token_count=1000, thoughts_token_count=0,
            candidates_tokens_details=[SimpleNamespace(modality="IMAGE", token_count=1000)]))


def edit_fixture():
    image = np.full((128, 96, 3), 130, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[85:100, 35:70] = 255
    return image, mask


def edit_response(value=140):
    image, mask = edit_fixture()
    image[mask > 0] = value
    return image_response(image=image)


def test_retry_attempts_count_against_hard_call_limit(tmp_path, cfg):
    class Throttle(Exception):
        code = 429
    cfg.update(max_calls=2, max_retries=5)
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [Throttle("secret-key"), Throttle("secret-key"), edit_response()])
    with pytest.raises(BudgetExceeded, match="max_calls"):
        gateway.edit_frame(*edit_fixture())
    assert len(fake.requests) == 2
    assert gateway.stats["failures"] == 2
    assert "secret-key" not in gateway.ledger_path.read_text()


def test_monetary_reservations_are_not_refunded_on_retry(tmp_path, cfg):
    cfg.update(max_cost_usd=.5, reserve_image_usd=.3, max_retries=2)
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [TimeoutError("private-url?key=x"), edit_response()])
    with pytest.raises(BudgetExceeded, match="reservation"):
        gateway.edit_frame(*edit_fixture())
    assert len(fake.requests) == 1
    assert gateway.stats["reserved_cost_usd"] == .3
    assert gateway.stats["cost_estimate_complete"] is False
    restarted = GeminiGateway(cfg, tmp_path)
    install_fake(restarted, [edit_response()])
    with pytest.raises(BudgetExceeded):
        restarted.edit_frame(*edit_fixture(), attempt=1)


def test_unknown_model_pricing_fails_closed(tmp_path, cfg):
    cfg["image_model"] = "new-model-without-known-price"
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [edit_response()])
    with pytest.raises(BudgetExceeded, match="prices"):
        gateway.edit_frame(*edit_fixture())
    assert not fake.requests
    assert gateway.stats["calls"] == 0


def test_missing_key_does_not_consume_budget(tmp_path, cfg):
    gateway = GeminiGateway(cfg, tmp_path)
    with pytest.raises(GeminiUnavailable, match="GEMINI_API_KEY"):
        gateway.edit_frame(*edit_fixture())
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
    SimpleNamespace(parts=[]),
    SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(mime_type="image/png", data=b"invalid"))]),
])
def test_invalid_image_output_never_becomes_cached_success(tmp_path, cfg, response):
    gateway = GeminiGateway(cfg, tmp_path)
    install_fake(gateway, [response])
    with pytest.raises(GeminiError):
        gateway.edit_frame(*edit_fixture())
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
        gateway.edit_frame(*edit_fixture())
    assert "PRIVATE_SECRET" not in str(failure.value)
    assert "PRIVATE_SECRET" not in gateway.ledger_path.read_text()
    assert len(fake.requests) == 1


def test_full_frame_request_contains_both_overlay_regions(tmp_path, cfg):
    gateway = GeminiGateway(cfg, tmp_path)
    image = np.full((640, 360, 3), 130, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[30:50, 25:75] = 255
    mask[550:570, 110:220] = 255
    edited = image.copy()
    edited[mask > 0] = 140
    fake = install_fake(gateway, [image_response(image=edited)])
    assert np.array_equal(gateway.edit_frame(image, mask), edited)
    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request["config"].response_modalities == ["IMAGE"]
    assert request["config"].automatic_continuation is False
    assert request["config"].automatic_function_calling.disable is True
    assert request["config"].image_config.aspect_ratio == "9:16"
    from vfs.gemini import _decode_image
    parts = request["contents"][0].parts
    blobs = [part.inline_data.data for part in parts if part.inline_data]
    assert np.array_equal(_decode_image(blobs[0]), image)
    assert np.array_equal(_decode_image(blobs[1])[:, :, 0], mask)
    assert gateway.last_edit_info["proposal_only"]
    resumed = GeminiGateway(cfg, tmp_path)
    assert np.array_equal(resumed.edit_frame(image, mask), edited)
    assert resumed.stats["api_calls"] == 1
    assert resumed.stats["cache_hits"] == 1
    assert resumed.stats["prompt_tokens"] == 200
    assert resumed.stats["output_tokens"] == 1000


@pytest.mark.parametrize("wrong", ["aspect", "alignment"])
def test_full_frame_rejects_wrong_geometry(tmp_path, cfg, wrong):
    gateway = GeminiGateway(cfg, tmp_path)
    image = np.full((240, 320, 3), 130, np.uint8)
    shape = (128, 256, 3) if wrong == "aspect" else image.shape
    install_fake(gateway, [image_response(shape)])
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[150:170, 100:200] = 255
    with pytest.raises(GeminiError, match="aspect ratio|could not be aligned"):
        gateway.edit_frame(image, mask)
    assert gateway.stats["api_calls"] == 1


def test_mock_is_explicit_and_does_not_pollute_live_cache(tmp_path, cfg):
    cfg["mock"] = True
    fake = GeminiGateway(cfg, tmp_path)
    assert fake.edit_frame(*edit_fixture()).shape == edit_fixture()[0].shape
    assert fake.last_edit_info["mock"] is True
    assert fake.stats["api_calls"] == 0
    assert fake.stats["mock_calls"] == 1
    cfg["mock"] = False
    real = GeminiGateway(cfg, tmp_path)
    with pytest.raises(GeminiUnavailable):
        real.edit_frame(*edit_fixture())
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
    first = install_fake(one, [edit_response()])
    second = install_fake(two, [edit_response()])
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda gateway: gateway.edit_frame(*edit_fixture()), [one, two]))
    assert np.array_equal(results[0], results[1])
    assert len(first.requests) + len(second.requests) == 1
    assert one.stats["calls"] == 1


def test_changed_model_invalidates_response_cache(tmp_path, cfg):
    cfg["mock"] = True
    one = GeminiGateway(cfg, tmp_path)
    one.edit_frame(*edit_fixture())
    cfg["image_model"] = "test-new-model"
    two = GeminiGateway(cfg, tmp_path)
    two.edit_frame(*edit_fixture())
    assert two.stats["calls"] == 2


def test_corrupted_response_cache_is_replaced_without_resetting_budget(tmp_path, cfg):
    gateway = GeminiGateway(cfg, tmp_path)
    fake = install_fake(gateway, [edit_response(), edit_response(150)])
    gateway.edit_frame(*edit_fixture())
    cache = next(p for p in gateway.path.glob("*.json") if p != gateway.ledger_path)
    value = json.loads(cache.read_text())
    value["value"] = "corrupt"
    cache.write_text(json.dumps(value))
    assert gateway.edit_frame(*edit_fixture())[90, 45, 0] == 150
    assert len(fake.requests) == 2
    assert gateway.stats["calls"] == 2


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


@pytest.mark.parametrize("shift", [2, 20])
def test_full_frame_checks_alignment_before_compositing(tmp_path, cfg, shift):
    gateway = GeminiGateway(cfg, tmp_path)
    rng = np.random.default_rng(37)
    image = np.clip(rng.normal(128, 55, (640, 360, 3)), 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[420:455, 80:165] = 255
    proposal = image.copy()
    proposal[mask > 0] = 200
    shifted = cv2.warpAffine(proposal, np.float32([[1, 0, shift], [0, 1, 0]]),
                             (360, 640), borderMode=cv2.BORDER_REPLICATE)
    install_fake(gateway, [image_response(image=shifted)])
    if shift > 2:
        with pytest.raises(GeminiError, match="alignment|aligned"):
            gateway.edit_frame(image, mask)
    else:
        aligned = gateway.edit_frame(image, mask)
        assert gateway.last_edit_info["alignment"] == "affine"
        assert np.all(aligned[425:450, 85:160] == 200)
