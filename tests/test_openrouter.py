"""OpenRouter (free hosted models) as the LLM substrate: rate limits, fallbacks, headers, key hygiene."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from backend import settings as settings_module
from backend.settings import AnalysisConfig, Settings, VerifierConfig
from backend.verification import llm as llm_module
from backend.verification.llm import Endpoint, LLMClient
from tests.fake_openai import FakeOpenAI

ROOT = Path(__file__).resolve().parents[1]
MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    waits: list[float] = []

    async def instant(seconds):
        waits.append(seconds)

    monkeypatch.setattr(llm_module, "_sleep", instant)
    return waits


def endpoint(server: FakeOpenAI, **kw) -> Endpoint:
    return Endpoint(provider="openrouter", model="vendor/primary:free", base_url=server.base_url, api_key="sk-or-v1-fake0000000000", timeout_s=10, **kw)


async def test_a_reasoning_trace_cut_off_by_the_token_limit_is_not_an_answer(fake_openai):
    fake_openai.script({"status": 200, "body": {"choices": [{"finish_reason": "length", "message": {"content": None, "reasoning": "The user wants the exact word from the image. First,"}}]}})
    reply = await LLMClient(endpoint(fake_openai)).complete(MSG, retries=0)
    assert not reply.ok and reply.error == "empty completion"
    fake_openai.script({"status": 200, "body": {"choices": [{"finish_reason": "stop", "message": {"content": "", "reasoning": "PELICAN"}}]}})
    assert (await LLMClient(endpoint(fake_openai)).complete(MSG, retries=0)).text == "PELICAN"


async def test_a_429_waits_as_told_then_succeeds(fake_openai, no_sleep):
    fake_openai.script({"status": 429, "body": {"error": {"message": "slow down"}}, "headers": {"Retry-After": "3"}}, "fine")
    reply = await LLMClient(endpoint(fake_openai)).complete(MSG, retries=0)
    assert reply.ok and reply.text == "fine"
    assert 3.0 in no_sleep, "Retry-After must be honoured"
    assert len(fake_openai.requests) == 2


async def test_a_rate_limited_model_falls_back_to_the_next_one(fake_openai):
    fake_openai.script(lambda body: {"status": 429, "body": "limited"} if body["model"] == "vendor/primary:free" else "from the fallback")
    ep = endpoint(fake_openai, fallback_models=("vendor/second:free",))
    reply = await LLMClient(ep).complete(MSG, retries=0)
    assert reply.ok and reply.text == "from the fallback" and reply.model == "vendor/second:free"


async def test_a_429_that_never_clears_is_reported_not_hung(fake_openai):
    fake_openai.script({"status": 429, "body": "limited"})
    reply = await LLMClient(endpoint(fake_openai)).complete(MSG, retries=0)
    assert not reply.ok and "HTTP 429" in reply.error and reply.status == 429
    assert len(fake_openai.requests) <= 4, "bounded: one try plus at most two rate-limit waits"


async def test_openrouter_upstream_errors_arrive_as_http_200_and_are_handled(fake_openai):
    fake_openai.script({"status": 200, "body": {"error": {"code": 429, "message": "Provider returned error"}}}, "ok now")
    reply = await LLMClient(endpoint(fake_openai)).complete(MSG, retries=1)
    assert reply.ok and reply.text == "ok now"


async def test_a_bad_key_is_fatal_and_does_not_try_fallbacks(fake_openai):
    fake_openai.script({"status": 401, "body": "bad key sk-or-v1-leakedleakedleaked"})
    ep = endpoint(fake_openai, fallback_models=("vendor/second:free",))
    reply = await LLMClient(ep).complete(MSG, retries=2)
    assert not reply.ok and reply.fatal and len(fake_openai.requests) == 1
    assert "leakedleaked" not in reply.error, "a key echoed by the server must not reach an error string"


async def test_an_unknown_model_moves_on_to_the_fallback(fake_openai):
    fake_openai.script(lambda body: {"status": 404, "body": "no endpoints"} if body["model"].endswith("primary:free") else "alt")
    reply = await LLMClient(endpoint(fake_openai, fallback_models=("vendor/second:free",))).complete(MSG)
    assert reply.ok and reply.text == "alt"


async def test_openrouter_attribution_headers_default_and_override(fake_openai):
    fake_openai.script("a", "b")
    await LLMClient(endpoint(fake_openai)).complete(MSG)
    sent = fake_openai.requests[-1]["_headers"]
    assert sent["x-title"] == "OmniBrain" and sent["http-referer"] == "http://localhost"
    await LLMClient(endpoint(fake_openai, headers={"HTTP-Referer": "https://me.example", "X-Title": "Mine"})).complete(MSG)
    sent = fake_openai.requests[-1]["_headers"]
    assert sent["x-title"] == "Mine" and sent["http-referer"] == "https://me.example"


def test_the_key_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-fromenv0000000")
    cfg = VerifierConfig(provider="openrouter", model="m:free", base_url="")
    ep = Endpoint.from_config(cfg)
    assert ep.api_key == "sk-or-v1-fromenv0000000" and ep.base_url == "https://openrouter.ai/api/v1"
    cfg.api_key = "explicit"
    assert Endpoint.from_config(cfg).api_key == "explicit"


async def test_health_is_honest_about_openrouter(fake_openai):
    fake_openai.models = ["vendor/primary:free", "vendor/other"]
    ok = await LLMClient(endpoint(fake_openai)).health()
    assert ok["state"] == "ready" and "free tier" in ok["detail"]
    paid_only = endpoint(fake_openai)
    paid_only.model = "vendor/other:free"
    assert (await LLMClient(paid_only).health())["state"] == "model_missing"
    near_miss = endpoint(fake_openai)
    near_miss.model = "vendor/other:free"
    fake_openai.models = ["vendor/other"]
    assert (await LLMClient(near_miss).health())["state"] == "model_missing", "a paid sibling is not the free model"
    fake_openai.auth_status = 401
    rejected = await LLMClient(endpoint(fake_openai)).health()
    assert rejected["state"] == "auth_failed" and "rejected" in rejected["detail"]
    keyless = endpoint(fake_openai)
    keyless.api_key = ""
    assert (await LLMClient(keyless).health())["state"] == "auth_failed"


def test_analysis_and_vision_can_be_configured_from_the_environment(monkeypatch):
    monkeypatch.setenv("OMNIBRAIN_ANALYSIS_PROVIDER", "openrouter")
    monkeypatch.setenv("OMNIBRAIN_ANALYSIS_MODEL", "vendor/x:free")
    monkeypatch.setenv("OMNIBRAIN_VISION_MODEL", "vendor/v:free")
    loaded = Settings.model_validate(settings_module._apply_env({}))
    assert loaded.analysis.provider == "openrouter" and loaded.analysis.model == "vendor/x:free"
    assert loaded.vision.model == "vendor/v:free"


def test_the_example_config_has_an_openrouter_block_and_no_secret():
    text = (ROOT / "config" / "settings.example.yaml").read_text(encoding="utf-8")
    assert "openrouter" in text and ":free" in text
    assert "sk-or-" not in text
    loaded = yaml.safe_load(text)
    for section in ("verifier", "analysis"):
        assert loaded[section]["api_key"] in ("", None)


def test_no_openrouter_key_is_committed_anywhere():
    """The key lives only in gitignored config/settings.yaml (or the environment)."""
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    offenders = []
    for name in files:
        path = ROOT / name
        if not path.is_file() or path.suffix in {".png", ".jpg", ".ico", ".db"}:
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        import re

        if re.search(r"sk-or-v1-[0-9a-f]{20,}", body):
            offenders.append(name)
    assert not offenders, offenders


def test_log_redaction_covers_openrouter_keys():
    from backend.logs import redact

    fake = "sk-or-" + "v1-" + "0123456789abcdef" * 2  # built at runtime so this file holds no key-shaped literal
    line = f"calling openrouter with Authorization: Bearer {fake} and api_key=sk-or-" + "v1-" + "abcdefabcdefabcdef"
    out = redact(line)
    assert "sk-or-v1-0123" not in out and "abcdefabcdef" not in out

def test_omnibrain_config_points_the_loader_at_another_file(tmp_path, monkeypatch):
    """Lets the eval run with a trimmed provider list without touching config/settings.yaml."""
    alt = tmp_path / "alt.yaml"
    alt.write_text("research:\n  max_rounds: 1\n", encoding="utf-8")
    monkeypatch.setenv("OMNIBRAIN_CONFIG", str(alt))
    assert settings_module.load_settings().research.max_rounds == 1


async def test_json_cut_off_by_the_token_limit_is_retried_with_more_room(fake_openai):
    cut = {"status": 200, "body": {"choices": [{"finish_reason": "length", "message": {"content": '{"verdicts": [{"claim_id": "a", "verdict": "sup'}}]}}
    fake_openai.script(cut, '{"verdicts": [], "answer": "ok"}')
    parsed, reply = await LLMClient(endpoint(fake_openai)).complete_json(MSG)
    assert parsed == {"verdicts": [], "answer": "ok"}
    first, second = fake_openai.requests[0], fake_openai.requests[1]
    assert second["max_tokens"] > first["max_tokens"], "the retry must allow a longer reply"


async def test_complete_json_does_not_retry_when_the_reply_was_not_truncated(fake_openai):
    fake_openai.script("not json at all")
    parsed, _reply = await LLMClient(endpoint(fake_openai)).complete_json(MSG)
    assert parsed is None and len(fake_openai.requests) == 1