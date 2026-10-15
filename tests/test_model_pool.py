"""Free-model pool: 429 -> next model at once, a slow model gets a racer, 404 / unlisted models are dropped (2026-10-15)."""
from __future__ import annotations

import asyncio
import time

import pytest

from backend.verification import llm
from backend.verification.llm import Endpoint, LLMClient, LLMReply


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    llm.reset_cooldowns()
    llm._LISTED["http://x"] = frozenset()  # never touch the network in tests
    yield
    llm.reset_cooldowns()


def _client(monkeypatch, behaviour, models=("a", "b", "c")):
    calls: list[str] = []

    async def fake(self, model, *a, **k):
        calls.append(model)
        return await behaviour(model)

    monkeypatch.setattr(LLMClient, "_complete_with", fake)
    ep = Endpoint(provider="openrouter", model=models[0], base_url="http://x", fallback_models=tuple(models[1:]))
    return LLMClient(ep), calls


def test_a_rate_limited_model_hands_over_at_once(monkeypatch):
    async def b(model):
        if model == "a":
            return LLMReply(text="", ok=False, error="HTTP 429", model=model, status=429)
        return LLMReply(text="{}", ok=True, model=model, status=200)

    client, calls = _client(monkeypatch, b)
    t0 = time.monotonic()
    reply = asyncio.run(client.complete([{"role": "user", "content": "x"}]))
    assert reply.model == "b" and calls == ["a", "b"] and time.monotonic() - t0 < 1


def test_a_slow_model_gets_a_second_one_racing_it(monkeypatch):
    monkeypatch.setattr(llm, "HEDGE_AFTER_S", 0.2)

    async def b(model):
        if model == "a":
            await asyncio.sleep(5)
        return LLMReply(text="{}", ok=True, model=model, status=200)

    client, calls = _client(monkeypatch, b)
    t0 = time.monotonic()
    reply = asyncio.run(client.complete([{"role": "user", "content": "x"}]))
    assert reply.model == "b" and calls == ["a", "b"] and time.monotonic() - t0 < 2


def test_a_404_model_is_never_asked_again(monkeypatch):
    async def b(model):
        if model == "a":
            return LLMReply(text="", ok=False, error="HTTP 404", model=model, status=404)
        return LLMReply(text="{}", ok=True, model=model, status=200)

    client, calls = _client(monkeypatch, b)
    asyncio.run(client.complete([{"role": "user", "content": "x"}]))
    asyncio.run(client.complete([{"role": "user", "content": "x"}]))
    assert calls == ["a", "b", "b"]


def test_models_the_server_no_longer_lists_are_dropped_but_never_all(monkeypatch):
    llm._LISTED["http://x"] = frozenset({"b", "c"})
    assert llm.live_models("http://x", ["a", "b", "c"]) == ["b", "c"]
    assert llm.live_models("http://x", ["gone"]) == ["gone"]


def test_the_configured_pools_hold_no_dead_models():
    text = open("config/settings.yaml", encoding="utf-8").read() if __import__("os").path.exists("config/settings.yaml") else ""
    assert "apodex/apodex-1.1-mini" not in text and "qwen/qwen3.8-27b" not in text
