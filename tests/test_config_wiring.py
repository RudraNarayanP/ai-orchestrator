"""Config knobs and dead code (backlog item 8).

Every setting either does something or is gone. The fictional ones are deleted; the
ones worth having (max_events_per_job, per_provider_concurrency, requires_login)
are wired and tested here, and a guard fails if a new setting is added that nothing
reads.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
import yaml

from backend import settings as settings_module
from backend.models import FinalAnswer, JobEvent, ProviderResponse, ProviderStatus
from backend.orchestrator.runner import ResearchRunner
from backend.settings import BrowserConfig, ProviderConfig, ResearchConfig, SearchConfig, Settings, StorageConfig
from backend.storage.db import Store
from tests.conftest import FakeAdapter, base_settings

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------ deleted knobs / code


def test_fictional_knobs_are_gone():
    assert "user_agent_seed" not in BrowserConfig.model_fields
    assert "http_fallback" not in SearchConfig.model_fields
    for name in ("tab_role", "weight"):
        assert name not in ProviderConfig.model_fields, name
    assert "tone_note" not in FinalAnswer.model_fields
    # timeouts live in browser/adapters/selectors.py (hard_timeout_ms), where they are actually enforced
    assert "timeout_s" not in ProviderConfig.model_fields
    assert "hard_response_timeout_s" not in ResearchConfig.model_fields


def test_a_stale_settings_file_with_the_old_keys_still_loads():
    """The user's own config/settings.yaml still carries the deleted keys; it must not break startup."""
    raw = {
        "providers": {"chatgpt": {"url": "https://chatgpt.com/", "label": "ChatGPT", "tab_role": "own_window", "weight": 1.0, "requires_login": True}},
        "browser": {"user_agent_seed": None},
        "search": {"http_fallback": True},
    }
    loaded = Settings.model_validate(raw)
    assert loaded.providers["chatgpt"].requires_login is True and not hasattr(loaded.providers["chatgpt"], "weight")


def test_the_example_config_has_no_dead_keys():
    text = (ROOT / "config" / "settings.example.yaml").read_text(encoding="utf-8")
    for key in ("user_agent_seed", "http_fallback", "tab_role", "weight"):
        assert not re.search(rf"^\s*{key}\s*:", text, re.M), key
    Settings.model_validate(yaml.safe_load(text))


def test_dead_adapter_code_is_gone():
    from backend.research import claims
    from browser.adapters import base, gemini, le_chat

    assert not hasattr(claims, "downgrade_unsupported") and not hasattr(claims, "NO_WEB_RE")
    for module in (gemini, le_chat, base):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for dead in ("_grounding_seen", "_last_browsing", "_typing_tried"):
            assert dead not in source, f"{dead} in {module.__name__}"
    assert "_assess_web_research" not in Path(gemini.__file__).read_text(encoding="utf-8")


def test_every_setting_is_read_somewhere():
    """Guard against fiction: a field nothing outside settings.py reads fails this test."""
    sources = []
    for folder in ("backend", "browser"):
        for path in (ROOT / folder).rglob("*.py"):
            if path.name != "settings.py":
                sources.append(path.read_text(encoding="utf-8"))
    sources.append((ROOT / "run.py").read_text(encoding="utf-8"))
    sources.append((ROOT / "frontend" / "app.js").read_text(encoding="utf-8"))
    blob = "\n".join(sources)
    dead = []
    for model in (BrowserConfig, SearchConfig, StorageConfig, ResearchConfig, ProviderConfig):
        for field in model.model_fields:
            if not re.search(rf"[.\"']{re.escape(field)}\b", blob):
                dead.append(f"{model.__name__}.{field}")
    assert not dead, f"settings nothing reads (wire or delete): {dead}"


# ------------------------------------------------------------ search.fetch_body_chars


async def test_fetch_body_chars_reaches_the_page_fetch(monkeypatch):
    from backend.evidence import sources
    from backend.evidence.sources import FetchedPage

    seen: list[int] = []

    async def fake_fetch(url, *, timeout_s=30, max_chars=12000, browser_fetch=None):
        seen.append(max_chars)
        return FetchedPage(url=url, ok=False, error="stub")

    monkeypatch.setattr(sources, "fetch_page", fake_fetch)
    await sources.gather_from_links("j", [{"href": "https://a.example/x"}], max_chars=777)
    assert seen == [777]


async def test_the_evidence_pool_passes_the_configured_body_size(monkeypatch):
    from backend.evidence import pool as pool_module
    from backend.evidence import search_http
    from backend.models import Citation, ResearchMode

    captured: list[int] = []

    async def fake_gather(job_id, links, **kwargs):
        captured.append(kwargs.get("max_chars"))
        return []

    async def no_search(query, **kw):
        return [], {"tried": [], "used": None}

    monkeypatch.setattr(pool_module, "gather_from_links", fake_gather)
    monkeypatch.setattr(search_http, "search", no_search)
    response = ProviderResponse(job_id="j", provider="chatgpt", prompt="p", status=ProviderStatus.COMPLETED,
                                citations=[Citation(url="https://a.example/x", title="t")])
    settings = base_settings(search={"engines": ["google"], "max_results": 5, "fetch_body_chars": 555})
    await pool_module.build_pool(job_id="j", question="q", claims=[], responses=[response], mode=ResearchMode.STANDARD, settings=settings, engine=None)
    assert captured and set(captured) == {555}


# ------------------------------------------------------------ max_events_per_job


def _store(tmp_path, cap):
    return Store(base_settings(storage={"db_path": str(tmp_path / "e.db"), "max_events_per_job": cap}))


def test_event_cap_drops_chatter_but_keeps_the_events_that_close_a_job(tmp_path):
    store = _store(tmp_path, 5)
    stored = [store.add_event("job_a", JobEvent(kind="provider", message=f"m{i}")) for i in range(12)]
    assert stored == [True] * 5 + [False] * 7
    for kind in ("final", "error", "done"):
        assert store.add_event("job_a", JobEvent(kind=kind, message=kind)) is True
    kinds = [e["kind"] for e in store.events_since("job_a")]
    assert kinds.count("provider") == 5 and kinds[-3:] == ["final", "error", "done"]
    # another job has its own budget
    assert store.add_event("job_b", JobEvent(kind="provider", message="x")) is True


def test_event_cap_survives_a_restart(tmp_path):
    first = _store(tmp_path, 3)
    for i in range(3):
        first.add_event("job_a", JobEvent(kind="status", message=str(i)))
    second = _store(tmp_path, 3)  # fresh process: the count comes from the database
    assert second.add_event("job_a", JobEvent(kind="status", message="over")) is False
    assert len(second.events_since("job_a")) == 3


def test_event_cap_zero_means_unlimited(tmp_path):
    store = _store(tmp_path, 0)
    assert all(store.add_event("job_a", JobEvent(kind="status", message=str(i))) for i in range(50))


def test_event_cap_warns_once_in_the_log(tmp_path, caplog):
    store = _store(tmp_path, 1)
    with caplog.at_level("WARNING", logger="omnibrain.store"):
        for i in range(5):
            store.add_event("job_a", JobEvent(kind="status", message=str(i)))
    assert len([r for r in caplog.records if "max_events_per_job" in r.getMessage()]) == 1


# ------------------------------------------------------- per_provider_concurrency


class Slow(FakeAdapter):
    """Records how many questions are in flight at once."""

    active = 0
    peak = 0

    async def ask(self, job_id, prompt, round_no=1, emit=None):
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)
        try:
            await asyncio.sleep(0.05)
            return await super().ask(job_id, prompt, round_no, emit)
        finally:
            type(self).active -= 1


def _slow_runner(limit):
    class Mine(Slow):
        active = 0
        peak = 0

    settings = base_settings(research={"mode": "STANDARD", "max_rounds": 3, "max_workers": 6, "min_independent_sources": 2, "per_provider_concurrency": limit})
    adapter = Mine("chatgpt", {"answer": "ok"}, settings)
    return ResearchRunner(settings, {"chatgpt": adapter}, engine=None), Mine


async def test_one_site_is_asked_one_question_at_a_time_by_default():
    runner, adapter = _slow_runner(1)
    await asyncio.gather(*(runner._ask("chatgpt", f"q{i}", 1) for i in range(4)))
    assert adapter.peak == 1


async def test_per_provider_concurrency_can_be_raised():
    runner, adapter = _slow_runner(2)
    await asyncio.gather(*(runner._ask("chatgpt", f"q{i}", 1) for i in range(4)))
    assert adapter.peak == 2


async def test_different_providers_still_run_in_parallel():
    settings = base_settings(research={"mode": "STANDARD", "max_rounds": 3, "max_workers": 4, "min_independent_sources": 2, "per_provider_concurrency": 1})

    class Mine(Slow):
        active = 0
        peak = 0

    adapters = {n: Mine(n, {"answer": "ok"}, settings) for n in ("chatgpt", "gemini", "copilot")}
    runner = ResearchRunner(settings, adapters, engine=None)
    await asyncio.gather(*(runner._ask(n, "q", 1) for n in adapters))
    assert Mine.peak == 3


# -------------------------------------------------------------------- requires_login


def test_run_py_doctor_only_nags_about_sign_in_for_providers_that_need_one():
    source = (ROOT / "run.py").read_text(encoding="utf-8")
    assert "cfg.requires_login" in source