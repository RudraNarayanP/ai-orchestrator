"""Test scaffolding: a scripted world of providers and sources.

Nothing here touches a browser or the network. Adapters return pre-written answers
and the evidence layer is replaced by a script that says which cited page confirms
or fails which claim -- which is exactly the seam that lets us test "most of the
models were wrong and the ledger still got it right".
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.evidence import pool as pool_module
from backend.models import (
    Citation,
    Evidence,
    ProviderResponse,
    ProviderStatus,
    ResearchMode,
    SourceCheckStatus,
    SourceTier,
    WebResearchStatus,
)
from backend.settings import ProviderConfig, Settings

# A world: provider name -> the answer it should give.
WORLD: dict[str, dict[str, Any]] = {}


def make_response(provider: str, script: dict[str, Any], job_id: str, round_no: int) -> ProviderResponse:
    status = ProviderStatus(script.get("status", "completed"))
    citations = [
        Citation(url=c["url"], title=c.get("title"), snippet=c.get("snippet"), provider=provider)
        for c in script.get("citations", [])
    ]
    answer = script.get("answer", "")
    response = ProviderResponse(
        job_id=job_id,
        round=round_no,
        provider=provider,
        prompt="",
        answer_text=answer,
        raw_text=answer,
        citations=citations,
        status=status,
        error=script.get("error"),
        web_research_status=script.get("web_research", WebResearchStatus.PERFORMED if citations else WebResearchStatus.FAILED_OR_UNCLEAR),
    )
    response.pages_visited = [c.url for c in citations]
    return response


class FakeAdapter:
    def __init__(self, provider: str, script: dict[str, Any], settings: Settings) -> None:
        self.provider = provider
        self.script = script
        self.settings = settings
        self.calls: list[dict[str, Any]] = []

    async def ask(self, job_id: str, prompt: str, round_no: int = 1, emit: Any = None) -> ProviderResponse:
        self.calls.append({"prompt": prompt, "round": round_no, "job_id": job_id})
        if self.script.get("raises"):
            raise RuntimeError(self.script["raises"])
        if emit is not None:
            await emit("provider", f"{self.provider}: answering from script", self.provider, round_no)
        rounds = self.script.get("rounds")
        chosen = rounds[round_no - 1] if rounds and round_no - 1 < len(rounds) else self.script
        return make_response(self.provider, chosen, job_id, round_no)


def base_settings(**overrides: Any) -> Settings:
    """A settings object with every provider we care about enabled and no network use."""
    raw: dict[str, Any] = {
        "providers": {
            name: {"enabled": True, "label": name.title(), "url": f"https://{name}.test/"}
            for name in ["chatgpt", "gemini", "copilot", "google_ai", "le_chat", "qwen", "deepseek", "meta_ai", "pi", "search"]
        },
        "verifier": {"provider": "disabled", "model": "none", "base_url": ""},
        "analysis": {"provider": "disabled", "model": "none", "base_url": ""},
        "research": {"mode": "STANDARD", "max_rounds": 3, "max_workers": 4, "min_independent_sources": 2, "min_provider_spacing_s": 0},
        "search": {"engines": ["google"], "max_results": 5},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    return Settings.model_validate(raw)


@pytest.fixture
def fake_openai():
    """A scripted OpenAI-compatible server on localhost (no Ollama needed)."""
    from tests.fake_openai import FakeOpenAI

    server = FakeOpenAI().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def settings() -> Settings:
    return base_settings()


@pytest.fixture
def net(world):
    """Alias so tests read as "the network said this"."""
    return world


@pytest.fixture
def world(monkeypatch):
    """Install the scripted evidence ledger.

    ``confirm(url)`` says what happens when we actually open a cited page:
    confirmed, mismatch (the page does not contain the claim), hallucinated,
    outdated, blocked, unreachable -- plus the tier and polarity that drive the
    verdict.
    """

    class Ledger:
        def __init__(self) -> None:
            self.script: dict[str, dict[str, Any]] = {}
            self.results: dict[str, list[dict[str, Any]]] = {}

        def add_results(self, query: str, items: list[dict[str, Any]]) -> None:
            """What the discovery engine should return for a given query."""
            self.results[query] = items

        def confirm(
            self,
            url: str,
            *,
            tier: SourceTier = SourceTier.JOURNALISM,
            polarity: str = "support",
            published: str | None = "2026-06-01",
            claim_id: str | None = None,
            excerpt: str | None = None,
        ) -> dict[str, Any]:
            self.script[url] = {
                "tier": tier,
                "polarity": polarity,
                "published": published,
                "status": SourceCheckStatus.CONFIRMED,
                "claim_id": claim_id,
                "excerpt": excerpt,
            }
            return self.script[url]

        def fail(
            self,
            url: str,
            status: SourceCheckStatus = SourceCheckStatus.MISMATCH,
            **kwargs: Any,
        ) -> dict[str, Any]:
            entry = self.script.setdefault(url, {})
            entry.update({"status": status, **kwargs})
            entry.setdefault("tier", SourceTier.UNKNOWN)
            entry.setdefault("polarity", "support")
            return entry

        def search(self, url: str, *, title: str = "", snippet: str = "", **kw: Any) -> dict[str, Any]:
            entry = self.confirm(url, **kw)
            entry["title"] = title
            entry["snippet"] = snippet
            entry["origin"] = "search"
            return entry

    ledger = Ledger()

    async def fake_gather(job_id, links, *, max_pages=20, concurrency=5, browser_fetch=None, round_no=1, origin="provider", max_chars=12000, attribute_to=None):
        out: list[Evidence] = []
        for link in links:
            url = link.get("href") or link.get("url")
            if not url:
                continue
            spec = ledger.script.get(url, {})
            status = spec.get("status", SourceCheckStatus.UNREACHABLE)
            out.append(
                Evidence(
                    job_id=job_id,
                    round=round_no,
                    claim_id=link.get("claim_id") or spec.get("claim_id"),
                    url=url,
                    title=link.get("title") or spec.get("title"),
                    domain=(url.split("/")[2] if "://" in url else url).lower(),
                    snippet=link.get("snippet") or spec.get("snippet"),
                    published=spec.get("published"),
                    tier=spec.get("tier", SourceTier.UNKNOWN),
                    polarity=spec.get("polarity", "support"),
                    check_status=status,
                    check_notes=spec.get("notes", f"scripted {status.value}"),
                    verbatim_excerpt=spec.get("excerpt") if status == SourceCheckStatus.CONFIRMED else None,
                    origin=link.get("origin", origin),
                )
            )
        return out

    async def fake_search(query, *, engines=None, limit=8, timeout_s=25):
        """Offline stand-in for the HTTP discovery transport.

        Tests that care about search results register them per query; everything
        else gets an empty answer without ever touching the network.
        """
        items = ledger.results.get(query) or []
        for item in items:
            ledger.script.setdefault(item["href"], {"status": SourceCheckStatus.CONFIRMED, "tier": SourceTier.JOURNALISM, "polarity": "support", "published": "2026-06-01"})
        return list(items), {"tried": [{"engine": "script"}], "used": "script" if items else None}

    monkeypatch.setattr(pool_module, "gather_from_links", fake_gather)
    from backend.evidence import search_http

    monkeypatch.setattr(search_http, "search", fake_search)
    return ledger


def adapters_from(world_scripts: dict[str, dict[str, Any]], settings: Settings) -> dict[str, FakeAdapter]:
    return {name: FakeAdapter(name, script, settings) for name, script in world_scripts.items()}
