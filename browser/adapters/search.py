"""Search adapter: real web research in a dedicated window.

This is not a chat provider. It is the layer that produces *independent
evidence* -- the thing model consensus can never give us -- so it returns result
links plus visible snippets rather than an opinion.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from backend.models import Citation, ProviderResponse, ProviderStatus, new_id
from browser.adapters.base import ChatAdapter, DOMUnavailable

ENGINE_TEMPLATES = {
    "google": "https://www.google.com/search?q={query}&num={num}&hl=en",
    "duckduckgo": "https://html.duckduckgo.com/html/?q={query}",
    "bing": "https://www.bing.com/search?q={query}",
}


class SearchAdapter(ChatAdapter):
    submit_verify = False  # the results page keeps the query in the search box: the composer never empties
    name = "search"
    is_chat = False

    def __init__(self, *args: Any, engine: str = "google", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # search_engine, not engine: ChatAdapter already uses self.engine for the
        # browser engine, and shadowing it here silently broke every query.
        self.search_engine = engine if engine in ENGINE_TEMPLATES else "google"

    async def ask(self, job_id: str, prompt: str, round_no: int = 1, emit=None) -> ProviderResponse:
        """Try each configured engine until one returns real results.

        Google renders a consent-gated shell on a cold profile and DuckDuckGo's
        html endpoint is a plain page; the order that works depends on where you
        are and what Google decided to show today, so neither is trusted blindly.
        """
        emit = emit or (lambda *a, **k: asyncio.sleep(0))
        engines = [self.search_engine] + [e for e in self.settings.search.engines if e != self.search_engine and e in ENGINE_TEMPLATES]
        last: ProviderResponse | None = None
        for engine in engines:
            self.search_engine = engine
            response = await self._ask_once(job_id, prompt, round_no, emit)
            if response.citations:
                return response
            last = response
            if engine != engines[-1]:
                await emit("provider", f"{self.provider}: {engine} gave nothing usable, trying {engines[engines.index(engine)+1]}", self.provider, round_no)
        return last or ProviderResponse(job_id=job_id, round=round_no, provider=self.provider, prompt=prompt)

    async def _ask_once(self, job_id: str, prompt: str, round_no: int, emit) -> ProviderResponse:
        emit = emit or (lambda *a, **k: asyncio.sleep(0))
        response = ProviderResponse(
            id=new_id("resp"),
            job_id=job_id,
            round=round_no,
            provider=self.provider,
            prompt=prompt,
            started_at=time.time(),
            status=ProviderStatus.CONNECTED,
        )
        page = None
        try:
            page = await self._query_page(prompt)
            await self._settle(page)
            ready = await self.prepare(page, emit, round_no)
            if not ready.get("ok"):
                response.note(
                    ProviderStatus.LOGGED_OUT if ready.get("state") == "login_wall" else ProviderStatus.BROKEN,
                    error=f"search readiness={ready.get('state')}",
                    detail=(ready.get("bodyHead") or "")[:200],
                )
                return self._finish(response)

            baseline = await self._call(page, "baseline", self._sel_dict) or {"count": 0, "lastText": ""}
            typed = await self._type(page, prompt)
            if not typed:
                response.note(ProviderStatus.BROKEN, error="no search box found")
                return self._finish(response)
            await self._submit(page, prompt)

            harvested = await self._await_results(page, baseline, emit, round_no)
            if not harvested:
                response.note(ProviderStatus.TIMEOUT, error="no results appeared")
                return self._finish(response)

            items = harvested.get("items") or []
            response.citations = [
                Citation(
                    url=i["href"],
                    title=(i.get("title") or "")[:220] or None,
                    snippet=(i.get("snippet") or "")[:400] or None,
                    provider=self.search_engine,
                )
                for i in items
                if i.get("href")
            ]
            response.answer_text = self._digest(prompt, items)
            response.raw_text = response.answer_text
            response.ui_url = page.url
            response.status = ProviderStatus.COMPLETED
            from backend.models import WebResearchStatus

            response.web_research_status = (
                WebResearchStatus.PERFORMED if response.citations else WebResearchStatus.FAILED_OR_UNCLEAR
            )
        except DOMUnavailable as exc:
            response.note(ProviderStatus.BROKEN, error=str(exc)[:300])
        except Exception as exc:  # noqa: BLE001
            response.note(ProviderStatus.FAILED, error=f"{type(exc).__name__}: {exc}"[:300])
        if page is not None:
            try:
                response.detail = (response.detail or "") + f" [{self.search_engine}]"
            except Exception:  # noqa: BLE001
                pass
        return self._finish(response)

    async def _query_page(self, query: str):
        from urllib.parse import quote_plus

        url = ENGINE_TEMPLATES[self.search_engine].format(query=quote_plus(query), num=self.settings.search.max_results)
        page = await self.engine.open_research_page(self.provider, url, key=f"srch_{abs(hash(query)) % 997}")
        await self._install(page)
        return page

    async def _await_results(self, page, baseline: dict[str, Any], emit, round_no: int) -> dict[str, Any] | None:
        deadline = time.time() + self.settings.search.per_query_timeout_s
        best: dict[str, Any] | None = None
        stable = 0
        last_count = -1
        while time.time() < deadline:
            self._check_cancel()
            try:
                got = await self._call(page, "harvestResults", self._sel_dict, self.settings.search.max_results * 2)
            except DOMUnavailable:
                return best
            items = (got or {}).get("items") or []
            if items:
                best = got
                if len(items) == last_count:
                    stable += 1
                else:
                    stable = 0
                last_count = len(items)
                if stable >= self.settings.research.response_stable_rounds:
                    return got
            elif (got or {}).get("pageText") and "captcha" in str(got.get("pageText", "")).lower():
                await emit("provider", f"{self.provider}: search blocked by a human check -- not bypassing it", self.provider, round_no)
                got = got or {}
                got["items"] = []
                return got
            await self._sleep(0.5)
        return best

    @staticmethod
    def _digest(query: str, items: list[dict[str, Any]]) -> str:
        lines = [f"WEB RESULTS for: {query}", ""]
        for n, item in enumerate(items[:10], start=1):
            title = (item.get("title") or "").strip() or item.get("host", "untitled")
            snippet = " ".join((item.get("snippet") or "").split())[:300]
            lines.append(f"{n}. {title}\n   {item.get('href')}\n   {snippet}")
        return "\n".join(lines).strip()
