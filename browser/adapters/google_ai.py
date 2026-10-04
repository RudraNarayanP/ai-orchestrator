"""Google AI Mode adapter (the AI layer on top of Search, ``udm=50``).

Deliberately distinct from the Gemini chat app: AI Mode answers are assembled
from live search, so they take much longer and stream in numbered blocks that
load lazily. A gentle scroll during the wait is what makes the later blocks
render at all.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from backend.models import ProviderResponse, ProviderStatus, new_id
from browser.adapters.base import ChatAdapter, DOMUnavailable


class GoogleAIAdapter(ChatAdapter):
    name = "google_ai"

    async def _attempt(self, *, page_setup: bool, response: ProviderResponse, prompt: str, round_no: int, emit: Any) -> bool:
        page = await self._page(fresh=page_setup)
        await self._settle(page)
        ready = await self.prepare(page, emit, round_no)
        if not ready.get("ok"):
            mapping = {
                "login_wall": ProviderStatus.LOGGED_OUT,
                "blocked": ProviderStatus.FAILED,
                "rate_limited": ProviderStatus.RATE_LIMITED,
            }
            response.note(
                mapping.get(ready.get("state", ""), ProviderStatus.BROKEN),
                error=f"ai-mode readiness={ready.get('state')}",
                detail=(ready.get("bodyHead") or "")[:240],
            )
            return False

        baseline = await self._call(page, "baseline", self._sel_dict) or {"count": 0, "lastText": ""}
        if not await self._type(page, prompt):
            response.note(ProviderStatus.BROKEN, error="no AI Mode input found")
            return False
        await self._submit(page, prompt)

        deadline = time.time() + self.sel.hard_timeout_ms / 1000.0
        # Logged out, AI Mode can sit on animated dots forever (observed: nothing after 400 s). It is slow when it
        # works, but if no text at all has appeared by never_started_ms there is nothing to wait for.
        give_up_if_empty = time.time() + self.sel.never_started_ms / 1000.0
        best: dict[str, Any] | None = None
        last_len = 0
        last_change = time.time()
        while time.time() < deadline:
            if not best and time.time() > give_up_if_empty:
                break
            try:
                capture = await self._call(page, "capture", self._sel_dict, baseline)
            except DOMUnavailable as exc:
                response.note(ProviderStatus.BROKEN, error=str(exc)[:200])
                return False
            text = ((capture or {}).get("text") or "").strip()
            length = int((capture or {}).get("plainLength") or 0)
            if length > last_len:
                last_len = length
                last_change = time.time()
                best = capture
                # Lazy blocks below the fold never render unless asked for.
                try:
                    await page.mouse.wheel(0, 900)
                except Exception:  # noqa: BLE001
                    pass
            elif best and (time.time() - last_change) > self.sel.stable_ms / 1000.0 and not (capture or {}).get("busy"):
                best = capture or best
                break
            else:
                await asyncio.sleep(self.settings.research.response_stability_poll_ms / 1000.0)
                continue
            await asyncio.sleep(self.settings.research.response_stability_poll_ms / 1000.0)

        if not best or not (best.get("text") or "").strip():
            # Say what the page WAS showing: live (2026-10-04) it sat on the animated "thinking" dots for 140 s with only
            # the screen-reader line "AI Mode response is ready" in the DOM and no answer text at all.
            seen = ""
            try:
                seen = " ".join(((await page.evaluate("() => (document.body ? document.body.innerText : '')")) or "").split())[-200:]
            except Exception:  # noqa: BLE001
                pass
            response.note(
                ProviderStatus.BROKEN,
                error="no-response-element: AI Mode produced no answer block",
                detail=f"page showed only: {seen}" if seen else None,
            )
            return False
        from backend.models import Citation

        response.raw_text = (best.get("text") or "").strip()
        response.answer_text = self._clean(response.raw_text)
        response.citations = [
            Citation(**self._citation_shape(l)) for l in (best.get("links") or []) if l.get("href")
        ]
        response.ui_url = page.url
        response.status = ProviderStatus.COMPLETED
        from backend.models import WebResearchStatus

        response.web_research_status = (
            WebResearchStatus.PERFORMED if response.citations else WebResearchStatus.UNKNOWN
        )
        if len(response.answer_text) < 60:
            response.detail = "AI Mode returned only a stub; treated as weak"
        return True
