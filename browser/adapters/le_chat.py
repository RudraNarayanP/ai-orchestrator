"""Le Chat (Mistral) adapter.

Le Chat ships an explicit web-browsing toggle. Turning it on is ordinary use of
the product's own affordance -- it is how the answer becomes independently
sourced rather than parametric memory.
"""

from __future__ import annotations

import re
from typing import Any

from backend.models import ProviderResponse
from browser.adapters.base import ChatAdapter

BROWSING_LABELS = ("web search", "search the web", "browse", "search", "internet")


class LeChatAdapter(ChatAdapter):
    name = "le_chat"

    async def prepare(self, page, emit: Any, round_no: int) -> dict[str, Any]:
        state = await super().prepare(page, emit, round_no)
        if not state.get("ok"):
            return state
        try:
            toggled = await page.evaluate(
                "(labels) => { const hit = [];"
                "for (const b of document.querySelectorAll('button,[role=button],[aria-label]')) {"
                "const t = ((b.getAttribute('aria-label')||'') + ' ' + (b.innerText||'')).toLowerCase();"
                "if (!t.trim()) continue;"
                "if (labels.some(l => t.includes(l))) {"
                "const on = b.getAttribute('aria-pressed') === 'true' || b.className.toLowerCase().includes('active');"
                "if (!on) { try { b.click(); hit.push(t.slice(0,40)); } catch (e) {} } } }"
                "return hit; }",
                list(BROWSING_LABELS),
            )
        except Exception:  # noqa: BLE001
            toggled = []
        if toggled:
            await page.wait_for_timeout(400)
            state["browsing_enabled"] = toggled[0]
            await emit("provider", "le_chat: web browsing toggled on", self.provider, round_no)
        return state

    def _assess_web_research(self, response: ProviderResponse) -> None:
        super()._assess_web_research(response)
        if getattr(self, "_last_browsing", False):
            from backend.models import WebResearchStatus

            if response.web_research_status == WebResearchStatus.FAILED_OR_UNCLEAR:
                response.web_research_status = WebResearchStatus.UNKNOWN
