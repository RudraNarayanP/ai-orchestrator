"""Gemini adapter.

The composer is a Quill contenteditable (``.ql-editor``), which silently drops
``textContent`` writes -- handled by the cascade in the base adapter. What is
Gemini-specific is the answer-mode switch: a mode chosen before typing changes
both latency and whether it browses at all, so the requested mode is confirmed
rather than assumed.
"""

from __future__ import annotations

from typing import Any

from backend.models import ProviderResponse
from browser.adapters.base import ChatAdapter


class GeminiAdapter(ChatAdapter):
    name = "gemini"

    async def prepare(self, page, emit: Any, round_no: int) -> dict[str, Any]:
        state = await super().prepare(page, emit, round_no)
        if not state.get("ok"):
            return state
        wanted = (self.cfg.extra or {}).get("mode")
        if not wanted:
            return state
        labels = {"thinking": ["Thinking"], "pro": ["Pro"], "fast": ["Fast"]}
        for needle in labels.get(str(wanted), []):
            try:
                clicked = await page.evaluate(
                    "(needle) => { const out = [];"
                    "for (const b of document.querySelectorAll('button, [role=button], mat-button-toggle')) {"
                    "const t = (b.innerText||'').trim(); if (!t) continue;"
                    "if (t.toLowerCase().includes(needle.toLowerCase())) { b.click(); out.push(t.slice(0,40)); break; } }"
                    "return out; }",
                    needle,
                )
            except Exception:  # noqa: BLE001
                clicked = []
            if clicked:
                await page.wait_for_timeout(700)
                state["mode_selected"] = clicked[0]
                await emit("provider", f"gemini: mode -> {clicked[0]}", self.provider, round_no)
                break
        return state

    def _assess_web_research(self, response: ProviderResponse) -> None:
        super()._assess_web_research(response)
        # Gemini's grounding chip is the reliable tell that it actually searched.
        if response.web_research_status.value in {"failed_or_unclear", "unknown"} and getattr(
            self, "_grounding_seen", False
        ):
            from backend.models import WebResearchStatus

            response.web_research_status = WebResearchStatus.PERFORMED
