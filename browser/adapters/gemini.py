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
