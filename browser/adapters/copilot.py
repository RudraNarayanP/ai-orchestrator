"""Microsoft Copilot adapter.

Cold profiles hit Copilot's consent/tracking wall before the composer exists, and
the answer streams inside Copilot's own ``#bnp-container`` rather than a generic
transcript. Both are handled here.
"""

from __future__ import annotations

from typing import Any

from browser.adapters.base import ChatAdapter


class CopilotAdapter(ChatAdapter):
    name = "copilot"

    async def prepare(self, page, emit: Any, round_no: int) -> dict[str, Any]:
        state = await super().prepare(page, emit, round_no)
        if state.get("ok"):
            return state
        # Copilot's wall is a dialog whose accept button only appears once the
        # dialog script has run, so a single dismiss pass routinely misses it.
        for _ in range(3):
            try:
                await page.evaluate(
                    "() => { const want = /(accept all|i agree|allow all|got it|close|no thanks)/i;"
                    "for (const b of document.querySelectorAll('button,[role=button],input[type=submit]')) {"
                    "const t = (b.innerText||b.value||b.getAttribute('aria-label')||'');"
                    "if (want.test(t)) { try { b.click(); } catch (e) {} } } }"
                )
            except Exception:  # noqa: BLE001
                break
            await page.wait_for_timeout(900)
            state = await self.readiness(page)
            if state.get("inputHere"):
                state["ok"] = True
                await emit("provider", "copilot: cleared the consent wall", self.provider, round_no)
                return state
        return state
