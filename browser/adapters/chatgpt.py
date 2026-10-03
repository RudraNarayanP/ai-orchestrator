"""ChatGPT adapter.

Only real quirk that matters: the citation list arrives as its *own* assistant
turn after the prose answer. Capturing only the newest block therefore returns a
bare "Sources" list instead of the answer, so the answer block and the reference
block are both harvested and then merged.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from backend.models import ProviderResponse, ProviderStatus
from browser.adapters.base import ChatAdapter


class ChatGPTAdapter(ChatAdapter):
    name = "chatgpt"

    async def _attempt(self, *, page_setup: bool, response: ProviderResponse, prompt: str, round_no: int, emit: Any) -> bool:
        ok = await super()._attempt(
            page_setup=page_setup, response=response, prompt=prompt, round_no=round_no, emit=emit
        )
        if not ok:
            return False
        page = await self._page()
        # Give the trailing reference block a moment to land before we freeze
        # the response, without waiting so long that a normal answer is delayed.
        deadline = time.time() + 6
        before = len(response.citations)
        while time.time() < deadline:
            try:
                capture = await self._call(page, "capture", self._sel_dict, {"count": 0, "lastText": ""})
            except Exception:  # noqa: BLE001
                break
            links = (capture or {}).get("links") or []
            if len(links) > before:
                from backend.models import Citation

                seen = {c.url for c in response.citations}
                for link in links:
                    if link.get("href") and link["href"] not in seen:
                        seen.add(link["href"])
                        response.citations.append(Citation(**self._citation_shape(link)))
                await asyncio.sleep(1.2)
                before = len(links)
            else:
                break
        if response.status == ProviderStatus.RESPONDING:
            response.status = ProviderStatus.COMPLETED
        return True
