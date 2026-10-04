"""ChatGPT adapter.

Only real quirk that matters: the citation list arrives as its *own* assistant
turn after the prose answer. Capturing only the newest block therefore returns a
bare "Sources" list instead of the answer, so the answer block and the reference
block are both harvested and then merged.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from backend.models import ProviderResponse, ProviderStatus
from browser.adapters.base import ChatAdapter


def clean_source_url(href: str) -> str:
    """The page's own address: ChatGPT appends ``utm_source=chatgpt.com`` to every source link it shows."""
    parts = urlsplit(href or "")
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not (k == "utm_source" and "chatgpt" in v.lower())]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


SOURCES_DIALOG_JS = """() => {
  const d = document.getElementById("assistant-sources-dialog");
  if (!d) return [];
  return [...d.querySelectorAll("a[href^=http]")].map(a => ({href: a.href, text: (a.innerText || "").trim()}));
}"""


class ChatGPTAdapter(ChatAdapter):
    name = "chatgpt"

    async def _harvest_source_chips(self, page, response: ProviderResponse) -> int:
        """Logged-out ChatGPT shows its sources as chips ("legislation.gov.uk") that are buttons, not links: the URLs
        sit in a dialog that opens when a chip is clicked (live 2026-10-04: answer text had the site names, citations
        were empty). Opening that dialog is the product's own read-only "show sources" control."""
        try:
            chips = page.locator("button[aria-controls='assistant-sources-dialog']")
            if await chips.count() == 0:
                return 0
            await chips.last.click(timeout=4000)
            await page.wait_for_timeout(1500)
            found = await page.evaluate(SOURCES_DIALOG_JS)
        except Exception:  # noqa: BLE001 -- sources are a bonus; never fail the answer over them
            return 0
        finally:
            try:
                await page.keyboard.press("Escape")
            except Exception:  # noqa: BLE001
                pass
        from backend.models import Citation

        seen = {c.url for c in response.citations}
        added = 0
        for item in found or []:
            url = clean_source_url(item.get("href") or "")
            if not url or url in seen or self._is_first_party(url):
                continue
            lines = [ln.strip() for ln in (item.get("text") or "").splitlines() if ln.strip()]
            lines = [ln for ln in lines if len(ln) > 1]  # the favicon letter chip ("L")
            seen.add(url)
            response.citations.append(Citation(**self._citation_shape({"href": url, "title": lines[-1] if lines else None, "snippet": " - ".join(lines)})))
            added += 1
        return added

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
        if not any(not self._is_first_party(c.url) for c in response.citations):
            if await self._harvest_source_chips(page, response):
                self._assess_web_research(response)
        if response.status == ProviderStatus.RESPONDING:
            response.status = ProviderStatus.COMPLETED
        return True
