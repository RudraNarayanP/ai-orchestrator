"""Google AI Mode adapter (the AI layer on top of Search, ``udm=50``).

Deliberately distinct from the Gemini chat app: AI Mode answers are assembled
from live search, so they take much longer and stream in numbered blocks that
load lazily. A gentle scroll during the wait is what makes the later blocks
render at all.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from backend.models import ProviderResponse, ProviderStatus, new_id
from browser.adapters.base import PROGRESS_ONLY_RE, ChatAdapter, DOMUnavailable

# The answer-block selectors miss AI Mode when the prose is on the page but not inside #search.
# Live (job_261010221023_1edcb248, 2026-10-10): body text ended with the TUM admission sentences and
# "AI Mode response is ready", and the run stored that only as a failure note.
_VISIBLE_ANSWER_JS = r"""() => {
  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const noise = /AI Mode response is ready|Skip to (previous|next) prompt|\bShow all\b|\bShow more\b/gi;
  const clean = (s) => norm(String(s || '').replace(noise, ' '));
  // No size check: a background tab often reports a 0x0 box while innerText still holds the answer
  // (live job_261010221023_1edcb248, the TUM sentences were in the body and the selector missed them).
  let best = '';
  for (const el of document.querySelectorAll('p, div, section, article, main')) {
    if (el.querySelector('textarea, input[name="q"], [role="search"]')) continue;
    let st;
    try { st = getComputedStyle(el); } catch (e) { continue; }
    if (st.display === 'none' || st.visibility === 'hidden') continue;
    const t = clean(el.innerText);
    if (t.length < 80) continue;
    if (t.length > best.length) best = t;
  }
  if (best) return best.slice(0, 8000);
  const raw = (document.body && document.body.innerText) || '';
  const parts = raw.split(/Skip to previous prompt|AI Mode response is ready/i);
  const before = parts.length > 1 ? parts[parts.length - 2] : raw;
  const tail = clean(before).slice(-8000);
  return tail.length >= 80 ? tail : '';
}"""


# Google paints a source chip in the prose itself: "required. Instagram +1 However".
_INLINE_CHIP = re.compile(r"(?<=[.!?])\s+\S(?:(?!\s\+\d+\b).){0,70}?\s\+\d+\b")


def drop_prompt_echo(text: str, prompt: str) -> str:
    """Remove the question the page echoed. A partial cut leaves the answer starting mid-word.

    The page often shows only the question, while the typed prompt is that question plus
    research instructions. Drop the longest prefix of the page text that also sits in the prompt.
    """
    flat = " ".join((text or "").split())
    needle = " ".join((prompt or "").split())
    if len(needle) < 40 or len(flat) < 40:
        return flat
    low_p = needle.lower()
    # The typed prompt can be echoed in pieces (the question, then the instructions).
    # Peel every leading run that also appears in the prompt, until the answer itself remains.
    for _ in range(8):
        low_t = flat.lower()
        if len(low_t) < 40:
            break
        origin = low_p.find(low_t[:40])
        if origin < 0:
            break
        i = 0
        j = origin
        while i < len(low_t) and j < len(low_p) and low_t[i] == low_p[j]:
            i += 1
            j += 1
        if i < 40:
            break
        flat = flat[i:].strip()
    return flat


def strip_inline_chips(text: str) -> str:
    out = text or ""
    for _ in range(12):
        nxt = _INLINE_CHIP.sub("", out)
        if nxt == out:
            break
        out = nxt
    return " ".join(out.split())


class GoogleAIAdapter(ChatAdapter):
    name = "google_ai"
    submit_verify = False  # the AI Mode page keeps the query in its box after sending

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
            # "Searching" is the status line under the box, not the answer (live follow-up, 2026-10-10).
            if text and PROGRESS_ONLY_RE.match(text):
                text = ""
                length = 0
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
                if text:
                    best = capture
                break
            else:
                if not text:
                    visible = strip_inline_chips(drop_prompt_echo(await self._visible_answer(page), prompt))
                    if PROGRESS_ONLY_RE.match(visible or ""):
                        visible = ""
                    if visible and len(visible) >= 80 and len(visible) > last_len:
                        last_len = len(visible)
                        last_change = time.time()
                        best = {"text": visible, "links": [], "from_page_text": True}
                await asyncio.sleep(self.settings.research.response_stability_poll_ms / 1000.0)
                continue
            await asyncio.sleep(self.settings.research.response_stability_poll_ms / 1000.0)

        if best and PROGRESS_ONLY_RE.match((best.get("text") or "").strip()):
            best = None
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
        if best.get("from_page_text"):
            response.raw_text = strip_inline_chips(drop_prompt_echo(response.raw_text, prompt))
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
        if best.get("from_page_text"):
            response.detail = "answer read from the visible page; the answer-block selector matched nothing"
        elif len(response.answer_text) < 60:
            response.detail = "AI Mode returned only a stub; treated as weak"
        return True

    async def _visible_answer(self, page) -> str:
        """Prose that is on screen when the answer-block selector matched nothing."""
        try:
            raw = await page.evaluate(_VISIBLE_ANSWER_JS)
        except Exception:  # noqa: BLE001 -- a failed read leaves the old "no answer block" result
            return ""
        return " ".join((raw or "").split())
