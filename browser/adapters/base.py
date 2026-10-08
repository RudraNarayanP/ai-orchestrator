"""Generic chat-UI adapter: the mechanics every provider shares.

Completion detection is the hard part, and the design follows what prior
browser-council projects converged on after getting it wrong in production:

* text stability is the primary "done" signal, not the Stop button -- a stale
  stop selector must never cause a false timeout;
* busy signals can only *extend* the wait, up to a force-capture ceiling;
* a page that produced no new message at all is reported as BROKEN (selector
  drift), which is a different failure from TIMEOUT;
* unfocused windows render slower, so every threshold doubles while
  ``document.visibilityState === 'hidden'``, and we periodically nudge focus.

Adapters subclass this and override only what differs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict
from typing import Any, Awaitable, Callable

from backend.browser.engine import BrowserEngine
from backend.browser.live_chrome import LiveChromeNeedsUser, needs_user_in
from backend.cancel import CancelToken, JobCancelled
from backend.models import Citation, ProviderResponse, ProviderStatus, new_id
from backend.settings import ProviderConfig, Settings
from browser.adapters.dom_library import DOM_LIBRARY_JS
from browser.adapters.selectors import SelectorSet, selectors_for

EventHook = Callable[..., Awaitable[None]]

RATE_LIMIT_RE = re.compile(
    r"(rate ?limit|too many requests|slow down|try again in \d|you'?ve reached|"
    r"daily (limit|cap)|out of (free )?credits|upgrade to (pro|plus|premium)|"
    r"please wait a (moment|few)|temporarily unavailable)",
    re.I,
)
LOGIN_RE = re.compile(
    r"(sign in|log in|create (a |an )?account|continue with (google|facebook|apple|microsoft|github)|"
    r"get started for free|must (log in|sign in)|please (log|sign) ?in)",
    re.I,
)
# An age gate asks the person for a personal declaration; we never answer it for them.
BROKEN_RE = re.compile(r"(access denied|are you a robot|unusual traffic|verify you are a human|confirm your age|what year were you born|date of birth|verify your age|what should i call you|preferred name)", re.I)
# Signals that the provider actually went and looked something up (section 17).
WEB_RESEARCH_RE = re.compile(
    r"(searched the web|browsed|browsing|searching the web|web results|"
    r"sources\b|citations\b|looked up|according to \d+ sources|grounded|"
    r"deep research|researching|finding (out|information))",
    re.I,
)
# The section headings our research prompt asks for, in capitals; a lowercase "evidence" is just a word.
GLUED_HEADING_RE = re.compile(
    r"(\S)(DIRECT ANSWER|KEY CLAIMS|EVIDENCE|SOURCE LINKS|SOURCE DATES|UNCERTAINTIES|CONTRADICTORY EVIDENCE|WHAT I MAY BE WRONG ABOUT)(?=[ \t]*(?:\n|$))"
)
STALE_UI_NOISE = re.compile(
    r"^\s*(copied!?|copy|regenerate|good response|bad response|share|more|show more|"
    r"voice input|try again|retry|feedback|was this helpful\??|(chatgpt|gemini|copilot|you) said:?|#{1,6}\s*:?)\s*$",
    re.I,
)

# Selector-free entry point: several of these sites prefill the composer from a
# query parameter. It is the last-resort path when the composer cannot be typed
# into, and it is only accepted after the page is read back and the text is
# actually there -- a wrong parameter name is simply ignored.
URL_PARAMS: dict[str, list[str]] = {
    "chatgpt": ["q", "prompt"],
    "gemini": ["q", "prompt"],
    "copilot": ["q", "prompt", "showconv"],
    "meta_ai": ["q", "prompt"],
    "le_chat": ["text", "prompt", "q"],
    "pi": ["text", "q"],
    "qwen": ["prompt", "q"],
    "deepseek": ["prompt", "q"],
    "*": ["q", "prompt", "text", "query"],
}


def _now() -> float:
    return time.time()


_CHIP_TAIL_RE = re.compile(
    r"^(?P<body>.*?[.!?)\]])[ \t]+(?P<label>[A-Z][\w&'.-]*(?:[ \t]+[A-Z0-9][\w&'.-]*){0,3})(?P<plus>[ \t]*\+\d+)?[ \t]*$"
)


_BULLET_RE = re.compile(r"\s*(?:[-*\u2022]|\d+[.)])\s")
_BULLET_TAIL_RE = re.compile(r"^(?P<body>.{20,}?[.!?])[ \t]+(?P<label>[A-Z0-9][^.!?]{0,48})$")


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def strip_chip_labels(text: str, citations: list[Any] | None = None) -> str:
    """Remove inline citation-chip labels the site renders after a sentence ("... comfort. Headphones Addict").

    They are UI, not prose, and left in they get glued onto claims. A trailing capitalised label is dropped only
    when it is a "+N" overflow chip, or when it names one of the response's own citations (domain or title).
    """
    from urllib.parse import urlsplit

    names: list[str] = []
    for c in citations or []:
        url = getattr(c, "url", None) or (c.get("url") if isinstance(c, dict) else "") or ""
        title = getattr(c, "title", None) or (c.get("title") if isinstance(c, dict) else "") or ""
        host = (urlsplit(url).hostname or "").lower()
        host = re.sub(r"^www\.", "", host)
        stem = host.split(".")[0] if host else ""
        for n in (_squash(stem), _squash(title)):
            if len(n) >= 4:
                names.append(n)
    out = []
    for line in (text or "").split("\n"):
        # A list item is one sentence ending in punctuation. If a short unpunctuated tail follows its full stop
        # ("... March 31, 1889. La tour Eiffel"), that tail is a source chip whatever its capitalisation.
        tail = _BULLET_TAIL_RE.match(line.rstrip()) if _BULLET_RE.match(line) else None
        if tail and len(tail.group("label").split()) <= 4 and not re.search(r"\d", tail.group("label")):
            out.append(tail.group("body"))
            continue
        m = _CHIP_TAIL_RE.match(line.rstrip())
        if m:
            label = _squash(m.group("label"))
            known = len(label) >= 4 and any(label in n or n in label for n in names)
            # A list item is one sentence; a short capitalised tail after its full stop is a chip, even when the
            # site gave us no links to match it against (logged-out ChatGPT renders chips as bare text).
            bullet = bool(re.match(r"\s*(?:[-*\u2022]|\d+[.)])\s", line)) and len(m.group("label").split()) <= 2
            if m.group("plus") or known or bullet:
                line = m.group("body")
        out.append(line)
    return "\n".join(out)


class ChatAdapter:
    name = "generic"
    #: chat UIs use ask(); the search adapter overrides this to False.
    is_chat = True

    def __init__(
        self,
        engine: BrowserEngine,
        settings: Settings,
        provider: str,
        cfg: ProviderConfig,
        selectors: SelectorSet | None = None,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.provider = provider
        self.cfg = cfg
        self.sel = selectors or selectors_for(provider)
        self._sel_dict = asdict(self.sel)
        self._pages_with_library: set[int] = set()
        self._prompt_sent = False
        # research id -> URL of that research's conversation on this site. One
        # question never shares a chat with another question; follow-ups within
        # one research continue the thread recorded here.
        self._threads: dict[str, str] = {}
        self._continue_thread = False
        self._research_id = ""
        self._tab_dirty = False
        self.cancel_token: CancelToken | None = None  # shared with the job; set by the runner
        self.vision_client = None  # tests inject a scripted client; otherwise built from settings.vision

    # ---------------------------------------------------------------- plumbing

    def _check_cancel(self) -> None:
        if self.cancel_token is not None:
            self.cancel_token.raise_if_cancelled()

    async def _sleep(self, seconds: float) -> None:
        """Sleep that a cancelled job cuts short."""
        if self.cancel_token is not None:
            await self.cancel_token.sleep(seconds)
        else:
            await asyncio.sleep(seconds)

    async def _close_tab_on_cancel(self) -> None:
        try:
            await self.engine.close_tab(self.provider)
        except Exception:  # noqa: BLE001 -- cancelling must not raise anything new
            pass

    async def _page(self, fresh: bool = False):
        url = self.cfg.new_chat_url or self.cfg.url
        if fresh and self.cfg.new_chat_url:
            url = self.cfg.new_chat_url
        page = await self.engine.open_research_page(self.provider, url)
        await self._install(page)
        return page

    async def _install(self, page) -> None:
        key = id(page)
        if key in self._pages_with_library:
            try:
                if await page.evaluate("!!(window.__omnibrain && window.__omnibrain.version === 3)"):
                    return
            except Exception:  # noqa: BLE001
                pass
            self._pages_with_library.discard(key)
        await page.evaluate(DOM_LIBRARY_JS)
        self._pages_with_library.add(key)

    async def _call(self, page, fn: str, *args: Any) -> Any:
        await self._install(page)
        script = f"(args) => window.__omnibrain['{fn}'](...args)"
        try:
            return await page.evaluate(script, list(args))
        except Exception as exc:  # noqa: BLE001
            # Navigation or a torn DOM: re-inject once and retry, then give up.
            self._pages_with_library.discard(id(page))
            try:
                await self._install(page)
                return await page.evaluate(script, list(args))
            except Exception as exc2:  # noqa: BLE001
                raise DOMUnavailable(f"{fn}: {exc2}") from exc

    def _threshold(self, name: str, hidden: bool) -> int:
        base = int(getattr(self.sel, name, 3000) or 3000)
        return base * 2 if hidden else base

    # ----------------------------------------------------------- readiness

    async def readiness(self, page) -> dict[str, Any]:
        payload = dict(self._sel_dict)
        payload["chat_mode"] = self.is_chat
        state = await self._call(page, "ready", payload)
        if not state:
            return {"state": "unknown", "reason": "probe returned nothing"}
        body_head = (state.get("bodyHead") or "") + " " + (state.get("title") or "")
        if state.get("blocked") or BROKEN_RE.search(body_head):
            state["state"] = "blocked"
        elif state.get("rateLimited"):
            state["state"] = "rate_limited"
        elif not state.get("inputHere") and (state.get("loginWall") or LOGIN_RE.search(body_head)):
            state["state"] = "login_wall"
        elif state.get("inputHere"):
            state["state"] = "ready"
        return state

    async def _settle(self, page) -> None:
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=self.settings.browser.nav_timeout_ms)
        except Exception:  # noqa: BLE001
            pass
        await page.wait_for_timeout(self.settings.browser.settle_ms)

    async def _escalate_focus(self, page) -> bool:
        """Live Chrome only: allow (once) bringing this provider's tab to the front after it failed in the background."""
        escalate = getattr(self.engine, "escalate_focus", None)
        if escalate is None or not hasattr(page, "check_gate"):
            return False
        first = escalate(self.provider)
        try:
            await page.bring_to_front()
        except Exception:  # noqa: BLE001
            pass
        return first

    def _dismiss_cfg(self, page) -> dict[str, Any]:
        """On the person's own Chrome (live driver) banners may be closed but never accepted / agreed to."""
        if hasattr(page, "check_gate"):
            return {**self._sel_dict, "safe_dismiss": True}
        return self._sel_dict

    async def prepare(self, page, emit: EventHook, round_no: int) -> dict[str, Any]:
        """Dismiss chrome that blocks the composer, then confirm we can type."""
        state: dict[str, Any] = {"ok": False}
        for attempt in range(3):
            try:
                clicked = await self._call(page, "dismiss", self._dismiss_cfg(page))
            except DOMUnavailable as exc:
                if needs_user_in(exc):
                    raise  # live-Chrome driver: a login/captcha/age page is the user's, reported by _ask_dom
                state = {"state": "broken", "reason": str(exc)}
                await emit("provider", f"{self.provider}: DOM unavailable ({exc})", self.provider, round_no)
                return state
            if clicked:
                await page.wait_for_timeout(600)
                await emit("provider", f"{self.provider}: dismissed {'/'.join(clicked[:2])}", self.provider, round_no)
            state = await self.readiness(page)
            if state["state"] == "ready":
                check_gate = getattr(page, "check_gate", None)
                if check_gate is not None:
                    await check_gate()  # live Chrome: a consent / announcement dialog over the page is the person's to answer
                state["ok"] = True
                return state
            if state["state"] in {"login_wall", "blocked", "rate_limited"}:
                return state
            if attempt == 1 and hasattr(page, "check_gate"):
                # Live Chrome drives new tabs in the background and never raises them by default. A few apps (meta.ai)
                # render nothing until the tab is shown: only after the composer failed to appear twice, raise THIS tab.
                await self._escalate_focus(page)
            await page.wait_for_timeout(1200 * (attempt + 1))
        return state

    # ------------------------------------------------------------------ ask

    async def ask(
        self,
        job_id: str,
        prompt: str,
        round_no: int = 1,
        emit: EventHook | None = None,
        continue_thread: bool = False,
    ) -> ProviderResponse:
        """DOM path first; the vision fallback only if that path ends BROKEN.

        ``job_id`` is the research id. Without ``continue_thread`` the prompt goes
        into a brand-new conversation; with it, into the conversation this
        research already has on this site (a new one if there is none).
        """
        emit = emit or (lambda *a, **k: asyncio.sleep(0))
        self._prompt_sent = False
        self._research_id = job_id
        self._continue_thread = bool(continue_thread and job_id in self._threads)
        response = await self._ask_dom(job_id, prompt, round_no, emit)
        if self._vision_eligible(response):
            response = await self._vision_fallback(response, prompt, round_no, emit)
        return response

    def _vision_eligible(self, response: ProviderResponse) -> bool:
        """Selector drift only, and only before anything was sent.

        BROKEN after the prompt went out ("no new message appeared") is not retried
        through vision: that would submit the question a second time.
        """
        return (
            self.is_chat
            and response.status == ProviderStatus.BROKEN
            and not self._prompt_sent
            and (getattr(self, "vision_client", None) is not None or self.settings.vision.provider != "disabled")
        )

    async def _vision_fallback(self, response: ProviderResponse, prompt: str, round_no: int, emit: EventHook) -> ProviderResponse:
        from backend.browser.vision import VisionFallback, VisionOutcome, apply_outcome

        fallback = VisionFallback(self.settings, client=getattr(self, "vision_client", None))
        if not fallback.available:
            return response
        dom_error = response.error
        await self._safe_emit(emit, "provider", f"{self.provider}: DOM path broken ({dom_error}); trying vision fallback", self.provider, round_no)
        try:
            page = await self.engine.open_research_page(self.provider, self.cfg.new_chat_url or self.cfg.url)
            outcome = await fallback.run(page, prompt, emit, self.provider, round_no)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- the fallback must never kill the job either
            outcome = VisionOutcome("failed", f"{type(exc).__name__}: {exc}"[:200])
        apply_outcome(response, outcome, prompt)
        await self._safe_emit(emit, "provider", f"{self.provider}: vision fallback -> {outcome.kind} {outcome.reason}".strip(), self.provider, round_no)
        return self._finish(response)

    async def _ask_dom(self, job_id: str, prompt: str, round_no: int, emit: EventHook) -> ProviderResponse:
        response = ProviderResponse(
            id=new_id("resp"),
            job_id=job_id,
            round=round_no,
            provider=self.provider,
            prompt=prompt,
            started_at=_now(),
            status=ProviderStatus.LAUNCHING,
        )
        attempts = max(1, (self.cfg.max_retries or 0) + 1)
        for attempt in range(attempts):
            if attempt and getattr(self.engine, "live", False):
                self.engine.escalate_focus(self.provider)  # a failed background attempt: the retry may raise this one tab
            try:
                outcome = await self._attempt(page_setup=bool(attempt), response=response, prompt=prompt, round_no=round_no, emit=emit)
            except asyncio.CancelledError:
                response.note(ProviderStatus.FAILED, error="cancelled")
                await self._close_tab_on_cancel()
                raise
            except DOMUnavailable as exc:
                gate = needs_user_in(exc)
                if gate:
                    return self._needs_user(response, gate)
                response.note(ProviderStatus.BROKEN, error=str(exc)[:400])
                return self._finish(response)
            except Exception as exc:  # noqa: BLE001 -- one provider must never kill the job
                gate = needs_user_in(exc)
                if gate:
                    return self._needs_user(response, gate)
                logging.getLogger("omnibrain.adapters").warning("%s attempt %d raised", self.provider, attempt + 1, exc_info=True)
                response.note(ProviderStatus.FAILED, error=f"{type(exc).__name__}: {exc}"[:400])
                await emit("provider", f"{self.provider}: attempt {attempt + 1} failed ({type(exc).__name__})", self.provider, round_no)
                if attempt + 1 < attempts:
                    response.status = ProviderStatus.LAUNCHING
                    continue
                return self._finish(response)
            if outcome:
                return self._finish(response)
            if response.status == ProviderStatus.RATE_LIMITED:
                # Never retry straight into a rate limit; the runner backs this site off.
                return self._finish(response)
            if (response.error or "").startswith(("readiness=login_wall", "readiness=blocked", "no-response-element: AI Mode")):
                # A login wall, captcha or age gate does not go away by asking again; report it and move on.
                return self._finish(response)
            if attempt + 1 >= attempts:
                # Out of attempts: keep the terminal status the last attempt set
                # (BROKEN, LOGGED_OUT, ...) instead of resetting it to LAUNCHING.
                return self._finish(response)
            response.status = ProviderStatus.LAUNCHING
            await emit("provider", f"{self.provider}: retrying ({response.error or 'no usable answer'})", self.provider, round_no)
        return self._finish(response)

    def _needs_user(self, response: ProviderResponse, gate: LiveChromeNeedsUser) -> ProviderResponse:
        """Live-Chrome driver reached a login / captcha / consent / age page: report it, never touch it."""
        login = gate.kind == "login"
        response.note(
            ProviderStatus.LOGGED_OUT if login else ProviderStatus.FAILED,
            error=f"readiness={'login_wall' if login else 'blocked'} ({gate})"[:400],
            detail=gate.url[:300],
        )
        return self._finish(response)

    async def _attempt(self, *, page_setup: bool, response: ProviderResponse, prompt: str, round_no: int, emit: EventHook) -> bool:
        self._check_cancel()
        page = await self._page(fresh=page_setup)
        await self._settle(page)
        page = await self._open_conversation(page)
        response.ui_url = page.url

        ready = await self.prepare(page, emit, round_no)
        if not ready.get("ok"):
            state = ready.get("state", "unknown")
            mapping = {
                "login_wall": ProviderStatus.LOGGED_OUT,
                "blocked": ProviderStatus.FAILED,
                "rate_limited": ProviderStatus.RATE_LIMITED,
            }
            response.note(
                mapping.get(state, ProviderStatus.BROKEN),
                error=f"readiness={state}" + (f" ({ready['gate']})" if state == "blocked" and ready.get("gate") else ""),
                detail=(ready.get("bodyHead") or "")[:300],
            )
            await emit("provider", f"{self.provider}: {response.status.value} — {state}", self.provider, round_no)
            return False

        # Typing and sending happen with this tab in front and the others waiting their turn: a background tab in
        # the shared window can swallow the keystrokes and then show no answer at all (seen live with three
        # providers asked at once). Once the prompt is sent the lock is released and the answers stream in parallel.
        lock = getattr(self.engine, "_focus_lock", None)
        if lock is not None:
            await lock.acquire()
        try:
            if lock is not None or getattr(self.engine, "live", False):
                # live Chrome: no shared lock (tabs are driven in parallel); just show this tab if foreground mode
                try:
                    await page.bring_to_front()
                except Exception:  # noqa: BLE001
                    pass
            baseline = await self._call(page, "baseline", self._sel_dict) or {"count": 0, "lastText": ""}
            await emit("provider", f"{self.provider}: composing prompt", self.provider, round_no)
            typed = await self._type_with_fallback(page, prompt)
            if not typed and await self._escalate_focus(page):
                typed = await self._type_with_fallback(page, prompt)  # live Chrome: retry once with this tab in front
            if not typed:
                response.note(ProviderStatus.BROKEN, error="could not place text in the composer")
                return False
            if getattr(self, "_navigated_for_prompt", False):
                # The URL prefill replaced the document, so the old baseline is stale.
                baseline = await self._call(page, "baseline", self._sel_dict) or baseline
                ready = await self.readiness(page)
                if not ready.get("inputHere"):
                    response.note(ProviderStatus.BROKEN, error="composer vanished after URL prefill")
                    return False

            await emit("provider", f"{self.provider}: prompt sent — waiting for answer", self.provider, round_no)
            await self._submit(page, prompt)
            if lock is not None:
                await asyncio.sleep(1.5)  # let the site take the message before the next tab comes forward
        finally:
            if lock is not None and lock.locked():
                lock.release()
        if getattr(self, "quick_answer_allowed", False):
            try:
                shortcut = await self._call(page, "quickAnswer", self._sel_dict)
            except Exception:  # noqa: BLE001
                shortcut = {}
            if isinstance(shortcut, dict) and shortcut.get("ok"):
                await emit("provider", f"{self.provider}: clicked '{shortcut.get('label')}' (QUICK mode)", self.provider, round_no)

        capture, status, error = await self._await_completion(page, baseline, response, emit, round_no)
        try:
            response.model_label = await self._call(page, "modelLabel")
        except Exception:  # noqa: BLE001
            response.model_label = None
        if status is not ProviderStatus.COMPLETED:
            response.note(status, error=error)
            # A timeout still yields usable partial text, so we harvest below.
        text = ""
        links: list[dict[str, Any]] = []
        if capture:
            text = (capture.get("text") or "").strip()
            links = capture.get("links") or []
        if not text and baseline.get("lastText"):
            # Guard against reading the previous turn as the new answer.
            response.note(ProviderStatus.FAILED, error="only the pre-existing message was visible")
            return False

        if not text:
            # Nothing was captured. Before calling that an empty answer, look at where the page is now: sending the
            # first message can redirect to a sign-in page (Le Chat, 2026-10-04) and the real state is LOGGED_OUT.
            try:
                after = await self.readiness(page)
            except Exception:  # noqa: BLE001 -- the diagnosis is best effort; the old message still applies
                after = {}
            if after.get("state") in {"login_wall", "blocked", "rate_limited"}:
                mapping = {"login_wall": ProviderStatus.LOGGED_OUT, "blocked": ProviderStatus.FAILED, "rate_limited": ProviderStatus.RATE_LIMITED}
                response.note(
                    mapping[after["state"]],
                    error=f"readiness={after['state']} after sending (page moved to: {page.url[:80]})",
                    detail=(after.get("bodyHead") or "")[:300],
                )
                return False
        response.answer_text = self._clean(text, prompt)
        response.raw_text = text
        items = (capture or {}).get("items") or []
        if items and all(i.get("mediaOnly") for i in items) and not response.answer_text:
            response.answer_text = "[provider returned image or non-text output; nothing to capture]"
            response.detail = "media-only answer, no text captured"
        response.citations = [Citation(**self._citation_shape(l)) for l in links if l and l.get("href")]
        response.fingerprint = hashlib.sha256((response.answer_text or "").encode("utf-8", "ignore")).hexdigest()[:16]
        self._assess_web_research(response)
        if not response.answer_text:
            response.note(ProviderStatus.FAILED, error="answer captured but empty after cleaning")
            return False
        if status == ProviderStatus.TIMEOUT and len(response.answer_text) < 40:
            response.note(ProviderStatus.TIMEOUT, error="timed out with only a fragment")
            return False
        response.status = status if status.terminal else ProviderStatus.COMPLETED
        await self._remember_thread_when_known(page, response)
        if status == ProviderStatus.TIMEOUT and len(response.answer_text) >= 200:
            response.detail = "answer truncated by timeout but usable"
            response.status = ProviderStatus.COMPLETED
        return True

    async def _open_conversation(self, page):
        """New research -> new chat; follow-up -> back to this research's own chat.

        Reusing the provider's tab is fine, but typing the next question into the
        previous question's conversation is not: the old answer is still on screen
        and can be read back as the new one.
        """
        new_url = self.cfg.new_chat_url or self.cfg.url
        try:
            if self._continue_thread:
                target = self._threads.get(self._research_id) or ""
                if target and target != new_url and page.url != target:
                    await page.goto(target, wait_until="domcontentloaded", timeout=self.settings.browser.nav_timeout_ms)
                    await self._settle(page)
            elif self._tab_dirty or (page.url or "").rstrip("/") != (new_url or "").rstrip("/"):
                await page.goto(new_url, wait_until="domcontentloaded", timeout=self.settings.browser.nav_timeout_ms)
                await self._settle(page)
                self._tab_dirty = False
        except Exception:  # noqa: BLE001 -- a failed navigation is reported by the readiness check that follows
            pass
        return page

    url_wait_s: float = 8.0

    async def _conversation_url(self, page) -> str:
        """The chat's own URL. Some sites (Gemini) put the conversation id in the address a moment AFTER the answer,
        so a read taken the instant the answer lands still shows the bare new-chat page."""
        base = (self.cfg.new_chat_url or self.cfg.url or "").rstrip("/")
        deadline = time.time() + self.url_wait_s
        while (page.url or "").rstrip("/") == base and time.time() < deadline and not self._continue_thread:
            await asyncio.sleep(0.4)
        if (page.url or "").rstrip("/") == base:
            found = await self._conversation_url_from_dom(page)
            if found:
                return found
        return page.url

    async def _conversation_url_from_dom(self, page) -> str | None:
        """Fallback when the address bar never changes: the site's own link to the open chat, if it shows one."""
        try:
            href = await page.evaluate(
                "() => { const a = document.querySelector('a[aria-current=\"page\"][href], a[aria-selected=\"true\"][href]');"
                " return a ? a.href : null; }"
            )
        except Exception:  # noqa: BLE001
            return None
        base = (self.cfg.new_chat_url or self.cfg.url or "").rstrip("/")
        return href if href and href.rstrip("/") != base and href.startswith("http") else None

    async def _remember_thread_when_known(self, page, response: ProviderResponse) -> None:
        url = await self._conversation_url(page)
        self._remember_thread(page, response, url)
        if (url or "").rstrip("/") == (self.cfg.new_chat_url or self.cfg.url or "").rstrip("/"):
            response.detail = ((response.detail or "") + " conversation id not exposed by the site; follow-ups continue in the open tab").strip()

    def _remember_thread(self, page, response: ProviderResponse, url: str | None = None) -> None:
        url = url or page.url
        self._tab_dirty = True
        response.conversation_url = url
        response.continued = self._continue_thread
        self._threads[self._research_id] = url
        while len(self._threads) > 20:
            self._threads.pop(next(iter(self._threads)))

    def _finish(self, response: ProviderResponse) -> ProviderResponse:
        if response.status == ProviderStatus.COMPLETED and response.answer_text:
            response.answer_text = strip_chip_labels(response.answer_text, response.citations)
        response.finished_at = _now()
        response.duration_s = round((response.finished_at - (response.started_at or response.finished_at)), 2)
        return response

    # ------------------------------------------------------------- interaction

    async def _type_with_fallback(self, page, prompt: str) -> bool:
        """Type into the composer; if that fails, try the URL prefill route.

        Returns True only when the text has been *read back* from the page, so a
        parameter the site ignores can never look like success.
        """
        if await self._type(page, prompt):
            self._navigated_for_prompt = False
            return True
        from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

        base = self.cfg.new_chat_url or self.cfg.url
        for param in URL_PARAMS.get(self.provider, URL_PARAMS["*"]):
            try:
                parts = urlsplit(base)
                query = dict(parse_qsl(parts.query))
                query[param] = prompt
                url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
                await page.goto(url, wait_until="domcontentloaded", timeout=self.settings.browser.nav_timeout_ms)
            except Exception:  # noqa: BLE001
                continue
            await page.wait_for_timeout(1500)
            try:
                await self._call(page, "dismiss", self._dismiss_cfg(page))
            except Exception:  # noqa: BLE001
                pass
            if await self._verify_composer(page, prompt):
                self._navigated_for_prompt = True
                return True
        self._navigated_for_prompt = False
        return False

    async def _type(self, page, prompt: str) -> bool:
        located = await self._call(page, "locateInput", self._sel_dict) or []
        if not located:
            return False
        handle = await page.evaluate_handle(
            "(cfg) => window.__omnibrain.pickEl(cfg, 'input')", self._sel_dict
        )
        element = handle.as_element()
        if element is None:
            return False
        try:
            await element.scroll_into_view_if_needed(timeout=4000)
            await element.click(timeout=4000)
        except Exception:  # noqa: BLE001
            try:
                await element.focus()
            except Exception:  # noqa: BLE001
                return False

        # Method 0: genuine browser input through CDP.
        ok = False
        try:
            await page.keyboard.insert_text(prompt)
            ok = await self._verify_composer(page, prompt)
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            result = await self._call(page, "typeInto", element, prompt)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {}
            ok = bool((result or {}).get("ok")) or await self._verify_composer(page, prompt)
        if not ok:
            # Last resort: real keystrokes. Slow but indistinguishable to the page.
            try:
                await page.keyboard.type(prompt, delay=6)
                ok = await self._verify_composer(page, prompt)
            except Exception:  # noqa: BLE001
                ok = False
        return bool(ok)

    async def _verify_composer(self, page, prompt: str) -> bool:
        probe = prompt[:24].strip()
        if not probe:
            return False
        try:
            found = await page.evaluate(
                "(needle) => { const n = (needle||'').toLowerCase();"
                "const els = [...document.querySelectorAll('textarea,[contenteditable=true],[role=textbox],input[type=text]')];"
                "for (const el of els) { const v = (el.value || el.innerText || el.textContent || '').toLowerCase().replace(/\\s+/g,' ');"
                "if (v.includes(n.trim())) return true; } return false; }",
                probe,
            )
        except Exception:  # noqa: BLE001
            found = False
        if found:
            return True
        # Very long prompts get truncated or normalised by editors; accept when
        # the composer holds a meaningful fraction of it.
        try:
            length = await page.evaluate(
                "() => { let m = 0; for (const el of document.querySelectorAll('textarea,[contenteditable=true],[role=textbox]')) {"
                "const v = (el.value || el.innerText || '').length; if (v > m) m = v; } return m; }"
            )
        except Exception:  # noqa: BLE001
            length = 0
        return bool(length and length >= min(40, int(len(prompt) * 0.4)))

    async def _submit(self, page, prompt: str = "") -> None:
        self._prompt_sent = True
        clicked = await self._call(page, "clickSend", self._sel_dict)
        if isinstance(clicked, str):
            try:
                clicked = json.loads(clicked)
            except json.JSONDecodeError:
                clicked = {}
        if clicked and clicked.get("ok"):
            # A scripted click is not a trusted event: some sites (seen live: chat.deepseek.com in a real Chrome)
            # ignore it and leave the prompt sitting in the composer. If it is still there, press Enter like a person.
            if not prompt:
                return
            try:
                await page.wait_for_timeout(1500)
                unsent = await self._verify_composer(page, prompt)
            except Exception:  # noqa: BLE001
                return
            if not unsent:
                return
        try:
            await page.keyboard.press("Enter")
        except Exception:  # noqa: BLE001
            await self._call(page, "pressEnter", None)

    # ------------------------------------------------------------- completion

    async def _await_completion(self, page, baseline: dict[str, Any], response: ProviderResponse, emit: EventHook, round_no: int) -> tuple[dict[str, Any] | None, ProviderStatus, str | None]:
        rc = self.settings.research
        poll_ms = max(300, rc.response_stability_poll_ms)
        started = _now()
        hard_deadline = started + self._threshold("hard_timeout_ms", False) / 1000.0

        last_sig = ""
        last_change = started
        stable_since: float | None = None
        best: dict[str, Any] | None = None
        saw_growth = False
        hidden = False
        last_nudge = started
        last_beat = started
        polls = 0

        while True:
            self._check_cancel()
            now = _now()
            elapsed = now - started
            try:
                capture = await self._call(page, "capture", self._sel_dict, baseline)
            except DOMUnavailable as exc:
                return best, ProviderStatus.BROKEN, f"page detached mid-answer: {exc}"
            polls += 1
            if not isinstance(capture, dict):
                await self._sleep(poll_ms / 1000)
                continue

            hidden = capture.get("visibility") == "hidden"
            text = (capture.get("text") or "").strip()
            plain_len = int(capture.get("plainLength") or 0)
            busy = capture.get("busy") or []

            if text and text != (baseline.get("lastText") or "")[: len(text)]:
                if plain_len > 0 and (not best or plain_len > int(best.get("plainLength") or 0)):
                    best = capture
                    saw_growth = True

            sig = f"{capture.get('blockCount')}|{plain_len}|{hashlib.sha256(text.encode('utf-8','ignore')).hexdigest()[:12]}"
            if sig != last_sig:
                last_sig = sig
                last_change = now
                stable_since = None
                if busy and response.status in {ProviderStatus.SEARCHING, ProviderStatus.RESPONDING, ProviderStatus.CONNECTED}:
                    pass
                if busy:
                    if response.status != ProviderStatus.RESPONDING:
                        await self._safe_emit(emit, "provider", f"{self.provider}: generating…", self.provider, round_no)
                        response.status = ProviderStatus.RESPONDING
                elif text:
                    if response.status != ProviderStatus.RESPONDING:
                        response.status = ProviderStatus.RESPONDING
            else:
                if stable_since is None:
                    stable_since = now

            quiet = now - last_change
            stable_ms = self._threshold("stable_ms", hidden)
            force_ms = self._threshold("force_capture_ms", hidden)
            never_ms = self._threshold("never_started_ms", hidden)
            ref_ms = self._threshold("source_block_ms", hidden)

            if not saw_growth and elapsed * 1000 > never_ms:
                return None, ProviderStatus.BROKEN, "no-response-element: no new assistant message appeared"

            reference_only = bool(best and best.get("items") and all(i.get("referenceOnly") for i in (best.get("items") or [])))
            needed = ref_ms if reference_only else stable_ms
            force_budget = (self._threshold("reference_force_ms", hidden) if reference_only else force_ms) / 1000.0
            stuck = self._threshold("stuck_ms", hidden) / 1000.0
            tiny = self._threshold("tiny_fragment_ms", hidden) / 1000.0

            # Decision order mirrors jumas45/no-api-llm-council's waitResult():
            # quiet-and-idle wins, a stuck busy flag can only delay to stuck_ms,
            # and nothing outruns the force budget.
            if saw_growth and text:
                if not busy and quiet >= needed / 1000.0:
                    return best, ProviderStatus.COMPLETED, None
                if quiet >= stuck:
                    # Text has been still for stuck_ms; a still-lit "generating"
                    # indicator is a stale UI artefact, not a reason to keep waiting.
                    return best, ProviderStatus.COMPLETED, None
            if saw_growth and plain_len and plain_len < 80 and not busy and elapsed > tiny:
                return best, ProviderStatus.COMPLETED, None
            if best and elapsed > force_budget:
                return best, ProviderStatus.COMPLETED, None
            if busy and now - last_beat > 12:
                last_beat = now
                await self._safe_emit(emit, "provider", f"{self.provider}: still streaming ({int(elapsed)}s)", self.provider, round_no)
            if elapsed > hard_deadline:
                status = ProviderStatus.TIMEOUT
                await self._safe_emit(emit, "provider", f"{self.provider}: timed out after {int(elapsed)}s", self.provider, round_no)
                return best, status, f"hard timeout after {int(elapsed)}s"

            # A background tab in the shared window stops repainting its stream.
            # Nudge it forward only when text has genuinely stalled, and through
            # the engine's lock so parallel providers are not fighting for focus.
            stalled = (now - last_change) > self.settings.browser.focus_nudge_after_s
            if hidden and stalled and now - last_nudge > self.settings.browser.focus_nudge_after_s:
                last_nudge = now
                await self.engine.focus_tab(page)
                await self._sleep(0.25)
                try:
                    await page.mouse.wheel(0, 1)
                    await page.mouse.wheel(0, -1)
                except Exception:  # noqa: BLE001
                    pass

            await self._sleep(poll_ms / 1000)

    @staticmethod
    async def _safe_emit(emit: EventHook, *args: Any) -> None:
        try:
            out = emit(*args)
            if asyncio.iscoroutine(out):
                await out
        except Exception:  # noqa: BLE001
            pass

    # --------------------------------------------------------------- extraction

    def _clean(self, text: str, prompt: str = "") -> str:
        lines = []
        for line in (text or "").splitlines():
            if STALE_UI_NOISE.match(line):
                continue
            lines.append(line.rstrip())
        out = "\n".join(lines).strip()
        # Live Gemini output glued the next heading onto the previous line ("...984 feet).EVIDENCE").
        out = GLUED_HEADING_RE.sub(r"\1\n\2", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        # Sites echo the prompt back as the first line of the answer, or as a
        # "Conversation so far" preamble in debate-style threads. Capturing that
        # as the answer would feed a provider its own words and count it as a
        # second source. (echo handling per AmT42 / jumas45 sanitizers.)
        if prompt:
            head = " ".join(prompt.split())[:60].lower()
            lowered = " ".join(out.split()).lower()
            if head and lowered.startswith(head[:40]):
                cut = out.lower().find(" ".join(prompt.split())[40:80].lower())
                out = out[cut:].lstrip() if cut > 0 else out[len(head):].lstrip()
            for marker in ("conversation so far", "here's a new message", "previous answer"):
                idx = out.lower().rfind(marker)
                if idx > 0 and idx < len(out) - 80:
                    out = out[idx:].split("\n", 1)[-1].strip()
        return out.strip()

    @staticmethod
    def _citation_shape(link: dict[str, Any]) -> dict[str, Any]:
        return {
            "url": link.get("href") or "",
            "title": (link.get("title") or "")[:220] or None,
            "snippet": (link.get("snippet") or "")[:400] or None,
            "published": None,
            "provider": None,
            "marker": link.get("marker"),
        }

    def _assess_web_research(self, response: ProviderResponse) -> None:
        from backend.models import WebResearchStatus

        blob = f"{response.answer_text} {json.dumps([c.model_dump() for c in response.citations])[:4000]}"
        signals = [m.group(0) for m in WEB_RESEARCH_RE.finditer(blob)][:6]
        response.web_research_signals = signals
        external = [c for c in response.citations if not self._is_first_party(c.url)]

        if response.status != ProviderStatus.COMPLETED:
            response.web_research_status = WebResearchStatus.UNKNOWN
        elif external:
            response.web_research_status = WebResearchStatus.PERFORMED
        elif len(signals) >= 2:
            response.web_research_status = WebResearchStatus.PERFORMED
        elif signals:
            response.web_research_status = WebResearchStatus.UNKNOWN
        else:
            response.web_research_status = WebResearchStatus.FAILED_OR_UNCLEAR

    @staticmethod
    def _is_first_party(url: str) -> bool:
        return bool(
            re.match(
                r"^https?://([a-z0-9.-]*\.)?(chatgpt\.com|openai\.com|gemini\.google|gemini\.com|copilot\.microsoft|"
                r"bing\.com|meta\.ai|mistral\.ai|chat\.mistral\.ai|pi\.ai|google\.com/search)\b",
                (url or "").lower(),
            )
        )


class DOMUnavailable(RuntimeError):
    """The page's JS world went away (navigation, crash, torn DOM)."""
