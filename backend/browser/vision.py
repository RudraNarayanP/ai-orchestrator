"""Vision fallback (spec 3, Level 2): read the page as pixels.

This runs ONLY when the DOM path has already reported BROKEN (selector drift) and
the prompt has not been sent yet. It is a way to keep working when a site changes
its markup or draws its UI on a canvas -- it is never a way past a site's controls:

* a captcha / human-check is detected from the DOM *and* from what the vision
  model sees; either is enough to stop. We never click it, never solve it;
* a login, sign-up or payment control is never clicked. Every click target is
  checked against the live DOM (``elementFromPoint``) before the mouse moves, so
  a model that points at the wrong thing cannot make us press it;
* the cap is ``vision.max_calls`` model requests per attempt.

An answer obtained this way is marked ``detail="vision fallback"`` and carries no
citations (nothing was read from the DOM), so downstream weighting treats it as an
unsourced transcription that still needs independent evidence.

Nothing here has been verified against a real vision model on a real site; the
tests drive it with a scripted client against local fixtures (see AGENTS.md).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from backend.models import ProviderResponse, ProviderStatus, WebResearchStatus
from backend.verification.llm import Endpoint, LLMClient, extract_json

VISION_DETAIL = "vision fallback"

# Anything that looks like a control we must never operate.
DENY_RE = re.compile(
    r"(captcha|recaptcha|hcaptcha|turnstile|i'?m not a robot|not a robot|verify (that )?you('?re| are) (a )?human|"
    r"log ?in|sign ?in|sign ?up|register|create (an? )?account|continue with|forgot password|password|"
    r"\bpay(ment)?\b|checkout|\bbuy\b|purchase|subscribe|upgrade|billing|credit card|card number|cvv|"
    r"place order|add to cart|donate|\bgo pro\b|get plus)",
    re.I,
)
CAPTCHA_SCENES = {"captcha", "human_check", "robot_check", "challenge"}
LOGIN_SCENES = {"login", "signin", "sign_in", "signup", "sign_up"}
PAYMENT_SCENES = {"payment", "checkout", "paywall"}

HAZARD_JS = r"""
() => {
  const reasons = [];
  const frameRe = /recaptcha|hcaptcha|turnstile|challenges\.cloudflare|arkoselabs|funcaptcha|geetest|captcha/i;
  for (const f of document.querySelectorAll('iframe')) {
    const hint = `${f.src || ''} ${f.title || ''} ${f.name || ''}`;
    if (frameRe.test(hint)) reasons.push('captcha iframe');
  }
  const sel = '[class*="captcha" i], [id*="captcha" i], .g-recaptcha, .h-captcha, .cf-turnstile, [data-sitekey]';
  if (document.querySelector(sel)) reasons.push('captcha element');
  const text = ((document.body && document.body.innerText) || '').slice(0, 6000);
  if (/captcha|verify (that )?you('re| are) (a )?human|are you a robot|i'?m not a robot|unusual traffic/i.test(text)) {
    reasons.push('human-check text');
  }
  const pwd = [...document.querySelectorAll('input[type=password]')].some(e => {
    const r = e.getBoundingClientRect(); return r.width > 4 && r.height > 4;
  });
  return {captcha: reasons, password: pwd, w: innerWidth, h: innerHeight};
}
"""

POINT_JS = r"""
(pt) => {
  const el = document.elementFromPoint(pt.x, pt.y);
  if (!el) return {tag: null, text: ''};
  const bits = [];
  let n = el;
  for (let i = 0; i < 5 && n; i++, n = n.parentElement) {
    const own = (n.children.length === 0 || (n.innerText || '').length < 60) ? (n.innerText || n.value || '') : '';
    bits.push([n.tagName.toLowerCase(), n.id || '', n.getAttribute('aria-label') || '', n.getAttribute('title') || '',
      n.getAttribute('name') || '', n.getAttribute('autocomplete') || '', n.getAttribute('data-testid') || '',
      n.getAttribute('href') || '', n.getAttribute('src') || '', (n.className && n.className.toString()) || '',
      own.slice(0, 80)].join(' '));
  }
  return {tag: el.tagName.toLowerCase(), type: el.getAttribute('type') || '', text: bits.join(' | ')};
}
"""

LOCATE_INSTRUCTION = """You are looking at a screenshot of a web page, {w}x{h} pixels. Coordinates are
pixels from the top-left of the image.

Decide what the page is and, if it is a chat or question box, where to type.
Answer with JSON only:
{{"scene": "chat" | "captcha" | "login" | "payment" | "other",
  "composer": {{"x": int, "y": int, "label": "what you see there"}} | null,
  "send": {{"x": int, "y": int, "label": "what you see there"}} | null}}

Rules: scene is "captcha" for ANY human check, image puzzle, or "I'm not a robot" box.
scene is "login" if the page is asking to sign in or create an account. "payment" for
any paywall, checkout or card form. Never give coordinates for those. "composer" is
the text box where a question is typed; "send" is the button that submits it, or null
if there is none (the Enter key will be used)."""

TRANSCRIBE_INSTRUCTION = """This is a screenshot of a chat page, {w}x{h} pixels, taken after a question was sent.
Transcribe the assistant's latest reply EXACTLY as written on screen. Do not
summarise, do not answer the question yourself, do not include the user's own message
or any buttons and menus.
Answer with JSON only:
{{"scene": "chat" | "captcha" | "login" | "payment" | "other",
  "answer": "the reply text, or empty if none is visible yet",
  "complete": true | false}}
"complete" is false if the reply is still being written (cursor, spinner, stop button,
text cut off mid-sentence)."""


class VisionClient(Protocol):
    async def ask(self, png: bytes, task: str, instruction: str) -> dict[str, Any] | None:  # pragma: no cover
        ...


class OpenAIVisionClient:
    """Screenshot -> JSON via any OpenAI-compatible endpoint with image input."""

    def __init__(self, cfg: Any) -> None:
        self.endpoint = Endpoint.from_config(cfg)
        self.llm = LLMClient(self.endpoint)
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.endpoint.enabled

    async def ask(self, png: bytes, task: str, instruction: str) -> dict[str, Any] | None:
        url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ]
        reply = await self.llm.complete(messages, retries=0)  # type: ignore[arg-type]
        if not reply.ok:
            self.last_error = reply.error
            return None
        parsed = extract_json(reply.text)
        if parsed is None:
            self.last_error = "vision model reply was not JSON"
        return parsed


@dataclass
class VisionOutcome:
    kind: str
    """answered | blocked | logged_out | unavailable | failed"""

    reason: str = ""
    answer: str = ""
    complete: bool = True
    calls: int = 0
    log: list[str] = field(default_factory=list)


Emit = Callable[..., Awaitable[None]]


def click_allowed(point_info: dict[str, Any], label: str = "") -> tuple[bool, str]:
    """Pure decision: may the mouse press whatever is at this point?"""
    blob = f"{point_info.get('text') or ''} {label or ''}"
    if point_info.get("tag") in {None, "iframe", "embed", "object"}:
        return False, "target is empty or inside a frame"
    if (point_info.get("type") or "").lower() == "password":
        return False, "target is a password field"
    hit = DENY_RE.search(blob)
    if hit:
        return False, f"target looks like a restricted control ({hit.group(0)!r})"
    return True, ""


class VisionFallback:
    def __init__(self, settings: Any, client: VisionClient | None = None) -> None:
        self.settings = settings
        self.cfg = settings.vision
        self.client: VisionClient | None = client
        if self.client is None:
            real = OpenAIVisionClient(self.cfg)
            self.client = real if real.enabled else None

    @property
    def available(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------ helpers

    async def _shot(self, page) -> bytes:
        return await page.screenshot(type="png", scale="css", full_page=False)

    async def _hazards(self, page) -> dict[str, Any]:
        try:
            return await page.evaluate(HAZARD_JS)
        except Exception:  # noqa: BLE001
            return {"captcha": [], "password": False, "w": 0, "h": 0}

    @staticmethod
    def _point(block: Any, w: int, h: int) -> tuple[int, int, str] | None:
        if not isinstance(block, dict):
            return None
        try:
            x, y = int(round(float(block.get("x")))), int(round(float(block.get("y"))))
        except (TypeError, ValueError):
            return None
        if not (0 <= x < w and 0 <= y < h):
            return None
        return x, y, str(block.get("label") or "")

    async def _vet(self, page, block: Any, w: int, h: int, what: str) -> tuple[tuple[int, int] | None, str]:
        """Resolve a model-supplied target and check it against the live DOM.

        Both targets are vetted before the first click, so a refused send button
        cannot leave a half-typed prompt behind.
        """
        pt = self._point(block, w, h)
        if pt is None:
            return None, f"no usable {what} coordinates"
        x, y, label = pt
        try:
            info = await page.evaluate(POINT_JS, {"x": x, "y": y})
        except Exception as exc:  # noqa: BLE001
            return None, f"could not inspect {what} target: {exc}"
        ok, why = click_allowed(info or {}, label)
        if not ok:
            return None, f"refused to click {what}: {why}"
        return (x, y), ""

    @staticmethod
    def _scene_outcome(scene: str, calls: int, log: list[str]) -> VisionOutcome | None:
        scene = (scene or "").strip().lower()
        if scene in CAPTCHA_SCENES:
            return VisionOutcome("blocked", "human check (captcha) on screen; not touched", calls=calls, log=log)
        if scene in PAYMENT_SCENES:
            return VisionOutcome("blocked", "payment or paywall on screen; not touched", calls=calls, log=log)
        if scene in LOGIN_SCENES:
            return VisionOutcome("logged_out", "login wall on screen; sign in yourself via run.py login", calls=calls, log=log)
        return None

    # --------------------------------------------------------------------- run

    async def run(self, page, prompt: str, emit: Emit | None = None, provider: str = "", round_no: int = 1) -> VisionOutcome:
        log: list[str] = []

        async def say(msg: str) -> None:
            log.append(msg)
            if emit is not None:
                try:
                    await emit("provider", f"{provider}: {msg}", provider, round_no)
                except Exception:  # noqa: BLE001
                    pass

        if self.client is None:
            return VisionOutcome("unavailable", "no vision endpoint configured", log=log)

        calls = 0
        max_calls = max(2, int(self.cfg.max_calls))

        hazards = await self._hazards(page)
        if hazards.get("captcha"):
            await say(f"vision fallback not used - human check on page ({hazards['captcha'][0]}); not bypassing it")
            return VisionOutcome("blocked", f"human check on page ({hazards['captcha'][0]})", log=log)
        if hazards.get("password"):
            return VisionOutcome("logged_out", "password field on page; sign in yourself via run.py login", log=log)

        w = int(hazards.get("w") or 0) or 1280
        h = int(hazards.get("h") or 0) or 800
        before = await self._shot(page)
        located = await self.client.ask(before, "locate", LOCATE_INSTRUCTION.format(w=w, h=h))
        calls += 1
        if not isinstance(located, dict):
            return VisionOutcome("failed", "vision model returned nothing usable", calls=calls, log=log)
        stop = self._scene_outcome(str(located.get("scene") or ""), calls, log)
        if stop:
            await say(f"vision fallback stopped: {stop.reason}")
            return stop

        composer, why = await self._vet(page, located.get("composer"), w, h, "composer")
        if composer is None:
            return VisionOutcome("failed", why, calls=calls, log=log)
        send_block = located.get("send")
        send = None
        if send_block:
            send, why = await self._vet(page, send_block, w, h, "send button")
            if send is None:
                return VisionOutcome("failed", why, calls=calls, log=log)

        await say("vision fallback: typing the prompt into the composer it located")
        await page.mouse.click(*composer)
        # One line: a newline keypress would submit a half-typed prompt.
        await page.keyboard.type(" ".join(prompt.split()), delay=2)
        if send:
            await page.mouse.click(*send)
        else:
            await page.keyboard.press("Enter")

        baseline_hash = hashlib.sha1(before).hexdigest()
        deadline = time.time() + float(self.cfg.wait_s)
        stable_s = float(self.cfg.stable_s)
        last_hash = ""
        last_change = time.time()
        changed = False
        best = ""
        while time.time() < deadline:
            await asyncio.sleep(0.8)
            hazards = await self._hazards(page)
            if hazards.get("captcha"):
                await say(f"human check appeared after sending ({hazards['captcha'][0]}); stopped, discarded")
                return VisionOutcome("blocked", f"human check appeared ({hazards['captcha'][0]})", calls=calls, log=log)
            shot = await self._shot(page)
            digest = hashlib.sha1(shot).hexdigest()
            if digest != last_hash:
                last_hash = digest
                last_change = time.time()
                if digest != baseline_hash:
                    changed = True
                continue
            if not changed or time.time() - last_change < stable_s:
                continue
            if calls >= max_calls:
                break
            reading = await self.client.ask(shot, "transcribe", TRANSCRIBE_INSTRUCTION.format(w=w, h=h))
            calls += 1
            last_change = time.time()  # do not re-ask the same frame in a tight loop
            if not isinstance(reading, dict):
                continue
            stop = self._scene_outcome(str(reading.get("scene") or ""), calls, log)
            if stop:
                return stop
            text = str(reading.get("answer") or "").strip()
            if text and self._is_echo(text, prompt):
                text = ""
            if text:
                best = text
            if text and reading.get("complete") is not False:
                return VisionOutcome("answered", answer=text, calls=calls, log=log)
        if best:
            return VisionOutcome("answered", "wait ended before the reply looked finished", answer=best, complete=False, calls=calls, log=log)
        return VisionOutcome("failed", "no reply became readable on screen", calls=calls, log=log)

    @staticmethod
    def _is_echo(text: str, prompt: str) -> bool:
        a = " ".join(text.lower().split())
        b = " ".join(prompt.lower().split())
        return bool(a) and (a == b or (len(a) > 20 and a in b))


def apply_outcome(response: ProviderResponse, outcome: VisionOutcome, prompt: str) -> bool:
    """Write a vision result into the provider response. Returns True if answered."""
    if outcome.kind == "answered":
        response.answer_text = outcome.answer
        response.raw_text = outcome.answer
        response.citations = []
        response.fingerprint = hashlib.sha256(outcome.answer.encode("utf-8", "ignore")).hexdigest()[:16]
        response.web_research_status = WebResearchStatus.UNKNOWN
        response.web_research_signals = []
        response.error = None
        response.status = ProviderStatus.COMPLETED
        response.detail = VISION_DETAIL if outcome.complete else f"{VISION_DETAIL} (possibly partial: {outcome.reason})"
        return True
    if outcome.kind == "blocked":
        response.note(ProviderStatus.FAILED, error="readiness=blocked", detail=f"{VISION_DETAIL}: {outcome.reason}")
        response.status = ProviderStatus.FAILED
    elif outcome.kind == "logged_out":
        response.note(ProviderStatus.LOGGED_OUT, error="readiness=login_wall", detail=f"{VISION_DETAIL}: {outcome.reason}")
    elif outcome.kind == "failed":
        response.detail = f"{VISION_DETAIL} failed: {outcome.reason}"
    return False