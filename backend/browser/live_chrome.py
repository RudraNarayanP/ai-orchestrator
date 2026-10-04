"""Optional live-Chrome driver (``browser.driver: chrome_use``).

OmniBrain's default driver is a dedicated Playwright Chrome profile that never touches
the browser you work in. This module is the *opt-in* alternative: it drives a tab in
your already-signed-in Chrome through the third-party ``chrome-use`` CLI (Apache-2.0,
https://github.com/leeguooooo/chrome-use) and its Chrome extension, so the provider
sites see the login you already have instead of a second one.

What this module guarantees, in OmniBrain's own code and independent of chrome-use:

* **Provider domains only.** Every command that names a URL, and every command that
  acts on a tab, is checked against a hard allowlist (the AI provider hosts plus
  ``google.com/search?udm=50``). The tab's *current* URL is re-read before each action,
  so a redirect or in-page navigation off the allowlist stops the run instead of being
  acted on.
* **New tabs only.** Tabs are created by this engine and are the only ones it ever
  selects. It never adopts, lists-and-picks, or reads one of your existing tabs.
* **A closed set of chrome-use verbs.** Cookies, state save/load, auth, storage,
  network routing, init scripts, humanize and every other flag are refused before a
  subprocess is started (see ``_check_argv``). The child process also gets
  ``AGENT_BROWSER_STEALTH=0`` and ``AGENT_BROWSER_HUMANIZE=off`` so chrome-use's
  webdriver-hiding and human-typing options stay off.
* **Gates belong to the human.** A captcha / Cloudflare check / age gate / consent or
  sign-in page is never touched: it raises ``LiveChromeNeedsUser`` and the adapter
  reports the provider as logged-out / blocked, exactly as it does for the default driver.

The shim (``LivePage``) implements only the slice of Playwright's ``Page`` the adapters
use, so no adapter changes are needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlsplit

from backend.settings import Settings

# ------------------------------------------------------------------ errors


class LiveChromeError(RuntimeError):
    """chrome-use failed, or answered something unusable."""


class LiveChromeUnavailable(LiveChromeError):
    """chrome-use is not installed / not runnable."""


class LiveChromeRefused(LiveChromeError):
    """OmniBrain's own policy refused the command (nothing was sent to chrome-use)."""


class LiveChromeNeedsUser(LiveChromeRefused):
    """The tab is on a login / captcha / consent / age page: a human has to act."""

    def __init__(self, kind: str, url: str, detail: str = "") -> None:
        self.kind = kind  # login | captcha | consent | age
        self.url = url
        self.detail = detail
        super().__init__(
            f"needs you: {kind} page at {_short(url)}"
            + (f" ({detail})" if detail else "")
            + " -- complete it yourself in Chrome; OmniBrain does not touch it"
        )


def extension_problem(status: dict[str, Any]) -> str | None:
    """Read ``chrome-use status --json`` data: why the extension route is not usable, or None."""
    ext = status.get("extension")
    if not isinstance(ext, dict):
        return None
    if ext.get("hostInstalled") is False:
        return "the chrome-use native-messaging host is not registered (run: chrome-use extension install)"
    if ext.get("hostHealthy") is False:
        return "the chrome-use native-host launcher is broken (run: chrome-use doctor)"
    if ext.get("relayUp") is False:
        return "Chrome is not connected through the chrome-use extension (open Chrome and make sure the extension is installed and enabled)"
    return None


def _short(url: str, n: int = 90) -> str:
    return (url or "")[:n]


def needs_user_in(exc: BaseException | None) -> LiveChromeNeedsUser | None:
    """Find a NeedsUser anywhere in an exception chain (adapters wrap errors)."""
    seen = 0
    while exc is not None and seen < 8:
        if isinstance(exc, LiveChromeNeedsUser):
            return exc
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return None


# ------------------------------------------------------------------ allowlist

# Exact hosts (a leading ``www.`` is tolerated). Subdomains are NOT implied.
AI_HOSTS: frozenset[str] = frozenset(
    {
        "chatgpt.com",
        "gemini.google.com",
        "copilot.microsoft.com",
        "meta.ai",
        "chat.mistral.ai",
        "pi.ai",
        "chat.deepseek.com",
        "chat.qwen.ai",
    }
)
# google.com is allowed ONLY for Google's AI Mode: /search?udm=50.
GOOGLE_AI_HOST = "google.com"

_GATE_URL = re.compile(
    r"(^|[./_-])(accounts|login|signin|sign-in|sign_in|auth|sso|oauth|consent|captcha|challenges?|verify|age)([./_?=&-]|$)",
    re.I,
)
_GATE_KIND = (
    ("captcha", re.compile(r"captcha|challenge|turnstile|cf-chl", re.I)),
    ("age", re.compile(r"(^|[./_-])age([./_?=&-]|$)|birth|adult", re.I)),
    ("consent", re.compile(r"consent|cookie|terms", re.I)),
)


def _normalise_host(host: str) -> str:
    host = (host or "").lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _parts(url: str):
    try:
        return urlsplit((url or "").strip())
    except ValueError:
        return None


def url_allowed(url: str) -> bool:
    """True only for https pages on a provider host (or Google AI Mode)."""
    p = _parts(url)
    if p is None or p.scheme != "https" or p.username or p.password:
        return False
    try:
        if p.port not in (None, 443):
            return False
    except ValueError:
        return False
    host = _normalise_host(p.hostname or "")
    if host in AI_HOSTS:
        return True
    if host == GOOGLE_AI_HOST:
        return p.path == "/search" and parse_qs(p.query).get("udm") == ["50"]
    return False


def classify_gate(url: str) -> str | None:
    """Best guess at *why* an off-allowlist URL is a human-only page, or None."""
    p = _parts(url)
    if p is None:
        return None
    probe = f"{p.hostname or ''}{p.path}"
    if not _GATE_URL.search(probe):
        return None
    for kind, rx in _GATE_KIND:
        if rx.search(probe):
            return kind
    return "login"


def assert_allowed_url(url: str, *, where: str = "") -> None:
    """Raise unless ``url`` is a provider page. Gate-looking URLs raise NeedsUser."""
    if url_allowed(url):
        return
    kind = classify_gate(url)
    if kind:
        raise LiveChromeNeedsUser(kind, url, where)
    raise LiveChromeRefused(
        f"refused: {_short(url)!r} is not an AI provider page"
        + (f" ({where})" if where else "")
        + "; the live-Chrome driver only touches chatgpt.com, gemini.google.com, copilot.microsoft.com, "
        "meta.ai, chat.mistral.ai, pi.ai, chat.deepseek.com, chat.qwen.ai and google.com/search?udm=50"
    )


# ------------------------------------------------------------------ command policy

# The only chrome-use verbs this module will ever run, and the only sub-forms of them.
_ALLOWED_TAB = {"new", "select", "list", "close"}
_ALLOWED_GET = {"url", "title"}
_ALLOWED_KEYBOARD = {"inserttext", "type"}
_ALLOWED_MOUSE = {"wheel"}
_ALLOWED_FLAGS = {"--activate", "--stdin", "--full"}
_KEY = re.compile(r"^[A-Za-z0-9_+]{1,40}$")
_TAB_ID = re.compile(r"^t\d{1,6}$")
_NUM = re.compile(r"^-?\d{1,6}$")
# Named so a reviewer (and a test) can see exactly what is out of bounds.
FORBIDDEN_VERBS = frozenset(
    {
        "cookies", "state", "auth", "storage", "network", "addinitscript", "addscript",
        "removeinitscript", "humanize", "adopt", "connect", "extension", "profiles", "set",
        "download", "download-url", "upload", "script", "jev", "batch", "react", "record",
        "trace", "profiler", "inspect", "install", "upgrade", "skill", "mcp", "find-url",
    }
)

# Child-process environment: switch OFF chrome-use's stealth and humanising knobs and
# strip the ones that would hide automation or change the browser identity.
# AUTO_CONNECT=1 makes chrome-use attach through the extension and fail (rather than quietly launch
# a throwaway browser of its own) when the extension is not reachable.
CHILD_ENV_FORCE = {"AGENT_BROWSER_STEALTH": "0", "AGENT_BROWSER_HUMANIZE": "off", "AGENT_BROWSER_AUTO_CONNECT": "1"}
CHILD_ENV_DROP = (
    "AGENT_BROWSER_HIDE_CANVAS", "AGENT_BROWSER_BLOCK_WEBRTC", "AGENT_BROWSER_TIMEZONE",
    "AGENT_BROWSER_LOCALE", "AGENT_BROWSER_USER_AGENT", "AGENT_BROWSER_PROFILE",
    "AGENT_BROWSER_ALLOWED_DOMAINS", "AGENT_BROWSER_INIT_SCRIPTS", "AGENT_BROWSER_EXTENSIONS",
    "AGENT_BROWSER_PROVIDER", "AGENT_BROWSER_ENABLE", "AGENT_BROWSER_SESSION",
    "AGENT_BROWSER_FORCE_LAUNCH", "AGENT_BROWSER_NO_AUTO_CONNECT", "AGENT_BROWSER_ENGINE", "AGENT_BROWSER_CDP",
    "AGENT_BROWSER_ARGS", "AGENT_BROWSER_EXECUTABLE_PATH", "AGENT_BROWSER_PROXY", "CI",
)


def _check_argv(args: Sequence[str]) -> None:
    """Refuse anything outside the closed command set BEFORE a process is started."""
    if not args:
        raise LiveChromeRefused("refused: empty chrome-use command")
    verb, rest = args[0], list(args[1:])
    if verb in FORBIDDEN_VERBS:
        raise LiveChromeRefused(f"refused: chrome-use {verb!r} is never used by OmniBrain")
    # ``keyboard type <text>`` carries free text; every other argument is checked as a possible flag.
    free_text = {2} if verb == "keyboard" and rest[:1] == ["type"] else set()
    for i, a in enumerate(args):
        if i in free_text:
            if a.startswith("-"):
                raise LiveChromeRefused("refused: typed text starting with '-' could be read as a chrome-use flag")
            continue
        if a.startswith("-") and a not in _ALLOWED_FLAGS and not _NUM.match(a):
            raise LiveChromeRefused(f"refused: chrome-use flag {a!r} is not allowed")

    def need(ok: bool, what: str) -> None:
        if not ok:
            raise LiveChromeRefused(f"refused: chrome-use {verb} {what}")

    if verb == "open":
        need(len(rest) == 1, "takes exactly one URL")
        assert_allowed_url(rest[0], where="open")
    elif verb == "tab":
        need(bool(rest) and rest[0] in _ALLOWED_TAB, f"only supports {sorted(_ALLOWED_TAB)}")
        sub, tail = rest[0], [r for r in rest[1:] if r != "--activate"]
        if sub == "new":
            need(len(tail) == 1, "needs exactly one provider URL (no blank or foreign tabs)")
            assert_allowed_url(tail[0], where="tab new")
        elif sub in {"select", "close"}:
            need(len(tail) == 1 and bool(_TAB_ID.match(tail[0])), "needs a tab id like t3 (a tab this run created)")
        else:
            need(not tail, "takes no arguments")
    elif verb == "get":
        need(len(rest) == 1 and rest[0] in _ALLOWED_GET, f"only supports {sorted(_ALLOWED_GET)}")
    elif verb == "eval":
        need(rest == ["--stdin"], "must read its script from stdin")
    elif verb == "keyboard":
        need(bool(rest) and rest[0] in _ALLOWED_KEYBOARD, f"only supports {sorted(_ALLOWED_KEYBOARD)}")
        if rest[0] == "inserttext":
            need(rest[1:] == ["--stdin"], "inserttext must read stdin")
        else:
            need(len(rest) == 2, "type takes exactly the text")
    elif verb == "press":
        need(len(rest) == 1 and bool(_KEY.match(rest[0])), "takes one key name")
    elif verb == "mouse":
        need(bool(rest) and rest[0] in _ALLOWED_MOUSE and all(_NUM.match(r) for r in rest[1:]), "only supports wheel <dy> <dx>")
    elif verb == "click":
        need(len(rest) == 2 and all(_NUM.match(r) for r in rest), "only takes viewport coordinates <x> <y>")
    elif verb == "screenshot":
        need(len([r for r in rest if r != "--full"]) <= 1, "takes at most a path")
    elif verb == "status":
        need(not rest, "takes no arguments")
    else:
        raise LiveChromeRefused(f"refused: chrome-use {verb!r} is not in OmniBrain's allowed command set")


def resolve_chrome_use(configured: str | None = None) -> str:
    """The chrome-use executable: the configured path, else PATH, else where
    scripts/install_chrome_use.ps1 puts it (%LOCALAPPDATA%\\Programs\\chrome-use)."""
    if configured:
        return configured
    found = shutil.which("chrome-use")
    if found:
        return found
    base = os.environ.get("LOCALAPPDATA")
    if base:
        candidate = Path(base) / "Programs" / "chrome-use" / "chrome-use.exe"
        if candidate.exists():
            return str(candidate)
    return "chrome-use"


class ChromeUseRunner:
    """Runs ``chrome-use --session S --json <args>`` and returns the ``data`` object."""

    def __init__(
        self,
        argv_prefix: Sequence[str] = ("chrome-use",),
        *,
        session: str = "omnibrain",
        browser: str | None = None,
        timeout_s: float = 60.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self.argv_prefix = list(argv_prefix)
        self.session = session
        self.browser = browser
        self.timeout_s = timeout_s
        self.extra_env = dict(env or {})
        self.calls = 0

    def build_argv(self, args: Sequence[str]) -> list[str]:
        argv = [*self.argv_prefix, "--session", self.session]
        if self.browser:
            argv += ["--browser", self.browser]
        return [*argv, "--json", *args]

    def build_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in CHILD_ENV_DROP}
        env.update(self.extra_env)
        env.update(CHILD_ENV_FORCE)
        return env

    def _run_sync(self, argv: list[str], stdin: str | None, timeout: float) -> subprocess.CompletedProcess:
        kwargs: dict[str, Any] = {"input": stdin.encode("utf-8")} if stdin is not None else {"stdin": subprocess.DEVNULL}
        return subprocess.run(argv, capture_output=True, timeout=timeout, env=self.build_env(), **kwargs)

    async def run(self, *args: str, stdin: str | None = None, timeout: float | None = None) -> dict[str, Any]:
        args = tuple(str(a) for a in args)
        _check_argv(args)  # policy first: nothing below runs for a refused command
        argv = self.build_argv(args)
        self.calls += 1
        try:
            done = await asyncio.to_thread(self._run_sync, argv, stdin, timeout or self.timeout_s)
        except FileNotFoundError as exc:
            raise LiveChromeUnavailable(
                "chrome-use is not installed (or not on PATH). Run scripts/install_chrome_use.ps1, "
                "or set browser.chrome_use_path in config/settings.yaml."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LiveChromeError(f"chrome-use {args[0]} timed out after {exc.timeout:g}s") from exc
        out = done.stdout.decode("utf-8", "replace")
        err = done.stderr.decode("utf-8", "replace").strip()
        payload = _parse_json_line(out)
        if payload is None:
            raise LiveChromeError(
                f"chrome-use {args[0]} returned no JSON (exit {done.returncode}): {(err or out).strip()[:300]}"
            )
        if done.returncode != 0 or not payload.get("success", False):
            message = str(payload.get("error") or err or f"exit {done.returncode}")[:400]
            if re.search(r"extension|relay|native|not connected|reach your chrome", message, re.I):
                message += " (is the chrome-use extension installed in Chrome? run: chrome-use status)"
            raise LiveChromeError(f"chrome-use {args[0]}: {message}")
        data = payload.get("data")
        return data if isinstance(data, dict) else {"value": data}


def _parse_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# ------------------------------------------------------------------ JS building

_UNSET = object()


class _Element:
    """A page element identified by the JS that finds it (handles don't cross the CLI)."""

    def __init__(self, page: "LivePage", expression: str, arg: Any = _UNSET) -> None:
        self.page, self.expression, self.arg = page, expression, arg

    def resolver(self) -> str:
        return build_call_js(self.expression, self.arg)

    # Playwright ElementHandle surface used by the adapters ---------------------
    async def scroll_into_view_if_needed(self, timeout: float | None = None) -> None:
        await self.page._eval_value(
            f"(async () => {{ const el = await {self.resolver()}; if (!el) throw new Error('element gone');"
            " (el.scrollIntoViewIfNeeded ? el.scrollIntoViewIfNeeded(true) : el.scrollIntoView({block:'center'})); return true; })()"
        )

    async def focus(self) -> None:
        await self.page._eval_value(
            f"(async () => {{ const el = await {self.resolver()}; if (!el) throw new Error('element gone'); el.focus(); return true; }})()"
        )

    async def click(self, timeout: float | None = None) -> None:
        rect = await self.page._eval_value(
            f"(async () => {{ const el = await {self.resolver()}; if (!el) throw new Error('element gone');"
            " (el.scrollIntoViewIfNeeded ? el.scrollIntoViewIfNeeded(true) : el.scrollIntoView({block:'center'}));"
            " const r = el.getBoundingClientRect(); return {x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width, h: r.height}; })()"
        )
        if not rect or not rect.get("w") or not rect.get("h"):
            raise LiveChromeError("click: element has no visible box")
        await self.page._click_xy(rect["x"], rect["y"])


class _Handle:
    def __init__(self, page: "LivePage", expression: str, arg: Any, is_element: bool) -> None:
        self._page, self._expression, self._arg, self._is_element = page, expression, arg, is_element

    def as_element(self) -> _Element | None:
        return _Element(self._page, self._expression, self._arg) if self._is_element else None


def _js_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _split_elements(value: Any, holes: list[_Element]) -> Any:
    if isinstance(value, _Element):
        holes.append(value)
        return {"__ob_el__": len(holes) - 1}
    if isinstance(value, (list, tuple)):
        return [_split_elements(v, holes) for v in value]
    if isinstance(value, dict):
        return {k: _split_elements(v, holes) for k, v in value.items()}
    return value


def build_call_js(expression: str, arg: Any = _UNSET) -> str:
    """A JS *expression* that evaluates to a Promise of ``expression(arg)``.

    Mirrors Playwright's ``page.evaluate``: a function source is called with ``arg``,
    anything else is taken as the value. The source is embedded directly (never passed
    to ``eval``/``new Function``) so strict page CSPs cannot block it.
    """
    holes: list[_Element] = []
    arg_json = "undefined" if arg is _UNSET else _js_literal(_split_elements(arg, holes))
    src = expression.strip().rstrip(";").strip()
    els = ", ".join(f"await {h.resolver()}" for h in holes)
    return (
        "(async () => {"
        f" const __els = [{els}];"
        " const __fill = (x) => (x && typeof x === 'object')"
        " ? ((typeof x.__ob_el__ === 'number') ? __els[x.__ob_el__] : (Array.isArray(x) ? x.map(__fill) : Object.fromEntries(Object.entries(x).map(([k, v]) => [k, __fill(v)]))))"
        " : x;"
        f" const __arg = __fill({arg_json});"
        f" const __v = ({src}\n);"
        " return (typeof __v === 'function') ? await __v(__arg) : await __v;"
        " })()"
    )


def wrap_for_cli(call_js: str) -> str:
    """Wrap so the CLI always returns a JSON string: ``{"ok":true,"v":..}`` or ``{"ok":false,"e":..}``."""
    return (
        "(async () => { try { const __r = await " + call_js + ";"
        " return JSON.stringify({ok: true, v: (__r === undefined ? null : __r)}); }"
        " catch (e) { return JSON.stringify({ok: false, e: String((e && e.message) || e)}); } })()"
    )


_CHALLENGE_JS = r"""(() => {
  const title = (document.title || '').slice(0, 160);
  const text = ((document.body && document.body.innerText) || '').slice(0, 4000);
  const frames = [...document.querySelectorAll('iframe')].map(f => f.src || '').filter(Boolean);
  const captchaFrame = frames.some(s => /recaptcha|hcaptcha|turnstile|challenges\.cloudflare|arkoselabs|funcaptcha|geetest/i.test(s));
  const cf = /just a moment|attention required|performing security verification|checking your browser/i.test(title + ' ' + text.slice(0, 600));
  const human = /verify (that )?you are (a )?human|are you a robot|press and hold|i'?m not a robot/i.test(text);
  const age = /(confirm|verify) (that )?you('| a)re (over |at least )?(18|eighteen)|age verification|date of birth/i.test(text.slice(0, 1500));
  return {captchaFrame, cf, human, age, title};
})()"""


# ------------------------------------------------------------------ page shim


class _Keyboard:
    def __init__(self, page: "LivePage") -> None:
        self._p = page

    async def insert_text(self, text: str) -> None:
        await self._p._act(("keyboard", "inserttext", "--stdin"), stdin=text, input_action=True)

    async def type(self, text: str, delay: float | None = None) -> None:
        for i in range(0, len(text), 4000):
            await self._p._act(("keyboard", "type", text[i : i + 4000]), input_action=True)

    async def press(self, key: str, delay: float | None = None) -> None:
        await self._p._act(("press", key), input_action=True)


class _Mouse:
    def __init__(self, page: "LivePage") -> None:
        self._p = page

    async def wheel(self, delta_x: float, delta_y: float) -> None:
        await self._p._act(("mouse", "wheel", str(int(delta_y)), str(int(delta_x))))

    async def click(self, x: float, y: float, **_: Any) -> None:
        await self._p._click_xy(x, y)


class _Locator:
    def __init__(self, page: "LivePage", selector: str, nth: int | None = None) -> None:
        self._p, self._sel, self._nth = page, selector, nth

    def _element_js(self) -> str:
        pick = "list[list.length - 1]" if self._nth == -1 else f"list[{int(self._nth or 0)}]"
        return f"() => {{ const list = Array.from(document.querySelectorAll({_js_literal(self._sel)})); return {pick} || null; }}"

    @property
    def last(self) -> "_Locator":
        return _Locator(self._p, self._sel, -1)

    @property
    def first(self) -> "_Locator":
        return _Locator(self._p, self._sel, 0)

    def nth(self, i: int) -> "_Locator":
        return _Locator(self._p, self._sel, i)

    async def count(self) -> int:
        return int(await self._p.evaluate(f"() => document.querySelectorAll({_js_literal(self._sel)}).length") or 0)

    async def click(self, timeout: float | None = None) -> None:
        await _Element(self._p, self._element_js()).click(timeout)


class LivePage:
    """The Playwright ``Page`` subset the adapters use, over one chrome-use tab."""

    def __init__(self, engine: "LiveChromeEngine", tab_id: str, url: str, key: str) -> None:
        self._engine, self.tab_id, self.key = engine, tab_id, key
        self._url = url
        self._closed = False
        self.keyboard = _Keyboard(self)
        self.mouse = _Mouse(self)

    # ---- state
    @property
    def url(self) -> str:
        return self._url

    def is_closed(self) -> bool:
        return self._closed

    # ---- the one gate every action passes through
    async def _act(
        self, args: Sequence[str], *, stdin: str | None = None, input_action: bool = False, timeout: float | None = None
    ) -> dict[str, Any]:
        if self._closed:
            raise LiveChromeError("tab was closed")
        async with self._engine._cmd_lock:
            await self._guard(input_action=input_action)
            return await self._engine.runner.run(*args, stdin=stdin, timeout=timeout)

    async def _guard(self, *, input_action: bool) -> None:
        """Select this tab, re-read its URL, refuse unless it is a provider page."""
        eng = self._engine
        await eng._select(self)
        data = await eng.runner.run("get", "url")
        url = str(data.get("url") or "")
        if not url:
            raise LiveChromeError("could not read the tab's URL; refusing to act blind")
        self._url = url
        if not url_allowed(url):
            eng._note(f"refused to act on {self.key}: tab is at {_short(url, 70)}")
            assert_allowed_url(url, where=f"{self.key} tab moved off the provider")
        if input_action:
            await self._refuse_if_challenge()

    async def _refuse_if_challenge(self) -> None:
        raw = await self._engine.runner.run("eval", "--stdin", stdin=wrap_for_cli(_CHALLENGE_JS))
        info = _unwrap(raw)
        if not isinstance(info, dict):
            return
        if info.get("captchaFrame") or info.get("cf") or info.get("human"):
            raise LiveChromeNeedsUser("captcha", self._url, str(info.get("title") or "")[:60])
        if info.get("age"):
            raise LiveChromeNeedsUser("age", self._url)

    # ---- evaluate
    async def _eval_value(self, call_js: str, *, input_action: bool = False) -> Any:
        raw = await self._act(("eval", "--stdin"), stdin=wrap_for_cli(call_js), input_action=input_action)
        return _unwrap(raw)

    async def evaluate(self, expression: str, arg: Any = _UNSET) -> Any:
        return await self._eval_value(build_call_js(expression, arg))

    async def evaluate_handle(self, expression: str, arg: Any = _UNSET) -> _Handle:
        call = build_call_js(expression, arg)
        is_el = await self._eval_value(
            f"(async () => {{ const v = await {call}; return (typeof Element !== 'undefined') && (v instanceof Element); }})()"
        )
        return _Handle(self, expression, arg, bool(is_el))

    def locator(self, selector: str) -> _Locator:
        return _Locator(self, selector)

    async def content(self) -> str:
        return str(await self.evaluate("() => document.documentElement ? document.documentElement.outerHTML : ''") or "")

    async def title(self) -> str:
        data = await self._act(("get", "title"))
        return str(data.get("title") or "")

    # ---- navigation / waiting
    async def goto(self, url: str, wait_until: str | None = None, timeout: float | None = None) -> None:
        assert_allowed_url(url, where="goto")
        async with self._engine._cmd_lock:
            await self._guard(input_action=False)
            await self._engine.runner.run("open", url, timeout=(timeout / 1000 + 15) if timeout else None)
            now = str((await self._engine.runner.run("get", "url")).get("url") or "")
            self._url = now or self._url
            assert_allowed_url(now, where="after navigation")

    async def wait_for_load_state(self, state: str = "load", timeout: float | None = None) -> None:
        want = ("complete",) if state == "load" else ("interactive", "complete")
        deadline = time.monotonic() + (timeout or 30000) / 1000
        while True:
            ready = await self.evaluate("() => document.readyState")
            if ready in want:
                return
            if time.monotonic() >= deadline:
                raise LiveChromeError(f"timeout waiting for load state {state!r}")
            await asyncio.sleep(0.3)

    async def wait_for_timeout(self, ms: float) -> None:
        await asyncio.sleep(max(0.0, ms) / 1000)
        try:  # keep ``page.url`` honest for the loops that poll it
            async with self._engine._cmd_lock:
                if not self._closed:
                    await self._engine._select(self)
                    self._url = str((await self._engine.runner.run("get", "url")).get("url") or self._url)
        except Exception:  # noqa: BLE001 -- a wait never fails on bookkeeping
            pass

    async def bring_to_front(self) -> None:
        if self._closed:
            return
        async with self._engine._cmd_lock:
            await self._guard(input_action=False)
            await self._engine._select(self, activate=True)

    # ---- pixels
    async def screenshot(self, path: str | None = None, type: str = "png", scale: str | None = None, full_page: bool = False, **_: Any) -> bytes:
        target = Path(path) if path else Path(tempfile.gettempdir()) / f"omnibrain_live_{os.getpid()}_{time.time_ns()}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        args = ["screenshot", str(target)] + (["--full"] if full_page else [])
        await self._act(args)
        data = target.read_bytes() if target.exists() else b""
        if not path:
            target.unlink(missing_ok=True)
        return data

    async def _click_xy(self, x: float, y: float) -> None:
        await self._act(("click", str(int(round(x))), str(int(round(y)))), input_action=True)

    async def close(self, run_before_unload: bool = False) -> None:
        if self._closed:
            return
        async with self._engine._cmd_lock:
            try:
                await self._engine.runner.run("tab", "close", self.tab_id)
            finally:
                self._closed = True
                self._engine._selected = None


def _unwrap(data: dict[str, Any]) -> Any:
    raw = data.get("result")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
    else:
        parsed = raw
    if isinstance(parsed, dict) and "ok" in parsed:
        if not parsed["ok"]:
            raise LiveChromeError(f"page script failed: {str(parsed.get('e'))[:300]}")
        return parsed.get("v")
    return parsed


# ------------------------------------------------------------------ engine


class LiveChromeEngine:
    """Drop-in for ``BrowserEngine``: one new tab per provider in your own Chrome."""

    live = True

    def __init__(self, settings: Settings, *, runner: ChromeUseRunner | None = None) -> None:
        self.settings = settings
        cfg = settings.browser
        self.runner = runner or ChromeUseRunner(
            [resolve_chrome_use(cfg.chrome_use_path)],
            session=cfg.chrome_use_session,
            browser=cfg.chrome_use_browser,
            timeout_s=cfg.chrome_use_timeout_s,
        )
        self._pages: dict[str, LivePage] = {}
        self._cmd_lock = asyncio.Lock()  # one chrome-use command at a time: it has ONE active tab
        self._focus_lock = asyncio.Lock()
        self._open_lock = asyncio.Lock()
        self._selected: str | None = None
        self._verified = False
        self.log: list[str] = []

    def _note(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        logging.getLogger("omnibrain.engine").info(message)
        self.log.append(line)
        if len(self.log) > 500:
            del self.log[: len(self.log) - 500]

    # ---- lifecycle
    async def start(self) -> None:
        try:
            status = await self.runner.run("status")
        except LiveChromeUnavailable:
            raise
        except LiveChromeError as exc:  # installed but unhealthy: say so, keep going
            self._note(f"live Chrome driver: chrome-use status was not healthy ({exc}); will retry on first use")
            return
        problem = extension_problem(status)
        self._note(
            f"live Chrome driver: chrome-use answered `status`"
            + (f" -- but {problem}; the first tab will fail until that is fixed" if problem else "")
        )

    async def _verify_live_chrome(self, page: "LivePage") -> None:
        """After the first tab: refuse to go on unless chrome-use is attached through the extension.

        (If it were not, chrome-use could be driving a browser it launched itself -- not your Chrome.)
        """
        if self._verified:
            return
        problem = extension_problem(await self.runner.run("status"))
        if problem:  # caller holds the command lock, so close through the runner directly
            try:
                await self.runner.run("tab", "close", page.tab_id)
            finally:
                page._closed = True
                self._selected = None
                self._pages.pop(page.key, None)
            raise LiveChromeUnavailable(f"refusing to continue: {problem}")
        self._verified = True

    async def stop(self, keep_windows: bool | None = None) -> None:
        keep = self.settings.browser.keep_windows_open if keep_windows is None else keep_windows
        if not keep:
            for page in list(self._pages.values()):
                try:
                    await page.close()
                except Exception as exc:  # noqa: BLE001
                    self._note(f"closing tab {page.tab_id} failed: {exc}")
        else:
            self._note("left OmniBrain's tabs open in your Chrome")
        self._pages.clear()
        self._selected = None

    async def _select(self, page: LivePage, *, activate: bool = False) -> None:
        """Make ``page`` chrome-use's active tab. Caller holds ``_cmd_lock``."""
        if self._selected == page.tab_id and not activate:
            return
        args = ["tab", "select", page.tab_id] + (["--activate"] if activate else [])
        try:
            await self.runner.run(*args)
        except LiveChromeError:
            self._selected = None
            raise
        self._selected = page.tab_id

    # ---- tabs
    async def open_research_page(self, provider: str, url: str, key: str | None = None) -> LivePage:
        assert_allowed_url(url, where=f"open for {provider}")  # a non-provider URL never reaches chrome-use
        tab_key = key or provider
        cached = self._pages.get(tab_key)
        if cached is not None and not cached.is_closed():
            return cached
        async with self._open_lock:
            cached = self._pages.get(tab_key)
            if cached is not None and not cached.is_closed():
                return cached
            async with self._cmd_lock:
                data = await self.runner.run("tab", "new", url)
                tab_id = str(data.get("tabId") or "")
                if not _TAB_ID.match(tab_id):
                    raise LiveChromeError(f"chrome-use tab new returned no usable tab id: {data!r}"[:300])
                self._selected = tab_id
                page = LivePage(self, tab_id, str(data.get("url") or url), tab_key)
                self._pages[tab_key] = page
                self._note(f"opened a new tab {tab_id} for {provider} in your Chrome")
                now = str((await self.runner.run("get", "url")).get("url") or page.url)
                page._url = now
                await self._verify_live_chrome(page)
            assert_allowed_url(now, where=f"{provider} tab after load")
            return page

    async def focus_tab(self, page: LivePage) -> None:
        async with self._focus_lock:
            try:
                await page.bring_to_front()
            except Exception as exc:  # noqa: BLE001
                self._note(f"focus nudge failed: {exc}")

    async def close_tab(self, provider: str, key: str | None = None) -> bool:
        page = self._pages.pop(key or provider, None)
        if page is None or page.is_closed():
            return False
        try:
            await page.close()
        except Exception as exc:  # noqa: BLE001
            self._note(f"closing the {provider} tab failed: {exc}")
            return False
        self._note(f"closed the {provider} tab (job cancelled)")
        return True

    async def close_provider(self, provider: str) -> None:
        await self.close_tab(provider)

    # ---- helpers the scripts use
    async def evaluate_json(self, page: LivePage, expression: str, arg: Any = None) -> Any:
        raw = await page.evaluate(expression, arg)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    async def snapshot(self, page: LivePage, name: str) -> Path:
        artifacts = Path(self.settings.storage.artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        path = artifacts / f"{name}_{int(time.time())}.png"
        try:
            await page.screenshot(path=str(path))
        except LiveChromeError as exc:
            self._note(f"screenshot failed for {name}: {exc}")
        return path

    async def probe(self, provider: str, url: str) -> dict[str, Any]:
        from backend.browser.engine import _PROBE_JS  # same observed-DOM probe as the default driver

        page = await self.open_research_page(provider, url)
        await page.wait_for_timeout(self.settings.browser.settle_ms * 2)
        dom = await page.evaluate(_PROBE_JS)
        dom["url_seen"] = page.url
        dom["title"] = await page.title()
        return dom
