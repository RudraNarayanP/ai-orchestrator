"""Browser engine.

One dedicated Chrome *window* per provider, each backed by its own persistent
``--user-data-dir`` under ``browser/profiles/``. This module never attaches to
a running Chrome and never touches the default profile, which means:

* it cannot disturb, read, or click inside the browser you work in;
* your daily profile is not locked, copied, or inspected;
* OmniBrain's automation is visible, not masked -- we do not strip
  ``--enable-automation`` or otherwise hide from a site's bot defences. If a
  site objects, the provider is reported as blocked/broken (see spec section 26).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from playwright.async_api import (
    BrowserContext,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeout,
    async_playwright,
)

from backend.models import ProviderStatus
from backend.settings import PROFILES_DIR, Settings

# Chrome refuses to share one user-data-dir between processes; that isolation
# is exactly what gives us a separate window per provider.
SAFE_PROFILE = re.compile(r"[^a-z0-9_-]+")


def profile_dir(name: str) -> Path:
    slug = SAFE_PROFILE.sub("_", name.strip().lower()) or "default"
    path = PROFILES_DIR / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class WindowSpec:
    index: int
    width: int
    height: int
    offset_x: int
    offset_y: int

    @property
    def position(self) -> tuple[int, int]:
        # tile rightwards, wrap to a second row so 8 windows stay reachable
        per_row = 4
        col = self.index % per_row
        row = self.index // per_row
        return (
            40 + self.offset_x + col * (self.width // 2 + 20),
            40 + self.offset_y + row * (self.height // 2 + 20),
        )


@dataclass
class OwnedSession:
    provider: str
    context: BrowserContext
    pages: dict[str, Page] = field(default_factory=dict)
    launched_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    attached: bool = False

    async def page(self, key: str = "chat", url: str | None = None) -> Page:
        if key in self.pages and not self.pages[key].is_closed():
            page = self.pages[key]
        else:
            page = await self.context.new_page()
            self.pages[key] = page
        if url and (not page.url or page.url == "about:blank"):
            await page.goto(url, wait_until="domcontentloaded")
        self.last_used = time.time()
        return page


class BrowserEngine:
    """One OmniBrain window, one tab per provider, tabs reused between runs.

    Modes:
      window_mode="single"       one window, one tab per provider (default)
      window_mode="per_provider" an isolated profile and window per site
      cdp_url=...                attach to a Chrome you started yourself

    There is deliberately no mode that reaches into your everyday browser by
    copying its session. Chrome 136+ blocks remote debugging on the default
    profile for exactly that reason, and the workaround -- duplicating a signed-in
    profile -- is credential handling I would rather not automate. Sign into the
    OmniBrain profile once instead, or point cdp_url at a Chrome you launched.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pw = None
        self._sessions: dict[str, OwnedSession] = {}
        self._launch_locks: dict[str, asyncio.Lock] = {}
        self._slots = asyncio.Semaphore(max(1, settings.browser.max_concurrent_profiles))
        self._focus_lock = asyncio.Lock()
        self._window_index = 0
        self.log: list[str] = []

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._pw is None:
            self._pw = await async_playwright().start()
            self._note("playwright driver started")

    async def stop(self, keep_windows: bool | None = None) -> None:
        keep = self.settings.browser.keep_windows_open if keep_windows is None else keep_windows
        if keep:
            # Profiles are persistent; leaving the windows open lets the user
            # see and take over any session that needs a manual click.
            pass
        for profile_key, session in list(self._sessions.items()):
            if session.attached:
                # A Chrome the user launched is not ours to close -- dropping the
                # connection is all we do, so their tabs survive untouched.
                self._note(f"detached from CDP browser ({profile_key}); left the user's tabs open")
                continue
            try:
                await session.context.close()
            except Exception as exc:  # noqa: BLE001
                self._note(f"context close failed for {profile_key}: {exc}")
        self._sessions.clear()
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None

    def _note(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        logging.getLogger("omnibrain.engine").info(message)
        self.log.append(line)
        if len(self.log) > 500:
            del self.log[: len(self.log) - 500]

    # ------------------------------------------------------------------ sessions

    def _profile_key(self, provider: str) -> str:
        return provider if self.settings.browser.window_mode == "per_provider" else "shared"

    async def session(self, provider: str, *, first_url: str) -> OwnedSession:
        """Return the window that hosts this provider, launching it if needed.

        In single mode all providers share one window, which is the point: a
        research run should not leave eight Chrome windows scattered on the desktop.
        """
        cfg = self.settings.browser
        key = self._profile_key(provider)
        existing = self._sessions.get(key)
        if existing and not self._context_dead(existing):
            existing.last_used = time.time()
            return existing

        lock = self._launch_locks.setdefault(key, asyncio.Lock())
        async with lock:
            existing = self._sessions.get(key)
            if existing and not self._context_dead(existing):
                return existing
            if existing:
                self._sessions.pop(key, None)
            return await self._launch(key, first_url)

    def _context_dead(self, session: OwnedSession) -> bool:
        try:
            browser = session.context.browser
            if browser is None:
                return False  # an attached context has no browser we own
            return not browser.is_connected()
        except Exception:  # noqa: BLE001
            return True

    async def _launch(self, profile_key: str, first_url: str) -> OwnedSession:
        await self.start()
        assert self._pw is not None
        cfg = self.settings.browser

        if cfg.cdp_url:
            browser = await self._pw.chromium.connect_over_cdp(cfg.cdp_url)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            session = OwnedSession(provider=profile_key, context=context, attached=True)
            self._sessions[profile_key] = session
            self._note(f"attached to an existing Chrome over CDP at {cfg.cdp_url}")
            return session

        single = cfg.window_mode != "per_provider"
        profile = profile_dir("omnibrain_shared" if single else f"omnibrain_{profile_key}")
        spec = WindowSpec(
            index=self._window_index,
            width=cfg.width,
            height=cfg.height,
            offset_x=cfg.window_offset_x,
            offset_y=cfg.window_offset_y,
        )
        self._window_index += 1
        left, top = spec.position

        async with self._slots:
            args = [
                # user_data_dir is passed positionally to launch_persistent_context;
                # repeating it as a flag is an error in Playwright >=1.60.
                f"--window-size={cfg.width},{cfg.height}",
                f"--window-position={left},{top}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-timer-throttling",
                # Without these three, an unfocused OmniBrain window throttles
                # its renderer and the provider stops streaming answers -- the
                # classic cause of truncated background responses. This affects
                # *our* windows' rendering only; it hides nothing from a site.
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-features=Translate",
                # Intentionally absent: anything that hides automation from the
                # target site. navigator.webdriver stays whatever Chrome sets.
                *cfg.args,
            ]
            kwargs: dict[str, Any] = {
                "headless": cfg.headless,
                "args": args,
                "viewport": None,
                "ignore_default_args": ["--hide-scrollbars"],
                "proxy": None,
            }
            if cfg.locale:
                kwargs["locale"] = cfg.locale
            if cfg.executable:
                kwargs["executable_path"] = cfg.executable
            elif cfg.channel and cfg.channel != "chromium":
                kwargs["channel"] = cfg.channel

            try:
                context = await self._pw.chromium.launch_persistent_context(str(profile), **kwargs)
            except PlaywrightError as exc:
                message = str(exc)
                if "Executable doesn't exist" in message or "channel" in message.lower():
                    self._note(f"chrome channel unavailable for {profile_key}, falling back to bundled chromium")
                    kwargs.pop("channel", None)
                    kwargs.pop("executable_path", None)
                    context = await self._pw.chromium.launch_persistent_context(str(profile), **kwargs)
                else:
                    raise

            session = OwnedSession(provider=profile_key, context=context)
            self._sessions[profile_key] = session
            self._note(f"launched {'shared window' if single else f'window for {profile_key}'} at {profile}")
            return session

    @staticmethod
    def _host_of(url: str) -> str:
        match = re.match(r"https?://([^/]+)", url or "")
        return (match.group(1) if match else "").lower().removeprefix("www.")

    @staticmethod
    def _is_blank(page: Page) -> bool:
        url = (page.url or "").lower()
        return url in {"", "about:blank"} or "new-tab-page" in url or "newtabpage" in url

    async def open_research_page(self, provider: str, url: str, key: str | None = None) -> Page:
        """Give this provider a tab in the OmniBrain window, reusing one if it exists.

        Reuse is the difference between "a browser with a few tabs open" and "eight
        new windows on the desktop every time you ask a question". The blank tab a
        fresh profile opens with is adopted rather than left sitting there.
        """
        cfg = self.settings.browser
        session = await self.session(provider, first_url=url)
        tab_key = key or provider
        cached = session.pages.get(tab_key)
        if cached is not None and not cached.is_closed():
            if self._is_blank(cached):
                await cached.goto(url, wait_until="domcontentloaded", timeout=cfg.nav_timeout_ms)
            session.last_used = time.time()
            return cached

        if cfg.reuse_tabs and key is None:
            want = self._host_of(url)
            owned = {
                id(page) for name, page in session.pages.items() if name != tab_key and not page.is_closed()
            }
            blank: Page | None = None
            for page in session.context.pages:
                try:
                    if page.is_closed() or id(page) in owned:
                        continue
                    if want and self._host_of(page.url) == want:
                        # A tab already claimed by another provider is not up for
                        # grabs; two sites sharing a host must not share a composer.
                        session.pages[tab_key] = page
                        self._note(f"reusing the open {provider} tab instead of opening another")
                        return page
                    if blank is None and self._is_blank(page):
                        blank = page
                except Exception:  # noqa: BLE001
                    continue
            if blank is not None:
                session.pages[tab_key] = blank
                await blank.goto(url, wait_until="domcontentloaded", timeout=cfg.nav_timeout_ms)
                self._note(f"adopted the window's blank tab for {provider}")
                return blank

        page = await session.page(tab_key, url=url)
        if page.url in {"", "about:blank"} or (url and self._host_of(page.url) != self._host_of(url)):
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=cfg.nav_timeout_ms)
            except PlaywrightTimeout:
                self._note(f"navigation to {url} timed out; continuing with what loaded")
        return page

    async def focus_tab(self, page: Page) -> None:
        """Bring one tab forward, serialised so parallel providers do not fight."""
        async with self._focus_lock:
            try:
                await page.bring_to_front()
            except Exception as exc:  # noqa: BLE001
                self._note(f"focus nudge failed: {exc}")

    async def close_tab(self, provider: str, key: str | None = None) -> bool:
        """Close this provider's tab (used when a job is cancelled mid-answer).

        Only tabs in a window OmniBrain owns are closed; in CDP-attach mode the tabs
        belong to the user's own Chrome and are left alone.
        """
        session = self._sessions.get(self._profile_key(provider))
        if session is None or session.attached:
            return False
        page = session.pages.pop(key or provider, None)
        if page is None or page.is_closed():
            return False
        try:
            await page.close(run_before_unload=False)
        except Exception as exc:  # noqa: BLE001
            self._note(f"closing the {provider} tab failed: {exc}")
            return False
        self._note(f"closed the {provider} tab (job cancelled)")
        return True

    async def close_provider(self, provider: str) -> None:
        session = self._sessions.pop(provider, None)
        if session:
            try:
                await session.context.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ helpers

    async def evaluate_json(self, page: Page, expression: str, arg: Any = None) -> Any:
        raw = await page.evaluate(expression, arg)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    async def snapshot(self, page: Page, name: str) -> Path:
        artifacts = Path(self.settings.storage.artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        path = artifacts / f"{name}_{int(time.time())}.png"
        try:
            await page.screenshot(path=str(path), full_page=False)
        except PlaywrightError as exc:
            self._note(f"screenshot failed for {name}: {exc}")
        return path

    async def probe(self, provider: str, url: str) -> dict[str, Any]:
        """Open the dedicated window and report what the page actually contains.

        Used at install time so adapters are written against observed DOM,
        never against remembered or guessed selectors.
        """
        page = await self.open_research_page(provider, url)
        await page.wait_for_timeout(self.settings.browser.settle_ms * 2)
        dom = await page.evaluate(_PROBE_JS)
        dom["url_seen"] = page.url
        dom["title"] = await page.title()
        return dom


# Runs inside the page. Deliberately broad: we score candidates instead of
# assuming any site keeps a stable id.
_PROBE_JS = r"""
() => {
  const text = (el) => (el.innerText || el.textContent || el.getAttribute('aria-label') || el.placeholder || '').trim().slice(0, 90);
  const vis = (el) => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 8 && r.height > 4 && s.visibility !== 'hidden' && s.opacity !== '0' && r.top < innerHeight + 200 && r.bottom > -200;
  };
  const describe = (el, kind) => {
    const r = el.getBoundingClientRect();
    return {
      kind,
      tag: el.tagName.toLowerCase(),
      id: el.id || null,
      role: el.getAttribute('role'),
      testid: el.getAttribute('data-testid') || el.getAttribute('data-test-id') || null,
      aria: el.getAttribute('aria-label'),
      placeholder: el.getAttribute('placeholder'),
      classes: (el.className || '').toString().slice(0, 120),
      contenteditable: el.getAttribute('contenteditable'),
      name: el.getAttribute('name'),
      text: text(el),
      rect: {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)},
      visible: vis(el),
    };
  };
  const inputs = [...document.querySelectorAll('textarea, [contenteditable="true"], [contenteditable=""], input[type=text], [role="textbox"], [role="combobox"]')]
    .map(el => describe(el, el.tagName === 'TEXTAREA' ? 'textarea' : (el.getAttribute('contenteditable') ? 'contenteditable' : 'textbox')))
    .filter(d => d.visible);
  const buttons = [...document.querySelectorAll('button, [role="button"], a[role="button"]')]
    .map(el => describe(el, 'button'))
    .filter(d => d.visible && /send|submit|ask|go\b|prompt|continue|generate|retry|try again|log ?in|sign ?in|continue with|allow|accept/i.test(`${d.aria||''} ${d.text} ${d.testid||''} ${d.classes}`));
  const shells = [...document.querySelectorAll('[role="log"], [aria-live], main, [class*="message"], [data-message-id], [class*="conversation"]')]
    .map(el => ({...describe(el, 'shell'), children: el.children ? el.children.length : 0}))
    .filter(d => d.visible).slice(0, 40);
  const banners = [...document.querySelectorAll('[role="alert"], [class*="cookie"], [class*="consent"], [class*="banner"]')]
    .map(el => describe(el, 'banner')).filter(d => d.visible).slice(0, 12);
  return {
    inputs,
    buttons,
    shells,
    banners,
    webdriver: !!navigator.webdriver,
    bodyLength: (document.body && document.body.innerText || '').length,
    bodyHead: (document.body && document.body.innerText || '').slice(0, 400),
  };
}
"""
