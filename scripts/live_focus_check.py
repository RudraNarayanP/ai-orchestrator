r"""Focus check: does OmniBrain's live driver ever take the foreground window from the person?  (Windows)

    .venv\Scripts\python.exe scripts/live_focus_check.py                 # open a tab per provider, read, screenshot, close
    .venv\Scripts\python.exe scripts/live_focus_check.py --providers chatgpt,gemini --ask "Reply with the single word: pong"

A 20 ms poller records GetForegroundWindow (handle, owning process, title) for the whole run and the titles of every
Chrome window at the start and the end (a Chrome window title is the title of its ACTIVE tab, so a changed title is a tab
that was switched to). Exit code 0 only if the foreground window never changed, apart from the providers named in
browser.chrome_use_front_providers, which are allowed to take it for an instant and must hand it back.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None


def foreground() -> tuple[int, int, str]:
    h = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(300)
    user32.GetWindowTextW(h, buf, 300)
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
    return int(h or 0), int(pid.value), buf.value


def zorder(limit: int = 14) -> list[str]:
    """The top visible windows, front first: "title [class] (minimised)". A window OmniBrain created that sits above
    the person's own window shows up here even when the foreground handle never changed."""
    rows: list[str] = []
    h = user32.GetTopWindow(None)
    while h and len(rows) < limit:
        if user32.IsWindowVisible(h):
            buf = ctypes.create_unicode_buffer(200)
            user32.GetWindowTextW(h, buf, 200)
            cls = ctypes.create_unicode_buffer(80)
            user32.GetClassNameW(h, cls, 80)
            if buf.value:
                rows.append(f"{buf.value[:60]} [{cls.value}]" + (" (minimised)" if user32.IsIconic(h) else ""))
        h = user32.GetWindow(h, 2)  # GW_HWNDNEXT
    return rows


def chrome_titles() -> list[str]:
    out: list[str] = []
    proto = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        cls = ctypes.create_unicode_buffer(100)
        user32.GetClassNameW(hwnd, cls, 100)
        if cls.value == "Chrome_WidgetWin_1" and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(300)
            user32.GetWindowTextW(hwnd, buf, 300)
            if buf.value:
                out.append(buf.value)
        return True

    user32.EnumWindows(proto(cb), 0)
    return sorted(out)


class Monitor:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self._stop = threading.Event()
        self.start_fg = foreground()
        self.start_titles = chrome_titles()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.t0 = time.monotonic()

    def _run(self) -> None:
        last = self.start_fg[0]
        while not self._stop.is_set():
            h, pid, title = foreground()
            if h != last:
                self.events.append({"t": round(time.monotonic() - self.t0, 3), "hwnd": h, "pid": pid, "title": title[:80]})
                last = h
            time.sleep(0.02)

    def __enter__(self) -> "Monitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(2)
        self.end_fg = foreground()
        self.end_titles = chrome_titles()

    def mark(self, label: str) -> None:
        self.events.append({"t": round(time.monotonic() - self.t0, 3), "mark": label, "chrome_titles": chrome_titles(), "zorder": zorder()})

    def report(self) -> dict:
        changes = [e for e in self.events if "hwnd" in e]
        return {
            "foreground_at_start": {"hwnd": self.start_fg[0], "title": self.start_fg[2][:80]},
            "foreground_at_end": {"hwnd": self.end_fg[0], "title": self.end_fg[2][:80]},
            "foreground_changes": changes,
            "marks": [e for e in self.events if "mark" in e],
            "never_changed": not changes and self.start_fg[0] == self.end_fg[0],
        }


async def drive(providers: list[str], ask: str | None, mon: Monitor, raise_front: bool = False) -> None:
    from backend.browser.factory import create_engine
    from backend.settings import load_settings

    settings = load_settings()
    settings.browser.driver = "chrome_use"
    engine = create_engine(settings)
    await engine.start()
    pages = {}
    try:
        for name in providers:
            cfg = settings.providers[name]
            mon.mark(f"before tab new {name}")
            pages[name] = await engine.open_research_page(name, cfg.url)
            mon.mark(f"after tab new {name}")
        await asyncio.sleep(2.0)
        for name, page in pages.items():
            await page.evaluate("document.title")
            mon.mark(f"after evaluate {name}")
            await page.screenshot()
            mon.mark(f"after screenshot {name}")
        if raise_front:  # deliberately raise each allowed provider, as typing/sending does, then check it is handed back
            for name, page in pages.items():
                mon.mark(f"before front instant {name}")
                async with engine.foreground_instant(name):
                    await page.bring_to_front()
                    await asyncio.sleep(1.0)
                    mon.mark(f"inside front instant {name}")
                await asyncio.sleep(0.4)
                mon.mark(f"after front instant {name}")
        if ask:
            from backend.providers.registry import ProviderCatalog  # noqa: PLC0415

            catalog = ProviderCatalog(settings, engine)

            async def one(name: str) -> None:
                resp = await catalog.adapter(name).ask("focus_check", ask, 1)
                mon.mark(f"after ask {name}: {resp.status.value} {(resp.answer_text or resp.error or '')[:40]!r}")

            await asyncio.gather(*(one(n) for n in pages))
    finally:
        await engine.stop(keep_windows=False)
        mon.mark("after stop (tabs closed)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="chatgpt,deepseek")
    ap.add_argument("--ask", default=None)
    ap.add_argument("--raise-front", action="store_true", help="raise each provider that is allowed to the front and check it is handed back")
    args = ap.parse_args()
    if user32 is None:
        print("Windows only")
        return 2
    names = [p.strip() for p in args.providers.split(",") if p.strip()]
    with Monitor() as mon:
        try:
            asyncio.run(drive(names, args.ask, mon, args.raise_front))
        except Exception as exc:  # noqa: BLE001
            mon.mark(f"error {type(exc).__name__}: {exc}"[:200])
    rep = mon.report()
    print(json.dumps(rep, indent=1, ensure_ascii=True))
    if args.raise_front:  # the instant itself may move the foreground; the end state must be the person's window again
        return 0 if rep["foreground_at_start"]["hwnd"] == rep["foreground_at_end"]["hwnd"] else 1
    return 0 if rep["never_changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())