r"""Focus check: does OmniBrain's live driver ever take the foreground window from the person?  (Windows)

    .venv\Scripts\python.exe scripts/live_focus_check.py                 # open a tab per provider, read, screenshot, close
    .venv\Scripts\python.exe scripts/live_focus_check.py --providers chatgpt,gemini --ask "Reply with the single word: pong"

A 20 ms poller records GetForegroundWindow (handle, owning process, class, title) for the whole run and the titles of every
Chrome window at the start and the end (a Chrome window title is the title of its ACTIVE tab, so a changed title is a tab
that was switched to). Exit code 0 only if no window OmniBrain could have raised came to the front -- a browser, a console,
a PowerShell/cmd/Python window or a File Explorer window. Changes owned by anything else (the person's Task Manager, the
shell's Task Switching, the agent's own app window) are recorded as noise: they are real foreground changes and they are not
evidence about OmniBrain, and a gate that cannot tell them apart can never pass on a desktop someone is actually using.
Still fails when the providers named in browser.chrome_use_front_providers take the front and do not hand it back.
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


def window_class(hwnd: int) -> str:
    cls = ctypes.create_unicode_buffer(80)
    try:
        user32.GetClassNameW(wintypes.HWND(hwnd), cls, 80)
    except Exception:  # noqa: BLE001 -- a handle that vanished mid-poll is not a violation
        return ""
    return cls.value


def process_name(pid: int) -> str:
    """The image name of the process owning a window, "" when it cannot be read."""
    if not pid:
        return ""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000  # noqa: N806
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(300)
        buf = ctypes.create_unicode_buffer(300)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return Path(buf.value).name.lower()
    except Exception:  # noqa: BLE001
        return ""
    finally:
        kernel32.CloseHandle(handle)
    return ""


# What counts as OmniBrain raising a window over the person: a browser the driver could have brought forward,
# a console of any kind (hard rule 1: no console, cmd or Explorer windows), or a File Explorer window.
BROWSER_IMAGES = {"chrome.exe", "msedge.exe", "brave.exe", "vivaldi.exe", "opera.exe", "chromium.exe"}
CONSOLE_IMAGES = {"conhost.exe", "cmd.exe", "powershell.exe", "pwsh.exe", "windowsterminal.exe", "wt.exe",
                  "python.exe", "pythonw.exe", "py.exe", "openconsole.exe"}
EXPLORER_WINDOW_CLASSES = {"CabinetWClass", "ExploreWClass"}


def classify_owner(pid_name: str, cls: str, title: str = "") -> tuple[str, str]:
    """("violation", why) when only OmniBrain could have raised that window, ("noise", why) otherwise."""
    name = (pid_name or "").lower()
    if name in BROWSER_IMAGES:
        return "violation", f"a browser window came to the front ({name})"
    if name in CONSOLE_IMAGES:
        return "violation", f"a console window came to the front ({name})"
    if cls in EXPLORER_WINDOW_CLASSES:
        return "violation", f"a File Explorer window came to the front ({cls})"
    if not name and cls == "Chrome_WidgetWin_1":
        # the owning process could not be read: a Chrome-class window is judged by its class, so an
        # unattributable Chrome activation is treated as the violation it most likely is
        return "violation", f"a Chrome-class window came to the front and its process could not be read ({title[:40]})"
    return "noise", f"{name or 'unknown process'} / {cls or 'no class'}"


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
                cls, name = window_class(h), process_name(pid)
                kind, why = classify_owner(name, cls, title)
                self.events.append({"t": round(time.monotonic() - self.t0, 3), "hwnd": h, "pid": pid,
                                    "process": name, "class": cls, "title": title[:80], "kind_verdict": kind, "why": why})
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
        violations = [e for e in changes if e.get("kind_verdict") == "violation"]
        noise = [e for e in changes if e.get("kind_verdict") != "violation"]
        start_cls, start_proc = window_class(self.start_fg[0]), process_name(self.start_fg[1])
        return {
            "foreground_at_start": {"hwnd": self.start_fg[0], "pid": self.start_fg[1], "process": start_proc,
                                    "class": start_cls, "title": self.start_fg[2][:80]},
            "foreground_at_end": {"hwnd": self.end_fg[0], "pid": self.end_fg[1],
                                  "process": process_name(self.end_fg[1]), "class": window_class(self.end_fg[0]),
                                  "title": self.end_fg[2][:80]},
            "foreground_changes": changes,
            "violations": violations,
            "unrelated_changes": noise,
            "marks": [e for e in self.events if "mark" in e],
            "never_changed": not changes and self.start_fg[0] == self.end_fg[0],
            "clean": not violations,
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
    if rep["violations"]:
        for v in rep["violations"]:
            print(f"VIOLATION at t={v['t']}s: {v['why']} -- {v['title']!r}")
    if rep["unrelated_changes"]:
        print(f"{len(rep['unrelated_changes'])} foreground change(s) by other windows (not OmniBrain): "
              + ", ".join(sorted({(c.get('process') or '?') for c in rep['unrelated_changes']})))
    if args.raise_front:  # the instant itself may move the foreground; the end state must be the person's window again
        return 0 if rep["clean"] and rep["foreground_at_start"]["hwnd"] == rep["foreground_at_end"]["hwnd"] else 1
    return 0 if rep["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())