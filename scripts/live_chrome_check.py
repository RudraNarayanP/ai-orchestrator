"""Live check of the opt-in live-Chrome driver (``browser.driver: chrome_use``).

    python scripts/live_chrome_check.py                      # ChatGPT: open a NEW tab, read its state, close it
    python scripts/live_chrome_check.py gemini               # any provider key from config/settings.yaml
    python scripts/live_chrome_check.py chatgpt --ask "Reply with the single word: pong"
    python scripts/live_chrome_check.py chatgpt --keep-tab   # leave the tab open afterwards

Default is read-only: it asks chrome-use for its status, opens one new tab on the provider's own page in your
everyday Chrome, runs OmniBrain's readiness probe (ready / login wall / blocked), and closes that tab. Nothing is
typed or sent unless you pass ``--ask``. The driver is forced on for this run only (config is not edited), is
limited to the AI provider domains, and never touches your other tabs. A login / captcha / age page is reported
as needing you -- complete it yourself in Chrome and run it again.

Needs, once: scripts/install_chrome_use.ps1, ``chrome-use extension install``, and the chrome-use extension
added in Chrome (see the printed hints if a step is missing).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from backend.browser.factory import create_engine  # noqa: E402
from backend.browser.live_chrome import (  # noqa: E402
    LiveChromeError,
    LiveChromeNeedsUser,
    LiveChromeUnavailable,
    extension_problem,
)
from backend.providers.registry import ProviderCatalog  # noqa: E402
from backend.settings import load_settings  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("provider", nargs="?", default="chatgpt")
    ap.add_argument("--ask", default="", help="also send this one question (default: read-only, nothing is sent)")
    ap.add_argument("--keep-tab", action="store_true", help="leave the OmniBrain tab open afterwards")
    ap.add_argument("--chrome-use", default="", help="full path to chrome-use.exe if it is not on PATH")
    ap.add_argument("--timeout", type=int, default=180, help="seconds to wait for an answer with --ask")
    args = ap.parse_args()

    settings = load_settings()
    cfg = settings.providers.get(args.provider)
    if cfg is None:
        print(f"unknown provider {args.provider!r}; known: {', '.join(settings.providers)}")
        return 2
    settings.browser.driver = "chrome_use"  # this run only
    if args.chrome_use:
        settings.browser.chrome_use_path = args.chrome_use
    cfg.enabled = True

    engine = create_engine(settings)
    print(f"[1/4] chrome-use: {engine.runner.argv_prefix[0]}")
    try:
        await engine.start()
        status = await engine.runner.run("status")
        ext = status.get("extension") or {}
        print(f"      chrome-use {status.get('cliVersion')}; host installed={ext.get('hostInstalled')} healthy={ext.get('hostHealthy')} relay up={ext.get('relayUp')} extension={ext.get('liveVersion')} profile={ext.get('profileEmail') or ext.get('profileId')}")
        problem = extension_problem(status)
        if problem:
            print(f"      WARNING: {problem}\n      (continuing: the first tab will tell for sure; OmniBrain refuses to go on if Chrome is not attached via the extension)")
    except LiveChromeUnavailable as exc:
        print(f"      NOT READY: {exc}\n      -> run scripts\\install_chrome_use.ps1 first")
        return 3
    except LiveChromeError as exc:
        print(f"      NOT READY: {exc}\n      -> run: chrome-use extension install   and add the extension in Chrome")
        return 3

    code = 0
    try:
        print(f"[2/4] opening a NEW tab for {args.provider} ({cfg.url}) in your Chrome")
        page = await engine.open_research_page(args.provider, cfg.new_chat_url or cfg.url)
        await page.wait_for_timeout(settings.browser.settle_ms * 2)
        print(f"      url   : {page.url}")
        print(f"      title : {await page.title()}")

        print("[3/4] readiness (read-only DOM probe)")
        adapter = ProviderCatalog(settings, engine).adapter(args.provider)
        state = await adapter.readiness(page)
        for _ in range(8):  # heavy single-page apps (meta.ai) need a few seconds before their composer exists
            if state.get("state") in ("ready", "login_wall", "blocked", "rate_limited"):
                break
            await page.wait_for_timeout(2500)
            if _ == 0:  # background tabs: some apps render nothing until the tab is shown
                await page.bring_to_front()
            state = await adapter.readiness(page)
        print(f"      state : {state.get('state')}  (input visible: {state.get('inputHere')}, login wall: {state.get('loginWall')})")
        if state.get("state") != "ready":
            print("      -> not ready: if this is a sign-in / captcha / age page, complete it yourself in that tab, then re-run.")
            code = 1

        if args.ask and state.get("state") == "ready":
            print(f"[4/4] asking: {args.ask!r}")
            t0 = time.time()
            r = await asyncio.wait_for(adapter.ask(f"live_check_{int(t0)}", args.ask, 1), timeout=args.timeout)
            print(f"      status: {r.status.value} in {time.time() - t0:.0f}s  {r.error or ''}")
            print(f"      answer: {(r.answer_text or '')[:300]!r}")
            code = 0 if r.status.value == "completed" else 1
        else:
            print("[4/4] skipped (read-only run; pass --ask to send one question)")
    except LiveChromeNeedsUser as exc:
        print(f"      NEEDS YOU: {exc}")
        code = 1
    except LiveChromeError as exc:
        print(f"      FAILED: {type(exc).__name__}: {exc}")
        code = 1
    finally:
        await engine.stop(keep_windows=args.keep_tab)
        print(f"done: {engine.runner.calls} chrome-use calls; tab {'left open' if args.keep_tab else 'closed'}.")
        for line in engine.log[-6:]:
            print("  log:", line)
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
