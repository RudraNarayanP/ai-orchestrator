"""Open one provider in its own dedicated Chrome window and report the DOM.

    python scripts/probe.py                     # every enabled provider, in order
    python scripts/probe.py chatgpt --login     # leave the window up so you can sign in
    python scripts/probe.py gemini --url https://gemini.google.com/app

This is how ``browser/adapters/selectors.py`` stops being guesswork: run a probe
after signing in and it writes ``data/probe/<provider>_<ts>.json`` with the real
candidate elements, which is what gets promoted to ``verified: probe``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from backend.browser.engine import BrowserEngine  # noqa: E402
from backend.settings import DATA_DIR, load_settings  # noqa: E402


async def probe_one(engine: BrowserEngine, provider: str, url: str | None, login: bool, wait_ms: int) -> dict:
    settings = engine.settings
    cfg = settings.providers.get(provider)
    if cfg is None:
        return {"provider": provider, "error": f"no config entry for {provider}"}
    target = url or cfg.url
    result: dict = {"provider": provider, "target": target, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        page = await engine.open_research_page(provider, target)
        await page.wait_for_timeout(wait_ms)
        if login:
            print(f"\n[{provider}] window open at {page.url}")
            print("Sign in (or dismiss whatever it shows), then press Enter here to probe...")
            await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
        dom = await engine.probe(provider, page.url)
        if not dom.get("inputs") and not dom.get("bodyLength"):
            # Some SPAs only mount after interaction or a second paint.
            for _ in range(3):
                try:
                    await page.mouse.wheel(0, 300)
                    await page.mouse.wheel(0, -300)
                    await page.keyboard.press("Tab")
                except Exception:  # noqa: BLE001
                    pass
                await page.wait_for_timeout(max(1500, wait_ms // 3))
            second = await engine.probe(provider, page.url)
            second["before_warmup"] = {k: dom.get(k) for k in ("bodyLength", "url_seen", "title")}
            second["note"] = "empty on first paint; re-probed after scroll+Tab warm-up"
            if second.get("inputs") or second.get("bodyLength"):
                dom = second
        result["observed"] = dom
        shot = await engine.snapshot(page, f"probe_{provider}")
        result["screenshot"] = str(shot)
        composer = [i for i in dom.get("inputs", []) if i.get("visible")]
        result["summary"] = {
            "url_seen": dom.get("url_seen"),
            "title": dom.get("title"),
            "navigator_webdriver": dom.get("webdriver"),
            "visible_inputs": [
                {k: v for k, v in c.items() if k in {"tag", "id", "role", "testid", "aria", "placeholder", "contenteditable", "classes", "rect"}}
                for c in composer[:6]
            ],
            "candidate_buttons": [
                {k: v for k, v in b.items() if k in {"text", "aria", "testid", "classes", "rect"}}
                for b in dom.get("buttons", [])[:12]
            ],
            "shells": [{"classes": s.get("classes"), "children": s.get("children"), "tag": s.get("tag")} for s in dom.get("shells", [])[:10]],
            "banners": [{"text": b.get("text"), "classes": b.get("classes")} for b in dom.get("banners", [])[:6]],
        }
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


async def main() -> int:
    ap = argparse.ArgumentParser(description="Probe provider DOM in an isolated Chrome window")
    ap.add_argument("providers", nargs="*", help="provider keys; default = all enabled")
    ap.add_argument("--url", help="override the configured start URL")
    ap.add_argument("--login", action="store_true", help="pause for manual sign-in before probing")
    ap.add_argument("--close", action="store_true", help="close windows when done (default: leave open)")
    ap.add_argument("--wait", type=int, default=6000, help="ms to let the page settle before probing (SPAs need more)")
    args = ap.parse_args()

    settings = load_settings()
    providers = args.providers or [k for k, c in settings.providers.items() if c.enabled]
    out_dir = DATA_DIR / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = BrowserEngine(settings)
    await engine.start()
    failures = 0
    for provider in providers:
        print(f"\n=== probing {provider} ===")
        result = await probe_one(engine, provider, args.url, args.login, args.wait)
        if result.get("summary"):
            s = result["summary"]
            note = (result.get("observed") or {}).get("note")
            if note:
                print(f"  NOTE: {note}")
            print(f"  url: {s['url_seen']}")
            print(f"  title: {s['title']}")
            print(f"  navigator.webdriver = {s['navigator_webdriver']}")
            print(f"  visible inputs: {len(s['visible_inputs'])}")
            for item in s["visible_inputs"][:3]:
                print(f"    - {item.get('tag')} id={item.get('id')} role={item.get('role')} "
                      f"testid={item.get('testid')} aria={item.get('aria')} ph={item.get('placeholder')} "
                      f"ce={item.get('contenteditable')} rect={item.get('rect')}")
            print(f"  candidate buttons: {len(s['candidate_buttons'])}")
            for b in s["candidate_buttons"][:5]:
                print(f"    - text={b.get('text')!r} aria={b.get('aria')!r} testid={b.get('testid')!r} cls={str(b.get('classes'))[:60]!r}")
            if s.get("banners"):
                print(f"  banners: {[b.get('text') for b in s['banners']]}")
        else:
            failures += 1
            print(f"  ERROR: {result.get('error')}")
        path = out_dir / f"{provider}_{int(time.time())}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {path}")

    if args.close:
        await engine.stop(keep_windows=False)
    else:
        print("\nWindows left open (use --close to shut them).")
    return 1 if failures == len(providers) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
