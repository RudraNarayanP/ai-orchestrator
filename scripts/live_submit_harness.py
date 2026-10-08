"""Live submit harness (debug): ask providers a short and/or a long multi-line prompt, all providers IN PARALLEL.

Usage: python scripts/live_submit_harness.py chatgpt,gemini,deepseek,meta_ai [short|long|both]
Each provider gets its own tab (own chrome-use session) in the user's Chrome; every input command is logged per session.
"""
import asyncio, sys, time
sys.path.insert(0, ".")
from backend.browser.factory import create_engine
from backend.browser.live_chrome import ChromeUseRunner
from backend.providers.registry import ProviderCatalog
from backend.settings import load_settings

LONG = ("You are one of several research assistants.\n\nQuestion: In which year was the Eiffel Tower completed?\n\n"
        "Rules:\n- Give the answer first in one short sentence.\n- Then name one source.\n- If unsure, say so.")
SHORT = "Reply with the single word: pong"
LOG: dict[str, list[str]] = {}
_real = ChromeUseRunner.run


async def _spy(self, *a, **kw):
    if a and a[0] in ("keyboard", "press", "click"):
        LOG.setdefault(self.session, []).append(" ".join(a[:3]) + (f" <stdin {len(kw.get('stdin') or '')} chars>" if kw.get("stdin") else ""))
    return await _real(self, *a, **kw)


async def one(cat, prov, which, results):
    ad = cat.adapter(prov)
    ad.sel.hard_timeout_ms = 90000
    for name, q in (("short", SHORT), ("long", LONG)):
        if which not in ("both", name):
            continue
        t0 = time.time()
        try:
            r = await ad.ask(f"h_{name}", q, 1)
            results.append((prov, name, r.status.value, round(time.time() - t0), r.error, (r.answer_text or "")[:160]))
        except Exception as exc:  # noqa: BLE001
            results.append((prov, name, "crashed", round(time.time() - t0), repr(exc)[:200], ""))


async def main():
    provs = [p for p in sys.argv[1].split(",") if p]
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    s = load_settings(); s.browser.driver = "chrome_use"
    eng = create_engine(s); await eng.start()
    ChromeUseRunner.run = _spy
    results: list = []
    t0 = time.time()
    try:
        cat = ProviderCatalog(s, eng)
        await asyncio.gather(*(one(cat, p, which, results) for p in provs))
    finally:
        await eng.stop(keep_windows=False)
    print(f"total wall time {time.time()-t0:.0f}s for {len(provs)} provider(s) in parallel")
    for prov, name, status, secs, err, ans in results:
        print(f"[{prov}/{name}] {status} in {secs}s err={err!r}")
        print("   answer:", repr(ans))
    for sess, cmds in LOG.items():
        print(f"   inputs[{sess}]:", cmds)

asyncio.run(main())
