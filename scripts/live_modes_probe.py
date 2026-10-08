"""Live mode-toggle probe (debug): list the Thinking/Search/Deep Research toggles beside each provider's composer.

Usage: python scripts/live_modes_probe.py chatgpt,gemini,deepseek [--roundtrip deepseek:thinking]
Read-only unless --roundtrip is given: then that one toggle is switched on, verified, switched back off, verified.
Plan-gated, disabled or unreadable toggles are never clicked.
"""
import asyncio, json, sys
sys.path.insert(0, ".")
from backend.browser.factory import create_engine
from backend.providers.registry import ProviderCatalog
from backend.settings import load_settings


async def probe(cat, prov, roundtrip):
    ad = cat.adapter(prov)
    out = {"provider": prov}
    try:
        page = await ad._page(fresh=False)
        await ad._settle(page)
        ready = await ad.prepare(page, None, 1) if False else await ad.readiness(page)
        out["readiness"] = ready.get("state")
        await asyncio.sleep(2.0)
        out["modes"] = await ad.modes(page)
        out["menus"] = await ad.mode_menus(page)
        for spec in roundtrip:
            p, mode = spec.split(":")
            if p != prov:
                continue
            # flip it and flip it back, so the user's own setting is left exactly as it was
            before = next((m.get("state") for m in out["modes"] if m.get("mode") == mode), None)
            if before not in ("on", "off"):
                out.setdefault("roundtrips", []).append({"mode": mode, "status": f"not tried (state {before})"})
                continue
            first = await ad.set_mode(page, mode, before == "off")
            back = await ad.set_mode(page, mode, before == "on") if first.get("status") in ("on", "off") else {"status": "not needed"}
            out.setdefault("roundtrips", []).append({"mode": mode, "was": before, "flip": first.get("status"), "restore": back.get("status"), "label": first.get("label")})
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)[:200]
    return out


async def main():
    provs = sys.argv[1].split(",")
    roundtrip = [a for a in sys.argv[2:] if ":" in a]
    s = load_settings()
    eng = create_engine(s)
    await eng.start()
    cat = ProviderCatalog(s, eng)
    try:
        results = await asyncio.gather(*(probe(cat, p, roundtrip) for p in provs))
    finally:
        await eng.stop(keep_windows=False)
    for r in results:
        print(json.dumps(r, ensure_ascii=False))


asyncio.run(main())
