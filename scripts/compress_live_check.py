"""Live check for Compress: a real thread on ChatGPT + DeepSeek, Compress (full read of both tabs, OmniBrain's curator model
summarises, a fresh chat per provider), then each fresh chat is asked something only the old chats knew.
The foreground window is sampled the whole time; it must never change.
  python scripts/compress_live_check.py [providers...]      (default: chatgpt deepseek)
Writes data/eval/compress_live_check.json.
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.browser.factory import create_engine  # noqa: E402
from backend.memory.embed import HashEmbedder  # noqa: E402
from backend.providers.registry import ProviderCatalog, endpoint_for  # noqa: E402
from backend.settings import load_settings  # noqa: E402
from backend.thread.service import ThreadService, adapter_ask, endpoint_curator  # noqa: E402
from backend.thread.store import ThreadStore  # noqa: E402

FIRST = [
    "I'm planning a trip to Lisbon in May; my codeword for this conversation is TANGERINE-42. Reply in one short sentence.",
    "I prefer window seats and I don't want any layover in Frankfurt. Reply in one short sentence.",
    "Name one famous tram line in Lisbon. One short sentence.",
]
# long replies so the chat is several screens tall: the reader has to scroll (and ChatGPT virtualises the turns it is not showing)
LONG = [
    "In about 250 words, describe a full first day in Lisbon's Alfama district, morning to night.",
    "In about 250 words, describe a day trip from Lisbon to Sintra, with timings.",
    "In about 250 words, compare staying in Baixa, Chiado and Príncipe Real.",
    "In about 250 words, explain how Lisbon's public transport cards and fares work.",
]
SECOND = "Name one pastry Lisbon is known for. One short sentence."
CHECK = "Which month is my trip, what is my codeword, and which city should I avoid a layover in? One short sentence."


def fg() -> tuple[int, str]:
    u = ctypes.windll.user32
    h = u.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    u.GetWindowTextW(h, buf, 256)
    return int(h or 0), buf.value


async def main(providers: list[str]) -> int:
    settings = load_settings()
    db = ROOT / "data" / "tmp_compress_thread.db"
    for ext in ("", "-wal", "-shm"):
        Path(str(db) + ext).unlink(missing_ok=True)
    svc = ThreadService(ThreadStore(db, HashEmbedder()), curator=endpoint_curator(endpoint_for(settings, "analysis")))
    tid = svc.create_thread("compress live check")
    changes: list[dict] = []
    start = fg()
    stop = threading.Event()

    def watch() -> None:
        last = start
        while not stop.is_set():
            now = fg()
            if now[0] != last[0]:
                changes.append({"t": round(time.time(), 1), "from": last[1], "to": now[1]})
                last = now
            time.sleep(0.25)

    threading.Thread(target=watch, daemon=True).start()
    engine = create_engine(settings)
    await engine.start()
    report: dict = {"providers": providers, "turns": []}
    try:
        adapters = ProviderCatalog(settings, engine).all()
        ask = adapter_ask(adapters)

        async def say(p: str, text: str) -> None:
            t0 = time.monotonic()
            row = {"provider": p, "text": text}
            try:
                turn = await svc.chat(tid, text, p, ask)
                row.update(chat=turn.plan.segment_label, continued=turn.plan.continue_thread, reply=turn.reply[:300])
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            row["s"] = round(time.monotonic() - t0, 1)
            report["turns"].append(row)
            print(json.dumps(row), flush=True)

        for text in FIRST + (LONG if "--long" in sys.argv else []):
            await say(providers[0], text)
        for p in providers[1:]:
            await say(p, SECOND)

        async def read(p: str):
            seg = next((s for s in reversed(svc.store.segments(tid)) if s.provider == p), None)
            t0 = time.monotonic()
            got = await adapters[p].read_transcript(seg.provider_key if seg else "")
            print(json.dumps({"read": p, "s": round(time.monotonic() - t0, 1), "turns": [(t["role"], t["text"][:60]) for t in (got or {}).get("turns", [])],
                              **{k: v for k, v in (got or {}).items() if k != "turns"}}), flush=True)
            return got

        async def ask_logged(p, key, prompt, cont):
            t0 = time.monotonic()
            print(json.dumps({"priming": p, "chars": len(prompt)}), flush=True)
            r = await ask(p, key, prompt, cont)
            print(json.dumps({"primed": p, "s": round(time.monotonic() - t0, 1), "reply": r[:100]}), flush=True)
            return r

        t0 = time.monotonic()
        out = await svc.compress(tid, providers, ask_logged, read)
        out["total_s"] = round(time.monotonic() - t0, 1)
        report["compress"] = out
        print(json.dumps({k: v for k, v in out.items() if k != "summary"}, indent=1), flush=True)
        print("SUMMARY SENT:\n" + out["summary"][:3000], flush=True)
        for p in providers:
            await say(p, CHECK)
    finally:
        await engine.stop(keep_windows=True)
        stop.set()
        svc.store.close()
    report["foreground"] = {"start": start[1], "end": fg()[1], "changes": changes}
    out_path = ROOT / "data" / "eval" / "compress_live_check.json"
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report["foreground"], ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main([a for a in sys.argv[1:] if not a.startswith("--")] or ["chatgpt", "deepseek"])))

