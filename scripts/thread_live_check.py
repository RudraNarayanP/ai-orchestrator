"""Live check: a thread keeps going across a REAL chat rotation and a REAL provider switch.

Throw-away thread file; real Chrome providers. A tiny context limit forces ChatGPT to rotate after a few turns; the first question
in the new chat asks for facts only the earlier chat knew. Then the thread switches to Gemini and asks for another.
  python scripts/thread_live_check.py
Writes data/eval/thread_live_check.json.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.browser.engine import BrowserEngine  # noqa: E402
from backend.memory.embed import HashEmbedder  # noqa: E402
from backend.providers.registry import ProviderCatalog  # noqa: E402
from backend.settings import load_settings  # noqa: E402
from backend.thread.service import ContextManager, ThreadService, adapter_ask  # noqa: E402
from backend.thread.store import ThreadStore  # noqa: E402

TURNS = [
    ("chatgpt", "Hi! I'm planning a trip to Lisbon in May, and my codeword for this conversation is TANGERINE-42. Reply in one short sentence."),
    ("chatgpt", "I prefer window seats and I don't want any layover in Frankfurt. Reply in one short sentence."),
    ("chatgpt", "Name one famous tram line in Lisbon. One short sentence."),
    ("chatgpt", "Name one pastry Lisbon is known for. One short sentence."),
    ("chatgpt", "Which month is my trip, and what is my codeword? One short sentence."),
    ("gemini", "Which seat do I prefer, and which city should I avoid a layover in? One short sentence."),
]


async def main() -> int:
    settings = load_settings()
    db = ROOT / "data" / "tmp_live_thread.db"
    for ext in ("", "-wal", "-shm"):
        Path(str(db) + ext).unlink(missing_ok=True)
    svc = ThreadService(ThreadStore(db, HashEmbedder()), context=ContextManager({"chatgpt": 250}, rotate_at=0.5, reply_reserve=20), packet_budget=500)
    tid = svc.create_thread()
    engine = BrowserEngine(settings)
    await engine.start()
    report = {"turns": []}
    try:
        ask = adapter_ask(ProviderCatalog(settings, engine).all())
        for provider, text in TURNS:
            row = {"provider": provider, "text": text}
            try:
                turn = await svc.chat(tid, text, provider, ask)
                row.update(chat=turn.plan.segment_label, rotated=turn.plan.rotated, reason=turn.plan.reason, continue_thread=turn.plan.continue_thread,
                           packet_tokens=turn.plan.packet.tokens if turn.plan.packet else None, reply=turn.reply[:300])
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            report["turns"].append(row)
            print(json.dumps(row), flush=True)
        by = report["turns"]
        report["checks"] = {
            "rotated_at_least_once": any(t.get("rotated") for t in by),
            "chatgpt_rotations": sum(1 for t in by if t.get("rotated") and t["provider"] == "chatgpt"),
            "chat_after_rotation_knew_codeword_and_month": any(t.get("rotated") and t.get("reason") == "context_limit" for t in by[:5]) and not by[4].get("continue_thread") and "TANGERINE" in by[4].get("reply", "").upper() and "may" in by[4].get("reply", "").lower(),
            "switch_to_gemini_knew_seat": "window" in by[5].get("reply", "").lower(),
            "raw_messages_kept": svc.store.count(tid),
            "chats": [svc.label(s) for s in svc.store.segments(tid)],
        }
        print(json.dumps(report["checks"], indent=1))
    finally:
        await engine.stop(keep_windows=False)
        svc.store.close()
        for ext in ("", "-wal", "-shm"):
            Path(str(db) + ext).unlink(missing_ok=True)
    (ROOT / "data" / "eval" / "thread_live_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))