"""Synthetic long-thread benchmark: python scripts/thread_bench.py [messages=100000] [embedder=hash|fastembed]

Builds one thread of N messages in segments of 25 (each closed, summarised and folded into the thread state), plants needle facts at
random depths, then measures recall (hit@12, hit@1, latency), the cost of preparing a turn / a rotation, and the research-job packet.
Writes data/eval/thread_bench_<N>.json.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.memory.embed import get_embedder  # noqa: E402
from backend.thread.service import ContextManager, ThreadService  # noqa: E402
from backend.thread.store import ThreadStore  # noqa: E402

TOPICS = ["guitar strings", "pasta recipe", "tax filing", "kernel module", "marathon training", "tulip bulbs", "mortgage rates", "chess openings",
          "solar panels", "python packaging", "sourdough starter", "visa paperwork", "router firmware", "piano scales", "bike maintenance", "database indexes"]
NEEDLES = [
    ("What was the Reykjavik hotel booking reference?", "The Reykjavik hotel booking reference is QX-7731 for the aurora trip", "QX-7731"),
    ("Which dentist did I choose in Lisbon?", "I chose Dr Almeida in Lisbon as my dentist because of the Saturday hours", "Almeida"),
    ("What did we decide about the Munich flight?", "Let's go with Lufthansa direct to Munich and avoid Frankfurt", "Lufthansa"),
    ("What is the wifi password at the cabin?", "The cabin wifi password is pinecone-4421 as the host told us", "pinecone"),
    ("Which tent did you recommend for Patagonia?", "For Patagonia I recommend the Hilleberg Nallo because of the wind", "Hilleberg"),
    ("What was the Blue Card salary threshold?", "The German Blue Card minimum salary is about 45,300 EUR", "45,300"),
    ("What is my sister's birthday?", "My sister Anya's birthday is on 17 November", "17 November"),
    ("Which database did we choose for Atlas?", "We chose PostgreSQL 16 for the Atlas project database", "PostgreSQL 16"),
    ("What was the marathon target time?", "My marathon target time is 3 hours 45 minutes in Valencia", "Valencia"),
    ("What did the landlord say about the deposit?", "The landlord said the deposit of 1,800 EUR is returned within 30 days", "1,800"),
]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    emb = sys.argv[2] if len(sys.argv) > 2 else "hash"
    rnd = random.Random(7)
    tmp = Path(tempfile.mkdtemp(prefix="threadbench_")) / "t.db"
    store = ThreadStore(tmp, get_embedder(emb))
    svc = ThreadService(store, context=ContextManager({"chatgpt": 32000}), packet_budget=6000)
    tid = svc.create_thread(project="bench")
    now = time.time()
    planted: dict[int, tuple[str, str, str]] = {}
    for i, nd in enumerate(NEEDLES):
        planted[rnd.randrange(0, n)] = nd
    seg_size = 25
    t0 = time.perf_counter()
    close_times: list[float] = []
    made = 0
    while made < n:
        seg = store.open_segment(tid, "chatgpt", "bench")
        rows = []
        for j in range(min(seg_size, n - made)):
            i = made + j
            if i in planted:
                text = planted[i][1]
            else:
                a, b = rnd.choice(TOPICS), rnd.choice(TOPICS)
                text = f"{a} discussion {i}: details about {b} and item {i} with some further {rnd.choice(TOPICS)} remarks"
            rows.append(dict(role="user" if i % 2 == 0 else "assistant", content=text, provider="chatgpt", segment_id=seg.segment_id,
                             ts=now - (n - i) * 180))
        store.add_messages(tid, rows)
        made += len(rows)
        c0 = time.perf_counter()
        svc.close_segment(store.segment(seg.segment_id))  # type: ignore[arg-type]
        close_times.append(time.perf_counter() - c0)
    build_s = time.perf_counter() - t0

    # --- recall
    store._vec.clear()  # cold: vectors load from the database
    c0 = time.perf_counter()
    svc.recall(tid, NEEDLES[0][0])
    cold = time.perf_counter() - c0
    lat, hit12, hit1 = [], 0, 0
    for q, _txt, key in NEEDLES * 3:
        c0 = time.perf_counter()
        found, _s = svc.recall(tid, q)
        lat.append(time.perf_counter() - c0)
    for q, _txt, key in NEEDLES:
        found, _s = svc.recall(tid, "remember " + q)
        hit12 += any(key in f.message.content for f in found)
        hit1 += bool(found) and key in found[0].message.content

    # --- turns
    turn_lat, rot_lat = [], []
    seg = store.open_segment(tid, "chatgpt", "bench")
    for k in range(60):
        c0 = time.perf_counter()
        p = svc.plan_turn(tid, f"continuing question {k} about {rnd.choice(TOPICS)}", "chatgpt")
        dt = time.perf_counter() - c0
        (rot_lat if p.rotated else turn_lat).append(dt)
        svc.record_reply(tid, "short reply " + "word " * 40, "chatgpt", segment_id=p.segment_id)
    sw = []
    for prov in ("gemini", "chatgpt", "gemini"):
        c0 = time.perf_counter()
        p = svc.plan_turn(tid, "switching provider now", prov)
        sw.append(time.perf_counter() - c0)
    job = []
    for _ in range(10):
        c0 = time.perf_counter()
        pk = svc.context_for_job(tid, "continue the thread please")
        job.append(time.perf_counter() - c0)
    rc = []
    for _ in range(5):
        c0 = time.perf_counter()
        p = svc.plan_turn(tid, "Remember that thing about the Reykjavik hotel 5 months ago?", "gemini" if _ % 2 == 0 else "chatgpt")
        rc.append(time.perf_counter() - c0)
    size = os.path.getsize(tmp) / 1e6
    out = {
        "messages": n, "embedder": store.embedder.name, "segments": len(store.segments(tid)), "build_s": round(build_s, 1), "db_mb": round(size, 1),
        "segment_close_ms_p50": round(statistics.median(close_times) * 1000, 1), "segment_close_ms_p95": round(pct(close_times, .95) * 1000, 1),
        "state_json_kb": round(len(json.dumps(store.get_thread(tid)["state"])) / 1024, 1),
        "recall_cold_ms": round(cold * 1000, 1), "recall_warm_ms_p50": round(statistics.median(lat) * 1000, 1), "recall_warm_ms_p95": round(pct(lat, .95) * 1000, 1),
        "recall_hit_at_12": f"{hit12}/{len(NEEDLES)}", "recall_hit_at_1": f"{hit1}/{len(NEEDLES)}",
        "plan_turn_ms_p50": round(statistics.median(turn_lat) * 1000, 2), "plan_turn_ms_p95": round(pct(turn_lat, .95) * 1000, 2),
        "rotation_turn_ms_p50": round(statistics.median(rot_lat) * 1000, 1) if rot_lat else None, "rotations": len(rot_lat),
        "provider_switch_ms_p50": round(statistics.median(sw) * 1000, 1), "job_context_ms_p50": round(statistics.median(job) * 1000, 1),
        "job_context_ms_p95": round(pct(job, .95) * 1000, 1), "job_packet_tokens": pk.tokens if pk else None,
        "recall_turn_ms_p50": round(statistics.median(rc) * 1000, 1), "raw_messages_intact": store.count(tid) >= n,
    }
    path = ROOT / "data" / "eval" / f"thread_bench_{n}_{emb}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())