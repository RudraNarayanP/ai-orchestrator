"""Vector-engine micro-benchmark for the memory store decision (data/MEMORY_DESIGN.md).

Compares, on this machine, at N memories x D dims (random unit vectors; latency only, quality is measured separately):
  numpy exact (float32, RAM)  |  numpy exact int8 (RAM)  |  sqlite-vec vec0 exact (on disk)  |  sqlite-vec prefiltered by project
Cold = fresh process/connection first query; warm = median of 50 queries. Writes data/eval/memory_bench_engines.json
"""
from __future__ import annotations

import json, os, sqlite3, statistics, sys, tempfile, time
import numpy as np
import sqlite_vec

D = int(os.environ.get("DIM", "384"))
SIZES = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1000,10000,100000").split(",")]
K = 20
rng = np.random.default_rng(7)


def unit(a):
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def ms(t0):
    return (time.perf_counter() - t0) * 1000


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def bench(n):
    out = {"n": n, "dim": D}
    vecs = unit(rng.standard_normal((n, D), dtype=np.float32))
    proj = rng.integers(0, 50, n)  # 50 projects -> a project filter keeps ~2%
    qs = unit(rng.standard_normal((60, D), dtype=np.float32))
    # numpy float32
    t0 = time.perf_counter(); _ = vecs @ qs[0]; out["np_f32_cold_ms"] = ms(t0)
    ts = []
    for q in qs[1:51]:
        t0 = time.perf_counter(); s = vecs @ q; idx = np.argpartition(-s, K)[:K]; idx = idx[np.argsort(-s[idx])]; ts.append(ms(t0))
    out["np_f32_warm_p50_ms"], out["np_f32_warm_p95_ms"] = statistics.median(ts), pct(ts, 0.95)
    ref = idx
    # numpy int8
    v8 = np.clip(np.round(vecs * 127), -127, 127).astype(np.int8)
    ts = []
    for q in qs[1:51]:
        q8 = np.clip(np.round(q * 127), -127, 127).astype(np.int8).astype(np.int16)
        t0 = time.perf_counter(); s = v8.astype(np.int16) @ q8 if n <= 20000 else (v8 @ q8.astype(np.int8)); i2 = np.argpartition(-s, K)[:K]; ts.append(ms(t0))
    out["np_i8_warm_p50_ms"] = statistics.median(ts)
    # numpy prefiltered by project (index array of the project's rows)
    rows = {p: np.nonzero(proj == p)[0] for p in range(50)}
    ts = []
    for i, q in enumerate(qs[1:51]):
        t0 = time.perf_counter(); r = rows[i % 50]; s = vecs[r] @ q; _ = np.argsort(-s)[:K]; ts.append(ms(t0))
    out["np_prefilter_p50_ms"] = statistics.median(ts)
    # sqlite-vec
    path = os.path.join(tempfile.gettempdir(), f"vec_bench_{n}.db")
    if os.path.exists(path): os.remove(path)
    db = sqlite3.connect(path); db.enable_load_extension(True); sqlite_vec.load(db); db.enable_load_extension(False)
    db.execute(f"create virtual table v using vec0(embedding float[{D}])")
    db.execute("create table mem(id integer primary key, project integer)")
    t0 = time.perf_counter()
    db.execute("begin")
    for s0 in range(0, n, 5000):
        chunk = vecs[s0:s0 + 5000]
        db.executemany("insert into v(rowid, embedding) values (?, ?)", [(s0 + i + 1, sqlite_vec.serialize_float32(chunk[i].tolist())) for i in range(len(chunk))])
        db.executemany("insert into mem values (?, ?)", [(s0 + i + 1, int(proj[s0 + i])) for i in range(len(chunk))])
    db.execute("commit")
    out["sqlitevec_insert_total_s"] = round(time.perf_counter() - t0, 2)
    db.close()
    # cold: new connection, first query
    db = sqlite3.connect(path); db.enable_load_extension(True); sqlite_vec.load(db); db.enable_load_extension(False)
    qb = [sqlite_vec.serialize_float32(q.tolist()) for q in qs]
    t0 = time.perf_counter(); db.execute("select rowid, distance from v where embedding match ? and k = ?", (qb[0], K)).fetchall(); out["sqlitevec_cold_ms"] = ms(t0)
    ts = []
    for b in qb[1:51]:
        t0 = time.perf_counter(); r = db.execute("select rowid, distance from v where embedding match ? and k = ?", (b, K)).fetchall(); ts.append(ms(t0))
    out["sqlitevec_warm_p50_ms"], out["sqlitevec_warm_p95_ms"] = statistics.median(ts), pct(ts, 0.95)
    got = {r[0] - 1 for r in r}
    out["sqlitevec_matches_numpy_top20"] = len(got & set(ref.tolist())) / K if False else None
    ts = []
    try:
        for i, b in enumerate(qb[1:51]):
            t0 = time.perf_counter()
            db.execute("select rowid, distance from v where embedding match ? and k = ? and rowid in (select id from mem where project = ?)", (b, K, i % 50)).fetchall()
            ts.append(ms(t0))
        out["sqlitevec_prefilter_p50_ms"] = statistics.median(ts)
    except Exception as e:
        out["sqlitevec_prefilter_error"] = str(e)[:120]
    out["db_mb"] = round(os.path.getsize(path) / 1e6, 1)
    db.close(); os.remove(path)
    return out


results = []
for n in SIZES:
    t0 = time.perf_counter()
    r = bench(n)
    r["bench_s"] = round(time.perf_counter() - t0, 1)
    results.append(r)
    print(json.dumps(r), flush=True)
    path = os.path.join("data", "eval", f"memory_bench_engines_{'_'.join(map(str, SIZES))}.json")
    json.dump({"machine": os.environ.get("COMPUTERNAME"), "python": sys.version.split()[0], "results": results}, open(path, "w"), indent=1)