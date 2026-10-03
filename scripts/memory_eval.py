"""Memory evaluation: retrieval quality, behaviour scenarios, and scale.

  python scripts/memory_eval.py quality  [--embedders hash,fastembed]
  python scripts/memory_eval.py scenarios
  python scripts/memory_eval.py scale 1000,10000,100000 [--embedder hash]

Everything is synthetic or hand-written - no personal data. Results land in data/eval/memory_*.json.
Relevance is judged by planted ids / expected substrings, never by the system under test.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.memory import MemoryService, MemoryStore  # noqa: E402
from backend.memory.embed import HashEmbedder, get_embedder  # noqa: E402
from backend.memory.retrieve import Retriever  # noqa: E402
from backend.memory.schema import Memory, MemoryType, Source  # noqa: E402
from backend.memory.service import approx_tokens  # noqa: E402

OUT = ROOT / "data" / "eval"

# ---------------------------------------------------------------- hand-written quality corpus
CORPUS = [
    # (content, type, source, slot, entities)
    ("Lives in Kyiv", "fact", "user_explicit", "residence", []),
    ("Works at Acme Robotics as a backend developer", "fact", "user_explicit", "employer", ["Acme Robotics"]),
    ("Prefers concise answers without preamble", "preference", "user_explicit", "pref:answer_style", []),
    ("Is allergic to peanuts", "fact", "user_explicit", None, []),
    ("Owns a cat called Miso", "fact", "user_explicit", None, ["Miso"]),
    ("Plays the cello on weekends", "fact", "user_explicit", None, []),
    ("Is learning Rust", "goal", "user_explicit", "goal:learning rust", ["Rust"]),
    ("Has a brother called Taras who lives in Lviv", "fact", "user_explicit", None, ["Taras", "Lviv"]),
    ("Is vegetarian and avoids fish", "fact", "user_explicit", "diet", []),
    ("Studies for a master's degree in statistics at night", "fact", "user_explicit", "education", []),
    ("Speaks Ukrainian, English and some Polish", "fact", "user_explicit", "languages", []),
    ("Uses a Linux laptop with Fedora for development", "fact", "user_explicit", None, ["Fedora"]),
    ("Dislikes long video tutorials, prefers written guides", "preference", "user_explicit", None, []),
    ("Wants to run a half marathon in spring", "goal", "user_explicit", "goal:half marathon", []),
    ("Likes dark roast coffee", "preference", "model_inferred", None, []),
    ("Is training for a half marathon three times a week", "episodic", "model_inferred", None, []),
    ("Budget for a new monitor is around 400 euros", "fact", "user_explicit", None, []),
    ("Prefers metric units", "preference", "user_explicit", "pref:units", []),
    ("Birthday is on 14 March", "fact", "user_explicit", "birthday", []),
    ("Takes the morning train to work", "fact", "user_explicit", None, []),
    ("Is building a personal budgeting app in Python", "goal", "user_explicit", "goal:budgeting app", ["Python"]),
    ("Has a peanut-free kitchen at home because of a family member's allergy", "fact", "user_explicit", None, []),
    ("Reads science fiction, especially Stanislaw Lem", "preference", "model_inferred", None, ["Stanislaw Lem"]),
    ("Girlfriend is called Olena", "fact", "user_explicit", None, ["Olena"]),
    ("Uses VS Code with the vim extension", "fact", "user_explicit", None, ["VS Code"]),
    ("Wants answers to include sources", "preference", "user_explicit", "pref:sources", []),
    ("Is saving money for a trip to Japan next year", "goal", "user_explicit", "goal:japan trip", ["Japan"]),
    ("Does not drink alcohol", "fact", "user_explicit", None, []),
    ("Team at work has six engineers and a product manager called Daria", "fact", "user_explicit", None, ["Daria"]),
    ("Has a nine-year-old bike with a broken derailleur", "fact", "user_explicit", None, []),
]
P_ATLAS = ("Atlas", [
    "Project Atlas database is PostgreSQL 16", "Project Atlas deploys with Docker on a single VPS", "Project Atlas frontend uses React and Vite",
])
P_BOREALIS = ("Borealis", ["Project Borealis database is MongoDB", "Project Borealis is written in Go"])

# (query, substrings of memories that are relevant; at least one should be retrieved) - paraphrased on purpose
POSITIVE = [
    ("where do I live?", ["Kyiv"]), ("which city am I based in", ["Kyiv"]),
    ("what do I do for work?", ["Acme"]), ("who is my employer", ["Acme"]),
    ("how should you format your replies to me?", ["concise"]), ("keep it brief please, how do I like answers", ["concise"]),
    ("suggest a snack for me, I have an allergy", ["peanut"]), ("can you recommend a dessert I can eat safely", ["peanut"]),
    ("what is my cat's name?", ["Miso"]), ("tell me about my pet", ["Miso"]),
    ("what instrument do I play", ["cello"]), ("what do I do on weekends for music", ["cello"]),
    ("what programming language am I learning", ["Rust"]),
    ("where does my brother live", ["Taras"]), ("do I have siblings", ["brother"]),
    ("what should I cook tonight for dinner", ["vegetarian"]), ("restaurant ideas for me, I do not eat meat", ["vegetarian"]),
    ("what am I studying", ["statistics"]), ("which languages do I speak", ["Ukrainian"]),
    ("what computer do I use", ["Fedora"]), ("which operating system is my laptop running", ["Fedora"]),
    ("should you send me a youtube course or an article", ["written guides"]),
    ("what race am I preparing for", ["half marathon"]),
    ("how much can I spend on a screen", ["monitor"]),
    ("which units should you use for distances", ["metric"]),
    ("when is my birthday", ["14 March"]),
    ("how do I get to work", ["train"]),
    ("what am I building in Python", ["budgeting"]),
    ("what books do I like", ["Lem"]), ("what is my partner called", ["Olena"]),
    ("which editor do I use for coding", ["VS Code"]),
    ("do I want citations in answers", ["sources"]),
    ("where am I travelling next year", ["Japan"]),
    ("do I drink alcohol", ["alcohol"]),
    ("who is the product manager of my team", ["Daria"]),
    ("what is wrong with my bicycle", ["derailleur"]),
]
# project-scoped positives: (query, project, substring)
PROJECT_POSITIVE = [
    ("which database does the project use", "Atlas", "PostgreSQL"), ("how is Atlas deployed", "Atlas", "Docker"),
    ("what is the frontend stack", "Atlas", "React"), ("what language is it written in", "Borealis", "Go"),
]
# general questions: nothing about the user is relevant -> zero hits expected
GENERAL = [
    "What is the capital of France?", "how tall is Mount Everest", "explain quantum tunnelling", "who won the 1998 world cup",
    "what is 17 * 23", "define photosynthesis", "when did the Berlin wall fall", "how many moons does Jupiter have",
    "what is the boiling point of ethanol", "summarise the plot of Hamlet", "what is the speed of light",
    "how does a transformer neural network work", "who wrote Pride and Prejudice", "what is the GDP of Germany",
    "translate 'good morning' into Spanish", "what is the difference between TCP and UDP",
]
# personal-sounding but unrelated to anything stored -> a careful system injects nothing
DISTRACTORS = [
    "what should I name my new goldfish tank plant", "give me tips for a job interview at a bank",
    "how do I fix a leaking kitchen tap", "recommend a podcast about ancient history",
    "what is a good gift for my grandmother", "how do I train for a swimming competition",
    "can you help me write a poem about autumn", "how do I repot an orchid",
]


def build(embedder, with_projects=True) -> MemoryStore:
    s = MemoryStore(":memory:", embedder)
    for content, typ, src, slot, ents in CORPUS:
        s.add(content, memory_type=typ, source=src, slot=slot, entities=ents)
    if with_projects:
        for proj, items in (P_ATLAS, P_BOREALIS):
            for c in items:
                s.add(c, memory_type="project", project=proj)
    return s


def judge(hits, subs):
    texts = [h.memory.content for h in hits]
    for rank, t in enumerate(texts, 1):
        if any(x.lower() in t.lower() for x in subs):
            return rank
    return None


def run_quality_for(embedder, floor):
    s = build(embedder)
    r = Retriever(s)
    rr, p_at, got, toks, lat = [], [], 0, [], []
    for q, subs in POSITIVE:
        t0 = time.perf_counter()
        res = r.retrieve(q, min_relevance=floor)
        lat.append((time.perf_counter() - t0) * 1000)
        rank = judge(res.hits, subs)
        rr.append(1 / rank if rank else 0.0)
        got += bool(rank)
        rel = sum(1 for h in res.hits if any(x.lower() in h.memory.content.lower() for x in subs))
        p_at.append(rel / len(res.hits) if res.hits else 0.0)
        toks.append(approx_tokens("\n".join(h.memory.content for h in res.hits)))
    pr = 0
    for q, proj, sub in PROJECT_POSITIVE:
        res = r.retrieve(q, project=proj, min_relevance=floor)
        pr += bool(judge(res.hits, [sub]))
    gen_quiet = sum(1 for q in GENERAL if not r.retrieve(q, min_relevance=floor).hits)
    dis_quiet = sum(1 for q in DISTRACTORS if not r.retrieve(q, min_relevance=floor).hits)
    n = len(POSITIVE)
    s.close()
    return {
        "floor": floor, "recall@budget": round(got / n, 3), "MRR": round(statistics.mean(rr), 3),
        "precision_of_returned": round(statistics.mean(p_at), 3), "project_recall": round(pr / len(PROJECT_POSITIVE), 3),
        "general_suppressed": round(gen_quiet / len(GENERAL), 3), "distractor_suppressed": round(dis_quiet / len(DISTRACTORS), 3),
        "avg_injected_tokens": round(statistics.mean(toks), 1), "p50_ms": round(statistics.median(lat), 2),
    }


def cmd_quality(args):
    out = {"n_positive": len(POSITIVE), "n_project": len(PROJECT_POSITIVE), "n_general": len(GENERAL), "n_distractor": len(DISTRACTORS),
           "corpus_size": len(CORPUS) + len(P_ATLAS[1]) + len(P_BOREALIS[1]), "embedders": {}}
    for kind in args.embedders.split(","):
        t0 = time.perf_counter()
        emb = get_embedder(kind, "BAAI/bge-small-en-v1.5") if kind != "hash" else HashEmbedder()
        load_s = time.perf_counter() - t0
        floors = [0.15, 0.18, 0.22, 0.26, 0.30, 0.35] if kind == "hash" else [0.45, 0.50, 0.55, 0.58, 0.62, 0.66, 0.70]
        rows = [run_quality_for(emb, f) for f in floors]
        out["embedders"][kind] = {"name": emb.name, "load_s": round(load_s, 2), "default_floor": emb.floor, "sweep": rows}
        print(f"\n== {kind} ({emb.name}) load {load_s:.1f}s")
        print("floor  recall  MRR   prec  proj  general_quiet  distractor_quiet  tokens  p50ms")
        for x in rows:
            print(f"{x['floor']:.2f}   {x['recall@budget']:.2f}   {x['MRR']:.2f}  {x['precision_of_returned']:.2f}  {x['project_recall']:.2f}   {x['general_suppressed']:.2f}          {x['distractor_suppressed']:.2f}             {x['avg_injected_tokens']:>5}  {x['p50_ms']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "memory_quality.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- behaviour scenarios
def cmd_scenarios(args):
    emb = get_embedder(args.embedder, "BAAI/bge-small-en-v1.5") if args.embedder != "hash" else HashEmbedder()
    results: list[dict] = []

    def check(group, name, ok, detail=""):
        results.append({"group": group, "name": name, "pass": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + f"[{group}] {name} {detail}")

    def fresh():
        return MemoryService(MemoryStore(":memory:", emb))

    # --- correction / supersede / temporal
    v = fresh()
    v.learn("I live in Kyiv")
    v.learn("Actually I live in Lviv now")
    act = [m.content for m in v.store.list(status="ACTIVE") if m.slot == "residence"]
    check("correction", "newest statement replaces the old one", act == ["Lives in Lviv"], str(act))
    check("correction", "never two ACTIVE in one slot (invariants)", v.store.check_invariants() == [])
    _b, now = v.context_for("where do I live?")
    check("temporal", "current question gets only the current fact", [h.memory.content for h in now.hits] == ["Lives in Lviv"], str([h.memory.content for h in now.hits]))
    _b, hist = v.context_for("where did I live before?")
    check("temporal", "history question can see the superseded fact", any("Kyiv" in h.memory.content for h in hist.hits))
    v.learn("My goal is to finish the Rust book")
    v.learn("I finished the Rust book")
    _b, g = v.context_for("what are my goals with the Rust book?")
    check("temporal", "finished goal is no longer injected", not any(h.memory.memory_type == MemoryType.GOAL and h.memory.goal_active for h in g.hits))
    # --- explicit vs inferred
    v = fresh()
    v.store.add("Lives in Kyiv", slot="residence", source=Source.USER_EXPLICIT)
    r = v.store.add("Lives in Odesa", slot="residence", source=Source.MODEL_INFERRED)
    check("source", "inferred never overrides explicit", r.action == "rejected")
    v.store.add("Likes strong tea in the evening", source=Source.MODEL_INFERRED, slot="x1")
    v.store.add("Drinks strong tea in the evening every day", source=Source.USER_EXPLICIT, slot="x2")
    _b, t = v.context_for("what tea do I drink in the evening?")
    srcs = [h.memory.source.value for h in t.hits]
    check("source", "explicit outranks inferred at equal relevance", bool(srcs) and srcs[0] == "user_explicit", str(srcs))
    # --- project isolation
    v = fresh()
    v.store.add("Project Atlas database is PostgreSQL 16", memory_type="project", project="Atlas")
    v.store.add("Project Borealis database is MongoDB", memory_type="project", project="Borealis")
    _b, a = v.context_for("which database does the project use?", project="Atlas")
    _b2, n = v.context_for("which database does the project use?")
    check("isolation", "Atlas question never sees Borealis memory", not any("Borealis" in h.memory.content for h in a.hits))
    check("isolation", "no-project question sees no project memory", not any(h.memory.project for h in n.hits))
    # --- opt-out, secrets, delete
    v = fresh()
    v.learn("Don't remember this: my sister is called Anya")
    v.learn("my password is hunter2")
    check("privacy", "opt-out and secrets store nothing", v.store.count() == 0)
    r = v.remember("Secret hobby is falconry")
    v.store.delete(r.memory.memory_id)
    leaked = v.store.fts_search(["falconry"]) or [x for x in v.store.export() if "falconry" in json.dumps(x)]
    check("privacy", "deleted memory is gone from rows, FTS, vectors and export", not leaked)
    # --- cross-provider consistency (same block for every provider; deterministic)
    v = build(emb)
    svc = MemoryService(v)
    blocks = {svc.context_for("what is my cat's name?")[0] for _ in range(5)}
    check("consistency", "identical context block on repeated calls (every provider receives the same text)", len(blocks) == 1)
    # --- memory never evidence: run the isolation tests
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_memory_isolation.py", "--tb=line"], cwd=ROOT, capture_output=True, text=True)
    last = [ln for ln in p.stdout.splitlines() if "passed" in ln or "failed" in ln][-1:] or ["?"]
    check("evidence", "memory never reaches claims/evidence/final answer; all providers get the same block", p.returncode == 0, last[0])
    out = {"embedder": emb.name, "passed": sum(r["pass"] for r in results), "total": len(results), "results": results}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"memory_scenarios_{args.embedder}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{out['passed']}/{out['total']} scenarios passed ({emb.name})")


# ---------------------------------------------------------------- scale
FIRST = "Anna Boris Clara Dmytro Eva Felix Greta Hugo Iryna Jan Kira Leo Mila Nikita Oksana Pavlo Quinn Rita Stefan Tanya Uma Viktor Wanda Yuri Zoya".split()
LAST = "Adler Bondar Corva Danko Esser Fenn Gorin Havel Ivanek Jarek Kovac Lindt Marek Novak Orlov Petrov Quade Rusko Sorel Tomas".split()
TEAMS = "platform payments search growth infra data mobile security design support billing analytics".split()
TOOLS = "Jira Notion Figma Grafana Terraform Kafka Redis Airflow Sentry Postman Slack Linear Datadog Snowflake".split()
CITIES = "Berlin Oslo Porto Graz Turin Bergen Gdansk Cork Lyon Split Brno Riga Tartu Malmo".split()
FOODS = "ramen pierogi risotto falafel paella dumplings gnocchi curry tacos pho".split()
SPORTS = "tennis rowing climbing volleyball fencing curling badminton skiing".split()
TEMPL = [
    "Colleague {n} works on the {t} team and uses {x}", "Colleague {n} lives in {c} and likes {f}", "Neighbour {n} plays {s} on Saturdays",
    "Meeting with {n} about the {t} roadmap was moved to {x}", "{n} recommended trying {f} at the place in {c}", "Client {n} asked for a {t} report in {x}",
    "Cousin {n} moved to {c} last year", "The {t} team switched from {x} to {x2}", "Friend {n} is training for a {s} tournament in {c}",
    "Ticket about {x} was reassigned to {n} from the {t} team", "Reminder that {n} prefers {f} for lunch", "Landlord {n} lives in {c}",
]


def synth_text(rng):
    return rng.choice(TEMPL).format(n=f"{rng.choice(FIRST)} {rng.choice(LAST)}", t=rng.choice(TEAMS), x=rng.choice(TOOLS), x2=rng.choice(TOOLS),
                                    c=rng.choice(CITIES), f=rng.choice(FOODS), s=rng.choice(SPORTS))


def build_scale(path: Path, n: int, emb, seed=11):
    rng = random.Random(seed)
    s = MemoryStore(path, emb)
    now = time.time()
    t0 = time.perf_counter()
    planted = [c for c, *_ in CORPUS]
    texts = planted + [synth_text(rng) for _ in range(max(0, n - len(planted)))]
    types = [MemoryType.FACT, MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.EPISODIC, MemoryType.GOAL, MemoryType.PROJECT]
    batch = 5000
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        vecs = emb.embed(chunk)
        mems = []
        for j, (txt, vec) in enumerate(zip(chunk, vecs)):
            if i + j < len(CORPUS):
                _c, typ, src, slot, ents = CORPUS[i + j]
                m = Memory(content=txt, memory_type=MemoryType(typ), source=Source(src), slot=slot, entities=ents, importance=0.8)
            else:
                typ = rng.choice(types)
                proj = rng.choice(["Atlas", "Borealis", "Cirrus", "Delta"]) if typ == MemoryType.PROJECT else None
                m = Memory(content=txt, memory_type=typ, project=proj, scope="project" if proj else "global", entities=[],
                           source=rng.choice([Source.USER_EXPLICIT, Source.MODEL_INFERRED]), importance=round(rng.uniform(0.2, 0.7), 2),
                           confidence=round(rng.uniform(0.5, 0.95), 2), created_at=now - rng.uniform(0, 700 * 86400), updated_at=now - rng.uniform(0, 300 * 86400))
            m.embedding = vec.astype(np.float32).tobytes()
            mems.append(m)
        s.bulk_insert(mems)
    return s, time.perf_counter() - t0


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def measure(path, emb, floor=None):
    s0 = time.perf_counter()
    s = MemoryStore(path, emb)
    r = Retriever(s)
    first = r.retrieve("where do I live?", min_relevance=floor)
    cold_ms = (time.perf_counter() - s0) * 1000
    lat, cand, final, rr, prec, toks = [], [], [], [], [], []
    stages: dict[str, list[float]] = {}
    got = 0
    for q, subs in POSITIVE * 2:
        t0 = time.perf_counter()
        res = r.retrieve(q, min_relevance=floor)
        lat.append((time.perf_counter() - t0) * 1000)
        cand.append(res.candidates)
        final.append(len(res.hits))
        rank = judge(res.hits, subs)
        rr.append(1 / rank if rank else 0.0)
        got += bool(rank)
        prec.append(sum(1 for h in res.hits if any(x.lower() in h.memory.content.lower() for x in subs)) / len(res.hits) if res.hits else 0.0)
        toks.append(approx_tokens("\n".join(h.memory.content for h in res.hits)))
        for k, v in res.stages.items():
            stages.setdefault(k, []).append(v)
    glat = []
    gen_quiet = 0
    for q in GENERAL * 2:
        t0 = time.perf_counter()
        gen_quiet += not r.retrieve(q, min_relevance=floor).hits
        glat.append((time.perf_counter() - t0) * 1000)
    dis_quiet = sum(1 for q in DISTRACTORS if not r.retrieve(q, min_relevance=floor).hits)
    n_pos = len(POSITIVE) * 2
    out = {
        "cold_first_query_ms": round(cold_ms, 1), "warm_p50_ms": round(statistics.median(lat), 2), "warm_p95_ms": round(pct(lat, 0.95), 2),
        "general_question_p50_ms": round(statistics.median(glat), 3),
        "avg_candidates": round(statistics.mean(cand), 1), "avg_final_hits": round(statistics.mean(final), 2),
        "recall": round(got / n_pos, 3), "MRR": round(statistics.mean(rr), 3), "precision_of_returned": round(statistics.mean(prec), 3),
        "general_suppressed": round(gen_quiet / (len(GENERAL) * 2), 3), "distractor_suppressed": round(dis_quiet / len(DISTRACTORS), 3),
        "avg_injected_tokens": round(statistics.mean(toks), 1),
        "stage_ms": {k: round(statistics.mean(v), 2) for k, v in stages.items()},
        "db_MB": round(Path(path).stat().st_size / 1e6, 1), "ram_matrix_MB": round(len(s._ids) * emb.dim * 4 / 1e6, 1),
    }
    s.close()
    return out


def cmd_scale(args):
    emb = get_embedder(args.embedder, "BAAI/bge-small-en-v1.5") if args.embedder != "hash" else HashEmbedder()
    floor = args.floor
    sizes = [int(x) for x in args.sizes.split(",")]
    results = {"embedder": emb.name, "floor": floor if floor is not None else emb.floor, "sizes": {}}
    for n in sizes:
        path = Path(args.tmp) / f"mem_scale_{n}.db"
        for ext in ("", "-wal", "-shm"):
            Path(str(path) + ext).unlink(missing_ok=True)
        print(f"building N={n} ...", flush=True)
        s, build_s = build_scale(path, n, emb)
        s.close()
        m = measure(path, emb, floor)
        m["build_s"] = round(build_s, 1)
        results["sizes"][n] = m
        print(json.dumps({n: m}), flush=True)
        path.unlink(missing_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_".join(str(n) for n in sizes)
    (OUT / f"memory_scale_{args.embedder}_{tag}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality"); q.add_argument("--embedders", default="hash,fastembed"); q.set_defaults(fn=cmd_quality)
    sc = sub.add_parser("scenarios"); sc.add_argument("--embedder", default="hash"); sc.set_defaults(fn=cmd_scenarios)
    sl = sub.add_parser("scale"); sl.add_argument("sizes"); sl.add_argument("--embedder", default="hash")
    sl.add_argument("--floor", type=float, default=None); sl.add_argument("--tmp", default=str(ROOT / "data" / "tmp_scale")); sl.set_defaults(fn=cmd_scale)
    args = ap.parse_args()
    Path(getattr(args, "tmp", ROOT / "data" / "tmp_scale")).mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()