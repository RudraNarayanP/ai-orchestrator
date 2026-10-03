r"""Repeatable hard-question evaluation.  One command:

    .venv\Scripts\python.exe scripts/eval_hard.py            # all questions
    .venv\Scripts\python.exe scripts/eval_hard.py --only uk-dpa-age,ua-marriage-age
    .venv\Scripts\python.exe scripts/eval_hard.py --all-providers --max-rounds 3

It starts its own server on a free port (so nothing else is disturbed), asks every question in
scripts/eval_questions.yaml through the real pipeline, and writes data/eval/eval_<timestamp>.json and .md.

For each question it records the final answer; whether a primary source was opened and cited; how refutation
went; any figure in the answer that no opened page contains (the failure we hunt: a claim that slipped through
unsupported); voice compliance (backend/research/style.py); and runtime.  By default the providers known to be
walled when logged out are switched off for speed (see SKIP_BY_DEFAULT); --all-providers keeps everything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):  # cp1252 consoles choke on the emojis in the house voice
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import yaml  # noqa: E402

from backend.research.style import voice_report  # noqa: E402

QUESTIONS = ROOT / "scripts" / "eval_questions.yaml"
OUT_DIR = ROOT / "data" / "eval"
# Walled or broken when logged out in the 2026-10-03 live runs; asking them only burns minutes.
SKIP_BY_DEFAULT = ["copilot", "meta_ai", "le_chat", "pi", "qwen", "deepseek", "google_ai"]
IDK_RE = re.compile(r"^\s*(i don'?t know|i couldn'?t verify|i can'?t verify)", re.I)

# --------------------------------------------------------------------------------------------- scoring


def _norm_num(token: str) -> str:
    return re.sub(r"[,\s]", "", token).lower().rstrip(".")


FIGURE_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s?%|\d+(?:\.\d+)?|(?:section|s\.|article|art\.)\s?\d+[a-z]?)(?![\w])",
    re.I,
)


def figures_in(text: str) -> list[str]:
    """Checkable figures in an answer: numbers, percentages, section/article numbers."""
    seen: dict[str, None] = {}
    for m in FIGURE_RE.finditer(text or ""):
        token = _norm_num(m.group(1))
        if token:
            seen[token] = None
    return list(seen)


def unsupported_figures(answer: str, corpus: str, question: str = "") -> list[str]:
    """Figures stated in the answer that appear neither in any opened page text nor in the question."""
    haystack = re.sub(r"[,\s]", "", (corpus + " " + question).lower())
    out = []
    for fig in figures_in(answer):
        bare = re.sub(r"^(section|s\.|article|art\.)", "", fig)
        if bare in {"1", "2"}:  # "1 of 2" style noise is not a claim
            continue
        if bare not in haystack and fig not in haystack:
            out.append(fig)
    return out


def judge(q: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """Outcome for one question: PASS / WARN / FAIL with the reasons."""
    answer = rec.get("answer") or ""
    kind = q.get("kind", "answer")
    expect = q.get("expect") or {}
    reasons: list[str] = []
    verdict = "PASS"
    idk = bool(IDK_RE.match(answer))
    for pat in expect.get("must_not_match", []):
        if re.search(pat, answer, re.I | re.S):
            reasons.append(f"asserts something it must not ({pat})")
            verdict = "FAIL"
    if kind == "unanswerable":
        if not idk:
            reasons.append("answered a question that has no public answer")
            verdict = "FAIL"
        elif not re.search(r"\w{4,}", answer.split(".", 1)[-1] if "." in answer else ""):
            reasons.append("don't-know without a reason")
            verdict = "WARN" if verdict == "PASS" else verdict
    else:
        misses = [p for p in expect.get("must_match", []) if not re.search(p, answer, re.I | re.S)]
        if idk:
            reasons.append("said it could not answer an answerable question")
            verdict = "WARN" if verdict == "PASS" else verdict
        elif misses:
            reasons.append("missing expected content: " + ", ".join(misses))
            verdict = "FAIL"
        if q.get("primary_domains") and not idk and not rec.get("primary_cited"):
            reasons.append("no primary source cited")
            verdict = "WARN" if verdict == "PASS" else verdict
    if rec.get("unsupported_figures") and not idk:
        reasons.append("figures with no opened page behind them: " + ", ".join(rec["unsupported_figures"]))
        verdict = "FAIL" if kind in {"false_premise", "myth"} or len(rec["unsupported_figures"]) > 1 else ("WARN" if verdict == "PASS" else verdict)
    if not rec.get("voice", {}).get("ok", True):
        reasons.append("voice: " + "; ".join(rec["voice"]["problems"]))
        verdict = "WARN" if verdict == "PASS" else verdict
    return {"verdict": verdict, "reasons": reasons}


def domain_matches(domain: str, wanted: list[str]) -> bool:
    d = (domain or "").lower().removeprefix("www.")
    return any(d == w or d.endswith("." + w) for w in wanted)


def analyse(q: dict[str, Any], snapshot: dict[str, Any], runtime_s: float) -> dict[str, Any]:
    """Turn one job snapshot into the evaluation record."""
    live = snapshot.get("live") or snapshot
    final = live.get("final") or {}
    answer = final.get("answer") or ""
    evidence = live.get("evidence") or []
    wanted = q.get("primary_domains") or []
    opened = [e for e in evidence if domain_matches(e.get("domain") or "", wanted)]
    cited_urls = [s.get("url") for s in final.get("sources") or []]
    cited_primary = [u for u in cited_urls if domain_matches(re.sub(r"^https?://", "", u or "").split("/")[0], wanted)]
    confirmed_text = " ".join(
        " ".join(str(e.get(k) or "") for k in ("snippet", "verbatim_excerpt", "title"))
        for e in evidence
        if e.get("check_status") in {"confirmed", "outdated"}
    )
    claim_text = " ".join(c.get("claim", "") for c in live.get("claims") or [])
    # only text from pages we opened counts as support; provider answers are what we are checking
    unsupported = unsupported_figures(answer, confirmed_text, q["question"])
    reports = live.get("reports") or []
    verdicts = (reports[-1].get("verdicts") if reports else []) or []
    refuting = [e for e in evidence if e.get("polarity") == "refute"]
    rec = {
        "id": q["id"],
        "category": q["category"],
        "kind": q["kind"],
        "question": q["question"],
        "answer": answer,
        "why": final.get("why"),
        "confidence": final.get("confidence_label"),
        "caveats": final.get("caveats"),
        "sources": cited_urls,
        "primary_domains": wanted,
        "primary_opened": [e.get("url") for e in opened if e.get("url")][:6],
        "primary_cited": cited_primary,
        "pages_opened": len([e for e in evidence if e.get("check_status") not in (None, "not_checked")]),
        "pages_confirmed": len([e for e in evidence if e.get("check_status") == "confirmed"]),
        "refutation": {
            "refuting_pages": len(refuting),
            "refuted_claims": len([v for v in verdicts if v.get("verdict") == "refuted"]),
            "contested_claims": len([v for v in verdicts if v.get("verdict") == "contested"]),
        },
        "providers_used": final.get("providers_used"),
        "providers_failed": final.get("providers_failed"),
        "rounds": final.get("rounds_run"),
        "verifier_model": (reports[-1].get("verifier_model") if reports else None),
        "unsupported_figures": unsupported,
        "claim_text_len": len(claim_text),
        "voice": voice_report(answer),
        "runtime_s": round(runtime_s, 1),
    }
    rec["judgement"] = judge(q, rec)
    return rec


def render_markdown(stamp: str, records: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        f"# OmniBrain hard-question evaluation {stamp}",
        "",
        f"Providers: {', '.join(meta.get('providers', []))}. Verifier: {meta.get('verifier')}. Mode: {meta.get('mode')}, max rounds {meta.get('max_rounds')}.",
        "",
        "| id | kind | outcome | primary opened / cited | unsupported figures | voice | runtime | answer |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        ans = (r["answer"] or "").replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {r['id']} | {r['kind']} | **{r['judgement']['verdict']}** | "
            f"{len(r['primary_opened'])} / {len(r['primary_cited'])} | {', '.join(r['unsupported_figures']) or '-'} | "
            f"{'ok' if r['voice']['ok'] else 'no'} | {int(r['runtime_s'])}s | {ans[:170]} |"
        )
    tally: dict[str, int] = {}
    for r in records:
        tally[r["judgement"]["verdict"]] = tally.get(r["judgement"]["verdict"], 0) + 1
    lines += ["", "Totals: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())), ""]
    lines.append("## Details")
    for r in records:
        lines += [
            "",
            f"### {r['id']} ({r['kind']}) -- {r['judgement']['verdict']}",
            f"Q: {r['question']}",
            "",
            f"A: {r['answer']}",
            "",
            f"- why: {r['why'] or '-'}",
            f"- confidence: {r['confidence']}; rounds {r['rounds']}; used {r['providers_used']}; failed {r['providers_failed']}",
            f"- primary sources opened: {r['primary_opened'] or 'none'}; cited: {r['primary_cited'] or 'none'}",
            f"- refutation: {r['refutation']}",
            f"- reasons: {'; '.join(r['judgement']['reasons']) or 'none'}",
            f"- caveats: {r['caveats']}",
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------- running


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(base: str, path: str, body: dict | None = None, timeout: int = 60) -> Any:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ask(base: str, question: str, mode: str, max_rounds: int, timeout_s: int) -> tuple[dict[str, Any], float]:
    started = time.time()
    job = call(base, "/api/jobs", {"question": question, "mode": mode, "max_rounds": max_rounds, "conversation_id": None})
    jid = job["job_id"]
    while time.time() - started < timeout_s:
        time.sleep(4)
        snap = call(base, f"/api/jobs/{jid}")
        live = snap.get("live") or snap
        if live.get("final") and (snap.get("status") in {"completed", "failed", "cancelled"} or live.get("status") in {"completed", "failed", "cancelled"}):
            return snap, time.time() - started
    return call(base, f"/api/jobs/{jid}"), time.time() - started


SECRET_FOR_CHILD = {"key": ""}


def trimmed_config(skip: list[str], dest: Path) -> list[str]:
    cfg_path = ROOT / "config" / "settings.yaml"
    if not cfg_path.exists():
        cfg_path = ROOT / "config" / "settings.example.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    for name in skip:
        if name in (raw.get("providers") or {}):
            raw["providers"][name]["enabled"] = False
    # The temp config never holds the API key: it is handed to the child through the environment, so a
    # killed run cannot leave a key on disk.
    for role in ("verifier", "analysis", "vision"):
        if isinstance(raw.get(role), dict) and raw[role].get("api_key"):
            SECRET_FOR_CHILD["key"] = SECRET_FOR_CHILD["key"] or str(raw[role]["api_key"])
            raw[role]["api_key"] = ""
    dest.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return [n for n, p in (raw.get("providers") or {}).items() if p.get("enabled")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated question ids")
    ap.add_argument("--category", default="", help="uk-law | ua-law | university | research")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="STANDARD")
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per question")
    ap.add_argument("--all-providers", action="store_true", help="do not switch off the providers known to be walled logged out")
    ap.add_argument("--questions", default=str(QUESTIONS))
    args = ap.parse_args()

    questions = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("no questions selected")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_file = OUT_DIR / f"_config_{stamp}.yaml"
    providers = trimmed_config([] if args.all_providers else SKIP_BY_DEFAULT, cfg_file)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ, OMNIBRAIN_CONFIG=str(cfg_file), PYTHONIOENCODING="utf-8")
    if SECRET_FOR_CHILD["key"] and not env.get("OPENROUTER_API_KEY"):
        env["OPENROUTER_API_KEY"] = SECRET_FOR_CHILD["key"]
    log = open(OUT_DIR / f"_server_{stamp}.log", "wb")
    server = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "serve", "--port", str(port)], cwd=ROOT, env=env, stdout=log, stderr=log)
    records: list[dict[str, Any]] = []
    try:
        for _ in range(60):
            try:
                call(base, "/api/doctor", timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        doctor = call(base, "/api/doctor", timeout=60)
        verifier = f"{doctor.get('verifier', {}).get('state')}: {doctor.get('verifier', {}).get('detail')}"
        print(f"server on {port}; providers {providers}; verifier {verifier}", flush=True)
        meta = {"providers": providers, "verifier": verifier, "mode": args.mode, "max_rounds": args.max_rounds}
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q['id']}: {q['question'][:90]}", flush=True)
            try:
                snap, secs = ask(base, q["question"], args.mode, args.max_rounds, args.timeout)
                rec = analyse(q, snap, secs)
            except Exception as exc:  # noqa: BLE001
                rec = {"id": q["id"], "category": q["category"], "kind": q["kind"], "question": q["question"], "answer": "", "why": None,
                       "confidence": None, "caveats": [], "sources": [], "primary_domains": [], "primary_opened": [], "primary_cited": [],
                       "pages_opened": 0, "pages_confirmed": 0, "refutation": {}, "providers_used": [], "providers_failed": [], "rounds": 0,
                       "verifier_model": None, "unsupported_figures": [], "claim_text_len": 0, "voice": {"ok": False, "problems": ["no answer"]},
                       "runtime_s": 0, "judgement": {"verdict": "FAIL", "reasons": [f"run error: {type(exc).__name__}: {exc}"]}}
            records.append(rec)
            print(f"    -> {rec['judgement']['verdict']} in {int(rec['runtime_s'])}s: {rec['answer'][:120]!r} {rec['judgement']['reasons']}", flush=True)
            (OUT_DIR / f"eval_{stamp}.json").write_text(json.dumps({"meta": meta, "records": records}, indent=1, ensure_ascii=False), encoding="utf-8")
        (OUT_DIR / f"eval_{stamp}.md").write_text(render_markdown(stamp, records, meta), encoding="utf-8")
        print(f"wrote {OUT_DIR / ('eval_' + stamp + '.md')}")
    finally:
        server.terminate()
        try:
            server.wait(10)
        except Exception:  # noqa: BLE001
            server.kill()
        log.close()
        subprocess.run([sys.executable, str(ROOT / "run.py"), "close"], cwd=ROOT, capture_output=True)
        cfg_file.unlink(missing_ok=True)
    return 0 if all(r["judgement"]["verdict"] != "FAIL" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())