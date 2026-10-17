"""Benchmark the free OpenRouter pool for the curator (adversarial reviewer) seat.

Runs the REAL curator prompt -- rebuilt from stored jobs through
``Verifier._material``, so each model sees exactly what a live review sees --
against every ``:free`` model, one at a time, and measures what this product
actually needs from that seat:

  latency          wall-clock seconds for the review call
  parses           the reply is usable JSON (``extract_json``)
  clean            it parsed WITHOUT a rescued tail (``LLMReply.repaired``) --
                   a repaired reply can never end a run COMPLETED
  coverage         verdicts returned / claims the prompt showed
  answer           the synthesis text came back non-empty

Free models are rate limited, so calls are sequential. Cost: roughly
n_models x n_cases x (latency + gap) -- the script prints progress per call so a
background run can be read from its log.

    .venv/Scripts/python.exe scripts/curator_bench.py                 # all jobs with >=6 claims
    .venv/Scripts/python.exe scripts/curator_bench.py --only job_261016150809_fa891ed9
    .venv/Scripts/python.exe scripts/curator_bench.py --models a:free,b:free
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import urllib.request  # noqa: E402

from backend.models import Claim, Disagreement, Evidence, ProviderResponse  # noqa: E402
from backend.settings import load_settings  # noqa: E402
from backend.storage.db import Store  # noqa: E402
from backend.verification.llm import Endpoint, LLMClient, _COOLING  # noqa: E402
from backend.verification.verifier import Verifier  # noqa: E402

# A moderation classifier cannot review claims; everything else on the free tier is a candidate.
EXCLUDE = {"nvidia/nemotron-3.5-content-safety:free"}


def free_models(base_url: str, key: str) -> list[str]:
    req = urllib.request.Request(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310 -- configured base_url, https
        data = json.load(response)
    return sorted(m["id"] for m in data.get("data", []) if str(m.get("id", "")).endswith(":free") and m["id"] not in EXCLUDE)


def material_for(store: Store, job_id: str) -> tuple[list[dict[str, str]], list[str]]:
    """The exact curator messages for a stored job, plus the claim ids it showed."""
    snap = store.job_snapshot(job_id)
    if not snap:
        raise SystemExit(f"no such job: {job_id}")
    claims = [Claim.model_validate(c) for c in snap["claims"]]
    evidence = [Evidence.model_validate(e) for e in snap["evidence"]]
    responses = [ProviderResponse.model_validate(r) for r in snap["responses"]]
    disagreements = [Disagreement.model_validate(d) for d in snap["disagreements"]]
    verifier = Verifier(Endpoint(provider="disabled", model="", base_url=""))
    shown = [c.id for c in verifier.focus(claims, evidence)[0]]
    messages = [
        {"role": "system", "content": verifier._system()},
        {"role": "user", "content": verifier._material(snap["question"], claims, evidence, responses, disagreements, snap.get("rounds_run") or 1)},
    ]
    return messages, shown


def candidate_jobs(store: Store, min_claims: int) -> list[dict[str, Any]]:
    rows = store.list_jobs(limit=200)
    out = []
    for row in rows:
        job_id = row.get("id") or row.get("job_id")
        snap = store.job_snapshot(job_id)
        if not snap or len(snap["claims"]) < min_claims:
            continue
        out.append({"job_id": job_id, "question": snap["question"], "claims": len(snap["claims"])})
    return out


async def run_case(endpoint_cfg: Any, model: str, messages: list[dict[str, str]], temperature: float, timeout_s: int) -> dict[str, Any]:
    ep = Endpoint(
        provider="openrouter",
        model=model,
        base_url=endpoint_cfg.base_url,
        api_key=endpoint_cfg.api_key,
        temperature=temperature,
        max_tokens=int(endpoint_cfg.max_tokens),
        timeout_s=timeout_s,
        fallback_models=(),
        headers=dict(getattr(endpoint_cfg, "headers", None) or {}) or None,
        reasoning=str(getattr(endpoint_cfg, "reasoning", "off") or "off"),
    )
    client = LLMClient(ep)
    started = time.monotonic()
    try:
        parsed, reply = await asyncio.wait_for(client.complete_json(messages, temperature=temperature), timeout=timeout_s)
    except asyncio.TimeoutError:
        return {"model": model, "seconds": round(time.monotonic() - started, 1), "error": f"bench timeout {timeout_s}s"}
    except Exception as exc:  # noqa: BLE001
        return {"model": model, "seconds": round(time.monotonic() - started, 1), "error": f"{type(exc).__name__}: {exc}"}
    verdicts = (parsed or {}).get("verdicts") or []
    return {
        "model": model,
        "seconds": round(reply.latency_s or (time.monotonic() - started), 1),
        "ok": reply.ok,
        "status": reply.status,
        "error": reply.error,
        "truncated": reply.truncated,
        "repaired": reply.repaired,
        "parses": parsed is not None,
        "verdicts": len(verdicts),
        "answered": bool(str((parsed or {}).get("answer") or "").strip()),
        "chars": len(reply.text or ""),
        "usage": reply.usage,
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=[], help="job id (repeatable); default: auto-pick")
    ap.add_argument("--models", default="", help="comma-separated model ids; default: every :free model")
    ap.add_argument("--min-claims", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=3, help="how many stored jobs to replay")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--gap", type=float, default=3.0, help="seconds between calls (free tier is rate limited)")
    args = ap.parse_args()

    settings = load_settings()
    store = Store(settings)
    cfg = settings.verifier
    models = [m.strip() for m in args.models.split(",") if m.strip()] or free_models(cfg.base_url, cfg.api_key)
    jobs = args.only or [c["job_id"] for c in candidate_jobs(store, args.min_claims)][: args.jobs]
    if not jobs:
        raise SystemExit("no stored job has >=%d claims" % args.min_claims)

    cases: list[tuple[str, list[dict[str, str]], list[str]]] = []
    for job_id in jobs:
        messages, shown = material_for(store, job_id)
        print(f"case {job_id}: {len(shown)} claims shown, {sum(len(m['content']) for m in messages)} chars of prompt", flush=True)
        cases.append((job_id, messages, shown))

    print(f"\n{len(models)} models x {len(cases)} cases, {args.timeout}s cap each\n")
    results: list[dict[str, Any]] = []
    for model in models:
        # the cooldown map is a runtime circuit breaker, not a benchmark result
        for key in [k for k in _COOLING if k[1] == model]:
            _COOLING.pop(key, None)
        row = {"model": model, "cases": []}
        for job_id, messages, shown in cases:
            res = await run_case(cfg, model, messages, cfg.temperature, args.timeout)
            res["job_id"] = job_id
            res["shown"] = len(shown)
            res["coverage"] = round(res.get("verdicts", 0) / len(shown), 2) if shown else 0
            row["cases"].append(res)
            flag = "ok " if res.get("parses") else ("ERR" if res.get("error") else "BAD")
            print(
                f"  {model:<48} {job_id[-8:]} {res['seconds']:6.1f}s {flag} "
                f"verdicts {res.get('verdicts', 0)}/{len(shown)} clean={not res.get('repaired') and not res.get('truncated')} "
                f"answer={res.get('answered')} {(res.get('error') or '')[:60]}",
                flush=True,
            )
            await asyncio.sleep(args.gap)
        worst = max((c["seconds"] for c in row["cases"]), default=0)
        ok_all = all(c.get("parses") for c in row["cases"])
        clean = all(c.get("parses") and not c.get("repaired") and not c.get("truncated") for c in row["cases"])
        cov = min((c["coverage"] for c in row["cases"]), default=0)
        row.update(worst_s=worst, parses_all=ok_all, clean_all=clean, min_coverage=cov)
        results.append(row)
        print(f"= {model:<48} worst {worst:6.1f}s parses={ok_all} clean={clean} coverage>={cov}\n")

    out = Path("data/eval") / f"curator_bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"models": results, "timeout": args.timeout}, indent=1, ensure_ascii=False), encoding="utf-8")

    print("=" * 96)
    print(f"{'model':<48} {'worst_s':>8} {'parses':>7} {'clean':>6} {'coverage':>9}  answer")
    for row in sorted(results, key=lambda r: (not r["clean_all"], r["worst_s"])):
        answered = all(c.get("answered") for c in row["cases"])
        print(f"{row['model']:<48} {row['worst_s']:>8.1f} {str(row['parses_all']):>7} {str(row['clean_all']):>6} {row['min_coverage']:>9.2f}  {answered}")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
