"""Replay build_final_answer for a stored job (assembly only, no browser)."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.models import ProviderResponse, ProviderStatus, VerifierReport
from backend.verification.verifier import build_final_answer, truth_state

jid = sys.argv[1] if len(sys.argv) > 1 else "job_261010234454_b92a0462"
db = ROOT / "data" / "omnibrain.db"
c = sqlite3.connect(db)
c.row_factory = sqlite3.Row
j = c.execute("SELECT question, rounds_run, final_json FROM jobs WHERE id=?", (jid,)).fetchone()
if not j:
    raise SystemExit(f"no job {jid}")
old = json.loads(j["final_json"] or "{}")
rep = json.loads(
    c.execute("SELECT report_json FROM reports WHERE job_id=? ORDER BY round DESC LIMIT 1", (jid,)).fetchone()["report_json"]
)
report = VerifierReport.model_validate(rep)
seen: set[str] = set()
responses: list[ProviderResponse] = []
for row in c.execute("SELECT provider, status, round FROM responses WHERE job_id=?", (jid,)):
    if row["provider"] in seen:
        continue
    seen.add(row["provider"])
    responses.append(
        ProviderResponse(
            job_id=jid,
            provider=row["provider"],
            round=row["round"],
            role="primary",
            status=ProviderStatus(row["status"]),
            prompt="",
            answer_text="",
        )
    )
q = j["question"]
state = truth_state(report, q)
final = build_final_answer(report, responses, j["rounds_run"], q)
print("=== OLD (stored) ===")
print(old.get("truth_state"), old.get("answer", "")[:280])
print()
print("=== REPLAY ===")
print("truth_state:", state, "->", final.truth_state)
print("ANSWER:", final.answer)
print()
print("WHY:", (final.why or "")[:700])
print("CAVEATS:", final.caveats)
low = final.answer.lower()
print()
print("CHECKS: has_13=", "13" in final.answer, "nah=", "nah, that" in low,
      "refusal=", "couldn't find that documented" in low, "disagree=", "disagree" in low)
