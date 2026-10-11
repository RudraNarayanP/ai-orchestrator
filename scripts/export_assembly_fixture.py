"""Export a sanitized assembly-replay fixture from a finished job in data/omnibrain.db.

    .venv\\Scripts\\python.exe scripts/export_assembly_fixture.py job_261010234454_b92a0462 \\
        tests/fixtures/assembly/dpa_iss_age_replay.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: export_assembly_fixture.py <job_id> <out.json>", file=sys.stderr)
        return 2
    jid, out = sys.argv[1], Path(sys.argv[2])
    db = ROOT / "data" / "omnibrain.db"
    if not db.exists():
        print(f"no database at {db}", file=sys.stderr)
        return 1
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    j = c.execute("SELECT question, rounds_run FROM jobs WHERE id=?", (jid,)).fetchone()
    if not j:
        print(f"unknown job {jid}", file=sys.stderr)
        return 1
    rep = json.loads(
        c.execute("SELECT report_json FROM reports WHERE job_id=? ORDER BY round DESC LIMIT 1", (jid,)).fetchone()[
            "report_json"
        ]
    )
    rep.pop("raw_output", None)
    rep["job_id"] = "fixture_dpa_iss_age_replay"
    seen: set[str] = set()
    responses = []
    for row in c.execute("SELECT provider, status, round FROM responses WHERE job_id=?", (jid,)):
        if row["provider"] in seen:
            continue
        seen.add(row["provider"])
        responses.append({"provider": row["provider"], "status": row["status"], "round": row["round"]})
    payload = {
        "source_job_id": jid,
        "question": j["question"],
        "rounds_run": j["rounds_run"],
        "report": rep,
        "responses": responses,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
