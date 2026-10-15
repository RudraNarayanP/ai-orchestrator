"""Claim rows belong to their own job -- the same claim text in two jobs is two rows.

Live (2026-10-15, the dpf question run twice on the same machine): claim ids hash the claim
text, and the claims table was keyed by id alone. The second run's "The European Commission
adopted the EU-US DPF adequacy decision on 10 July 2023" therefore stored no row at all -- the
row already existed, owned by the earlier job -- so its four evidence rows and its verdict
joined nothing inside that job, and the pages we had read looked unmatched. The second run
also overwrote the first run's status. Keyed by (job_id, id) both runs keep their own row.
"""

from __future__ import annotations

import sqlite3

from backend.models import Claim, ClaimStatus, Confidence, Job, ResearchMode
from backend.research.claims import stable_claim_id
from backend.storage.db import Store
from tests.conftest import base_settings

TEXT = "The European Commission adopted the EU-US DPF adequacy decision on 10 July 2023."


def _job(status: ClaimStatus, confidence: Confidence) -> Job:
    job = Job(question="Can an EU company transfer data to a DPF-certified US company without SCCs?", mode=ResearchMode.STANDARD)
    claim = Claim(id=stable_claim_id(TEXT), job_id=job.id, claim=TEXT, kind="fact",
                  provider_sources=["chatgpt"], status=status, confidence=confidence)
    job.claims = [claim]
    return job


def test_the_same_claim_in_two_jobs_keeps_two_rows(tmp_path):
    store = Store(base_settings(storage={"db_path": str(tmp_path / "x.db")}))
    first, second = _job(ClaimStatus.SUPPORTED, Confidence.HIGH), _job(ClaimStatus.REFUTED, Confidence.NONE)
    store.save_job(first)
    store.save_job(second)

    for job, expected in ((first, ClaimStatus.SUPPORTED), (second, ClaimStatus.REFUTED)):
        rows = store.job_snapshot(job.id)["claims"]
        assert len(rows) == 1, rows
        assert rows[0]["status"] == expected.value, (job.id, rows[0])
        assert rows[0]["id"] == first.claims[0].id == second.claims[0].id


def test_a_rerun_does_not_rewrite_the_earlier_audit_trail(tmp_path):
    store = Store(base_settings(storage={"db_path": str(tmp_path / "y.db")}))
    first = _job(ClaimStatus.SUPPORTED, Confidence.HIGH)
    store.save_job(first)
    store.save_job(_job(ClaimStatus.REFUTED, Confidence.NONE))
    row = store.job_snapshot(first.id)["claims"][0]
    assert row["status"] == "supported", row
    assert row["confidence"] == "high", row


def test_the_evidence_of_a_repeated_claim_still_joins_its_own_claim(tmp_path):
    """The dpf symptom: pages read, verdict recorded, and no claim row in that job to attach to."""
    from backend.models import Evidence, SourceCheckStatus, SourceTier

    store = Store(base_settings(storage={"db_path": str(tmp_path / "z.db")}))
    store.save_job(_job(ClaimStatus.SUPPORTED, Confidence.HIGH))
    job = _job(ClaimStatus.SUPPORTED, Confidence.HIGH)
    job.evidence = [
        Evidence(job_id=job.id, claim_id=job.claims[0].id, url="https://www.edpb.europa.eu/documents/other-guidance/x",
                 domain="www.edpb.europa.eu", tier=SourceTier.GOVERNMENT, check_status=SourceCheckStatus.CONFIRMED,
                 verbatim_excerpt="transfers may be based on the adequacy decision")
    ]
    store.save_job(job)
    snap = store.job_snapshot(job.id)
    attached = {e["claim_id"] for e in snap["evidence"] if e["check_status"] == "confirmed"}
    assert snap["claims"], "the repeated claim still has its own row in this job"
    assert attached & {c["id"] for c in snap["claims"]}, "the opened page joins a claim this job holds"


def test_an_existing_database_is_rekeyed_without_losing_rows(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """CREATE TABLE jobs (id TEXT PRIMARY KEY, question TEXT, mode TEXT, status TEXT, created_at REAL,
            updated_at REAL, finished_at REAL, level INTEGER, rounds_run INTEGER, max_rounds INTEGER,
            stop_reason TEXT, browser_sessions INTEGER, verifier_calls INTEGER, final_json TEXT,
            analysis_json TEXT, plan_json TEXT, error TEXT, answer_text TEXT, confidence TEXT);
           CREATE TABLE claims (
             id TEXT PRIMARY KEY, job_id TEXT, round INTEGER, claim TEXT, kind TEXT, topic TEXT,
             providers_json TEXT, web_json TEXT, supporting_json TEXT, contradicting_json TEXT,
             status TEXT, confidence TEXT, rationale TEXT);"""
    )
    claim_id = stable_claim_id(TEXT)
    conn.execute(
        "INSERT INTO claims (id, job_id, round, claim, kind, topic, providers_json, web_json, supporting_json,"
        " contradicting_json, status, confidence, rationale) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (claim_id, "job_old", 1, TEXT, "fact", "", "[]", "[]", "[]", "[]", "supported", "high", ""),
    )
    conn.commit()
    conn.close()

    store = Store(base_settings(storage={"db_path": str(path)}))
    key = sqlite3.connect(str(path)).execute("SELECT sql FROM sqlite_master WHERE name='claims'").fetchone()[0]
    assert "PRIMARY KEY (job_id, id)" in key, key

    old = _job(ClaimStatus.SUPPORTED, Confidence.HIGH)
    old.id = "job_old"
    old.claims[0].job_id = "job_old"
    store.save_job(old)
    row = store.job_snapshot("job_old")["claims"]
    assert [r["status"] for r in row] == ["supported"], row
    assert [r["claim"] for r in row] == [TEXT], row

    fresh = _job(ClaimStatus.REFUTED, Confidence.NONE)
    store.save_job(fresh)
    assert store.job_snapshot(fresh.id)["claims"][0]["status"] == "refuted"
    assert store.job_snapshot("job_old")["claims"][0]["status"] == "supported"
