"""History listing and job export (backlog items 13 and 14)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.api import app as api_app
from backend.export import inline, job_markdown
from backend.models import (
    Citation,
    Claim,
    ClaimStatus,
    Confidence,
    Disagreement,
    Evidence,
    FinalAnswer,
    Job,
    JobStatus,
    ResearchMode,
    SourceCheckStatus,
    SourceTier,
)
from tests.conftest import base_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    async def idle(self, job, emit):
        return None

    monkeypatch.setattr(api_app.JobManager, "_run", idle)
    app = api_app._make_app(base_settings(storage={"db_path": str(tmp_path / "h.db")}))
    with TestClient(app) as c:
        c.app_ = app
        yield c


def finished_job(question="Is it safe to take ibuprofen with warfarin?", **final_kw) -> Job:
    job = Job(question=question, mode=ResearchMode.STANDARD, max_rounds=2)
    claim = Claim(job_id=job.id, claim="Ibuprofen raises bleeding risk with warfarin.", kind="fact", provider_sources=["chatgpt", "gemini"], status=ClaimStatus.SUPPORTED)
    job.claims = [claim]
    job.evidence = [
        Evidence(job_id=job.id, claim_id=claim.id, url="https://nhs.example/warfarin", title="Warfarin and painkillers", domain="nhs.example",
                 tier=SourceTier.PRIMARY_OFFICIAL, check_status=SourceCheckStatus.CONFIRMED, verbatim_excerpt="avoid ibuprofen", published="2026-01-02")
    ]
    job.disagreements = [Disagreement(job_id=job.id, topic="dose", description="Sources differ on dose.", severity="low")]
    job.final = FinalAnswer(
        answer="Not without medical advice: ibuprofen raises bleeding risk with warfarin.",
        why="The NHS page says to avoid it.",
        confidence=Confidence.HIGH,
        confidence_label="High confidence",
        sources=[Citation(url="https://nhs.example/warfarin", title="Warfarin and painkillers", published="2026-01-02")],
        caveats=["Not medical advice."],
        rounds_run=2,
        **final_kw,
    )
    job.status = JobStatus.COMPLETED
    return job


# ----------------------------------------------------------------- history


def test_history_lists_newest_first_with_what_the_view_needs(client):
    store = client.app_.state.store
    first, second = finished_job("first question"), finished_job("second question")
    second.created_at = first.created_at + 10
    store.save_job(first)
    store.save_job(second)
    rows = client.get("/api/jobs").json()
    assert [r["question"] for r in rows][:2] == ["second question", "first question"]
    assert {"id", "question", "status", "created_at", "confidence", "answer_text", "conversation_id"} <= set(rows[0])


@pytest.mark.parametrize("limit, expected", [(1, 1), (0, 1), (-5, 1), (2, 2), (100000, 3)])
def test_history_limit_is_clamped(client, limit, expected):
    store = client.app_.state.store
    for n in range(3):
        j = finished_job(f"q{n}")
        j.created_at += n
        store.save_job(j)
    assert len(client.get(f"/api/jobs?limit={limit}").json()) == expected


def test_history_rejects_a_non_numeric_limit(client):
    assert client.get("/api/jobs?limit=abc").status_code == 422


# ------------------------------------------------------------------ export


def test_markdown_export_contains_the_answer_and_its_evidence(client):
    job = finished_job()
    client.app_.state.store.save_job(job)
    res = client.get(f"/api/jobs/{job.id}/export")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/markdown")
    assert f'filename="omnibrain-{job.id}.md"' in res.headers["content-disposition"]
    md = res.text
    for needle in [
        "# Is it safe to take ibuprofen with warfarin?",
        "## Answer", "ibuprofen raises bleeding risk", "**Confidence:** High confidence",
        "## Why", "## Sources", "[Warfarin and painkillers](https://nhs.example/warfarin)",
        "## Claim ledger", "| supported", "chatgpt, gemini",
        "## Evidence we opened (1)", "found on page", "avoid ibuprofen",
        "## Conflicts detected", "Sources differ on dose", "## Caveats",
    ]:
        assert needle in md, f"missing {needle!r} in:\n{md}"


def test_json_export_is_the_full_snapshot(client):
    job = finished_job()
    client.app_.state.store.save_job(job)
    res = client.get(f"/api/jobs/{job.id}/export?format=json")
    assert res.status_code == 200 and res.headers["content-type"].startswith("application/json")
    assert res.headers["content-disposition"].endswith('.json"')
    data = json.loads(res.text)
    assert data["id"] == job.id and data["final"]["answer"].startswith("Not without") and data["claims"] and data["evidence"]


def test_export_errors_are_clean(client):
    assert client.get("/api/jobs/nope/export").status_code == 404
    job = finished_job()
    client.app_.state.store.save_job(job)
    assert client.get(f"/api/jobs/{job.id}/export?format=pdf").status_code == 400


def test_exporting_a_job_that_is_still_running_is_refused(client):
    job = finished_job()
    job.final = None
    job.status = JobStatus.VERIFYING
    client.app_.state.store.save_job(job)
    client.app_.state.manager.jobs[job.id] = job
    assert client.get(f"/api/jobs/{job.id}/export").status_code == 409


def test_a_failed_job_still_exports_honestly(client):
    job = Job(question="q that failed", mode=ResearchMode.QUICK)
    job.status = JobStatus.FAILED
    job.error = "all providers failed"
    client.app_.state.store.save_job(job)
    md = client.get(f"/api/jobs/{job.id}/export").text
    assert "No answer was produced" in md and "# q that failed" in md


def test_hostile_text_cannot_inject_markup_into_the_export():
    snap = {
        "question": "# pwned\n\n<script>alert(1)</script>",
        "final": {
            "answer": "ok\n\n# Fake heading\n[click](javascript:alert(1))",
            "sources": [{"url": "javascript:alert(1)", "title": "evil](http://x) <b>"}, {"url": "https://ok.example/a b", "title": "t"}],
        },
        "evidence": [{"url": "https://ok.example/x", "title": "a\n## h", "check_status": "confirmed", "verbatim_excerpt": "x\n# y"}],
        "claims": [{"claim": "a | b\n| c", "status": "supported", "providers": ["p|q"]}],
    }
    md = job_markdown(snap)
    assert "<script>" not in md and "\n# Fake heading" not in md and "\n## h" not in md
    import re

    targets = re.findall(r"(?<!\\)\]\(([^)]*)\)", md)  # unescaped link targets, i.e. real links
    assert targets and all(t.startswith("https://") for t in targets), f"only validated http(s) URLs may be link targets: {targets}"
    assert "http://x" not in targets, "a title cannot close its own link and open another"
    assert md.count("\n# ") == 0 and md.startswith("# ")
    claim_row = next(line for line in md.splitlines() if line.startswith("| a"))
    assert claim_row.count("|") - claim_row.count("\\|") == 4, "pipes in a claim must not add table columns"


def test_inline_flattens_and_truncates():
    assert inline("a\n\n  b\t c") == "a b c"
    assert len(inline("x" * 1000, 50)) <= 51
    assert inline(None) == ""