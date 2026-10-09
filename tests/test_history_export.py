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
    RoundRecord,
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


# ----------------------------------------------------- `run.py ask` persistence


class _FakeEngine:
    def __init__(self, *args, **kwargs):
        self.stopped = False

    async def start(self):
        pass

    async def stop(self, keep_windows=True):
        self.stopped = True


class _FakeCatalog:
    def __init__(self, settings, engine):
        pass

    def all(self):
        return {}


def _patch_cli(monkeypatch, tmp_path, runner):
    """Point `run.py ask` at a throwaway database and a scripted runner: no browser, no network."""
    import run

    db = tmp_path / "cli.db"
    settings = base_settings(storage={"db_path": str(db)})
    monkeypatch.setattr(run, "load", lambda *a, **kw: settings)
    monkeypatch.setattr("backend.browser.factory.create_engine", lambda *a, **kw: _FakeEngine())
    monkeypatch.setattr("backend.providers.registry.ProviderCatalog", _FakeCatalog)
    monkeypatch.setattr("backend.providers.registry.endpoint_for", lambda *a, **kw: None)
    monkeypatch.setattr("backend.orchestrator.runner.ResearchRunner", runner)
    return str(db)


def test_cli_ask_stores_the_job_and_the_evidence_it_prints(monkeypatch, tmp_path, capsys):
    import asyncio
    import run

    class Runner:
        def __init__(self, settings, adapters, **kwargs):
            pass

        async def run(self, job):
            done = finished_job(job.question)
            job.final, job.claims, job.evidence = done.final, done.claims, done.evidence
            job.rounds = [RoundRecord(number=1), RoundRecord(number=2)]
            job.status = JobStatus.COMPLETED
            return job

    db = _patch_cli(monkeypatch, tmp_path, Runner)
    assert asyncio.run(run._one_shot("warfarin question", "STANDARD", None, None)) == 0

    from backend.storage.db import Store

    rows = Store(base_settings(storage={"db_path": db})).list_jobs(limit=5)
    assert [r["status"] for r in rows] == ["completed"]
    assert rows[0]["answer_text"].startswith("Not without medical advice")
    assert rows[0]["rounds_run"] == 2, "history must show the rounds that actually ran"
    assert db in capsys.readouterr().out, "the CLI must name the file it actually wrote"

    snapshot = Store(base_settings(storage={"db_path": db})).job_snapshot(rows[0]["id"])
    assert snapshot["claims"] and snapshot["evidence"], "the audit trail is stored too, not just the headline"


def test_cli_ask_leaves_a_row_when_the_run_dies_halfway(monkeypatch, tmp_path):
    """The job is stored before the browser work starts, so a crash mid-run is still in history."""
    import asyncio
    import pytest

    import run

    class Runner:
        def __init__(self, settings, adapters, **kwargs):
            pass

        async def run(self, job):
            raise RuntimeError("the browser connection went away")

    db = _patch_cli(monkeypatch, tmp_path, Runner)
    with pytest.raises(RuntimeError):
        asyncio.run(run._one_shot("a question that never finished", "STANDARD", None, None))

    from backend.storage.db import Store

    rows = Store(base_settings(storage={"db_path": db})).list_jobs(limit=5)
    assert len(rows) == 1 and rows[0]["status"] == "pending"


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


def test_an_incomplete_review_is_recorded_in_the_export():
    """`status: completed` alone makes a job whose review was cut off read as a finished one.

    The live truncated run (job_261009212243_87a51614) stored reviewer_status=COMPLETED with no
    verdicts; the corrected record has to say INCOMPLETE where a person reading the audit will see it.
    """
    md = job_markdown(
        {
            "question": "In which year was the Eiffel Tower completed?",
            "status": "completed",
            "final": {
                "answer": "Documented:\n- finished on 31 March 1889 (gov.example).",
                "reviewer_status": "INCOMPLETE",
                "synthesis_status": "FALLBACK",
                "fallback_reason": "curator review cut off mid-reply: 0 of 12 claims judged, 12 left unjudged",
            },
        }
    )
    assert "review: INCOMPLETE/FALLBACK" in md, md
    assert "## Review status" in md, md
    assert "cut off mid-reply" in md and "12 left unjudged" in md, md


def test_a_completed_review_records_its_status_and_adds_no_gap_section():
    md = job_markdown(
        {
            "question": "q",
            "status": "completed",
            "final": {"answer": "a", "reviewer_status": "COMPLETED", "synthesis_status": "CURATED"},
        }
    )
    assert "review: COMPLETED/CURATED" in md, md
    assert "## Review status" not in md, "a finished review is not reported as a gap"