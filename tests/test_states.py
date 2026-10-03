"""Explicit state, no invented explanations: the reviewer caveat appears only when the reviewer really failed."""

from __future__ import annotations

import json

from backend.evidence.sources import SourceTier
from backend.verification.llm import extract_json
from tests.conftest import base_settings
from tests.test_architecture import make_verifier, run
from tests.test_early_stop import LAW_Q, OPENED_PRIMARY, URL9, providers

REVIEWER_CAVEAT = "AI reviewer"


def _all_text(job):
    return " ".join(job.final.caveats + [job.final.why or ""])


async def test_a_reviewer_that_was_not_needed_is_not_reported_unavailable(net):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    job, _, _ = await run(providers("chatgpt"), {"chatgpt": OPENED_PRIMARY}, "What did Acme release?" if False else LAW_Q)
    f = job.final
    assert (f.research_status, f.reviewer_status, f.synthesis_status, f.fallback_reason) in {
        ("COMPLETED", "NOT_RUN", "DETERMINISTIC", ""),
        ("COMPLETED", "COMPLETED", "CURATED", ""),
    }
    assert REVIEWER_CAVEAT not in _all_text(job), "no reviewer failed, so no reviewer excuse"


async def test_a_429_is_reported_as_unavailable_with_its_reason(net, fake_openai, monkeypatch):
    from backend.verification import llm

    async def no_wait(seconds):
        return None

    monkeypatch.setattr(llm, "_sleep", no_wait)
    fake_openai.script({"status": 429, "body": "rate-limited upstream"})
    hedge = {"answer": "I couldn't verify that. I'm not sure.", "citations": []}
    job, _, _ = await run(providers("chatgpt", "gemini", "qwen"), {"chatgpt": hedge, "gemini": hedge, "qwen": hedge}, LAW_Q, verifier=make_verifier(fake_openai))
    reports = [r for r in job.reports if r.reviewer_status != "NOT_RUN"]
    assert reports and all(r.reviewer_status == "UNAVAILABLE" and r.synthesis_status == "FALLBACK" for r in reports)
    assert "429" in job.final.fallback_reason
    assert REVIEWER_CAVEAT in _all_text(job), "a real failure is explained"


async def test_unusable_reviewer_output_is_invalid_output_not_unavailable(net, fake_openai):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    fake_openai.script(lambda body: "I think the answer is probably 13, but here is no JSON at all.")
    hedge = {"answer": "I couldn't verify that. I'm not sure.", "citations": []}
    job, _, _ = await run(providers("chatgpt", "gemini", "qwen"), {"chatgpt": hedge, "gemini": hedge, "qwen": hedge}, LAW_Q, verifier=make_verifier(fake_openai))
    reports = [r for r in job.reports if r.reviewer_status != "NOT_RUN"]
    assert reports and all(r.reviewer_status == "INVALID_OUTPUT" and r.synthesis_status == "FALLBACK" and r.fallback_reason for r in reports)
    assert job.final.reviewer_status == "INVALID_OUTPUT"
    assert "wasn't available" not in _all_text(job), "it answered; it was not unavailable"


def test_a_reply_cut_off_by_max_tokens_is_repaired_to_the_last_complete_value():
    cut = '{"answer": "It is 13.", "confidence": "high", "needs_more_research": false, "verdicts": [{"claim_id": "a", "verdict": "supported", "reasoning": "ok"}, {"claim_id": "b", "verdict": "suppor'
    got = extract_json(cut)
    assert got and got["answer"] == "It is 13." and got["confidence"] == "high"
    assert [v["claim_id"] for v in got["verdicts"]] == ["a"], "only complete verdicts survive"
    assert extract_json("no json here") is None


def test_the_schema_puts_the_conclusion_first_so_a_cut_off_tail_loses_detail_not_the_answer():
    from backend.verification.verifier import VERIFIER_SCHEMA

    assert VERIFIER_SCHEMA.index('"answer"') < VERIFIER_SCHEMA.index('"verdicts"')
    assert json.loads('{"a": 1}')