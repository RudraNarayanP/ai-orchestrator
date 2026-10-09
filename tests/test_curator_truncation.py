"""A curator reply that never finished is an incomplete review, never a completed one.

Live (eiffel run, round 3, job_261009212243_87a51614): the model's JSON was cut off at
`"reasoning":`, the rescue kept the prose and dropped all 18 verdicts, and the job was stored
as `reviewer_status=COMPLETED`, `verdicts=[]`. A later rule then rewrote the answer to "None
of the pages I opened confirms an answer" -- in a run holding five confirmed evidence rows.
"""

from __future__ import annotations

import json
import re

from backend.models import (
    Claim,
    ClaimStatus,
    ClaimVerdict,
    Confidence,
    Job,
    ResearchMode,
    VerifierReport,
)
from backend.evidence.sources import tier_for  # noqa: F401
from backend.models import ClaimVerdict as _CV, Evidence, SourceTier
from backend.models import SourceCheckStatus
from backend.verification.llm import json_parses
from backend.verification.verifier import Verifier
from tests.conftest import base_settings
from tests.test_llm_paths import claim_and_evidence, endpoint_for

# The exact shape that got through: valid prefix, prose that reads fine, verdict list cut
# off mid-object.
TRUNCATED = (
    '{\n  "answer": "The Bolt costs $549.",\n  "confidence": "high",\n  "verdicts": [\n'
    '    {\n      "claim_id": "clm_whatever",\n      "claim": "The Acme Bolt costs $549.",\n'
    '      "verdict": "supported",\n      "confidence": "high",\n      "reasoning":'
)


def reply_for_named_claims(body: dict) -> str:
    """Judge the claims in this prompt -- unless the prompt is the big one, which gets cut off."""
    user = body["messages"][-1]["content"]
    ids = re.findall(r'"claim_id": "((?:clm|claim)_[0-9a-z_]+)"', user)
    if len(ids) > 6:  # only the whole-set prompt gets cut off; recovery batches are small enough to finish
        keep = ids[:3]
        head = ",\n".join(
            f'    {{"claim_id": "{cid}", "claim": "x", "verdict": "supported", "confidence": "high", "reasoning": "two pages"}}'
            for cid in keep
        )
        return '{\n  "answer": "The Bolt costs $549.",\n  "confidence": "high",\n  "verdicts": [\n' + head + ',\n    {\n      "claim_id": "x'
    rows = [
        {"claim_id": cid, "claim": "x", "verdict": "supported", "confidence": "high", "reasoning": "two pages"}
        for cid in ids
    ]
    return json.dumps({"answer": "The Bolt costs $549.", "confidence": "high", "verdicts": rows, "needs_more_research": False})


async def verify(server, replies, claims, evidence, question="How much is the Bolt?") -> VerifierReport:
    server.script(*replies)
    verifier = Verifier(endpoint_for(server), min_independent_sources=2)
    return await verifier.verify(
        job_id="j", question=question, round_no=1, claims=claims, evidence=evidence, responses=[], disagreements=[]
    )


# ---------------------------------------------------------------- detecting the cut-off


def test_a_rescued_tail_is_not_a_parseable_reply():
    assert not json_parses(TRUNCATED)
    assert json_parses('{"verdicts": [], "answer": "x"}')
    assert json_parses('```json\n{"verdicts": []}\n```')
    assert not json_parses("no json here")


async def test_a_cut_off_reply_is_never_recorded_as_a_completed_review(fake_openai):
    claim, evidence = claim_and_evidence(confirmed=5)
    report = await verify(fake_openai, [TRUNCATED], [claim], evidence)

    assert report.reviewer_status == "INCOMPLETE", report.reviewer_status
    assert report.synthesis_status == "FALLBACK"
    assert "cut off" in report.fallback_reason, report.fallback_reason
    assert report.raw_output.startswith("{") and '"reasoning":' in report.raw_output, "the broken reply is kept for diagnostics"
    assert claim.id in {v.claim_id for v in report.verdicts}, "the claim is still decided -- by the ledger"
    assert report.verdicts[0].verdict == ClaimStatus.SUPPORTED, report.verdicts[0].verdict
    assert report.confidence in {Confidence.HIGH, Confidence.MODERATE}, report.confidence
    assert not report.answer.lower().startswith(("i don't know", "i couldn't", "couldn't verify")), report.answer
    assert any("cut off" in c for c in report.caveats), report.caveats


async def test_a_valid_reply_with_no_verdicts_at_all_is_not_completed_either(fake_openai):
    claim, evidence = claim_and_evidence(confirmed=2)
    empty = json.dumps({"answer": "The Bolt costs $549.", "confidence": "high", "verdicts": []})
    report = await verify(fake_openai, [empty], [claim], evidence)

    assert report.reviewer_status == "INCOMPLETE"
    assert "no verdicts at all" in report.fallback_reason, report.fallback_reason
    assert report.verdicts and report.verdicts[0].claim_id == claim.id


async def test_bounded_batch_recovery_finishes_the_review(fake_openai):
    claims = [
        Claim(job_id="j", claim=f"The Acme Bolt costs ${549 + i}.", kind="statistic", provider_sources=["chatgpt"])
        for i in range(8)
    ]
    report = await verify(fake_openai, [reply_for_named_claims], claims, [])
    judged = {v.claim_id for v in report.verdicts}

    assert report.reviewer_status == "COMPLETED", (report.reviewer_status, report.fallback_reason)
    assert len(judged) == 8, sorted(judged)
    assert all(cid in judged for cid in [c.id for c in claims]), "every claim shown ends up judged"
    assert not any("unjudged" in u for u in report.unresolved), report.unresolved


async def test_recovery_is_bounded_and_reports_what_stays_unjudged(fake_openai):
    """When even the batches come back cut off, the pass is incomplete -- it does not loop."""
    claims = [
        Claim(job_id="j", claim=f"The Acme Bolt costs ${549 + i}.", kind="statistic", provider_sources=["chatgpt"])
        for i in range(8)
    ]
    report = await verify(fake_openai, [TRUNCATED], claims, [])

    assert report.reviewer_status == "INCOMPLETE"
    calls = len(fake_openai.requests)
    assert calls <= 1 + 1 + 2 + 2, f"recovery must be bounded, took {calls} calls"
    assert "left unjudged" in report.fallback_reason, report.fallback_reason


# ---------------------------------------------------------------- no false statements


async def test_confirmed_evidence_is_never_described_as_no_evidence(fake_openai):
    """The reported failure: confirmed rows existed, and the answer claimed none of the
    opened pages confirms anything. The wording must follow what the ledger holds.
    Here the pages confirm points that settle none of the judged claims."""
    claim, _ = claim_and_evidence(confirmed=0)
    evidence = [
        Evidence(
            job_id="j", claim_id=None, url=f"https://site{i}.example/other", domain=f"site{i}.example",
            tier=SourceTier.PRIMARY_OFFICIAL, check_status=SourceCheckStatus.CONFIRMED,
            verbatim_excerpt="something else entirely", published="2026-06-01",
        )
        for i in range(5)
    ]
    dodgy = json.dumps(
        {
            "answer": "The Bolt costs $549.",
            "confidence": "high",
            "verdicts": [
                {"claim_id": claim.id, "claim": claim.claim, "verdict": "unverified", "confidence": "low", "reasoning": "not sure"}
            ],
        }
    )
    report = await verify(fake_openai, [dodgy], [claim], evidence)

    assert report.confidence in {Confidence.LOW, Confidence.NONE}, report.confidence
    assert "None of the pages I opened confirms an answer" not in report.answer, report.answer
    assert "other points" in report.answer, report.answer
    assert not any("Not confirmed by any page we opened" in c for c in report.caveats), report.caveats


async def test_with_no_usable_evidence_the_plain_refusal_remains(fake_openai):
    claim, _ = claim_and_evidence(confirmed=0)
    dodgy = json.dumps(
        {
            "answer": "The Bolt costs $549.",
            "confidence": "high",
            "verdicts": [
                {"claim_id": claim.id, "claim": claim.claim, "verdict": "unverified", "confidence": "low", "reasoning": "nothing on the pages"}
            ],
        }
    )
    report = await verify(fake_openai, [dodgy], [claim], [])

    assert "None of the pages I opened confirms an answer" in report.answer, report.answer
    assert any("Not confirmed by any page we opened" in c for c in report.caveats), report.caveats


# ------------------------------------------------- an incomplete pass cannot erase one


def test_an_incomplete_final_pass_carries_the_earlier_verdicts_forward():
    from backend.orchestrator.runner import ResearchRunner

    claim = Claim(id="clm_a", job_id="j", claim="The Acme Bolt costs $549.", kind="statistic", provider_sources=["chatgpt"])
    job = Job(question="How much is the Bolt?", mode=ResearchMode.STANDARD)
    job.claims = [claim]
    done = VerifierReport(
        job_id=job.id, round=1, verdicts=[
            ClaimVerdict(claim_id=claim.id, claim=claim.claim, verdict=ClaimStatus.SUPPORTED, confidence=Confidence.HIGH, reasoning="two pages")
        ]
    )
    done.reviewer_status = "COMPLETED"
    cut_off = VerifierReport(job_id=job.id, round=2, verdicts=[], answer="")
    cut_off.reviewer_status, cut_off.synthesis_status, cut_off.fallback_reason = "INCOMPLETE", "FALLBACK", "curator review cut off mid-reply"
    job.reports = [done, cut_off]

    runner = ResearchRunner(base_settings(), {}, engine=None)
    finished = runner._complete(job, cut_off, [], rounds=2, stop="the last review was cut off")

    assert claim.id in {v.claim_id for v in cut_off.verdicts}, "the earlier verdict is carried forward"
    assert claim.status == ClaimStatus.SUPPORTED, "and it reaches the claim ledger"
    assert finished.final is not None


async def test_the_reader_is_told_the_review_never_finished(fake_openai):
    """INCOMPLETE cannot live only in the database row.

    The internal reason ("curator review cut off mid-reply: 0 of 12 claims judged") is vocabulary
    the voice rules strip from prose, and scrubbing it in silence would hand back an answer that
    looks fully reviewed. The gap has to arrive as a plain sentence.
    """
    from backend.verification.verifier import build_final_answer

    claim, evidence = claim_and_evidence(confirmed=5)
    report = await verify(fake_openai, [TRUNCATED], [claim], evidence)
    final = build_final_answer(report, [], 3, "How much is the Bolt?")

    assert report.reviewer_status == "INCOMPLETE"
    assert final.caveats, "the unfinished review reaches the user, it is not scrubbed as bookkeeping"
    caveat = final.caveats[0].lower()
    assert "reviewer" in caveat and ("cut off" in caveat or "incomplete" in caveat), final.caveats
    for internal in ("ledger", "verdict", "curator", "claim id", "round 3", "0 of 12"):
        assert internal not in caveat, final.caveats
