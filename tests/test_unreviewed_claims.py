"""A claim nobody checked is not a claim the sources fail to support.

Five states, and the two "we don't know" ones are not interchangeable:

  SUPPORTED              evidence and adjudication support the claim
  REFUTED                evidence supports the contrary
  INSUFFICIENT_EVIDENCE  we investigated, and the pages do not establish it
  NOT_REVIEWED           nothing adjudicated it -- it was never put to the curator
  (INCOMPLETE            the review itself failed or stopped: reviewer_status, not a claim status)

`Verifier.focus` caps one curator pass at MAX_CLAIMS, so a 25-claim run can end with a
"completed" review that never mentioned most of the claims. Live (job_261009210109_ccd924ec) the
same shape made a two-part question answer "Couldn't verify that one." over a date we held.
Rendering an omitted claim as "Not documented" reports our own gap as a fact about the evidence.
"""

from __future__ import annotations

import json
import re

from backend.export import job_markdown
from backend.models import (
    Claim,
    ClaimStatus,
    ClaimVerdict,
    Confidence,
    Evidence,
    SourceCheckStatus,
    SourceTier,
    VerifierReport,
)
from backend.research.claims import adjudicated_ids, annotate_unadjudicated, apply_verdicts
from backend.research.lint import evidence_report
from backend.verification.verifier import Verifier, build_final_answer
from tests.test_llm_paths import endpoint_for

TWO_PART = "When did the Act receive Royal Assent, and when did it come into force?"
ASSSENT = "The Act received Royal Assent on 23 May 2018."
FORCE = "Most of the Act's provisions came into force on 25 May 2018."
GOV = "https://www.gov.uk/government/collections/data-protection-act-2018"


def claim(text: str, cid: str, providers: tuple[str, ...] = ("chatgpt",)) -> Claim:
    return Claim(id=cid, job_id="j", claim=text, kind="date", provider_sources=list(providers))


def page(cid: str, n: int = 0) -> Evidence:
    return Evidence(
        job_id="j", claim_id=cid, url=f"{GOV}/{cid}/{n}", domain="gov.uk",
        tier=SourceTier.PRIMARY_OFFICIAL, check_status=SourceCheckStatus.CONFIRMED,
        verbatim_excerpt="the collection page states it",
    )


def judged(cid: str, text: str, value: ClaimStatus, evidence: list[str] = ()) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=cid, claim=text, verdict=value,
        confidence=Confidence.HIGH if value == ClaimStatus.SUPPORTED else Confidence.LOW,
        reasoning="two pages say it" if evidence else "the pages do not say it",
        strong_evidence=list(evidence),
    )


def answers_whatever_it_is_shown(body: dict) -> str:
    """A curator that adjudicates every claim in its prompt and never runs out of tokens."""
    ids = re.findall(r'"claim_id": "((?:clm|claim)_[0-9a-z_]+)"', body["messages"][-1]["content"])
    return json.dumps(
        {
            "answer": "The Act received Royal Assent on 23 May 2018.",
            "confidence": "moderate",
            "verdicts": [
                {"claim_id": i, "claim": "x", "verdict": "insufficient_evidence", "confidence": "low", "reasoning": "the page does not say"}
                for i in ids
            ],
        }
    )


async def run_curator(server, replies, claims, evidence, question=TWO_PART) -> VerifierReport:
    server.script(*replies)
    verifier = Verifier(endpoint_for(server), min_independent_sources=2)
    return await verifier.verify(
        job_id="j", question=question, round_no=1, claims=claims, evidence=evidence,
        responses=[], disagreements=[],
    )


# ------------------------------------------------------- more claims than one pass holds


async def test_claims_beyond_the_pass_are_put_to_the_curator_not_dropped(fake_openai):
    """25 claims, a 12-claim pass: the ones that answer the question and have pages attached are
    still reviewable, so they go in further batches instead of vanishing from the review."""
    over = Verifier.MAX_CLAIMS + 13
    claims = [claim(f"The Act received Royal Assent on 23 May 2018, wording {i}.", f"clm_{i:02d}") for i in range(over)]
    evidence = [page(c.id) for c in claims]  # every claim has a page of ours against it
    report = await run_curator(fake_openai, [answers_whatever_it_is_shown], claims, evidence)
    by_id = {v.claim_id: v for v in report.verdicts}

    assert over > Verifier.MAX_CLAIMS, "the fixture must exceed one curator pass"
    assert len(by_id) == over, "every claim ends the review with a state, not with an absent row"
    reviewed = adjudicated_ids(report.verdicts)
    assert len(reviewed) > Verifier.MAX_CLAIMS, (
        f"only the capped pass was adjudicated: {len(reviewed)} of {over}"
    )
    assert all(by_id[c.id].verdict is not ClaimStatus.NOT_REVIEWED for c in claims if c.id in reviewed)
    assert len(fake_openai.requests) <= 1 + Verifier.RECOVERY_CALLS + 1, "the extra review is bounded"


async def test_a_claim_with_no_page_against_it_is_never_called_insufficient(fake_openai):
    """Nothing was attached, so nothing was investigated: NOT_REVIEWED, never "the pages do not
    establish it" -- that sentence is about the evidence and the evidence was never consulted."""
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b")]
    report = await run_curator(
        fake_openai,
        [json.dumps({
            "answer": "The Act received Royal Assent on 23 May 2018.", "confidence": "moderate",
            "verdicts": [{"claim_id": "clm_a", "claim": ASSSENT, "verdict": "supported", "confidence": "high",
                          "reasoning": "gov.uk states it", "strong_evidence": [GOV]}],
        })],
        claims,
        [page("clm_a")],  # nothing at all attached to clm_b
    )
    by_id = {v.claim_id: v.verdict for v in report.verdicts}

    assert by_id["clm_a"] is ClaimStatus.SUPPORTED
    assert by_id["clm_b"] is ClaimStatus.NOT_REVIEWED, by_id
    assert report.verdicts[-1].confidence == Confidence.NONE


async def test_the_answer_says_not_checked_for_a_part_nobody_reviewed(fake_openai):
    """The exact live shape: one adjudicated part, one part the review never reached."""
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b")]
    report = await run_curator(
        fake_openai,
        [json.dumps({
            "answer": "The Act received Royal Assent on 23 May 2018.", "confidence": "low",
            "verdicts": [{"claim_id": "clm_a", "claim": ASSSENT, "verdict": "supported", "confidence": "moderate",
                          "reasoning": "gov.uk states it", "strong_evidence": [GOV]}],
        })],
        claims,
        [page("clm_a")],
    )
    final = build_final_answer(report, [], 1, TWO_PART)

    assert "23 May 2018" in final.answer and "gov.uk" in final.answer, final.answer
    assert "Not checked:" in final.answer and "come into force" in final.answer, final.answer
    assert "Not documented:" not in final.answer, "an omitted claim must not read as an evidentiary finding"
    assert "I didn't get that far." in final.answer


async def test_an_adjudicated_but_unproven_part_is_still_reported_as_not_documented(fake_openai):
    """The other side of the line: when the curator did check the pages and they did not say it,
    "not documented" is a true statement about the evidence and must stay."""
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b")]
    evidence = [page("clm_a"), page("clm_b")]
    report = await run_curator(
        fake_openai,
        [json.dumps({
            "answer": "The Act received Royal Assent on 23 May 2018.", "confidence": "low",
            "verdicts": [
                {"claim_id": "clm_a", "claim": ASSSENT, "verdict": "supported", "confidence": "moderate",
                 "reasoning": "gov.uk states it", "strong_evidence": [GOV]},
                {"claim_id": "clm_b", "claim": FORCE, "verdict": "insufficient_evidence", "confidence": "low",
                 "reasoning": "we opened the page and it does not give the commencement date"},
            ],
        })],
        claims,
        evidence,
    )
    final = build_final_answer(report, [], 1, TWO_PART)

    assert "Not documented:" in final.answer and "come into force" in final.answer, final.answer
    assert "I couldn't find that documented." in final.answer
    assert "Not checked:" not in final.answer, "this part was reviewed; it just is not established"


# ------------------------------------------------------------------ the truncated review


CUT_OFF = (
    '{"answer": "The Act received Royal Assent on 23 May 2018.", "confidence": "high", "verdicts": ['
    '{"claim_id": "clm_a", "claim": "x", "verdict": "supported", "confidence": "high", "reasoning":'
)


async def test_a_truncated_review_leaves_explicitly_unreviewed_rows(fake_openai):
    """A cut-off reply plus a claim nothing was ever attached to: every claim still ends with a
    state, and the one nobody investigated is NOT_REVIEWED rather than an absent row."""
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b"), claim("The Act has 200 sections.", "clm_c")]
    report = await run_curator(fake_openai, [CUT_OFF], claims, [page("clm_a"), page("clm_b")])
    states = {v.claim_id: v for v in report.verdicts}

    assert report.reviewer_status == "INCOMPLETE"
    assert report.synthesis_status == "FALLBACK"
    assert report.raw_output.endswith('"reasoning":'), "the broken reply is kept"
    assert set(states) == {"clm_a", "clm_b", "clm_c"}, "no claim is left without a state"
    assert states["clm_c"].verdict is ClaimStatus.NOT_REVIEWED
    assert states["clm_c"].confidence == Confidence.NONE
    assert states["clm_a"].verdict in {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}, "the ledger decides what it can"
    assert "Not documented:" not in build_final_answer(report, [], 1, TWO_PART).answer


# --------------------------------------------------------------------- mixed verdicts


def test_supported_contradicted_and_unchecked_parts_are_three_different_things():
    """One question, three parts, three different gaps -- none of them collapsing into another."""
    three_parts = "When did the Act receive Royal Assent, when did it come into force, and how many sections does it have?"
    report = VerifierReport(
        job_id="j", round=1, confidence=Confidence.LOW, answer="x",
        verdicts=[
            judged("clm_a", ASSSENT, ClaimStatus.SUPPORTED, [GOV]),
            judged("clm_b", FORCE, ClaimStatus.REFUTED, ["https://other.example/x"]),
            ClaimVerdict(claim_id="clm_c", claim="The Act has 200 sections.", verdict=ClaimStatus.NOT_REVIEWED,
                         confidence=Confidence.NONE, reasoning="never adjudicated"),
        ],
    )
    final = build_final_answer(report, [], 1, three_parts)

    assert "23 May 2018" in final.answer, final.answer
    assert "The pages we opened say the opposite:" in final.answer, final.answer
    assert "Not checked:" in final.answer and "sections" in final.answer.split("Not checked:")[1], final.answer
    assert "Not documented:" not in final.answer, "a contradicted part and an unchecked part are both findings, not absences"


def test_the_refuted_claim_is_never_promoted_by_its_neighbour():
    """A claim only moves on its own evidence: sitting beside a supported claim changes nothing."""
    report = VerifierReport(
        job_id="j", round=1, confidence=Confidence.LOW, answer="x",
        verdicts=[judged("clm_a", ASSSENT, ClaimStatus.SUPPORTED, [GOV]),
                  judged("clm_b", FORCE, ClaimStatus.NOT_REVIEWED)],
    )
    final = build_final_answer(report, [], 1, TWO_PART)

    assert "Not checked:" in final.answer, final.answer
    assert not any(v.verdict is ClaimStatus.SUPPORTED and v.claim_id == "clm_b" for v in report.verdicts)


# ------------------------------------------------------------------- budget runs out


async def test_when_the_extra_batches_run_out_the_review_says_so(fake_openai):
    """Beyond the recovery budget the claims were not even asked about. That is an incomplete
    review with a stated limit, not a completed one with silent holes."""
    over = Verifier.MAX_CLAIMS + 30
    claims = [claim(f"The Act came into force on 25 May 2018, wording {i}.", f"clm_{i:02d}") for i in range(over)]
    evidence = [page(c.id) for c in claims]
    report = await run_curator(fake_openai, [answers_whatever_it_is_shown], claims, evidence)

    unreviewed = [v for v in report.verdicts if v.verdict is ClaimStatus.NOT_REVIEWED]
    assert unreviewed, "some claims the budget could not reach"
    assert all(v.confidence == Confidence.NONE for v in unreviewed)
    assert report.reviewer_status == "INCOMPLETE", report.reviewer_status
    assert "limit" in report.fallback_reason, report.fallback_reason
    assert any("verifier review incomplete (claim limit reached)" in c for c in report.caveats), report.caveats
    calls = len(fake_openai.requests)
    assert calls <= 1 + 1 + 2 * Verifier.RECOVERY_CALLS, f"extra review must stay bounded, took {calls}"


def test_the_limit_caveat_reaches_the_reader_in_plain_words():
    report = VerifierReport(
        job_id="j", round=1, confidence=Confidence.MODERATE, answer="x",
        verdicts=[judged("clm_a", ASSSENT, ClaimStatus.SUPPORTED, [GOV])],
        reviewer_status="INCOMPLETE", synthesis_status="CURATED",
        fallback_reason="review limit reached: 18 claim(s) answering the question were never adjudicated",
        caveats=["verifier review incomplete (claim limit reached): 18 claim(s) were never adjudicated"],
    )
    final = build_final_answer(report, [], 2, TWO_PART)

    assert final.caveats, "the limit has to be visible, not only in the row"
    text = final.caveats[0].lower()
    assert "didn't get through all of it" in text or "not get to" in text, final.caveats
    for internal in ("ledger", "verdict", "curator", "claim", "18 claim"):
        assert internal not in text, final.caveats


# --------------------------------------------------------------------- ledger and export


def test_unjudged_claims_become_not_reviewed_not_unverified_and_never_insufficient():
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b"), claim("The Act has 200 sections.", "clm_c")]
    annotate_unadjudicated(claims, {"clm_a"})

    assert claims[0].status is ClaimStatus.UNVERIFIED, "an adjudicated claim is left to its verdict"
    assert claims[1].status is ClaimStatus.NOT_REVIEWED
    assert claims[2].status is ClaimStatus.NOT_REVIEWED
    assert "not among them" in claims[1].rationale


def test_a_later_pass_that_never_mentioned_a_claim_does_not_unsettled_it():
    """Every round rewrites the whole ledger, so a `not_reviewed` row means "not in THIS pass".

    Without the guard, a claim one curator pass settled with an opened page would come back
    unreviewed in the next round simply because focus dropped it -- erasing an established verdict.
    """
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b")]
    apply_verdicts(claims, [judged("clm_a", ASSSENT, ClaimStatus.SUPPORTED, [GOV])])
    assert claims[0].status is ClaimStatus.SUPPORTED

    apply_verdicts(
        claims,
        [ClaimVerdict(claim_id="clm_a", claim=ASSSENT, verdict=ClaimStatus.NOT_REVIEWED, confidence=Confidence.NONE, reasoning="not in this pass")],
    )
    assert claims[0].status is ClaimStatus.SUPPORTED, "not_reviewed describes a pass, not the claim's history"


def test_the_export_prints_the_two_gaps_differently():
    md = job_markdown(
        {
            "question": TWO_PART, "status": "completed",
            "final": {"answer": "Documented:\n- Royal Assent 23 May 2018 (gov.uk).", "reviewer_status": "COMPLETED", "synthesis_status": "CURATED"},
            "claims": [
                {"claim": ASSSENT, "status": "supported", "confidence": "high", "providers": ["chatgpt"]},
                {"claim": FORCE, "status": "insufficient_evidence", "confidence": "insufficient_evidence", "providers": ["gemini"]},
                {"claim": "The Act has 200 sections.", "status": "not_reviewed", "confidence": "insufficient_evidence", "providers": ["copilot"]},
            ],
        }
    )
    rows = [line.replace("\\", "") for line in md.splitlines() if line.startswith("| ")]
    ledger = [r for r in rows if r.startswith(("| The Act received", "| Most of", "| The Act has"))]
    assert len(ledger) == 3, md
    assert any("insufficient_evidence" in r for r in ledger), md
    assert any("not_reviewed" in r for r in ledger), md
    assert not any("not_reviewed / insufficient_evidence" in r for r in ledger), (
        "a claim nobody checked must not also be labelled as an evidentiary finding: "
        f"{[r for r in ledger if 'not_reviewed' in r]}"
    )


def test_evidence_report_keeps_undocumented_and_unchecked_apart():
    text = evidence_report(
        [("Royal Assent 23 May 2018", "gov.uk")],
        ["when did it come into force?"],
        [],
        ["how many sections does it have?"],
    )
    assert "Not documented:\n- when did it come into force? — I couldn't find that documented." in text
    assert "Not checked:\n- how many sections does it have? — I didn't get that far." in text
    assert "I couldn't find that documented." not in text.split("Not checked:")[1]
