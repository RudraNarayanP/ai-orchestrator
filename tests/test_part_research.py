"""A part of the question the user typed must not go unresearched because nobody asked about it.

The gap these tests pin: the level-3 loop only ever researched what the curator requested, and the
only part-level trigger (`analysis.sub_questions`) exists when the analysis model answered. So a
two-part question whose second half no provider ever claimed could end with `needs_more_research`
false, no follow-up, and an answer that called our own silence "Not documented".

Now the loop reads the question's own parts off the ledger, classifies each open part by the kind
of gap it is, and sends a targeted ask -- the existing conversation first, then the independents the
current policy picks -- within the same bounded budget. `not_adjudicated` is never sent to a
provider: pages are attached and another browser round cannot change that.
"""

from __future__ import annotations

import json
import re

from backend.models import (
    Claim,
    ClaimStatus,
    ClaimVerdict,
    Confidence,
    Evidence,
    Job,
    PartGap,
    ResearchMode,
    SourceCheckStatus,
    SourceTier,
    VerifierReport,
)
from backend.orchestrator.runner import ResearchRunner, _part_gaps
from backend.research.claims import stable_claim_id
from tests.conftest import adapters_from, base_settings

ASSSENT = "The Data Protection Act 2018 received Royal Assent on 23 May 2018."
FORCE = "Most of the Act's provisions came into force on 25 May 2018."
TWO_PART = "When did the UK Data Protection Act 2018 receive Royal Assent, and when did it come into force?"
GOV = "https://www.gov.uk/government/collections/data-protection-act-2018"


def claim(text: str, cid: str) -> Claim:
    return Claim(id=cid, job_id="j", claim=text, kind="date", provider_sources=["chatgpt"])


def page(cid: str, url: str, status: SourceCheckStatus = SourceCheckStatus.CONFIRMED) -> Evidence:
    return Evidence(
        job_id="j", claim_id=cid, url=url, domain=url.split("/")[2], tier=SourceTier.PRIMARY_OFFICIAL,
        check_status=status, verbatim_excerpt="the page states it",
    )


def verdict(cid: str, text: str, value: ClaimStatus) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=cid, claim=text, verdict=value,
        confidence=Confidence.HIGH if value is ClaimStatus.SUPPORTED else Confidence.LOW,
        reasoning="the page says so", strong_evidence=[GOV] if value is ClaimStatus.SUPPORTED else [],
    )


# ---------------------------------------------------- 1. which part is open, and what kind of gap


def test_a_part_no_page_was_ever_attached_to_is_a_missing_evidence_gap():
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b")]
    gaps = _part_gaps(TWO_PART, claims, [page("clm_a", GOV)], [verdict("clm_a", ASSSENT, ClaimStatus.SUPPORTED)])

    assert len(gaps) == 1, gaps
    assert "come into force" in gaps[0].part
    assert gaps[0].reason == "missing_evidence", "we hold nothing against that half"


def test_a_part_whose_page_refused_us_is_a_source_access_gap_not_a_missing_evidence_gap():
    """Different remedy: the source exists and we know where it is -- it refused us, so the ask is
    'open it yourself or find the official equivalent', not 'go find any source'."""
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b")]
    blocked = page("clm_b", "https://www.legislation.gov.uk/id/ukpga/2018/12", SourceCheckStatus.BLOCKED)
    gaps = _part_gaps(
        TWO_PART, claims, [page("clm_a", GOV), blocked],
        [verdict("clm_a", ASSSENT, ClaimStatus.SUPPORTED), verdict("clm_b", FORCE, ClaimStatus.NOT_REVIEWED)],
    )

    assert [g.reason for g in gaps] == ["source_blocked"], gaps


def test_pages_attached_but_never_weighed_is_our_review_gap_not_a_research_gap():
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b")]
    gaps = _part_gaps(
        TWO_PART, claims,
        [page("clm_a", GOV), page("clm_b", "https://example.gov.uk/commencement")],
        [verdict("clm_a", ASSSENT, ClaimStatus.SUPPORTED)],  # nothing ever adjudicated clm_b
    )

    assert [g.reason for g in gaps] == ["not_adjudicated"], gaps


def test_a_part_the_sources_disagree_about_is_a_contradiction():
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b")]
    gaps = _part_gaps(
        TWO_PART, claims, [page("clm_a", GOV), page("clm_b", "https://example.gov.uk/x")],
        [
            verdict("clm_a", ASSSENT, ClaimStatus.SUPPORTED),
            ClaimVerdict(claim_id="clm_b", claim=FORCE, verdict=ClaimStatus.CONTESTED, confidence=Confidence.LOW, reasoning="two pages differ"),
        ],
    )

    assert [g.reason for g in gaps] == ["contradiction"], gaps


def test_a_part_answered_by_an_opened_page_is_not_a_gap_at_all():
    claims = [claim(ASSSENT, "clm_a"), claim("Most provisions of the Act came into force on 25 May 2018.", "clm_b")]
    evidence = [page("clm_a", GOV), page("clm_b", "https://example.gov.uk/commence")]
    verdicts = [
        verdict("clm_a", claims[0].claim, ClaimStatus.SUPPORTED),
        verdict("clm_b", claims[1].claim, ClaimStatus.SUPPORTED),
    ]
    assert _part_gaps(TWO_PART, claims, evidence, verdicts) == []


def test_a_single_part_question_yields_no_part_gaps():
    """The claim-level machinery already covers a question that asks one thing; no extra rounds."""
    claims = [claim(ASSSENT, "clm_a")]
    assert _part_gaps("When did the DPA 2018 receive Royal Assent?", claims, [], []) == []


def test_a_mentioned_url_is_not_evidence_for_a_part():
    """A provider citing a link it never opened leaves the part unfilled -- the row is not_checked."""
    claims = [claim(ASSSENT, "clm_a"), claim(FORCE, "clm_b")]
    quote = Evidence(
        job_id="j", claim_id="clm_b", url="https://example.gov.uk/commence", domain="example.gov.uk",
        tier=SourceTier.PRIMARY_OFFICIAL, check_status=SourceCheckStatus.NOT_CHECKED,
        origin="provider_quote", verbatim_excerpt="it commenced on 25 May 2018",
    )
    gaps = _part_gaps(TWO_PART, claims, [page("clm_a", GOV), quote], [verdict("clm_a", ASSSENT, ClaimStatus.SUPPORTED)])

    assert [g.reason for g in gaps] == ["missing_evidence"], "second-hand text is not a page we read"


# ------------------------------------------------------ 2. routing: primary first, then independents


def _runner_with(primary: str = "chatgpt") -> ResearchRunner:
    scripts = {name: {"answer": "KEY CLAIMS\n1. x", "citations": []} for name in (primary, "gemini")}
    return ResearchRunner(base_settings(), adapters_from(scripts, base_settings()), engine=None)


def test_the_first_ask_about_a_part_goes_to_the_ai_already_in_the_conversation():
    runner = _runner_with("chatgpt")
    job = Job(question=TWO_PART, mode=ResearchMode.STANDARD)
    gap = PartGap(part="when did it come into force?", reason="missing_evidence", claim_ids=["clm_b"])

    asks = runner._part_follow_ups(job, [gap], 2, "chatgpt")

    assert len(asks) == 1
    assert "come into force" in asks[0].question
    assert asks[0].target_providers == ["chatgpt"], "same conversation first"
    assert "no source we opened states it" in asks[0].reason
    assert asks[0].claim_ids == ["clm_b"], "the ask carries the claim it is about, so provenance survives"


def test_the_second_ask_about_the_same_part_goes_wider():
    """The primary already tried and failed: the existing policy sends it to independents next."""
    runner = _runner_with("chatgpt")
    job = Job(question=TWO_PART, mode=ResearchMode.STANDARD)
    gap = PartGap(part="when did it come into force?", reason="missing_evidence", claim_ids=["clm_b"])
    first = runner._part_follow_ups(job, [gap], 2, "chatgpt")
    job.follow_ups.extend(first)

    second = runner._part_follow_ups(job, [gap], 3, "chatgpt")

    assert second[0].target_providers == [], "independents are chosen by the existing escalation policy"
    assert "independent researchers" in second[0].reason


def test_an_unadjudicated_part_is_never_put_to_a_provider():
    """Pages are attached; a browser round cannot weigh them, and spending one would look like progress."""
    runner = _runner_with("chatgpt")
    job = Job(question=TWO_PART, mode=ResearchMode.STANDARD)
    gap = PartGap(part="when did it come into force?", reason="not_adjudicated", claim_ids=["clm_b"])

    assert runner._part_follow_ups(job, [gap], 2, "chatgpt") == []


def test_the_ask_never_invents_an_answer_it_lacks():
    """Each wording asks for a source and to be told when none exists -- never to fill the hole."""
    runner = _runner_with("chatgpt")
    job = Job(question=TWO_PART, mode=ResearchMode.STANDARD)
    for reason in ("missing_evidence", "source_blocked", "contradiction"):
        asks = runner._part_follow_ups(job, [PartGap(part="when did it come into force?", reason=reason)], 2, "chatgpt")
        text = asks[0].question.lower()
        assert "open" in text, reason
        assert "say" in text, reason
        assert "probably" not in text and "most likely" not in text, reason


def test_an_unsettled_half_is_reported_even_when_the_loop_ended_early():
    """The early "the ledger is settled" returns never reach the research loop, so the record has to
    be made honest in `_complete` -- otherwise a run that stopped as soon as it had one good answer
    leaves the other half unmentioned, which is the same silence with a different cause."""
    runner = _runner_with("chatgpt")
    job = Job(question=TWO_PART, mode=ResearchMode.STANDARD)
    job.claims = [claim(ASSSENT, "clm_a")]
    job.evidence = [page("clm_a", GOV)]
    report = VerifierReport(
        job_id=job.id, round=1, confidence=Confidence.MODERATE, answer=ASSSENT,
        verdicts=[ClaimVerdict(claim_id="clm_a", claim=ASSSENT, verdict=ClaimStatus.PARTIALLY_SUPPORTED,
                               confidence=Confidence.MODERATE, reasoning="one opened page states it",
                               strong_evidence=[GOV])],
    )
    finished = runner._complete(job, report, [], rounds=1, stop="sufficient evidence without needing the verifier")
    gap_reasons = {g.reason for g in report.part_gaps}

    assert gap_reasons == {"missing_evidence"}, report.part_gaps
    assert finished.final.caveats and "come into force" in finished.final.caveats[0], finished.final.caveats
    assert "still open" in finished.stop_reason, finished.stop_reason
    assert "23 May 2018" in finished.final.answer, "the half that IS established is still answered"


def test_the_stop_reason_never_says_both_settled_and_open():
    """"Every material claim settled" beside "a part of the question is still open" is a record that
    contradicts itself: one is about the claims raised, the other about what was asked."""
    runner = _runner_with("chatgpt")
    job = Job(question=TWO_PART, mode=ResearchMode.STANDARD)
    job.claims = [claim(ASSSENT, "clm_a")]
    job.evidence = [page("clm_a", GOV)]
    report = VerifierReport(
        job_id=job.id, round=3, confidence=Confidence.MODERATE, answer=ASSSENT,
        verdicts=[ClaimVerdict(claim_id="clm_a", claim=ASSSENT, verdict=ClaimStatus.SUPPORTED,
                               confidence=Confidence.MODERATE, reasoning="opened pages state it",
                               strong_evidence=[GOV])],
    )
    stop = "stopped at max rounds (3) with every material claim settled by opened evidence"
    finished = runner._complete(job, report, [], rounds=3, stop=stop)

    assert "every material claim settled" not in finished.stop_reason, finished.stop_reason
    assert "the claims it raised settled by opened evidence" in finished.stop_reason, finished.stop_reason
    assert "still open" in finished.stop_reason and "come into force" in finished.stop_reason, finished.stop_reason


def test_a_settled_ledger_with_every_part_answered_adds_no_gap_note():
    """No noise when the question is fully settled: the record reads exactly as it did before."""
    runner = _runner_with("chatgpt")
    job = Job(question=TWO_PART, mode=ResearchMode.STANDARD)
    job.claims = [claim(ASSSENT, "clm_a"), claim("Most provisions of the Act came into force on 25 May 2018.", "clm_b")]
    job.evidence = [page("clm_a", GOV), page("clm_b", FORCE_PAGE)]
    report = VerifierReport(
        job_id=job.id, round=1, confidence=Confidence.MODERATE, answer=ASSSENT,
        verdicts=[
            ClaimVerdict(claim_id="clm_a", claim=ASSSENT, verdict=ClaimStatus.SUPPORTED, confidence=Confidence.MODERATE,
                         reasoning="opened pages state it", strong_evidence=[GOV]),
            ClaimVerdict(claim_id="clm_b", claim=job.claims[1].claim, verdict=ClaimStatus.SUPPORTED,
                         confidence=Confidence.MODERATE, reasoning="the commencement regulations state it",
                         strong_evidence=[FORCE_PAGE]),
        ],
    )
    finished = runner._complete(job, report, [], rounds=1, stop="evidence sufficient after expansion")

    assert report.part_gaps == [], report.part_gaps
    assert "still open" not in finished.stop_reason, finished.stop_reason
    assert not any("could not answer this part" in c for c in finished.final.caveats), finished.final.caveats


# --------------------------------------------------------------------- the research loop itself

ASSSENT_ONLY = "KEY CLAIMS\n1. " + ASSSENT
BOTH_CLAIMS = "KEY CLAIMS\n1. " + ASSSENT + "\n2. " + FORCE
WEAK = "KEY CLAIMS\n1. The Act was enacted in 2018."
FORCE_PAGE = "https://www.legislation.gov.uk/ukpga/2018/12/schedule"


async def run_loop(settings, scripts, question=TWO_PART, *, verifier=None, max_rounds=3):
    adapters = adapters_from(scripts, settings)
    runner = ResearchRunner(settings, adapters, engine=None, verifier=verifier)
    job = Job(question=question, mode=ResearchMode.STANDARD, max_rounds=max_rounds)
    return await runner.run(job), adapters


async def test_an_unanswered_half_becomes_a_targeted_ask_not_silence(settings, net):
    """The defect: part one answered, part two never claimed, and the loop stopped because the
    ledger asked for nothing. The user got a confident half-answer and no word about the other half."""
    net.confirm(GOV, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="received Royal Assent on 23 May 2018",
                claim_id=stable_claim_id(ASSSENT))
    scripts = {
        "chatgpt": {"answer": ASSSENT_ONLY, "citations": [{"url": GOV, "title": "DPA 2018 collection"}]},
        "gemini": {"answer": WEAK, "citations": []},
    }
    job, _ = await run_loop(settings, scripts)
    asks = [f for f in job.follow_ups if "come into force" in f.question]

    assert asks, "an open half of the user's question must become a research task"
    assert asks[0].target_providers == ["chatgpt"], "the AI already holding the conversation is asked first"
    assert any(f.target_providers == [] for f in asks), "and when it cannot settle the part, the independents get it"
    assert asks[0].claim_ids == [], "nothing has claimed that half yet -- the ask says so by carrying no claims"
    assert "come into force" in (job.stop_reason or ""), job.stop_reason
    assert job.final.caveats and "come into force" in job.final.caveats[0], job.final.caveats
    assert "No source I opened states it." in job.final.caveats[0], job.final.caveats


async def test_a_part_the_follow_up_establishes_stops_being_a_gap(settings, net):
    """The primary answers the half when it is asked: the gap closes and must not be reported again."""
    net.confirm(GOV, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="received Royal Assent on 23 May 2018",
                claim_id=stable_claim_id(ASSSENT))
    net.confirm(FORCE_PAGE, tier=SourceTier.GOVERNMENT, excerpt="Most provisions came into force on 25 May 2018.",
                claim_id=stable_claim_id(FORCE))
    scripts = {
        "chatgpt": {
            "rounds": [
                {"answer": ASSSENT_ONLY, "citations": [{"url": GOV, "title": "collection"}]},
                {"answer": "I could not find anything further.", "citations": []},
                {"answer": BOTH_CLAIMS, "citations": [{"url": GOV, "title": "collection"}, {"url": FORCE_PAGE, "title": "commencement regulations"}]},
            ]
        },
        "gemini": {"answer": WEAK, "citations": []},
    }
    job, _ = await run_loop(settings, scripts)
    seen_gaps = [g for r in job.reports for g in r.part_gaps]

    assert any(g.reason == "missing_evidence" for g in seen_gaps), "it began as an open part"
    assert job.reports[-1].part_gaps == [], f"once the follow-up establishes it the part is closed: {job.reports[-1].part_gaps}"
    assert "still open" not in (job.stop_reason or ""), job.stop_reason
    assert "25 May 2018" in job.final.answer, job.final.answer


async def test_a_refused_source_is_asked_about_as_a_refusal(settings, net):
    """"We could not read that page" is a different ask from "nobody looked for a source"."""
    net.confirm(GOV, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="received Royal Assent on 23 May 2018",
                claim_id=stable_claim_id(ASSSENT))
    net.fail(FORCE_PAGE, SourceCheckStatus.BLOCKED, claim_id=stable_claim_id(FORCE))
    scripts = {
        "chatgpt": {
            "answer": BOTH_CLAIMS,
            "citations": [{"url": GOV, "title": "collection"}, {"url": FORCE_PAGE, "title": "commencement"}],
        },
        "gemini": {"answer": "nothing", "citations": []},
    }
    job, _ = await run_loop(settings, scripts)
    asks = [f for f in job.follow_ups if "come into force" in f.question]
    final_gaps = job.reports[-1].part_gaps

    assert final_gaps, "the refused half is still open"
    assert final_gaps[0].reason == "source_blocked", final_gaps
    assert any("blocked or refused us" in f.question for f in asks), [f.question[:90] for f in asks]
    assert "refused us" in job.final.caveats[0], job.final.caveats


async def test_the_budget_bounds_it_and_never_re_asks_what_is_settled(settings, net):
    """max_rounds still decides when we stop; what changed is that the stop names the open half."""
    net.confirm(GOV, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="received Royal Assent on 23 May 2018",
                claim_id=stable_claim_id(ASSSENT))
    scripts = {
        "chatgpt": {"answer": ASSSENT_ONLY, "citations": [{"url": GOV, "title": "collection"}]},
        "gemini": {"answer": WEAK, "citations": []},
        "copilot": {"answer": WEAK, "citations": []},
    }
    job, _ = await run_loop(settings, scripts, max_rounds=2)

    assert len(job.rounds) <= 3, [r.number for r in job.rounds]
    assert "still open" in (job.stop_reason or ""), job.stop_reason
    assert not [f for f in job.follow_ups if "Royal Assent" in f.question], (
        "the half the pages already establish is never re-asked"
    )


async def test_a_recovered_pass_never_unsettles_an_earlier_supported_verdict(settings, net, fake_openai):
    """End-to-end carry-forward: the final pass is cut off mid-reply, and a claim an earlier
    completed pass settled must survive it as settled -- never absent, never not_reviewed."""
    from backend.verification.verifier import Verifier
    from backend.verification.llm import Endpoint

    net.confirm(GOV, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="received Royal Assent on 23 May 2018",
                claim_id=stable_claim_id(ASSSENT))
    scripts = {
        "chatgpt": {"answer": BOTH_CLAIMS, "citations": [{"url": GOV, "title": "collection"}]},
        "gemini": {"answer": WEAK, "citations": []},
    }

    def reply(body):
        user = body["messages"][-1]["content"]
        rows = re.findall(r'"claim_id": "([^"]+)",\s*"claim": "([^"]+)"', user)
        if len(rows) > 1:  # the multi-claim pass truncates for real
            return '{"answer": "x", "confidence": "low", "verdicts": [{"claim_id": "%s", "claim": "y", "verdict": "supported", "reasoning":' % rows[0][0]
        return json.dumps(
            {
                "verdicts": [
                    {"claim_id": cid, "claim": text, "verdict": "supported", "confidence": "high",
                     "reasoning": "the opened page states it", "strong_evidence": [GOV]}
                    for cid, text in rows
                ],
                "answer": ASSSENT, "why": "the page states it", "confidence": "high",
                "needs_more_research": False, "research_needed": [], "unresolved": [],
            }
        )

    fake_openai.script(reply)
    verifier = Verifier(
        Endpoint(provider="openai_compatible", model="fake-model", base_url=fake_openai.base_url, timeout_s=10),
        min_independent_sources=2,
    )
    job, _ = await run_loop(settings, scripts, verifier=verifier, max_rounds=2)
    assent = [c for c in job.claims if "Royal Assent" in c.claim]

    assert job.reports[-1].reviewer_status == "INCOMPLETE", job.reports[-1].reviewer_status
    assert assent, "the assent claim is still in the ledger"
    assert all(c.status is not ClaimStatus.NOT_REVIEWED for c in assent), [(c.claim[:40], c.status.value) for c in assent]
    assert any(v.verdict in {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED} for v in job.reports[-1].verdicts)
