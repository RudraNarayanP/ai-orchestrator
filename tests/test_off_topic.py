"""A reused chat window that answers an earlier question must not feed the ledger (live eval defect)."""

from __future__ import annotations

from backend.evidence.sources import SourceTier
from backend.models import ProviderStatus
from backend.research.router import off_topic, question_from_prompt
from tests.test_resilience import run_job

UA_Q = "Under the Family Code of Ukraine, what is the minimum marriageable age and can a court allow it earlier?"
STALE = (
    "DIRECT ANSWER\nThe Employment Rights Act 2025 received Royal Assent on 18 December 2025 and enacts "
    "structural amendments to the Employment Rights Act 1996, staged across 2026 and 2027 rather than 2028."
)
GOOD = "The minimum marriageable age in Ukraine is 18; under the Family Code a court can allow marriage from 16 if it is in the person's interests."


def test_off_topic_detects_a_reply_about_a_different_subject():
    assert off_topic(UA_Q, STALE)
    assert not off_topic(UA_Q, GOOD)


def test_off_topic_is_not_triggered_by_short_or_lowercase_answers():
    assert not off_topic(UA_Q, "I don't know.")
    assert not off_topic("how tall is mount everest in metres right now", "It is about 8,849 metres above sea level according to the 2020 survey by Nepal and China.")


def test_question_is_recovered_from_round_one_and_targeted_prompts():
    assert question_from_prompt(UA_Q + "\n\nSearch the web before answering.") == UA_Q
    assert question_from_prompt("Targeted verification task.\n\nOriginal question: " + UA_Q + "\n\nCheck this point") == UA_Q


async def test_a_stale_reply_is_dropped_before_it_reaches_the_ledger(settings, net):
    net.confirm("https://zakon.rada.gov.ua/laws/show/2947-14", tier=SourceTier.GOVERNMENT)
    scripts = {
        "chatgpt": {"answer": GOOD, "citations": [{"url": "https://zakon.rada.gov.ua/laws/show/2947-14", "title": "Family Code of Ukraine"}]},
        "gemini": {"answer": STALE, "citations": [{"url": "https://legislation.gov.uk/ukpga/2025/36", "title": "Employment Rights Act 2025"}]},
    }
    job, _adapters, _ = await run_job(settings, scripts, UA_Q)
    gem = [r for r in job.responses if r.provider == "gemini"]
    assert gem and all(r.status == ProviderStatus.FAILED and (r.error or "").startswith("off_topic") for r in gem)
    assert all(not r.answer_text for r in gem)
    assert not any("Employment Rights" in c.claim for c in job.claims)

def test_addresses_question_wants_the_asked_attribute_not_just_the_same_subject():
    from backend.research.router import addresses_question

    q = "What percentage of PhD vivas at the University of Manchester ended in outright failure last year?"
    assert not addresses_question(q, "The University of Manchester's current PhD regulations were last modified on 5 August 2026.")
    assert addresses_question(q, "In 2025, 3 percent of Manchester PhD vivas ended in outright failure.")
    assert addresses_question("When was the Eiffel Tower completed and how tall is it?", "The tower was completed on 31 March 1889.")


def test_model_free_answer_says_dont_know_when_the_only_supported_claims_are_beside_the_point():
    from backend.models import ClaimStatus, ClaimVerdict, Confidence, Evidence, SourceCheckStatus, SourceTier
    from backend.verification.llm import Endpoint
    from backend.verification.verifier import Verifier

    verifier = Verifier(Endpoint(provider="disabled", model="none", base_url=""), min_independent_sources=2)
    url = "https://www.manchester.ac.uk/regs"
    ev = Evidence(job_id="j", claim_id="c1", url=url, domain="www.manchester.ac.uk", tier=SourceTier.GOVERNMENT, check_status=SourceCheckStatus.CONFIRMED)
    verdict = ClaimVerdict(claim_id="c1", claim="The University of Manchester's PhD regulations were last modified on 5 August 2026.",
                           verdict=ClaimStatus.SUPPORTED, confidence=Confidence.HIGH, reasoning="r", strong_evidence=[url])
    q = "What percentage of PhD vivas at the University of Manchester ended in outright failure last year?"
    best = verifier._best_supported([], [verdict], [ev], q)
    assert best["answer"].startswith("I don't know.") and "won't guess" in best["answer"]
    assert best["confidence"] == Confidence.NONE and not best["sources"]
    # without the question the old behaviour is unchanged
    assert verifier._best_supported([], [verdict], [ev])["answer"].startswith("The University of Manchester")

def test_future_year_questions_are_not_answered_with_what_happened_earlier():
    from backend.research.router import addresses_question, future_year

    q = "Which amendments will Parliament make to the Employment Rights Act 1996 during 2028?"
    assert future_year(q, now_year=2026) == "2028" and future_year("What happened in 1995?") is None
    assert not addresses_question(q, "The Employment Rights Act 2025 amended the Employment Rights Act 1996 through numerous provisions.")


def test_ledger_supported_claims_about_other_years_become_a_reasoned_dont_know():
    from backend.models import Claim, ClaimStatus, ClaimVerdict, Confidence, Evidence, SourceCheckStatus, SourceTier, VerifierReport
    from backend.verification.llm import Endpoint
    from backend.verification.verifier import Verifier

    q = "Which amendments will Parliament make to the Employment Rights Act 1996 during 2028?"
    claim = Claim(job_id="j", id="c1", claim="The Employment Rights Act 2025 amended the Employment Rights Act 1996.", kind="legal")
    evs = [
        Evidence(job_id="j", claim_id="c1", url=f"https://www.legislation.gov.uk/ukpga/2025/36/{i}", domain=d, tier=SourceTier.GOVERNMENT, check_status=SourceCheckStatus.CONFIRMED)
        for i, d in enumerate(["www.legislation.gov.uk", "www.gov.uk"])
    ]
    verdict = ClaimVerdict(claim_id="c1", claim=claim.claim, verdict=ClaimStatus.SUPPORTED, confidence=Confidence.HIGH, reasoning="r", strong_evidence=[e.url for e in evs])
    report = VerifierReport(job_id="j", round=1, verdicts=[verdict], answer="The 2025 Act amended the 1996 Act in many ways.", confidence=Confidence.HIGH)
    Verifier(Endpoint(provider="disabled", model="none", base_url=""), min_independent_sources=2)._reconcile(report, [claim], evs, q)
    assert report.answer.startswith("I don't know.") and "2028" in report.answer and "hasn't happened yet" in report.answer
    assert not report.sources and report.confidence != Confidence.HIGH

def test_verifier_judges_at_most_twelve_claims_and_keeps_the_ones_with_pages():
    """Live: ~40 reworded claims made the verdict JSON outrun max_tokens and the run fell back to the model-free path."""
    from backend.models import Claim, Evidence, SourceCheckStatus
    from backend.verification.verifier import Verifier

    claims = [Claim(job_id="j", id=f"c{i}", claim=f"claim number {i} about the thing", kind="fact") for i in range(40)]
    backed = {"c33", "c38"}
    evidence = [Evidence(job_id="j", claim_id=cid, url=f"https://x.example/{cid}", check_status=SourceCheckStatus.CONFIRMED) for cid in backed]
    evidence.append(Evidence(job_id="j", claim_id=None, url="https://orphan.example/", check_status=SourceCheckStatus.CONFIRMED))
    kept, ev = Verifier.focus(claims, evidence)
    assert len(kept) == Verifier.MAX_CLAIMS
    assert backed <= {c.id for c in kept}, "claims with confirmed pages are judged first"
    assert {e.claim_id for e in ev if e.claim_id} == backed
    small, ev_small = Verifier.focus(claims[:5], evidence)
    assert len(small) == 5 and len(ev_small) == len(evidence)

def test_a_verdict_with_an_invented_claim_id_is_reattached_by_wording():
    """Live: nemotron answered with claim_id 'clm_answer'; every verdict was dropped and a supported answer became a don't-know."""
    from backend.models import Claim
    from backend.verification.llm import Endpoint
    from backend.verification.verifier import Verifier

    claims = [
        Claim(job_id="j", id="clm_real1", claim="Under the Data Protection Act 2018 the minimum age at which a child can consent to information society services is 13 years.", kind="legal"),
        Claim(job_id="j", id="clm_real2", claim="Preventive and counselling services are excluded from Article 8.", kind="legal"),
    ]
    v = Verifier(Endpoint(provider="disabled", model="none", base_url=""), min_independent_sources=2)
    parsed = {
        "verdicts": [
            {"claim_id": "clm_answer", "claim": "Under the Data Protection Act 2018, the minimum age at which a child can consent to information society services in the UK is 13 years.", "verdict": "supported", "confidence": "moderate"},
            {"claim_id": "clm_nonsense", "claim": "Totally unrelated statement about whales", "verdict": "supported", "confidence": "high"},
        ],
        "answer": "A child must be 13.",
        "confidence": "moderate",
    }
    report = v._from_model(parsed, job_id="j", round_no=1, claims=claims, evidence=[])
    assert [x.claim_id for x in report.verdicts] == ["clm_real1"], "re-attached by wording; the unrelated one is dropped"

def test_ledger_lifts_a_claim_the_model_under_rated_and_rebuilds_the_answer():
    """Live: nemotron marked Theft Act s.1 'unverifiable' although opened legislation pages quoted it; the answer became a don't-know."""
    from backend.models import Claim, ClaimStatus, ClaimVerdict, Confidence, Evidence, SourceCheckStatus, SourceTier, VerifierReport
    from backend.verification.llm import Endpoint
    from backend.verification.verifier import Verifier

    q = "How does section 1 of the Theft Act 1968 define theft?"
    claim = Claim(job_id="j", id="c1", claim="Section 1 of the Theft Act 1968 defines theft as dishonestly appropriating property belonging to another with the intention of permanently depriving the other of it.", kind="legal")
    evs = [
        Evidence(job_id="j", claim_id="c1", url=u, domain=d, tier=SourceTier.GOVERNMENT, check_status=SourceCheckStatus.CONFIRMED)
        for u, d in (("https://www.legislation.gov.uk/ukpga/1968/60/section/1", "www.legislation.gov.uk"), ("https://www.gov.uk/theft-act", "www.gov.uk"))
    ]
    verdict = ClaimVerdict(claim_id="c1", claim=claim.claim, verdict=ClaimStatus.INSUFFICIENT_EVIDENCE, confidence=Confidence.NONE, reasoning="r", problems=["unverifiable"])
    report = VerifierReport(job_id="j", round=1, verdicts=[verdict], answer="I don't know.", confidence=Confidence.LOW)
    Verifier(Endpoint(provider="disabled", model="none", base_url=""), min_independent_sources=2)._reconcile(report, [claim], evs, q)
    assert verdict.verdict == ClaimStatus.SUPPORTED and "unverifiable" not in verdict.problems
    assert report.answer.startswith("Section 1 of the Theft Act 1968") and report.sources
    assert report.confidence in {Confidence.HIGH, Confidence.MODERATE}


def test_ledger_does_not_lift_a_claim_with_no_confirmed_pages():
    from backend.models import Claim, ClaimStatus, ClaimVerdict, Confidence, VerifierReport
    from backend.verification.llm import Endpoint
    from backend.verification.verifier import Verifier

    claim = Claim(job_id="j", id="c1", claim="Section 1 of the Theft Act 1968 defines theft as appropriation.", kind="legal")
    verdict = ClaimVerdict(claim_id="c1", claim=claim.claim, verdict=ClaimStatus.INSUFFICIENT_EVIDENCE, confidence=Confidence.NONE, reasoning="r")
    report = VerifierReport(job_id="j", round=1, verdicts=[verdict], answer="I don't know.", confidence=Confidence.LOW)
    Verifier(Endpoint(provider="disabled", model="none", base_url=""), min_independent_sources=2)._reconcile(report, [claim], [], "How does section 1 of the Theft Act 1968 define theft?")
    assert verdict.verdict == ClaimStatus.INSUFFICIENT_EVIDENCE