"""A question that names four subjects must not be answered about whichever two ranked highest.

Live (job_261016150809_fa891ed9, 2026-10-16): "tell me about housing crisis in netherlands,
germany, france and italy" produced a two-sentence answer about Germany and France. Netherlands
and Italy were never mentioned -- not because nothing was found (the run extracted claims naming
all four, with the CBS shortage figure and the Milan rent-to-income ratio in them), but because
the answer layer capped every question at 2-3 sentences / 280 characters and the fallback path
capped itself at three claims. A silent cut of that kind reads as a complete answer.

Widening the budget is presentation only. It never raises confidence, never promotes an
unadjudicated claim, and never applies to a yes/no check, whose two-sentence lead is the point.
"""

from __future__ import annotations

from backend.models import ClaimStatus, ClaimVerdict, Confidence, VerifierReport
from backend.research.lint import asked_subjects
from backend.research.style import answer_budget, render_truth, tiny
from backend.verification.verifier import build_final_answer

BREADTH = "tell me about housing crisis in netherlands , germany , france and italy"
SINGLE = "Is the Netherlands the EU country with the highest renter share?"

# Each sentence states its figure inside a settled claim: anything the ledger does not carry is
# stripped by ground_answer before the length budget is ever reached (see the test below).
COUNTRY_CLAIMS = [
    "The Netherlands' own statistical office puts the housing shortage at about 384,000 homes, or 4.6% of the stock.",
    "Germany had the highest renter share in the EU in 2024, with 53% of its population renting.",
    "In France, 35% of renters told the 2023-2024 housing survey that their home was too expensive.",
    "In Italy the European Commission reports rent-to-income ratios reaching 76% in Milan and 65% in Rome.",
]
FOUR_SUBJECTS = " ".join(COUNTRY_CLAIMS)


def verdict(claim: str, cid: str, value: ClaimStatus = ClaimStatus.SUPPORTED) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=cid,
        claim=claim,
        verdict=value,
        confidence=Confidence.MODERATE,
        reasoning="an official page we opened states it",
        strong_evidence=["https://ec.europa.eu/eurostat/web/interactive-publications/housing-2025"],
    )


# ---------------------------------------------------------------- detection


def test_a_question_naming_several_subjects_is_detected_as_several():
    assert asked_subjects(BREADTH) == ["netherlands", "germany", "france", "italy"]


def test_a_question_about_one_thing_names_no_series():
    assert asked_subjects(SINGLE) == []
    assert asked_subjects("What is the maximum fine under the US federal GDPR?") == []


def test_the_budget_grows_with_the_question_and_only_there():
    assert answer_budget(SINGLE) == (3, 280), "an ordinary question keeps the old cap"
    sentences, chars = answer_budget(BREADTH)
    assert sentences >= 5 and chars >= 460


# ---------------------------------------------------------------- the answer keeps every subject


def test_an_answer_about_four_subjects_reaches_the_reader_rather_than_the_first_two():
    report = VerifierReport(
        job_id="j",
        round=1,
        verdicts=[verdict(claim, f"clm_{n}") for n, claim in enumerate(COUNTRY_CLAIMS)],
        confidence=Confidence.MODERATE,
        answer=FOUR_SUBJECTS,
        why="the pages we opened state this",
    )
    final = build_final_answer(report, [], 1, question=BREADTH)
    for subject in ("Netherlands", "Germany", "France", "Italy"):
        assert subject in final.answer, f"{subject} was cut out of the answer"
    assert len(final.answer) > 280, "the old character cap is still deciding what got said"


def test_a_wider_budget_does_not_let_an_ungrounded_figure_into_the_answer():
    """The breadth fix must not become a licence: a number no page stated still comes out."""
    report = VerifierReport(
        job_id="j",
        round=1,
        verdicts=[verdict("Germany had the highest renter share in the EU in 2024.", "clm_a")],
        confidence=Confidence.MODERATE,
        answer=FOUR_SUBJECTS,
        why="the pages we opened state this",
    )
    final = build_final_answer(report, [], 1, question=BREADTH)
    assert "384,000" not in final.answer, "a figure no settled claim or page states must not be printed"


def test_the_old_cap_would_have_dropped_two_of_the_four():
    """Pins the direction: the widening is what makes the difference, not a different answer."""
    cut = tiny(FOUR_SUBJECTS, max_sentences=3, max_chars=280)
    assert "Netherlands" in cut
    assert "Italy" not in cut


def test_a_yes_no_check_stays_two_sentences_however_much_evidence_there_is():
    shown = render_truth("TRUE", question=SINGLE, answer=FOUR_SUBJECTS)
    assert len(shown) <= 300
    assert len(shown.split(". ")) <= 3, "a verdict question is not owed a four-country essay"


def test_widening_the_answer_does_not_widen_the_verdict():
    """More prose must not become more confidence: an unadjudicated claim is still unadjudicated."""
    report = VerifierReport(
        job_id="j",
        round=1,
        verdicts=[
            verdict("Germany had the highest renter share in the EU in 2024.", "clm_a"),
            verdict("Italy saw a persistent contraction in housing investment.", "clm_b", ClaimStatus.NOT_REVIEWED),
        ],
        confidence=Confidence.MODERATE,
        answer=FOUR_SUBJECTS,
        why="the pages we opened state this",
    )
    final = build_final_answer(report, [], 1, question=BREADTH)
    assert final.confidence is Confidence.MODERATE
    unjudged = [v for v in report.verdicts if v.verdict is ClaimStatus.NOT_REVIEWED]
    assert unjudged and unjudged[0].confidence is not Confidence.HIGH
