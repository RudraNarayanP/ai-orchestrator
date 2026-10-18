"""A leading "No" that determines a noun is not a verdict.

Live (2026-10-18, eval_20261018_044504, Case C-703/25 P). The curator answered the question "Will
the Court of Justice uphold the EU-US Data Privacy Framework in Latombe's appeal?" with:

    "No one can say whether the Court of Justice will uphold the EU-US Data Privacy Framework in
     Case C-703/25 P. The appeal is pending; no final judgment has been delivered."

The user read:

    "Nah, that doesn't work like that. One can say whether the Court of Justice will uphold the
     EU-US Data Privacy Framework in Case C-703/25 P."

The word "No" was consumed as the yes/no lead, so the answer asserted the exact opposite of the
evidence -- that the outcome *can* be stated -- and then denied it in the next sentence. Deleting a
negation is the worst corruption this layer can commit, because nothing downstream can see it.

Fixing that alone flipped the failure rather than removing it: with no lead, the ledger's TRUE
rendered "Yeah, you're right." on a future ruling nobody can predict. Both halves are pinned here,
because the rule is "an open question never gets a yes or a no", not "a bare 'no' is a denial".
"""

from __future__ import annotations

from backend.models import ClaimStatus, ClaimVerdict, Confidence, VerifierReport
from backend.research.style import premise_state, render_truth, verdict_lead
from backend.verification.verifier import build_final_answer

APPEAL = "Will the Court of Justice uphold the EU-US Data Privacy Framework in Latombe's appeal, Case C-703/25 P?"
NO_ONE_CAN_SAY = (
    "No one can say whether the Court of Justice will uphold the EU-US Data Privacy Framework in Case "
    "C-703/25 P. The appeal is pending; no final judgment has been delivered."
)


# ------------------------------------------------------------------ the lead reader


def test_no_one_can_say_is_not_read_as_a_denial():
    assert not verdict_lead(NO_ONE_CAN_SAY)


def test_a_real_denial_is_still_read_as_a_denial():
    assert verdict_lead("No. The framework was upheld on appeal.")
    assert verdict_lead("No, that is out of date.")
    assert verdict_lead("Nah, the Act says the opposite.")
    assert verdict_lead("Yes, it came into force in 2018.")


def test_no_followed_by_a_noun_keeps_its_negation():
    """The determiner reading covers the shapes that actually appear in a legal answer."""
    for line in (
        "No one can say how the court will rule.",
        "No court has decided the point yet.",
        "No such case exists in the reports.",
        "No provision of the Act addresses it.",
        "No evidence was produced on that date.",
    ):
        assert not verdict_lead(line), line


def test_premise_state_does_not_flip_true_to_false_on_a_quantified_no():
    assert premise_state("TRUE", APPEAL, NO_ONE_CAN_SAY) == "OPEN", "an open question is never a yes or a no"


# ------------------------------------------------------------------ what reaches the user


def report() -> VerifierReport:
    return VerifierReport(
        job_id="j",
        round=1,
        verdicts=[
            ClaimVerdict(
                claim_id="clm_a",
                claim="The appeal in Case C-703/25 P is pending and no final judgment has been delivered.",
                verdict=ClaimStatus.SUPPORTED,
                confidence=Confidence.HIGH,
                reasoning="the Official Journal notice says so",
                strong_evidence=["https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:62025CN0703"],
            )
        ],
        confidence=Confidence.HIGH,
        answer=NO_ONE_CAN_SAY,
        why="the Official Journal notice states the appeal is pending",
    )


def test_the_answer_never_asserts_that_the_outcome_can_be_predicted():
    final = build_final_answer(report(), [], 1, question=APPEAL)
    assert "One can say" not in final.answer, final.answer
    assert "No one can say" in final.answer or "isn't settled" in final.answer, final.answer


def test_an_open_question_is_not_answered_with_a_confident_yes_or_no():
    final = build_final_answer(report(), [], 1, question=APPEAL)
    assert final.truth_state == "OPEN", final.truth_state
    assert not final.answer.lower().startswith(("yeah", "yes", "nah", "no ")), final.answer


def test_render_truth_on_an_open_state_says_so_and_adds_no_emoji():
    shown = render_truth("OPEN", question=APPEAL, answer=NO_ONE_CAN_SAY)
    assert shown.startswith("That one isn't settled yet.")
    assert "✅" not in shown and "❌" not in shown
