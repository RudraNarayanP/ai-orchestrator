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
from backend.research.style import premise_main, premise_state, render_truth, verdict_lead
from backend.verification.verifier import _aspect, build_final_answer

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


# ------------------------------------------- the premise of a conditional question is its main clause
#
# Live (2026-10-18, eval_20261018_044504, lg-gpl-internal-use). The question asks whether GPLv3
# obliges publishing source *if* the program is never distributed. The curator answered
# "No. GPLv3 does not require a company to publish ...". The answer reached the user as
# "Yeah, you're right." -- the tail splitter needed a comma before "if", so the condition's "never"
# stayed in the clause treated as the premise, the agreement check decided the premise was itself
# negative, and switched itself off.

GPL = (
    "Does the GNU GPL version 3 oblige a company to publish the source of its modified version "
    "if the program is only used inside the company and never given to anyone outside it?"
)
GPL_ANSWER = "No. GPLv3 does not require a company to publish the source code of a modified program that is used only internally."


def test_a_condition_never_belongs_to_the_premise():
    main = premise_main(GPL)
    assert "never" not in main, main
    assert main.startswith("Does the GNU GPL version 3 oblige")


def test_a_prepositional_without_is_part_of_the_premise_and_stays():
    """The opposite case, already decided: "training without a licence" asks about an absent licence."""
    assert "without a licence" in premise_main(
        "Is training a generative AI model on copyrighted books without a licence fair use under US law?"
    )


def test_a_conditional_question_denied_by_the_reviewer_leads_with_the_denial():
    assert premise_state("TRUE", GPL, GPL_ANSWER) == "FALSE"


def test_the_gpl_answer_reaches_the_user_as_a_no():
    report = VerifierReport(
        job_id="j",
        round=1,
        verdicts=[
            ClaimVerdict(
                claim_id="clm_g",
                claim="GPLv3 imposes no source-publishing duty on software that is never distributed outside the company.",
                verdict=ClaimStatus.SUPPORTED,
                confidence=Confidence.HIGH,
                reasoning="the FSF text on the copyleft terms says so",
                strong_evidence=["https://www.gnu.org/licenses/gpl-faq.html"],
            )
        ],
        confidence=Confidence.HIGH,
        answer=GPL_ANSWER,
        why="the license text conditions the duty on distribution",
    )
    final = build_final_answer(report, [], 1, question=GPL)
    assert final.truth_state == "FALSE", final.truth_state
    assert final.answer.lower().startswith(("no", "nah", "nope")), final.answer


# ------------------------------------------------------ an aspect must be something the user put in play
#
# Live (2026-10-18, lg-ai-training-fair-use): the question asked no number, and the answer opened
# "Yeah, the idea is right, but the number part is a bit off" -- because the supporting claim
# happened to contain "Section 107". A digit in our own evidence is not a number the user claimed.

FAIR_USE = "Is training a generative AI model on copyrighted books without a licence fair use under US law?"
SECTION_107 = "Section 107 requires a case-specific four-factor assessment; two 2025 rulings found fair use on their records."


def test_a_digit_in_our_evidence_is_not_a_number_the_user_claimed():
    report = VerifierReport(
        job_id="j", round=1,
        verdicts=[ClaimVerdict(
            claim_id="clm_f", claim=SECTION_107, verdict=ClaimStatus.PARTIALLY_SUPPORTED,
            confidence=Confidence.LOW, reasoning="the statute conditions it", problems=["figure_absent"],
        )],
        confidence=Confidence.MODERATE, answer=SECTION_107, why="section 107 text",
    )
    assert _aspect(report, FAIR_USE) != "number", "no number was asked, so none can be 'off'"
    final = build_final_answer(report, [], 1, question=FAIR_USE)
    assert "number part" not in final.answer, final.answer


def test_a_date_aspect_is_still_named_when_the_user_did_ask_a_date():
    report = VerifierReport(
        job_id="j", round=1,
        verdicts=[ClaimVerdict(
            claim_id="clm_d", claim="The Act commenced on 25 May 2018.", verdict=ClaimStatus.PARTIALLY_SUPPORTED,
            confidence=Confidence.LOW, reasoning="one page says it", problems=["figure_absent"],
        )],
        confidence=Confidence.MODERATE, answer="The Act commenced on 25 May 2018.", why="a page",
    )
    assert _aspect(report, "When did the Act come into force?") == "date"


def test_an_unnamed_aspect_reads_as_english_not_the_detail_part():
    assert "the detail part" not in render_truth("PARTLY", question=FAIR_USE, answer=SECTION_107, aspect="detail")
    assert "one detail" in render_truth("PARTLY", question=FAIR_USE, answer=SECTION_107, aspect="detail")
    assert "the date part" in render_truth("PARTLY", question=FAIR_USE, answer=SECTION_107, aspect="date")
