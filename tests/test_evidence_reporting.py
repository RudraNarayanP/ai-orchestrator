"""REPORT EVIDENCE, DO NOT INVENT OR LECTURE: the final-answer lint and the prompts that ask for it."""

from __future__ import annotations

import pytest

from backend.models import ClaimStatus, ClaimVerdict, Confidence, VerifierReport
from backend.research import style
from backend.research.lint import (
    UNKNOWN_PHRASES,
    asks_for_advice,
    asks_for_inference,
    check_structure,
    evidence_report,
    final_check,
    is_what_do_we_know,
    lint_answer,
)
from backend.verification.verifier import build_final_answer

# ---- the tuition example --------------------------------------------------------------------
BAD_TUITION = (
    "The sources I found don't say who paid Rahul's tuition. "
    "It's important to note that this doesn't prove his parents didn't pay. "
    "They probably didn't pay, so he likely took a loan. "
    "Perhaps because the family was short of money. "
    "You shouldn't assume anything either way, and it would be prudent to ask him directly."
)
GOOD_TUITION = "The sources I found don't say who paid Rahul's tuition."


def test_tuition_bad_answer_is_cut_back_to_the_good_one():
    res = lint_answer(BAD_TUITION, question="who paid Rahul's tuition?")
    assert res.text == GOOD_TUITION
    assert res.codes >= {"C_unknown_made_negative", "D_invented_explanation", "E_unrequested_advice", "F_boilerplate"}


def test_tuition_good_answer_is_untouched():
    res = lint_answer(GOOD_TUITION, question="who paid Rahul's tuition?")
    assert res.text == GOOD_TUITION and not res.findings


def test_unknown_is_never_turned_into_a_negative_even_when_stated_flatly():
    res = lint_answer("His parents did not pay the tuition.", question="who paid Rahul's tuition?", unknowns=["His parents paid the tuition"])
    assert "did not pay" not in res.text and "couldn't find that documented" in res.text
    assert "C_unknown_made_negative" in res.codes


def test_documented_and_undocumented_structure_survives_the_lint():
    text = ("Documented: the offer was rejected on 3 May 2026 (the user's own statement).\n"
            "Documented: tuition was GBP 25,000 (university fee page).\n"
            "Not documented: who would have paid it \u2014 I couldn't find that documented.")
    res = lint_answer(text, question="what do we actually know?", unknowns=["who would have paid it"])
    assert res.text == text and not res.findings


# ---- the obvious-inference example ----------------------------------------------------------
def test_obvious_contextual_inference_is_not_challenged():
    q = "My dad paid for my school, so I assume he can afford coaching too. How much is Allen coaching for class 11?"
    bad = "Allen's class 11 programme costs about INR 1.2 lakh a year. You shouldn't assume your dad can afford it just because he paid for school."
    res = lint_answer(bad, question=q)
    assert res.text == "Allen's class 11 programme costs about INR 1.2 lakh a year."


def test_a_real_factual_error_is_still_corrected():
    q = "Since Allen is in Delhi, how far is Delhi from Kota?"
    answer = "Allen's main centre is in Kota, not Delhi. Kota is about 465 km from Delhi."
    assert lint_answer(answer, question=q).text == answer


# ---- the lint follows the question ------------------------------------------------------------
@pytest.mark.parametrize("q,expected", [("Should I take the loan?", True), ("what do you recommend", True), ("who paid the tuition?", False)])
def test_advice_requests_are_recognised(q, expected):
    assert asks_for_advice(q) is expected


@pytest.mark.parametrize("q,expected", [("what do you think, could they afford it?", True), ("is it fair to say they were poor?", True), ("when did it happen?", False)])
def test_inference_requests_are_recognised(q, expected):
    assert asks_for_inference(q) is expected


def test_advice_is_allowed_when_asked_for():
    ans = "I'd recommend comparing the interest rates before you sign."
    assert lint_answer(ans, question="Should I take the loan?").text == ans


def test_advice_is_removed_when_not_asked_for():
    res = lint_answer("The loan's interest rate is 9.5%. I'd recommend comparing the interest rates before you sign.", question="What is the loan's interest rate?")
    assert res.text == "The loan's interest rate is 9.5%." and "E_unrequested_advice" in res.codes


def test_inference_is_allowed_when_asked_for():
    ans = "It's fair to conclude they could not afford it. That suggests they did not pay."
    assert lint_answer(ans, question="what do you think, could they afford it?").text == ans


def test_what_do_we_know_is_recognised():
    for q in ("what do we actually know about the offer?", "What's actually known about this?", "what do the records show"):
        assert is_what_do_we_know(q)
    assert not is_what_do_we_know("what can be proven beyond doubt?")


# ---- boilerplate ---------------------------------------------------------------------------------
@pytest.mark.parametrize("lead", ["It's important to note that", "It is important to note that", "It's worth mentioning that", "Worth mentioning:",
                                  "That being said,", "Ultimately,", "Keep in mind that", "Importantly,"])
def test_boilerplate_lead_ins_are_stripped_and_the_fact_stays(lead):
    res = lint_answer(f"{lead} the Act was passed in 2018.", question="when was the Act passed?")
    assert res.text == "The Act was passed in 2018." and "F_boilerplate" in res.codes


@pytest.mark.parametrize("filler", [
    "We cannot definitively say what happened.", "There are several possible explanations.", "This highlights the importance of due diligence.",
    "It would be prudent to double-check.",
])
def test_filler_sentences_are_dropped(filler):
    assert lint_answer(f"The fee was GBP 25,000. {filler}", question="what was the fee?").text == "The fee was GBP 25,000."


@pytest.mark.parametrize("line", [*UNKNOWN_PHRASES, style.TRUTH_UNVERIFIED, style.TRUTH_CONFLICT, "I couldn't verify that reliably.", "I don't know.",
                                  "I couldn't find a public source documenting that.", "The available sources don't specify that."])
def test_honest_uncertainty_is_never_weakened(line):
    res = lint_answer(line, question="who paid?")
    assert res.text == line and not res.findings


def test_a_lint_that_would_empty_the_answer_keeps_the_original():
    res = lint_answer("You shouldn't assume anything.", question="x")
    assert res.text.strip() and "G_longer_than_needed" in res.codes


def test_banned_phrases_cover_the_new_list():
    for p in ("It's important to note", "worth mentioning", "That being said", "Ultimately", "keep in mind", "We cannot definitively say",
              "several possible explanations", "highlights the importance of", "prudent to"):
        assert style.scrub(f"x. {p} y")[1], p
        assert not style.voice_report(f"The fee is 5. {p} it.")["ok"], p


# ---- structure ------------------------------------------------------------------------------------
def test_evidence_report_keeps_documented_and_undocumented_apart():
    r = evidence_report([("Parents spent about INR 4 lakh on Allen coaching", "the user's statement"), ("Land bought, roughly INR 50 lakh", "the user's statement")],
                        ["Who paid the university tuition"], ["The year the land was bought"])
    assert r.splitlines()[0] == "Documented:"
    assert "Not documented:" in r and r.count("I couldn't find that documented.") == 1
    assert "probably" not in r and "likely" not in r
    assert "The sources disagree on:" in r


def test_check_structure_flags_an_unknown_stated_as_a_claim_and_a_lost_fact():
    flags = check_structure("The parents paid the tuition. Land was bought.", ["Parents spent INR 4 lakh on coaching"], ["parents paid the tuition"])
    assert {f.code for f in flags} == {"B_structure"} and len(flags) == 2


def test_final_check_reports_each_question():
    bad = final_check(BAD_TUITION, question="who paid Rahul's tuition?", undocumented=["who paid Rahul's tuition"])
    assert bad.unknown_turned_negative and bad.invented_explanation and bad.unrequested_advice and bad.useless_disclaimer and not bad.ok
    assert bad.text == GOOD_TUITION
    good = final_check(GOOD_TUITION, question="who paid Rahul's tuition?", undocumented=["who paid Rahul's tuition"])
    assert good.ok and good.answered and good.facts_and_unknowns_separated
    assert final_check("x " * 500, question="what?").shorter_possible


# ---- wired into the final answer ---------------------------------------------------------------
def _report(answer, verdicts, conf=Confidence.MODERATE, **kw):
    return VerifierReport(job_id="j", verdicts=verdicts, answer=answer, confidence=conf, **kw)


def _v(claim, status, evidence=()):
    return ClaimVerdict(claim_id="c1", claim=claim, verdict=status, confidence=Confidence.MODERATE, reasoning="r", strong_evidence=list(evidence))


def test_final_answer_drops_lecture_and_keeps_the_fact():
    rep = _report("The fee was GBP 25,000. It's important to note that you shouldn't assume it was paid in full. Perhaps because of a scholarship.",
                  [_v("The fee was GBP 25,000", ClaimStatus.SUPPORTED, ["https://leeds.ac.uk/fees"])])
    out = build_final_answer(rep, [], 1, question="")
    assert out.answer.startswith("The fee was GBP 25,000.") and "assume" not in out.answer and "scholarship" not in out.answer


def test_final_answer_why_and_caveats_are_linted_too():
    rep = _report("The fee was GBP 25,000.", [_v("The fee was GBP 25,000", ClaimStatus.SUPPORTED)],
                  why="The university fee page lists it. Keep in mind that you should be careful about fees changing.")
    out = build_final_answer(rep, [], 1, question="")
    assert out.why == "The university fee page lists it."


def test_unverified_still_says_couldnt_verify_that_one():
    rep = _report("They probably didn't pay.", [_v("Parents paid the tuition", ClaimStatus.UNVERIFIED)], conf=Confidence.NONE)
    out = build_final_answer(rep, [], 1, question="did his parents pay the tuition?")
    assert out.answer == style.TRUTH_UNVERIFIED and out.truth_state == "UNVERIFIED"


def test_what_do_we_know_gives_documented_facts_then_undocumented():
    rep = _report("whatever", [
        _v("The offer was rejected on 3 May 2026", ClaimStatus.SUPPORTED, ["https://example.org/letter"]),
        _v("Tuition was GBP 25,000", ClaimStatus.SUPPORTED, ["https://www.leeds.ac.uk/fees"]),
        _v("His parents paid the tuition", ClaimStatus.UNVERIFIED),
    ])
    out = build_final_answer(rep, [], 1, question="what do we actually know about the offer?")
    lines = out.answer.splitlines()
    assert lines[0] == "Documented:" and "(example.org)" in out.answer and "(leeds.ac.uk)" in out.answer
    assert "Not documented:" in out.answer and "His parents paid the tuition \u2014 I couldn't find that documented." in out.answer
    assert "probably" not in out.answer.lower()


# ---- the prompts ask for it too ------------------------------------------------------------------------
def test_prompts_carry_the_rule():
    for p in (style.style_prompt(), style.style_prompt(include_verifier=True), style.VERIFIER_ROLE):
        assert "REPORT EVIDENCE. DO NOT INVENT OR LECTURE." in p
        assert "Absence of evidence is not evidence of absence" in p
        assert "probably didn't pay" in p
    from backend.research.prompts import ANTI_INJECTION

    assert "couldn't find that documented" in ANTI_INJECTION
