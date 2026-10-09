"""The user-facing truth states: TRUE / PARTLY / FALSE / CONFLICT / UNVERIFIED in 1-3 casual sentences.

The state is read off the evidence ledger's verdicts (verifier.truth_state); the wording never decides it, and
uncertainty is never rounded up."""

from __future__ import annotations

import pytest

from backend.models import ClaimStatus, ClaimVerdict, Confidence, VerifierReport
from backend.research.style import (
    BANNED_PHRASES,
    TRUTH_CONFLICT,
    TRUTH_UNVERIFIED,
    count_emojis,
    is_claim_check,
    render_truth,
    split_sentences,
    voice_report,
)
from backend.verification.verifier import build_final_answer, truth_state

CHECK = "Bro, is the minimum age to consent online in the UK 13?"
WH = "What is the minimum age to consent online in the UK?"


def verdict(status, claim="The minimum age is 13.", problems=()):
    return ClaimVerdict(claim_id="c", claim=claim, verdict=status, confidence=Confidence.HIGH, reasoning="r", problems=list(problems))


def report(*verdicts, confidence=Confidence.HIGH, answer="It is 13 under section 9.", disagreement=None):
    return VerifierReport(job_id="j", verdicts=list(verdicts), answer=answer, confidence=confidence, important_disagreement=disagreement)


# ---- the state comes from the ledger
@pytest.mark.parametrize(
    "rep,expected",
    [
        (report(verdict(ClaimStatus.SUPPORTED)), "TRUE"),
        (report(verdict(ClaimStatus.SUPPORTED), verdict(ClaimStatus.REFUTED, "The fine is 4,500.")), "PARTLY"),
        (report(verdict(ClaimStatus.PARTIALLY_SUPPORTED)), "PARTLY"),
        (report(verdict(ClaimStatus.REFUTED)), "FALSE"),
        (report(verdict(ClaimStatus.CONTESTED)), "CONFLICT"),
        (report(verdict(ClaimStatus.SUPPORTED), disagreement="two official pages differ"), "CONFLICT"),
        (report(verdict(ClaimStatus.INSUFFICIENT_EVIDENCE)), "UNVERIFIED"),
        (report(verdict(ClaimStatus.SUPPORTED), confidence=Confidence.LOW), "UNVERIFIED"),
        (report(confidence=Confidence.NONE), "UNVERIFIED"),
    ],
)
def test_truth_state_is_read_from_verdicts_not_tone(rep, expected):
    assert truth_state(rep) == expected


def test_a_cheerful_answer_text_cannot_turn_a_refuted_ledger_into_agreement():
    rep = report(verdict(ClaimStatus.REFUTED), answer="Yes, absolutely, you're totally right!")
    final = build_final_answer(rep, [], 1, CHECK)
    assert final.truth_state == "FALSE" and final.answer.startswith("Nah, that doesn't work like that")


# ---- wording
def test_agreement_is_two_words_of_substance():
    assert render_truth("TRUE", question="Is it 13?", answer="Yes. It is 13 under section 9 of the Data Protection Act 2018.").startswith("Yeah, you're right.")


def test_partly_names_the_aspect_and_the_corrected_fact():
    out = render_truth("PARTLY", question="Is the fine 3,000 and the age 13?", answer="The age is 13, but the maximum fine is 4,500 pounds.", aspect="number")
    assert out.startswith("Yeah, the idea is right, but the number part is a bit off")
    assert "4,500" in out or "age is 13" in out
    assert len(split_sentences(out)) <= 3


def test_wrong_is_plain():
    out = render_truth("FALSE", question="Does the Act ban VPNs?", answer="The Act doesn't ban VPNs.")
    assert out.startswith("Nah, that doesn't work like that.")


def test_conflict_and_unverified_are_fixed_lines_and_never_reworded():
    for q in (CHECK, WH):
        assert render_truth("CONFLICT", question=q, answer="It is 13, obviously.") == TRUTH_CONFLICT
        assert render_truth("UNVERIFIED", question=q, answer="It is 13, obviously.") == TRUTH_UNVERIFIED
    assert TRUTH_CONFLICT == "Hmm, I'm not sure \u2014 the sources disagree." and TRUTH_UNVERIFIED == "Couldn't verify that one."


def test_a_what_question_keeps_its_content_answer_instead_of_yeah_youre_right():
    assert not is_claim_check(WH) and is_claim_check(CHECK)
    out = render_truth("TRUE", question=WH, answer="It's 13 under section 9 of the Data Protection Act 2018.")
    assert out.startswith("It's 13") and "you're right" not in out


def test_casual_tone_is_matched_mildly_and_only_on_confident_states():
    assert render_truth("TRUE", question=CHECK, answer="Yes.").startswith("Yeah, you're right, bro.")
    assert "bro" not in render_truth("UNVERIFIED", question=CHECK).lower()
    assert "bro" not in render_truth("CONFLICT", question=CHECK).lower()
    assert "bro" not in render_truth("TRUE", question="Is it 13?", answer="Yes.").lower()


# ---- the contract: short, no filler, emoji only at the end of confident answers
@pytest.mark.parametrize("state", ["TRUE", "PARTLY", "FALSE", "CONFLICT", "UNVERIFIED"])
def test_every_state_is_short_and_free_of_banned_filler(state):
    long_answer = "It is 13. " + "This was established after a long review of many pages and several providers. " * 12
    out = render_truth(state, question=CHECK, answer=long_answer, aspect="date")
    assert len(split_sentences(out)) <= 3 and len(out) <= 280
    low = out.lower()
    assert not any(p.lower() in low for p in BANNED_PHRASES)
    for filler in ("as an ai", "important to note", "i cannot guarantee", "%", "round", "provider"):
        assert filler not in low
    assert voice_report(out)["ok"], voice_report(out)


def test_length_does_not_scale_with_internal_work():
    small = report(verdict(ClaimStatus.SUPPORTED))
    big = report(*[verdict(ClaimStatus.SUPPORTED, claim=f"Claim number {i} about the Act.") for i in range(40)], answer="It is 13. " + "More detail. " * 50)
    assert len(build_final_answer(small, [], 1, CHECK).answer) <= 60
    assert len(build_final_answer(big, [], 1, CHECK).answer) <= 60


def test_emoji_only_at_the_end_of_confident_answers():
    t = build_final_answer(report(verdict(ClaimStatus.SUPPORTED)), [], 1, CHECK).answer
    f = build_final_answer(report(verdict(ClaimStatus.REFUTED)), [], 1, CHECK).answer
    assert count_emojis(t) == 1 and t.rstrip().endswith("\u2705")
    assert count_emojis(f) == 1 and f.rstrip().endswith("\u274C")
    for rep in (report(verdict(ClaimStatus.CONTESTED)), report(verdict(ClaimStatus.INSUFFICIENT_EVIDENCE), confidence=Confidence.NONE),
                report(verdict(ClaimStatus.PARTIALLY_SUPPORTED))):
        assert count_emojis(build_final_answer(rep, [], 1, CHECK).answer) == 0


def test_without_a_question_the_old_path_is_unchanged():
    out = build_final_answer(report(verdict(ClaimStatus.SUPPORTED)), [], 1)
    assert out.truth_state == "" and out.answer.startswith("It is 13")

# ---- the user's claim vs the answer's claim (live: res-ten-percent-brain)
TEN = "Is it true that humans only use 10 percent of their brains?"


def test_supported_denial_of_the_users_claim_is_shown_as_false_not_agreement():
    rep = report(verdict(ClaimStatus.SUPPORTED, "Humans do not use only 10% of their brains."), answer="Humans do not use only 10% of their brains; they use all of it, though not all at once.")
    final = build_final_answer(rep, [], 1, TEN)
    assert final.truth_state == "FALSE"
    assert final.answer.startswith("Nah, that doesn't work like that")
    assert "you're right" not in final.answer.lower()


def test_a_leading_no_also_flips_a_supported_answer():
    rep = report(verdict(ClaimStatus.SUPPORTED), answer="No. The minimum age is 13 under section 9.")
    assert build_final_answer(rep, [], 1, "Is the minimum age 16?").truth_state == "FALSE"


def test_agreement_is_untouched_when_the_answer_affirms_or_the_question_is_negative():
    yes = report(verdict(ClaimStatus.SUPPORTED), answer="It is 13 under section 9.")
    assert build_final_answer(yes, [], 1, CHECK).truth_state == "TRUE"
    neg = report(verdict(ClaimStatus.SUPPORTED, "The Act does not ban VPNs."), answer="The Act does not ban VPNs.")
    assert build_final_answer(neg, [], 1, "Is it true the Act doesn't ban VPNs?").truth_state == "TRUE"


def test_uncertain_states_are_never_flipped_by_the_premise_check():
    from backend.research.style import premise_state

    for state in ("UNVERIFIED", "CONFLICT", "PARTLY", "FALSE"):
        assert premise_state(state, TEN, "Humans do not use only 10%.") == state

def test_appended_fact_starts_with_a_capital():
    out = render_truth("FALSE", question=TEN, answer="the claim that humans use only 10% of their brains is a myth.")
    assert out.startswith("Nah, that doesn't work like that. The claim")
    assert render_truth("TRUE", question=CHECK, answer="it is 13 under section 9.").endswith("It is 13 under section 9.")


def test_a_wh_question_after_a_leading_phrase_is_not_a_claim_check():
    """Live (uk-dpa-age, 2026-10-08): the answer opened "Yeah, you're right." for a "what is" question."""
    q = "Under the Data Protection Act 2018, what is the minimum age at which a child can consent to information society services in the UK?"
    assert not is_claim_check(q)
    out = render_truth("TRUE", question=q, answer="13 years old, under section 9 of the Data Protection Act 2018.")
    assert not out.startswith("Yeah") and "13" in out
    assert not is_claim_check("In 1889, how tall was the Eiffel Tower?")
    assert not is_claim_check("According to the ICO; who can give consent online?")
    assert is_claim_check("Is the Eiffel Tower taller than the Statue of Liberty, which is in New York?")
    assert is_claim_check("Under the DPA 2018 the age is 13, right?")
    assert is_claim_check("Is it true that, under the DPA, the age is 13?")


def test_a_wh_question_opening_with_a_preposition_is_not_a_claim_check():
    """Live (eiffel-year, 2026-10-09): "In which year was the Eiffel Tower completed?" got
    "Yeah, the idea is right, but the number part is a bit off" -- a claim correction aimed
    at a question that asserted nothing."""
    q = "In which year was the Eiffel Tower completed?"
    assert not is_claim_check(q)
    out = render_truth("PARTLY", question=q, answer="The Eiffel Tower was completed in 1889.", aspect="figure")
    assert not out.startswith("Yeah"), out
    assert "1889" in out, out
    for wh in (
        "On what date did the Act receive Royal Assent?",
        "By how much did the fee rise?",
        "Under which section is the deadline set?",
        "For how long does the exemption last?",
    ):
        assert not is_claim_check(wh), wh
    # a check that merely happens to open with a preposition is still a check
    assert is_claim_check("In 2024, did Parliament lower the age?")
    assert render_truth(
        "PARTLY", question="The tower was finished in 1887, right?", answer="It was completed in 1889.", aspect="date"
    ).startswith("Yeah, the idea is right")
