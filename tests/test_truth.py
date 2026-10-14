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


def test_open_claims_do_not_rewrite_a_supported_informational_answer():
    q = "Under the Data Protection Act 2018, what is the minimum age at which a child can consent to information society services in the UK?"
    rep = report(
        verdict(ClaimStatus.SUPPORTED, "The minimum age is 13 years."),
        verdict(ClaimStatus.NOT_REVIEWED, "Section 9 was omitted from the Data Protection Act 2018 after Brexit."),
        answer=(
            "Under the UK Data Protection Act 2018, a child can consent to information society services at age 13; "
            "the UK does not apply the GDPR default age of 16. The operative provision is section 9 of the "
            "Data Protection Act 2018, which substitutes 13 years for 16 years in Article 8(1) of the GDPR."
        ),
    )
    final = build_final_answer(rep, [], 3, q)
    assert "13" in final.answer
    assert "couldn't find that documented" not in final.answer.lower()


def test_refuting_a_hypothetical_figure_in_a_what_question_is_not_false_premise():
    q = (
        "What is the UK minimum age for ISS consent, and does the UK apply the GDPR default age of 16 "
        "or a different age?"
    )
    rep = report(
        verdict(ClaimStatus.SUPPORTED, "The minimum age is 13 years."),
        verdict(ClaimStatus.REFUTED, "The UK applies the GDPR default age of 16 for ISS consent."),
    )
    assert truth_state(rep, q) == "PARTLY"


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

    # PARTLY is no longer in this list: a partly supported answer that DENIES the premise leads with "Nah" (2026-10-12 legal run), see below.
    for state in ("UNVERIFIED", "CONFLICT", "FALSE"):
        assert premise_state(state, TEN, "Humans do not use only 10%.") == state

def test_appended_fact_starts_with_a_capital():
    out = render_truth("FALSE", question=TEN, answer="the claim that humans use only 10% of their brains is a myth.")
    assert out.startswith("Nah, that doesn't work like that. The claim")
    assert render_truth("TRUE", question=CHECK, answer="it is 13 under section 9.").endswith("It is 13 under section 9.")


HIPAA_LIABILITY_Q = (
    "A proprietary foundation model (Company A) is fine-tuned by Company B on patient records under a HIPAA BAA. "
    "Synthetic inference output leaks PHI to a third party via an API. "
    "Under a standard US commercial supply chain framework, who holds primary legal and regulatory liability "
    "for the PHI breach: (1) Company A, (2) Company B, or (3) does liability fracture into an unallocated "
    "black-box gap because the data was emergent?"
)


def test_a_long_scenario_with_who_holds_primary_liability_is_not_a_claim_check():
    """Live (job_261011002248): headline was only 'Yeah, you're right.'"""
    assert not is_claim_check(HIPAA_LIABILITY_Q)


def test_supported_allocation_answer_is_not_replaced_by_yeah_youre_right():
    q = HIPAA_LIABILITY_Q
    body = (
        "Company B, as covered entity and business associate, bears primary HIPAA breach-notification and "
        "safeguard duties for PHI it put into the fine-tuned system; Company A's role depends on its BAA and "
        "subcontractor status, and emergent synthetic output does not create an unallocated regulatory gap."
    )
    rep = report(verdict(ClaimStatus.SUPPORTED, "Company B bears primary HIPAA duties for the disclosed PHI."), answer=body)
    final = build_final_answer(rep, [], 1, q)
    assert final.truth_state == "TRUE"
    assert "you're right" not in final.answer.lower()
    assert "company b" in final.answer.lower()


def test_does_company_b_hold_liability_stays_a_claim_check():
    assert is_claim_check("Does Company B hold primary liability for this HIPAA breach?")


def test_replay_stored_hipaa_job_headline_is_not_agreement_only():
    """Assembly replay of job_261011002248 when the local audit DB is present."""
    from pathlib import Path

    import json
    import sqlite3

    from backend.models import ProviderResponse, ProviderStatus, VerifierReport

    db = Path(__file__).resolve().parents[1] / "data" / "omnibrain.db"
    if not db.exists():
        pytest.skip("no local omnibrain.db")
    jid = "job_261011002248_b5871cb3"
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    row = c.execute("SELECT question, rounds_run FROM jobs WHERE id=?", (jid,)).fetchone()
    if not row:
        pytest.skip("HIPAA smoke job not stored")
    rep = VerifierReport.model_validate(
        json.loads(c.execute("SELECT report_json FROM reports WHERE job_id=? ORDER BY round DESC LIMIT 1", (jid,)).fetchone()["report_json"])
    )
    responses = [
        ProviderResponse(
            job_id=jid,
            provider="chatgpt",
            round=1,
            role="primary",
            status=ProviderStatus.COMPLETED,
            prompt="",
            answer_text="",
        )
    ]
    final = build_final_answer(rep, responses, row["rounds_run"], row["question"])
    assert "you're right" not in final.answer.lower()
    assert len(final.answer.strip()) > 40


def test_a_mixed_premise_check_and_wh_question_stays_a_claim_check():
    q = "Is it true that UK law uses 16 as the minimum age for ISS consent, and what provision establishes the age?"
    assert is_claim_check(q)


def test_a_multi_part_what_and_yesno_question_is_not_a_claim_check():
    """Live (job_261010234454): wh + 'does the UK apply 16…' was misread as a premise check -> 'Nah, …'."""
    q = (
        "Under the Data Protection Act 2018, what is the minimum age at which a child can consent to "
        "information society services in the UK, and does the UK apply the GDPR default age of 16 for "
        "that purpose or a different age? Name the operative statutory provision."
    )
    assert not is_claim_check(q)


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


# ---- 2026-10-12 legal run: the lead follows what the settled answer says about the user's premise
from backend.research.style import premise_state as _premise


def test_a_partly_supported_answer_that_denies_the_premise_leads_with_no():
    q = "As of October 2026, are all the substantive obligations of the Act already in force?"
    a = "As of 12 October 2026, the substantive obligations of the Act are not yet in force."
    assert _premise("PARTLY", q, a) == "FALSE"
    assert render_truth("FALSE", question=q, answer=a).startswith("Nah")


def test_only_the_question_head_clause_sets_its_polarity():
    q = "Does the country enforce awards made in any Convention state, even one the government has not notified?"
    assert _premise("TRUE", q, "The country does not enforce awards from states that were not notified.") == "FALSE"
    assert _premise("TRUE", "Is it true the Act doesn't ban VPNs?", "The Act does not ban VPNs.") == "TRUE"


def test_an_open_legal_question_is_neither_yes_nor_no():
    q = "Is training a model on copyrighted books without a licence fair use under US law?"
    a = "It is not automatically fair use; it depends on the four statutory factors, and no court has yet established a blanket rule."
    assert _premise("PARTLY", q, a) == "OPEN"
    out = render_truth("OPEN", question=q, answer=a)
    assert out.startswith("That one isn't settled yet") and not out.lower().startswith(("yes", "yeah", "no", "nah"))


def test_a_premise_flip_from_partly_keeps_low_confidence():
    rep = report(verdict(ClaimStatus.PARTIALLY_SUPPORTED, "The obligations are not yet in force."), answer="The obligations are not yet in force.")
    fa = build_final_answer(rep, [], 1, "Are all the obligations of the Act already in force?")
    assert fa.answer.startswith("Nah") or fa.truth_state in {"UNVERIFIED", "CONFLICT"}, fa.answer
    assert "\u2705" not in fa.answer, "a flipped PARTLY never gains the high-confidence mark"


def test_a_negation_inside_the_main_clause_still_counts():
    """Regression found while re-checking PASS answers: "can parties choose the law of a country that has NO connection?" --
    the "no" is part of what is asked, so a supported "they may" stays "Yeah"."""
    q = "Under the Rome I Regulation, can the parties to a cross-border business-to-business contract choose the law of a country that has no connection to the contract?"
    a = "Parties may choose the law of a country with no connection to the contract, as the regulation does not require an objective connection."
    assert _premise("TRUE", q, a) == "TRUE"


# ---- 2026-10-12 re-run: the remaining FAILs, as general rules
def test_not_automatically_or_it_depends_never_leads_with_yes():
    q = "Is training a generative AI model on copyrighted books without a licence fair use under US law?"
    a = "Training a generative AI model on copyrighted books without a licence is not automatically fair use under US law; it depends on a case-specific four-factor analysis."
    assert _premise("TRUE", q, a) == "OPEN"
    assert not render_truth("OPEN", question=q, answer=a).lower().startswith(("yeah", "yes", "nah", "no"))


def test_a_question_about_the_law_now_leads_with_the_settled_claim_about_now():
    from backend.verification.verifier import lead_with_the_present

    q = "Which section of the Indian Penal Code, 1860 currently punishes theft?"
    rep = report(
        verdict(ClaimStatus.SUPPORTED, "Section 303 of the Bharatiya Nyaya Sanhita addresses theft under the current criminal law."),
        answer="Section 379 of the Indian Penal Code, 1860 prescribed the punishment for theft.",
    )
    out = lead_with_the_present(rep, q)
    assert out.startswith("Section 303 of the Bharatiya Nyaya Sanhita") and "379" in out, out
    # nothing unsettled is ever promoted
    unsettled = report(verdict(ClaimStatus.INSUFFICIENT_EVIDENCE, "Section 303 of the Bharatiya Nyaya Sanhita addresses theft under the current criminal law."),
                       answer="Section 379 of the Indian Penal Code, 1860 prescribed the punishment for theft.")
    assert lead_with_the_present(unsettled, q).startswith("Section 379")


def test_an_answer_about_a_different_law_is_led_by_the_settled_claim_about_the_named_law():
    from backend.verification.verifier import lead_with_the_named_law

    q = "What penalty does India's Digital India Act impose for a personal data breach?"
    rep = report(
        verdict(ClaimStatus.SUPPORTED, "India's Digital India Act does not have a verified, documented personal-data-breach penalty."),
        answer="The enacted Digital Personal Data Protection Act, 2023 provides penalties of up to 250 crore.",
    )
    out = lead_with_the_named_law(rep, q)
    assert out.startswith("India's Digital India Act does not have"), out
    weak = report(verdict(ClaimStatus.INSUFFICIENT_EVIDENCE, "The Digital India Act is a proposed initiative."),
                  answer="The enacted Digital Personal Data Protection Act, 2023 provides penalties.")
    assert lead_with_the_named_law(weak, q).startswith("The enacted"), "only a settled claim leads"


def test_a_round_ends_at_the_hard_ceiling_even_if_nobody_answered():
    import asyncio as _a
    import time as _t
    from types import SimpleNamespace as _NS
    from backend.orchestrator.runner import ResearchRunner

    runner = object.__new__(ResearchRunner)
    runner.settings = _NS(research=_NS(secondary_cap_s=0.1, secondary_hard_cap_s=0.4))

    async def emit(*a, **k):
        return None

    runner._emit = emit

    async def never():
        await _a.sleep(30)

    started = _t.monotonic()
    out = _a.run(runner._gather_capped([never()], ["le_chat"], 2))
    assert _t.monotonic() - started < 2 and out == [None]


def test_a_denial_in_a_later_sentence_about_the_asked_point_leads_with_no():
    q = "Can Parliament use Article 368 to amend the basic structure of the Indian Constitution?"
    a = ("Article 368 empowers Parliament to amend the Indian Constitution. "
         "Parliament cannot amend the Constitution in a manner that damages or destroys its basic structure.")
    assert _premise("TRUE", q, a) == "FALSE"
    # a denial about something the user did not ask stays a side remark
    assert _premise("TRUE", "Is it true the Eiffel Tower is in Paris?", "The Eiffel Tower is in Paris. It is not in Lyon.") == "TRUE"


def test_the_curator_is_told_to_lead_a_yes_no_question_with_yes_or_no():
    from backend.research.style import ANSWER_CONTRACT

    assert '"Yes" or "No" as the answer\'s first word' in ANSWER_CONTRACT


# ---- 2026-10-13: the headline must answer the asked question (Erie: "Yeah, you're right. Erie ... was decided on April 25, 1938.")
ERIE_Q = ("Under Erie Railroad Co. v. Tompkins, must a US federal court that hears a state-law claim under diversity jurisdiction "
          "apply federal general common law?")
DATE = "Erie Railroad Co. v. Tompkins was decided on April 25, 1938."


def test_an_off_question_headline_is_rebuilt_from_the_settled_claim_with_its_polarity():
    rep = report(verdict(ClaimStatus.SUPPORTED, DATE),
                 verdict(ClaimStatus.SUPPORTED, "A federal court hearing a state-law claim under diversity jurisdiction must not apply federal general common law."),
                 answer=DATE)
    fa = build_final_answer(rep, [], 1, ERIE_Q)
    assert fa.answer.startswith("Nah") and "general common law" in fa.answer, fa.answer


def test_an_off_question_headline_with_nothing_settled_on_the_point_is_couldnt_verify():
    fa = build_final_answer(report(verdict(ClaimStatus.SUPPORTED, DATE), answer=DATE), [], 1, ERIE_Q)
    assert fa.truth_state == "UNVERIFIED" and "you're right" not in fa.answer.lower(), fa.answer


def test_an_on_question_headline_is_left_alone():
    q = "Can Parliament use Article 368 to amend the basic structure of the Indian Constitution?"
    rep = report(verdict(ClaimStatus.SUPPORTED, "Parliament cannot amend the basic structure."), answer="No. Parliament cannot amend the basic structure.")
    assert build_final_answer(rep, [], 1, q).answer.startswith("Nah")


def test_the_curator_runs_at_temperature_zero():
    from backend.settings import Settings

    assert Settings().verifier.temperature == 0.0


# ---- 2026-10-14 stability run: Erie shown as "Yeah, you're right" in 2 of 3 runs
def test_a_contrast_rejecting_the_asked_object_is_a_no():
    from backend.research.style import premise_state

    q = ERIE_Q
    a = ("Under Erie, a U.S. federal court hearing a state-law claim under diversity jurisdiction must apply state "
         "substantive law, rather than federal general common law.")
    assert premise_state("TRUE", q, a) == "FALSE"
    assert premise_state("PARTLY", q, "Erie overruled Swift v. Tyson. A federal court must apply state law, not federal general common law.") == "FALSE"


def test_a_contrast_about_something_else_keeps_the_yes():
    from backend.research.style import premise_state

    q = "Can a company in the EU transfer personal data to a certified US company?"
    assert premise_state("TRUE", q, "Yes. It can transfer the data under the adequacy decision, rather than via SCCs.") == "TRUE"
