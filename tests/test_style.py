"""The house voice: direct, human, a little warm, a fitting emoji or two -- and never soft on "I don't know"."""

from __future__ import annotations

import pytest

from backend.models import Confidence, VerifierReport
from backend.research import style
from backend.research.style import count_emojis, humanize, style_prompt, voice_report
from backend.verification.verifier import build_final_answer


def test_a_settled_answer_gets_one_fitting_emoji_after_the_first_sentence():
    out = humanize("The Eiffel Tower was completed on March 31, 1889. It is 330 metres tall today.", "moderate")
    assert out.startswith("The Eiffel Tower was completed on March 31, 1889. ") and count_emojis(out) == 1
    assert "\U0001F5FC" in out, "a tower gets a tower"
    assert "\U0001F50B" in humanize("The battery lasts up to 30 hours with noise cancelling on.", "high")
    assert "\u2696" in humanize("Section 12 of the Act says the notice period is 28 days.", "high")
    assert "\U0001F393" in humanize("The university allows a student to appeal within 20 working days.", "moderate")


@pytest.mark.parametrize(
    "plain",
    [
        "I don't know.",
        "I don't know. No public page states that figure.",
        "I couldn't verify this reliably.",
        "I couldn't verify this reliably. Only one forum post mentions it.",
    ],
)
def test_warmth_never_touches_an_honest_dont_know(plain):
    assert humanize(plain, "low") == plain and humanize(plain, "insufficient_evidence") == plain
    assert voice_report(plain)["ok"]


def test_no_emoji_is_added_when_confidence_is_none_or_one_is_already_there():
    assert humanize("The price is $549.", "none") == "The price is $549."
    already = "It costs $549 \U0001F4B8."
    assert humanize(already, "high") == already


def test_a_correction_and_a_dispute_get_their_own_marks_and_stay_conclusion_first():
    no = humanize("No - that doesn't hold up. The page says the opposite.", "high")
    assert no.startswith("No - that doesn't hold up.") and "\U0001F6AB" in no
    dispute = humanize("The sources genuinely conflict on this: the date is either 1887 or 1889.", "low")
    assert dispute.startswith("The sources genuinely conflict") and "\U0001F914" in dispute


def test_the_final_answer_is_humanised_but_an_idk_is_not():
    settled = VerifierReport(job_id="j", answer="Sony rates it at 30 hours of battery life with noise cancelling on.", confidence=Confidence.HIGH)
    assert count_emojis(build_final_answer(settled, [], 1).answer) == 1
    idk = VerifierReport(job_id="j", answer="I don't know.", confidence=Confidence.NONE)
    assert build_final_answer(idk, [], 1).answer == "I don't know."


def test_voice_report_flags_corporate_hedging_and_internal_vocabulary():
    bad = voice_report("It is important to note that the ledger score for claim clm_ab12 in round 2 was 3.1.")
    assert not bad["ok"]
    joined = " ".join(bad["problems"])
    assert "internal vocabulary" in joined and "hedging opener" in joined
    assert voice_report("It was finished on 31 March 1889 and it's 330 m tall now \U0001F5FC")["ok"]
    assert not voice_report("Great question! Certainly! " + "x" * 50)["ok"]
    assert not voice_report("I don't know. \U0001F937")["ok"], "an emoji on a don't-know is a voice failure"
    assert not voice_report("Done \U0001F600\U0001F600\U0001F600\U0001F600")["ok"], "more than three emojis is decoration"
    assert not voice_report("x" * 1000)["ok"]


def test_the_prompts_carry_the_tone_rules_and_a_before_after_example():
    prompt = style_prompt()
    assert "at most two emojis" in prompt and "knowledgeable friend" in prompt
    assert "Before and after" in prompt and 'NEVER put an emoji on "I don\'t know."' in prompt
    assert "Warmth never softens" in style.VOICE
    assert "great question" in [p.lower() for p in style.BANNED_PHRASES]


def test_plain_caveats_reword_or_drop_internal_bookkeeping():
    from backend.research.style import plain_caveats, INTERNAL_VOCAB_RE

    raw = [
        "verifier model unavailable (cannot reach http://localhost:11434/v1); verdicts come from the evidence ledger, not from a language model",
        "(no model verifier active: cannot reach http://localhost:11434/v1)",
        "verifier answer overruled to low confidence by the ledger",
        " No claim survived the evidence ledger.",
        "Not confirmed by any page we opened: The minimum age is 13 years old.",
        "Only one independent site backs this.",
        "Only one independent site backs this.",
        "voice scrub removed: ledger",
    ]
    got = plain_caveats(raw)
    assert got[0].startswith("The AI reviewer wasn't available")
    assert "None of the claims could be confirmed from the pages I opened." in got
    assert any(c.startswith("The AI answers said The minimum age is 13 years old.") for c in got)
    assert got.count("Only one independent site backs this.") == 1, "duplicates collapse"
    assert not any(INTERNAL_VOCAB_RE.search(c) or "localhost" in c or "verifier" in c.lower() for c in got)


def test_final_answer_caveats_are_plain_language():
    from backend.models import Confidence, VerifierReport
    from backend.verification.verifier import build_final_answer

    report = VerifierReport(job_id="j", round=1, verdicts=[], answer="I couldn't verify this reliably.", confidence=Confidence.LOW)
    report.unresolved = ["verifier answer overruled to low confidence by the ledger"]
    report.caveats = ["(no model verifier active: cannot reach http://localhost:11434/v1)"]
    report.confidence_note = " No claim survived the evidence ledger."
    final = build_final_answer(report, [], 1)
    assert final.caveats == ["None of the claims could be confirmed from the pages I opened."]