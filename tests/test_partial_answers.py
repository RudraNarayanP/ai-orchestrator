"""A question that asks two things and settles one is not a question we know nothing about.

Live (job_261009210109_ccd924ec): the pages established that the Data Protection Act 2018
received Royal Assent on 23 May 2018 -- the curator judged that claim `supported` on an opened
gov.uk page -- and the run answered "Couldn't verify that one." because the commencement half
was never reached. Stating the settled part is not a weaker claim; hiding it is the lie.

The refusal is only replaced for the part the evidence actually answers. An unresolved half
never becomes a "no", and a settled half never lifts confidence in the whole.
"""

from __future__ import annotations

from backend.models import Claim, ClaimStatus, ClaimVerdict, Confidence, VerifierReport
from backend.research.lint import covers_part, question_parts
from backend.research.style import BANNED_PHRASES
from backend.verification.verifier import build_final_answer

TWO_PART = "When did the Data Protection Act 2018 receive Royal Assent, and when did it come into force?"
ASSENT = "The Data Protection Act 2018 received Royal Assent on 23 May 2018."
FORCE = "Most of the Act's provisions came into force on 25 May 2018."
GOV = "https://www.gov.uk/government/collections/data-protection-act-2018"


def verdict(claim: str, value: ClaimStatus, evidence: list[str], cid: str = "clm_a") -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=cid, claim=claim, verdict=value,
        confidence=Confidence.HIGH if value == ClaimStatus.SUPPORTED else Confidence.LOW,
        reasoning="the collection page states it" if evidence else "nothing we opened states it",
        strong_evidence=evidence,
    )


def report(*verdicts: ClaimVerdict, confidence: Confidence) -> VerifierReport:
    return VerifierReport(
        job_id="j", round=1, verdicts=list(verdicts), confidence=confidence,
        answer=" ".join(v.claim for v in verdicts), why="the collection page states it",
    )


# ------------------------------------------------------------------------ part splitting


def test_a_question_asking_two_things_splits_into_two_parts():
    assert question_parts(TWO_PART) == [
        "When did the Data Protection Act 2018 receive Royal Assent",
        "when did it come into force?",
    ]


def test_one_thing_asked_is_one_part():
    # "designed and built" is one predicate, not two questions; a preposition-led wh-clause is
    # a question even though it opens with "In".
    assert len(question_parts("Who designed and built the Eiffel Tower?")) == 1
    assert len(question_parts("When was the Eiffel Tower completed?")) == 1
    assert len(question_parts("In which year was the Eiffel Tower completed, and when did it open to the public?")) == 2


def test_a_relative_clause_after_a_comma_is_not_a_second_question():
    assert len(question_parts("The tower, which is in Paris, is how tall?")) == 1


def test_a_claim_only_answers_the_part_it_states():
    assent = "The construction of the Eiffel Tower was finished on March 31, 1889"
    parts = question_parts("In which year was the Eiffel Tower completed, and when did it open to the public?")
    assert covers_part(parts[0], assent)
    assert covers_part(parts[1], "The Eiffel Tower opened to the public on 15 May 1889.")
    assert not covers_part(parts[1], assent), "the completion date does not answer the opening half"
    # Sharing the subject is not answering: a page about the tower's height says nothing about a year.
    assert not covers_part(parts[0], "The Eiffel Tower is 1083 feet tall (333 meters).")


# ------------------------------------------------------------------------ the three shapes


def test_both_parts_settled_is_a_plain_answer():
    final = build_final_answer(
        report(
            verdict(ASSENT, ClaimStatus.SUPPORTED, [GOV]),
            verdict(FORCE, ClaimStatus.SUPPORTED, [GOV], cid="clm_b"),
            confidence=Confidence.HIGH,
        ),
        [], 1, TWO_PART,
    )
    assert "23 May 2018" in final.answer and "25 May 2018" in final.answer, final.answer
    assert "Not documented" not in final.answer, final.answer


def test_one_part_settled_states_it_and_names_the_other():
    final = build_final_answer(
        report(
            verdict(ASSENT, ClaimStatus.SUPPORTED, [GOV]),
            verdict(FORCE, ClaimStatus.INSUFFICIENT_EVIDENCE, [], cid="clm_b"),
            confidence=Confidence.LOW,
        ),
        [], 1, TWO_PART,
    )
    assert final.answer.startswith("Documented:"), final.answer
    assert "23 May 2018" in final.answer and "gov.uk" in final.answer, final.answer
    assert "Not documented:" in final.answer and "come into force" in final.answer, final.answer
    assert not final.answer.lower().startswith(("couldn't verify", "i couldn't verify", "i don't know")), final.answer
    assert final.confidence == Confidence.LOW, "documenting one part does not raise confidence in the whole"
    assert not any(p.lower() in final.answer.lower() for p in BANNED_PHRASES), final.answer


def test_the_live_shape_one_supported_verdict_and_never_judged_second_half():
    """job_261009210109_ccd924ec: ONE verdict, supported, no adjudication at all for the other
    half -- low confidence alone turned the whole thing into a flat "Couldn't verify that one."""
    final = build_final_answer(
        report(verdict(ASSENT, ClaimStatus.SUPPORTED, [GOV]), confidence=Confidence.LOW),
        [], 3, TWO_PART,
    )
    assert "23 May 2018" in final.answer, final.answer
    assert "come into force" in final.answer, "the half we could not settle is named, not hidden"
    assert "Not checked:" in final.answer and "I didn't get that far." in final.answer, final.answer
    assert "Not documented:" not in final.answer, "the review never reached that half; the sources were not consulted about it"
    assert final.answer != "Couldn't verify that one.", final.answer


def test_neither_part_settled_still_refuses():
    final = build_final_answer(
        report(
            verdict(ASSENT, ClaimStatus.INSUFFICIENT_EVIDENCE, []),
            verdict(FORCE, ClaimStatus.INSUFFICIENT_EVIDENCE, [], cid="clm_b"),
            confidence=Confidence.LOW,
        ),
        [], 1, TWO_PART,
    )
    assert final.answer == "Couldn't verify that one.", final.answer


def test_a_documented_fact_that_answers_neither_part_does_not_become_the_answer():
    """Ancillary evidence must not be dressed up as a response to the question asked."""
    final = build_final_answer(
        report(
            verdict("The Eiffel Tower is 333 metres tall.", ClaimStatus.SUPPORTED, [GOV]),
            confidence=Confidence.LOW,
        ),
        [], 1, "In which year was the Eiffel Tower completed, and when did it open to the public?",
    )
    assert final.answer == "Couldn't verify that one.", final.answer
    assert "333 metres" not in final.answer


def test_a_single_part_question_is_left_alone():
    """No ledger structure where nothing was asked twice: the normal answer/refusal still applies."""
    final = build_final_answer(
        report(verdict(ASSENT, ClaimStatus.SUPPORTED, [GOV]), confidence=Confidence.LOW),
        [], 1, "When did the Data Protection Act 2018 receive Royal Assent?",
    )
    assert "Documented:" not in final.answer, final.answer


# ------------------------------------------------------------------------ honesty guards


def test_an_unsettled_half_is_never_reported_as_false():
    final = build_final_answer(
        report(
            verdict(ASSENT, ClaimStatus.SUPPORTED, [GOV]),
            ClaimVerdict(claim_id="clm_b", claim=FORCE, verdict=ClaimStatus.UNVERIFIED, confidence=Confidence.LOW, reasoning="never reached"),
            confidence=Confidence.LOW,
        ),
        [], 1, TWO_PART,
    )
    assert "Not documented:" in final.answer and "23 May 2018" in final.answer, final.answer
    lowered = final.answer.lower()
    assert "nah" not in lowered and "doesn't work" not in lowered and "wrong" not in lowered, final.answer
    assert "false" not in lowered, final.answer


def test_a_part_the_sources_disagree_on_is_named_as_disagreement_not_as_undocumented():
    final = build_final_answer(
        report(
            verdict(ASSENT, ClaimStatus.SUPPORTED, [GOV]),
            verdict(FORCE, ClaimStatus.CONTESTED, [], cid="clm_b"),
            confidence=Confidence.LOW,
        ),
        [], 1, TWO_PART,
    )
    assert "23 May 2018" in final.answer, final.answer
    assert "The sources disagree on:" in final.answer, final.answer
    under = final.answer.split("The sources disagree on:")[1].strip().splitlines()
    assert len(under) == 1 and "into force" in under[0], final.answer
    assert "Not documented:" not in final.answer, "the disputed half is not also listed as undocumented"


def test_a_claim_check_keeps_its_own_wording():
    """Checking a premise is not a two-part question; the documented/undocumented list stays out."""
    final = build_final_answer(
        report(verdict(ASSENT, ClaimStatus.SUPPORTED, [GOV]), confidence=Confidence.HIGH),
        [], 1, "Under the DPA 2018 the age of consent is 16, right?",
    )
    assert final.answer.startswith("Yeah, you're right.") or final.truth_state == "TRUE", final.answer
    assert "Documented:" not in final.answer


def test_the_answer_stays_short_however_many_claims_the_ledger_holds():
    """One line per part asked, not one line per reworded claim (a live 25-claim review produced a
    23-line dump once the parts were mapped instead of listed)."""
    verdicts = [verdict(ASSENT, ClaimStatus.SUPPORTED, [GOV])] + [
        ClaimVerdict(
            claim_id=f"clm_{i}", claim=f"Detail {i} about the Act's commencement arrangements.",
            verdict=ClaimStatus.INSUFFICIENT_EVIDENCE, confidence=Confidence.LOW, reasoning="never reached",
        )
        for i in range(20)
    ]
    final = build_final_answer(report(*verdicts, confidence=Confidence.LOW), [], 1, TWO_PART)
    assert len(final.answer.splitlines()) <= 6, final.answer
    assert final.answer.count("Detail ") == 0, "the question's own halves are named, not 20 claim rewordings"


def test_the_settled_part_carries_its_source_not_a_bare_assertion():
    final = build_final_answer(
        report(verdict(ASSENT, ClaimStatus.SUPPORTED, [GOV]), confidence=Confidence.LOW),
        [], 1, TWO_PART,
    )
    assert "gov.uk" in final.answer, "provenance travels with the documented part"
