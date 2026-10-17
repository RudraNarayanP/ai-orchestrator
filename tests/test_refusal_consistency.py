"""A refusal must not be followed by prose describing an assertion it never made.

Live (2026-10-16, six jobs): the answer was "Couldn't verify that one." and the same row carried
why="5 pages we opened, from 3 independent sites, state this." plus the caveat "Only one
independent site backs this." Both describe an assertion that does not exist. It is the worst
shape this product can take -- a refusal the record appears to disprove, with no reason stated --
and it breaks non-negotiable 5, which says a refusal must carry why.

The fix is consistency only. It never touches the verdict, the state, or whether the run refused;
it stops the surrounding prose contradicting the answer, and makes the refusal say what the run
could not do. Saying "none of the pages states it" would be a claim about the evidence that this
layer is exactly unable to make, so the wording describes the run instead.
"""

from __future__ import annotations

from backend.research.style import is_refusal
from backend.verification.verifier import _REFUSAL_WHY, keep_a_refusal_consistent


def test_a_refusal_is_recognised():
    for line in ("Couldn't verify that one.", "I don't know.", "I'm not sure — the sources disagree.",
                 "I couldn't verify that reliably."):
        assert is_refusal(line), line


def test_an_answer_that_asserts_something_is_not_a_refusal():
    assert not is_refusal("Yeah, you're right. The IPO closed on 15 June 2026.")
    assert not is_refusal("No — that doesn't hold up. The Act says the opposite.")
    # A PARTLY headline asserts too, and _PLAIN_STARTS (used for emoji rules) lists it -- so this
    # is the case a shared list would have wrongly treated as a refusal.
    assert not is_refusal("Yeah, the idea is right, but the date part is a bit off.")
    assert not is_refusal("")


def test_a_reported_conflict_declines_to_take_a_position():
    """CONFLICT takes no factual side, so "pages state this" is just as wrong beside it."""
    assert is_refusal("Hmm, I'm not sure — the sources disagree.")


def test_the_why_no_longer_claims_pages_state_a_refused_answer():
    why, caveats = keep_a_refusal_consistent(
        "Couldn't verify that one.",
        "5 pages we opened, from 3 independent sites, state this.",
        ["Only one independent site backs this."],
    )
    assert "state this" not in why
    assert why == _REFUSAL_WHY, "the refusal now has to say why, and the why has to be true"
    assert caveats == [], "a caveat about what backs 'this' is meaningless beside a refusal"


def test_an_ordinary_answer_keeps_its_evidence_sentence_untouched():
    """The guard is only allowed to fire on a refusal: 'state this' is correct when an answer exists."""
    why, caveats = keep_a_refusal_consistent(
        "Yeah, looks that way — but I could only partly confirm it. An adequacy decision covers transfers.",
        "5 pages we opened, from 2 independent sites, state this.",
        ["Only one independent site backs this."],
    )
    assert why == "5 pages we opened, from 2 independent sites, state this."
    assert caveats == ["Only one independent site backs this."]


def test_a_refusal_with_no_evidence_sentence_stays_silent():
    """Nothing was opened and nothing was claimed: do not invent an explanation about pages."""
    why, caveats = keep_a_refusal_consistent("I don't know.", "", [])
    assert why == ""
    assert caveats == []


def test_a_genuine_reason_survives_the_cleanup():
    """The reviewer-cut-short caveat is a real reason and must stay -- non-negotiable 5 needs it."""
    why, caveats = keep_a_refusal_consistent(
        "Couldn't verify that one.",
        "4 pages we opened, from an independent site, state this.",
        ["The AI reviewer's pass was incomplete, so this rests only on the pages I could open and check."],
    )
    assert len(caveats) == 1
    assert "reviewer" in caveats[0]
