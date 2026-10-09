"""A date has to belong to the event the claim names -- clause by clause, on an explicit vocabulary.

Live (Eiffel Tower runs, 2026-10-09): the page said "Construction work began in January
1887 and was finished on 31 March 1889", and a claim that the tower was *completed* in 1887
matched it, because the whole page contains the words and the year. These tests pin the
behaviour that replaced that: bind where the page says so, refuse where it doesn't, and never
guess a role.
"""

from __future__ import annotations

from backend.evidence.events import EVENT_CUES, bind_event_dates, dated
from backend.evidence.sources import FetchedPage, check_support
from backend.models import SourceCheckStatus

EIFFEL = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. "
    "Construction work began in January 1887 and was finished on 31 March 1889. "
    "The tower was opened to the public on 15 May 1889. "
    "Renovations were completed on 24 June 1985."
)
ACT = (
    "The Data Protection Act 2018 received Royal Assent on 23 May 2018. "
    "Most of its provisions came into force on 25 May 2018."
)


def page(body: str, url: str = "https://example.org/page") -> FetchedPage:
    return FetchedPage(url=url, final_url=url, status=200, title="A page", text=body, ok=True)


def bind(claim: str, body: str = EIFFEL) -> str:
    return (bind_event_dates(claim, body) or {}).get("verdict", "not-applicable")


# ---------------------------------------------------------------- several dates, one page


def test_each_date_is_bound_to_the_event_the_page_gives_it():
    assert bind("Construction of the Eiffel Tower began in January 1887.") == "bound"
    assert bind("The Eiffel Tower was finished on 31 March 1889.") == "bound"
    assert bind("The Eiffel Tower was opened to the public on 15 May 1889.") == "bound"
    assert bind("Renovations were completed on 24 June 1985.") == "bound"


def test_swapping_two_events_dates_is_not_support():
    """The reversed-date case: the completion claim cannot borrow the opening date, and
    the opening claim cannot borrow the completion date."""
    assert bind("The Eiffel Tower was completed on 15 May 1889.") == "excluded"
    assert bind("The Eiffel Tower opened to the public on 31 March 1889.") == "excluded"


def test_a_start_year_never_evidences_a_completion():
    assert bind("The Eiffel Tower was completed in 1887.") == "excluded"
    assert bind("Construction of the Eiffel Tower began in 1889.") == "excluded"


def test_an_unrelated_date_on_the_page_is_not_borrowed():
    """1985 belongs to renovations; a completion claim dated 1985 must not use it even
    though the cue word 'completed' is on the page in that clause."""
    assert bind("The Eiffel Tower was completed on 24 June 1985.") == "excluded"


# ------------------------------------------------------------------------ paraphrase


def test_synonyms_of_the_same_event_bind():
    """'was finished' evidences 'completed' -- the cue sets are per event, not per word."""
    assert bind("The Eiffel Tower was completed on 31 March 1889.") == "bound"
    assert bind("The construction of the Eiffel Tower was finished on 31 March 1889.") == "bound"
    assert bind("Building work on the Eiffel Tower started in January 1887.") == "bound"


def test_statute_events_are_kept_apart():
    assert bind("The Act came into force on 25 May 2018.", ACT) == "bound"
    assert bind("The Act received Royal Assent on 23 May 2018.", ACT) == "bound"
    assert bind("The Act came into force on 23 May 2018.", ACT) == "excluded"
    assert bind("The Act received Royal Assent on 25 May 2018.", ACT) == "excluded"


# ------------------------------------------------------------- ambiguity: say so, don't guess


def test_a_date_with_no_event_words_is_left_unverified():
    """Requirement: when the relationship cannot be established, the claim stays
    unverified rather than being promoted by a coincidence of wording."""
    body = "The tower stands on the Champ de Mars. Records from 1887 and 1889 are held here."
    assert bind("The Eiffel Tower was completed in 1889.", body) == "unknown"


def test_a_year_the_page_never_gives_is_never_bound():
    assert bind("The Eiffel Tower was completed in 1900.") == "unknown"


def test_a_claim_naming_no_event_is_left_to_the_plain_checks():
    """No event, no role question -- the figure and wording rules alone decide."""
    assert bind("The Eiffel Tower is 330 metres tall.") == "not-applicable"
    assert bind("The Data Protection Act 2018 has 204 sections.", ACT) == "not-applicable"


# ----------------------------------------------------------------- through check_support


def test_check_support_confirms_only_the_event_it_states():
    ok = check_support("Construction of the Eiffel Tower began in January 1887.", page(EIFFEL))
    assert ok["status"] == SourceCheckStatus.CONFIRMED, ok
    assert "January 1887" in ok["excerpt"], ok["excerpt"]
    assert (ok["role"] or {}).get("verdict") == "bound"

    swapped = check_support("Construction of the Eiffel Tower began in March 1889.", page(EIFFEL))
    assert swapped["status"] == SourceCheckStatus.MISMATCH, swapped
    assert any("1889" in str(m) for m in swapped["missing"]), swapped["missing"]


def test_an_unbound_date_is_recorded_as_not_checked_not_confirmed():
    body = "The Eiffel Tower is on the Champ de Mars in Paris. Files from 1889 are listed below."
    out = check_support("The Eiffel Tower was completed in 1889.", page(body))
    assert out["status"] == SourceCheckStatus.NOT_CHECKED, out
    assert any("never states which event" in str(m) for m in out["missing"]), out["missing"]


def test_the_recorded_passage_keeps_the_pages_own_words():
    out = check_support("The Eiffel Tower was finished on 31 March 1889.", page(EIFFEL))
    assert "31 March 1889" in out["excerpt"], out["excerpt"]
    assert "1889-03-31" not in out["excerpt"], "a canonical key is not what the page printed"


# ------------------------------------------------------------------------- the vocabulary


def test_dated_covers_bare_years_and_full_dates_without_double_counting():
    keys = [key for _, _, key in dated("Work began in January 1887 and was finished on 31 March 1889.")]
    assert keys == ["1887-01", "1889-03-31"], keys


def test_the_cue_vocabulary_is_explicit_and_small():
    """Every accepted or refused date role rests on a listed word: nothing is inferred
    from tone, and the list stays auditable."""
    assert set(EVENT_CUES) == {"start", "end", "open", "force", "enacted", "published"}
    assert sum(len(cues) for cues in EVENT_CUES.values()) < 70
