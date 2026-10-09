"""Age only undermines a claim that could have changed since the page was written.

Live (DPA 2018 run, 2026-10-09): a gov.uk page published 23 May 2018 was the best evidence
that the Act "received Royal Assent on 23 May 2018", and the run discarded it as `outdated`
(3061 days old) and answered "Couldn't verify that one." A date fixed in the past cannot go
stale; a price, an officeholder or a rule in force can, and still gets checked.
"""

from __future__ import annotations

import asyncio

from backend.evidence import sources
from backend.evidence.events import claim_ages
from backend.evidence.sources import FetchedPage, gather_from_links
from backend.models import SourceCheckStatus
from backend.research import router

ACT_URL = "https://www.gov.uk/government/collections/data-protection-act-2018"
ACT_PAGE_TEXT = (
    "The Data Protection Act 2018 received Royal Assent on 23 May 2018 and came into force "
    "on 25 May 2018. Most of its provisions commenced alongside the UK GDPR."
)


def _gather(monkeypatch, *, text: str, published: str, claim_id: str, claim_text: str, title: str = "Data Protection Act 2018"):
    async def fake_fetch(url, **kwargs):
        return FetchedPage(url=ACT_URL, final_url=ACT_URL, status=200, title=title, text=text, published=published, ok=True)

    monkeypatch.setattr(sources, "fetch_page", fake_fetch)
    links = [{"href": ACT_URL, "title": title, "claim_id": claim_id, "claim_text": claim_text}]
    return asyncio.run(gather_from_links("j", links, attribute_to=[(claim_id, claim_text)], max_pages=1))


# ------------------------------------------------------------------- classification


def test_a_year_in_the_question_is_not_a_request_for_current_data():
    for q in (
        "When did the UK Data Protection Act 2018 receive Royal Assent?",
        "In which year was the Eiffel Tower completed?",
        "What happened in 1889 in Paris?",
        "When did the Act come into force?",
    ):
        assert not router.classify(q).needs_current_data, q


def test_present_asks_still_need_current_data():
    for q in (
        "Who is the current Prime Minister?",
        "What is the price of the Acme Bolt?",
        "How much does the subscription cost now?",
        "What is the latest version of Python?",
        "Who won the election?",
    ):
        assert router.classify(q).needs_current_data, q


def test_a_past_tense_price_question_is_history_not_current_data():
    assert not router.classify("What was the price of the Bolt in 2019?").needs_current_data
    assert router.classify("What is the price of the Bolt?").needs_current_data


def test_a_present_ask_outvotes_a_past_anchor_in_the_same_sentence():
    assert router.classify("Who is the Prime Minister today, and who was it in 2010?").needs_current_data


# ------------------------------------------------------------------------ claim_ages


def test_a_completed_past_event_does_not_age():
    assert claim_ages("The Data Protection Act 2018 received Royal Assent on 23 May 2018.") is False
    assert claim_ages("The Eiffel Tower was completed on 31 March 1889.") is False
    assert claim_ages("Construction began in January 1887.") is False


def test_states_of_affairs_still_age():
    """Freshness is scoped, never switched off."""
    assert claim_ages("Keir Starmer is the current Prime Minister.") is True
    assert claim_ages("The price of the Bolt is $499.") is True
    assert claim_ages("The organisation must respond within one calendar month.") is True
    assert claim_ages("The tower is 330 metres tall.") is True


def test_claims_with_no_date_and_future_dates_still_age():
    assert claim_ages("The Bolt weighs 1.2 kg.") is True
    assert claim_ages("The Act will come into force in 2030.", today_year=2026) is True


# --------------------------------------------------------------------- through gather


def test_an_old_page_still_confirms_a_fixed_past_event(monkeypatch):
    out = _gather(
        monkeypatch,
        text=ACT_PAGE_TEXT,
        published="2018-05-23",
        claim_id="clm_assent",
        claim_text="The Data Protection Act 2018 received Royal Assent on 23 May 2018.",
    )
    row = next(e for e in out if e.claim_id == "clm_assent")
    assert row.check_status == SourceCheckStatus.CONFIRMED, (row.check_status, row.check_notes)
    assert "fixed past event" in (row.check_notes or ""), row.check_notes
    assert "Royal Assent on 23 May 2018" in (row.verbatim_excerpt or ""), row.verbatim_excerpt


def test_an_old_page_is_still_treated_as_stale_for_a_present_state(monkeypatch):
    text = "As of the 2015 revision the price of the Bolt is $499, and it remains on sale."
    out = _gather(
        monkeypatch,
        text=text,
        published="2015-06-01",
        claim_id="clm_price",
        claim_text="The price of the Bolt is $499.",
        title="Acme Bolt price",
    )
    row = next(e for e in out if e.claim_id == "clm_price")
    assert row.check_status == SourceCheckStatus.OUTDATED, (row.check_status, row.check_notes)
    assert "days old" in (row.check_notes or ""), row.check_notes
