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


def _gather(monkeypatch, *, text: str, published: str, claim_id: str, claim_text: str, title: str = "Data Protection Act 2018", url: str = ACT_URL):
    async def fake_fetch(u, **kwargs):
        return FetchedPage(url=url, final_url=url, status=200, title=title, text=text, published=published, ok=True)

    monkeypatch.setattr(sources, "fetch_page", fake_fetch)
    links = [{"href": url, "title": title, "claim_id": claim_id, "claim_text": claim_text}]
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


# ------------------------------------------------- what the law provides does not age

RULE_TEXT = (
    "The Data Protection Act 2018 provides that a controller must maintain a record of processing "
    "activities and make it available to the supervisory authority on request."
)


def test_an_old_official_page_still_confirms_a_standing_rule(monkeypatch):
    """Live (2026-10-15, dpf): the Commission's own EU-US transfers page, published 2021, was the only
    official page read, and every row taken from it was voided as outdated -- so the claim stating what
    the adequacy decision provides had no usable evidence left. A page that quotes the instrument is the
    record of the rule; a newer source showing the rule changed is what displaces it, not the calendar."""
    claim = "The Data Protection Act 2018 provides that a controller must maintain a record of processing activities."
    out = _gather(
        monkeypatch,
        text=RULE_TEXT,
        published="2021-12-17T10:19:13+01:00",
        claim_id="clm_rule",
        claim_text=claim,
    )
    row = next(e for e in out if e.claim_id == "clm_rule")
    assert row.check_status == SourceCheckStatus.CONFIRMED, (row.check_status, row.check_notes)
    assert "standing" in (row.check_notes or ""), row.check_notes


def test_a_standing_rule_on_an_old_unofficial_page_still_ages(monkeypatch):
    """The exemption is about the official publisher's own words. A law-firm post from 2021 describing
    a rule is evidence of what it believed then, and stays age-sensitive."""
    claim = "The Data Protection Act 2018 provides that a controller must maintain a record of processing activities."
    out = _gather(
        monkeypatch,
        text=RULE_TEXT,
        published="2021-12-17",
        claim_id="clm_rule",
        claim_text=claim,
        url="https://www.blogs.example.com/dpa-2018-summary",
    )
    row = next(e for e in out if e.claim_id == "clm_rule")
    assert row.check_status == SourceCheckStatus.OUTDATED, (row.check_status, row.check_notes)


def test_a_rule_asked_about_in_force_still_ages_on_an_old_official_page(monkeypatch):
    """Whether the instrument is still in force is a state of affairs: for that, age is exactly the point."""
    claim = "The Data Protection Act 2018 remains in force."
    out = _gather(
        monkeypatch,
        text="The Data Protection Act 2018 remains in force and most of its provisions commenced on 25 May 2018.",
        published="2018-05-23",
        claim_id="clm_force",
        claim_text=claim,
    )
    row = next(e for e in out if e.claim_id == "clm_force")
    assert row.check_status == SourceCheckStatus.OUTDATED, (row.check_status, row.check_notes)


def test_the_standing_rule_predicate_asks_what_the_claim_asserts():
    from backend.evidence.sources import standing_rule

    assert standing_rule("Under the adequacy decision, transfers to certified US companies do not require standard contractual clauses.")
    assert standing_rule("GDPR Article 45(1) provides that such a transfer shall not require any specific authorisation.")
    assert standing_rule("The Data Protection Act 2018 provides that a controller must maintain a record of processing activities.")
    assert not standing_rule("Section 379 of the IPC currently punishes theft.")
    assert not standing_rule("The adequacy decision is still in force.")
    assert not standing_rule("The fine is 4% of turnover.")
    assert not standing_rule("Paris is a large city.")
    assert not standing_rule("Keir Starmer is the Prime Minister."), "an officeholder is not a rule"
