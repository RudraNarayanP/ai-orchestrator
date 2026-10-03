"""A page that settles several claims is filed under each of them (live eval defect).

Live: the DPA 2018 question opened legislation.gov.uk section 9 -- the statute that
answers it -- but the link carried no usable title/snippet, so the page was attached
to no claim, the ledger saw zero evidence for the headline claim, and the answer
degraded to "I couldn't verify this reliably." even though the primary source was open.
"""

from __future__ import annotations

import pytest

from backend.evidence import sources
from backend.evidence.sources import FetchedPage, gather_from_links
from backend.models import Evidence, SourceCheckStatus
from backend.orchestrator.runner import _merge_evidence

S9 = (
    "Section 9 Child's consent in relation to information society services. In Article 8(1) of the "
    "UK GDPR, references to 16 years are to be read as references to 13 years. A child under 13 "
    "needs the consent of the holder of parental responsibility."
)
URL = "https://www.legislation.gov.uk/ukpga/2018/12/section/9"


@pytest.fixture()
def fake_fetch(monkeypatch):
    async def fetch(url, **kwargs):
        return FetchedPage(url=url, final_url=url, status=200, title="s9", text=S9, ok=True)

    monkeypatch.setattr(sources, "fetch_page", fetch)


async def test_unattributed_page_is_filed_under_every_claim_it_supports(fake_fetch):
    targets = [
        ("c_age", "The minimum age of consent for information society services under the Act is 13 years."),
        ("c_parent", "A child under 13 needs consent of the holder of parental responsibility."),
        ("c_unrelated", "The Eiffel Tower was completed in 1889 in Paris."),
    ]
    got = await gather_from_links("j", [{"href": URL}], attribute_to=targets)
    filed = {e.claim_id for e in got if e.check_status == SourceCheckStatus.CONFIRMED}
    assert {"c_age", "c_parent"} <= filed
    assert "c_unrelated" not in filed, "a page is never attached to a claim it does not contain"


async def test_attribution_still_requires_the_figures_to_be_on_the_page(fake_fetch):
    got = await gather_from_links(
        "j", [{"href": URL}], attribute_to=[("c_wrong", "The minimum age of consent for information society services is 21 years.")]
    )
    assert "c_wrong" not in {e.claim_id for e in got if e.check_status == SourceCheckStatus.CONFIRMED}


async def test_without_targets_behaviour_is_unchanged(fake_fetch):
    got = await gather_from_links("j", [{"href": URL}])
    assert len(got) == 1 and got[0].claim_id is None


def _ev(claim_id, status=SourceCheckStatus.CONFIRMED, url=URL):
    return Evidence(job_id="j", claim_id=claim_id, url=url, check_status=status)


def test_merge_keeps_one_row_per_page_and_claim():
    merged = _merge_evidence([], [_ev("a"), _ev("b")])
    assert {e.claim_id for e in merged} == {"a", "b"}
    merged = _merge_evidence(merged, [_ev("a"), _ev("b")])
    assert len(merged) == 2, "the same page seen again must not duplicate"


def test_merge_adopts_an_orphan_page_into_the_claim_that_later_claims_it():
    merged = _merge_evidence([], [_ev(None)])
    merged = _merge_evidence(merged, [_ev("a")])
    assert len(merged) == 1 and merged[0].claim_id == "a"


def test_merge_does_not_duplicate_a_page_seen_again_without_a_claim():
    merged = _merge_evidence([], [_ev("a")])
    merged = _merge_evidence(merged, [_ev(None)])
    assert len(merged) == 1 and merged[0].claim_id == "a"

async def test_a_page_that_never_mentions_the_questions_subject_is_irrelevant(fake_fetch):
    """Live: California/Singapore/UK-SI pages 'supported' a Ukrainian Family Code claim that had lost the word Ukraine."""
    got = await gather_from_links(
        "j", [{"href": URL, "claim_id": "c1", "claim_text": "The marriageable age is set at 18 years for both."}], require_names=["ukrai"]
    )
    assert got and all(e.check_status == SourceCheckStatus.IRRELEVANT and e.polarity == "neutral" for e in got)
    assert "never mentions the question's subject" in (got[0].check_notes or "")


async def test_a_page_that_mentions_the_subject_is_kept(fake_fetch, monkeypatch):
    async def fetch(url, **kwargs):
        return FetchedPage(url=url, final_url=url, status=200, title="Ukraine Family Code", text="Family Code of Ukraine: the marriageable age is 18 years for women and men.", ok=True)

    monkeypatch.setattr(sources, "fetch_page", fetch)
    got = await gather_from_links(
        "j", [{"href": "https://zakon.example/fc", "claim_id": "c1", "claim_text": "The marriageable age is set at 18 years for both."}], require_names=["ukrai"]
    )
    assert got[0].check_status == SourceCheckStatus.CONFIRMED