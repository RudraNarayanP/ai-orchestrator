"""A long official statute is read at the provisions the claims name (live IPC-theft defect, 2026-10-16).

We opened the India Code PDF of the Bharatiya Nyaya Sanhita -- 112 pages, real text -- and kept 1,500 characters of
front matter, because the body cap was applied to the START of the document. "Theft is Section 303" therefore could
not be confirmed by anything we read, and the run refused an answerable question.
"""

from __future__ import annotations

import asyncio
import re

from backend.evidence import sources
from backend.evidence.sources import (
    FetchedPage,
    check_support,
    provision_numbers,
    statute_body,
)

HEAD = "THE BHARATIYA NYAYA SANHITA, 2023 (ACT NO. 45 OF 2023) LIST OF ABBREVIATIONS "
CONTENTS = "CHAPTER XVII OF OFFENCES AGAINST PROPERTY Of theft 303. Theft. 304. Snatching. 305. Theft in a dwelling house. "
BODY = (
    "303. Theft.--Whoever, intending to take dishonestly any movable property out of the possession of any person "
    "without that person's consent, moves that property in order to such taking, shall be punished with imprisonment "
    "of either description for a term which may extend to three years, or with fine which may extend to one thousand "
    "rupees, or with both. "
)


def statute(char_count: int = 400_000) -> str:
    """A statute-shaped document: front matter, then the arrangement of sections, then the sections themselves."""
    filler = "Preliminary definitions and short statements of general principles of the law. " * (char_count // 90)
    return f"{HEAD}{filler}{CONTENTS}{BODY}{filler}"


def test_provision_numbers_reads_the_sections_the_claims_name():
    assert provision_numbers("Section 303 of the BNS applies.", "Article 45(3) GDPR; s.116 CPC.") == ["303", "45", "116"]
    assert provision_numbers("the age is 13 and the fine is 4,500") == []  # figures are not provisions


def test_the_provision_itself_is_read_not_the_start_of_the_document():
    text = statute()
    body = statute_body(text, 12000, ["303"])
    assert len(body) <= 12000
    assert "intending to take dishonestly" in body, body[-400:]
    assert "may extend to three years" in body


def test_a_contents_entry_alone_does_not_stand_for_the_section():
    """The arrangement of sections says "303. Theft." and nothing about punishment: a claim about the penalty must
    be confirmed by the section, so the window taken is the one with running text after the number."""
    text = statute()
    body = statute_body(text, 1500, ["303"])
    assert "shall be punished" in body or "be punished" in body, body[-500:]
    assert body.count("303. Theft.") <= 2


def test_a_short_document_is_left_alone():
    short = f"{HEAD}{BODY}"
    assert statute_body(short, 12000, ["303"]) == short


def test_check_support_confirms_the_punishment_from_the_read_provision():
    page = FetchedPage(url="https://www.indiacode.nic.in/x.pdf", ok=True, status=200, title="BNS 2023",
                       text=statute_body(statute(), 12000, ["303"]))
    claim = "Section 303 of the Bharatiya Nyaya Sanhita, 2023 punishes theft with imprisonment up to three years or a fine."
    got = check_support(claim, page)
    assert got["status"].value == "confirmed", got
    assert "303" in (got["excerpt"] or "")


def test_gather_passes_the_provisions_it_was_given_down_to_the_fetch(monkeypatch):
    """The numbers come from the claims being checked, so the fetcher knows which part of a long document matters."""
    seen: dict[str, object] = {}

    async def fake_fetch(url, *, timeout_s=30, max_chars=12000, browser_fetch=None, provisions=None):
        seen["provisions"] = provisions
        seen["max_chars"] = max_chars
        return FetchedPage(url=url, ok=True, status=200, text="303. Theft.--punishment.", title="BNS")

    monkeypatch.setattr(sources, "fetch_page", fake_fetch)
    asyncio.run(
        sources.gather_from_links(
            "j1", [{"href": "https://x.test/a.pdf"}], attribute_to=[("clm_1", "Section 303 of the BNS punishes theft.")], max_pages=1
        )
    )
    assert seen.get("provisions") == ["303"], seen


MENU_HTML = """
<html><head><title>SpaceX IPO</title><script>var x=1;</script></head><body>
<nav>''' + ("Products Featured Products By Objective By Institutional By Investor Markets A-Z Screening Ideas My Watchlist " * 12) + '''</nav>
<main><h1>SpaceX priced its IPO at $120 a share</h1>
<p>The rocket company began trading on Nasdaq under the ticker SPCX on 12 June 2026 after raising
$85.7 billion in gross proceeds, the largest listing in the exchange's history. The company said demand
from investors exceeded the offer several times, and that Class A common shares carry one vote each.</p></main>
<footer>Cookie settings Privacy Terms Advertise Careers</footer>
</body></html>
"""


def test_the_pages_own_main_region_is_read_before_its_navigation():
    """The live Nasdaq newsroom page gave body text that was the mega-menu, so the sentence that settled the claim
    fell outside the 12,000-character body and the row attached to no claim at all."""
    from backend.evidence.sources import _html_to_text, _main_region

    main = _html_to_text(_main_region(MENU_HTML))
    assert "priced its IPO at $120" in main and "SPCX" in main
    assert "Featured Products" not in main


def test_a_page_read_without_a_claim_to_check_confirms_nothing():
    """"We opened it" is not "it says so": an unchecked row used to be filed as confirmed, which is how the answer
    could say pages we opened state the claim while no claim had been matched to any page."""
    from backend.evidence.sources import gather_from_links

    page = FetchedPage(url="https://x.test/a", ok=True, status=200, title="t",
                       text="Section 303 of the Bharatiya Nyaya Sanhita, 2023 punishes theft.")

    async def fake_fetch(url, **kw):
        return page

    async def run():
        import backend.evidence.sources as src

        original = src.fetch_page
        src.fetch_page = fake_fetch
        try:
            return await gather_from_links("j", [{"href": "https://x.test/a"}], max_pages=1)
        finally:
            src.fetch_page = original

    rows = asyncio.run(run())
    base = [r for r in rows if r.claim_id is None]
    assert base and base[0].check_status.value == "not_checked", [(r.check_status.value, r.claim_id) for r in rows]
