"""Product / service review behaviour (backlog item 4).

Recurring owner complaints are mined from review/community pages, tagged COMMUNITY,
and surfaced as a labelled caveat. They are never part of the answer, never evidence
for a claim, and a single page's grumble is not "recurring".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.evidence import reviews, sources
from backend.evidence.reviews import caveat_lines, gather_reviews, is_review_url, mine_complaints, review_queries, subject_for
from backend.models import ClaimStatus, Job, ResearchMode, ReviewFindings, SourceTier
from backend.orchestrator.runner import ResearchRunner
from tests.conftest import adapters_from, base_settings

FIXTURES = Path(__file__).parent / "fixtures" / "reviews"
URLS = {
    "https://www.reddit.com/r/gadgets/comments/acme_bolt_long_term": "reddit_thread.html",
    "https://www.trustpilot.com/review/acme-bolt.example": "trustpilot_page.html",
    "https://apps.apple.com/us/app/acme-bolt-companion/id1": "appstore_page.html",
}
QUESTION = "Is the Acme Bolt phone worth buying? Any problems in the reviews?"


def page_for(url: str, name: str | None = None) -> sources.FetchedPage:
    name = name or URLS[url]
    html = (FIXTURES / name).read_text(encoding="utf-8")
    text = sources._html_to_text(html)
    blocked = bool(sources.BLOCK_HINT_RE.search(text))
    return sources.FetchedPage(url=url, final_url=url, status=200, title=name, text=text, ok=not blocked, error="blocked" if blocked else None)


@pytest.fixture
def review_world(net, monkeypatch):
    """Search returns the review pages; fetching serves the fixtures (nothing touches the network)."""
    queries = review_queries("Acme Bolt", 3)
    net.add_results(queries[0], [{"href": "https://www.reddit.com/r/gadgets/comments/acme_bolt_long_term", "title": "Acme Bolt after 6 months", "snippet": "long term"}])
    net.add_results(queries[1], [{"href": "https://www.trustpilot.com/review/acme-bolt.example", "title": "Acme Bolt reviews", "snippet": "reviews"}])
    net.add_results(queries[2], [
        {"href": "https://apps.apple.com/us/app/acme-bolt-companion/id1", "title": "Acme Bolt Companion", "snippet": "app"},
        {"href": "https://shop.example/acme-bolt", "title": "Acme Bolt - buy now", "snippet": "shop"},  # not a review page
    ])
    fetched: list[str] = []

    async def fake_fetch(url, **kwargs):
        assert kwargs.get("max_chars") == 12000  # search.fetch_body_chars reaches the fetch
        fetched.append(url)
        return page_for(url)

    monkeypatch.setattr(reviews, "fetch_page", fake_fetch)
    net.fetched = fetched
    return net


# ----------------------------------------------------------------------- pieces


def test_subject_prefers_named_entities_then_strips_question_words():
    assert subject_for("Is the Acme Bolt phone worth buying?", ["Acme", "Bolt"]) == "Acme Bolt"
    assert subject_for("what is the best budget laptop under 500?", []).startswith("the best budget laptop")
    # a follow-up has no name of its own; it inherits the thread's subject, and a stray "Any" is not a product
    assert subject_for("What about its reviews?", ["Acme Bolt"]) == "Acme Bolt"
    assert subject_for("Is the Acme Bolt phone worth buying? Any problems?", ["Acme", "Bolt", "Any"]) == "Acme Bolt"


def test_review_urls_are_recognised_and_shops_are_not():
    assert is_review_url("https://www.reddit.com/r/x/comments/1/y")
    assert is_review_url("https://www.g2.com/products/acme/reviews")
    assert is_review_url("https://play.google.com/store/apps/details?id=a")
    assert is_review_url("https://blog.example/acme-long-term-review")
    assert not is_review_url("https://shop.example/acme-bolt")
    assert not is_review_url("https://notg2.com/")  # substring of a host is not the host


def test_review_hosts_are_community_tier():
    for url in ["https://www.g2.com/products/a", "https://www.trustpilot.com/review/a", "https://apps.apple.com/a", "https://www.reddit.com/r/a"]:
        assert sources.tier_for(url) == SourceTier.COMMUNITY, url


def test_only_complaints_that_recur_across_sources_are_reported():
    pages = [page_for(u) for u in URLS]
    found = {c.theme: c for c in mine_complaints(pages)}
    assert set(found) == {"battery life", "overheating"}, found.keys()
    assert found["battery life"].sources == 3 and found["overheating"].sources == 3
    # one grumble on one page is an anecdote
    assert "customer support" not in found
    # "No crashes though" is praise, and the app's one crash report is a single source
    assert "crashes and bugs" not in found
    assert "battery drains" in found["battery life"].example.lower()


def test_blocked_or_unreadable_pages_contribute_nothing():
    blocked = page_for("https://www.reddit.com/r/blocked", "blocked_page.html")
    assert not blocked.ok
    assert mine_complaints([blocked, blocked]) == []


def test_caveat_line_is_labelled_and_honest_in_every_state():
    none = caveat_lines(None)
    assert none == [] and caveat_lines(ReviewFindings()) == []
    found = ReviewFindings(attempted=True, pages_read=3, complaints=mine_complaints([page_for(u) for u in URLS]))
    line = caveat_lines(found)[0]
    assert line.startswith("Owner reports (community sources, not verified facts)") and "battery life (3 sources)" in line
    read_nothing = caveat_lines(ReviewFindings(attempted=True, pages_read=2))[0]
    assert "no complaint recurred" in read_nothing and "not the same as there being none" in read_nothing
    assert "couldn't read any review pages" in caveat_lines(ReviewFindings(attempted=True))[0]


async def test_gather_reviews_reads_only_review_pages(review_world):
    findings = await gather_reviews("Acme Bolt", base_settings())
    assert findings.pages_read == 3 and review_world.fetched and "https://shop.example/acme-bolt" not in review_world.fetched
    assert all(s["tier"] == SourceTier.COMMUNITY.value for s in findings.sources)
    assert {c.theme for c in findings.complaints} == {"battery life", "overheating"}


async def test_gather_reviews_degrades_when_search_and_fetch_fail(monkeypatch, net):
    async def boom(url, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(reviews, "fetch_page", boom)
    net.add_results(review_queries("Acme Bolt", 3)[0], [{"href": "https://www.reddit.com/r/a/comments/1", "title": "t", "snippet": "s"}])
    findings = await gather_reviews("Acme Bolt", base_settings())
    assert findings.pages_read == 0 and findings.complaints == [] and findings.sources[0]["read"] is False
    assert "couldn't read any review pages" in caveat_lines(findings)[0]


# ------------------------------------------------------------------- end to end


async def _run(settings, mode=ResearchMode.STANDARD, question=QUESTION):
    net_scripts = {
        "chatgpt": {"answer": "DIRECT ANSWER\nThe Acme Bolt phone launched in March 2026.\n\nKEY CLAIMS\n1. The Acme Bolt phone launched in March 2026.", "citations": []},
        "search": {"answer": "r", "citations": []},
    }
    adapters = adapters_from(net_scripts, settings)
    job = await ResearchRunner(settings, adapters, engine=None).run(Job(question=question, mode=mode, max_rounds=1))
    return job


async def test_product_question_gets_a_separate_owner_reports_line(review_world, settings):
    job = await _run(settings)
    assert job.analysis.intent == "product"
    assert job.reviews and job.reviews.attempted and job.reviews.subject == "Acme Bolt"
    caveats = " | ".join(job.final.caveats)
    assert "Owner reports (community sources, not verified facts)" in caveats and "battery life" in caveats
    # never blended into the answer
    answer = (job.final.answer + " " + job.final.why).lower()
    assert "battery" not in answer and "overheat" not in answer and "complain" not in answer
    # never evidence for a claim, never a verdict input
    assert not any(e.tier == SourceTier.COMMUNITY and "reddit" in (e.domain or "") for e in job.evidence)
    assert all(v.verdict != ClaimStatus.REFUTED for v in job.reports[-1].verdicts)


async def test_review_findings_survive_the_store(review_world, settings, tmp_path):
    from backend.storage.db import Store

    store = Store(base_settings(storage={"db_path": str(tmp_path / "r.db")}))
    job = await _run(settings)
    store.save_job(job)
    snap = store.job_snapshot(job.id)
    assert snap["reviews"]["attempted"] and snap["reviews"]["complaints"][0]["theme"] in {"battery life", "overheating"}


async def test_reviews_are_skipped_when_they_should_be(review_world):
    quick = await _run(base_settings(), mode=ResearchMode.QUICK)
    assert quick.reviews is None and not any("Owner reports" in c for c in quick.final.caveats)
    off = await _run(base_settings(search={"engines": ["google"], "max_results": 5, "review_queries": 0}))
    assert off.reviews is None
    not_product = await _run(base_settings(), question="When was the Acme Treaty signed in 2026?")
    assert not_product.analysis.intent != "product" and not_product.reviews is None
    assert not review_world.fetched