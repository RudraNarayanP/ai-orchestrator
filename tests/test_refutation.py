"""Refuting evidence (backlog item 2).

Before this, nothing in the codebase ever set ``Evidence.polarity = "refute"``, so
the verifier's REFUTED branch was unreachable from real data. These tests cover the
whole chain: the page-level detector, the counter-queries in the evidence pool, the
tier-weighted verdict, and the end-to-end case the backlog asks for -- a popular
claim repeated by several providers that a primary source contradicts.
"""

from __future__ import annotations

import pytest

from backend.evidence import pool as pool_module
from backend.evidence import search_http, sources
from backend.evidence.pool import build_pool, counter_query, counter_targets, is_material
from backend.evidence.sources import FetchedPage, check_refutation, gather_from_links
from backend.models import (
    Claim,
    ClaimStatus,
    Evidence,
    Job,
    ResearchMode,
    SourceCheckStatus,
    SourceTier,
)
from backend.orchestrator.runner import ResearchRunner
from backend.verification.llm import Endpoint
from backend.verification.verifier import Verifier
from tests.conftest import adapters_from, base_settings

CLAIM_499 = "The Acme Bolt costs $499."


def page(text: str, url: str = "https://store.example/bolt") -> FetchedPage:
    return FetchedPage(url=url, final_url=url, status=200, title="t", text=text, ok=True)


# ------------------------------------------------------------ page-level detector


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Acme confirmed the Bolt launches at $549 in the US.", "conflicting_figure"),
        ("Reports that the Acme Bolt costs $499 are false. The price is $549.", "explicit_correction"),
        ("The Acme Bolt costs $549, not $499, according to the store listing.", "explicit_correction"),
        ("Acme said the Bolt launches in 2024 at a price of $499 or 2025.", None),
    ],
)
def test_price_claim_is_refuted_only_by_a_page_that_says_otherwise(text, kind):
    found = check_refutation(CLAIM_499, page(text))
    assert (found or {}).get("kind") == kind, found


def test_page_that_agrees_or_is_off_topic_or_just_says_more_than_is_not_a_refutation():
    assert check_refutation(CLAIM_499, page("The Acme Bolt costs $499 at launch.")) is None
    assert check_refutation(CLAIM_499, page("The Zed Pro costs $549 and ships in March.")) is None
    # "more than $499" is not a correction of $499
    assert check_refutation(CLAIM_499, page("The Acme Bolt costs more than $499 once the case is added.")) is None
    assert check_refutation(CLAIM_499, page("")) is None


def test_year_and_negation_conflicts():
    year = check_refutation("Acme launched the Bolt in 2024.", page("Acme launched the Bolt in 2025 after delays."))
    assert year and year["kind"] == "conflicting_figure"
    neg = check_refutation(
        "Acme Bolt ships with a charger in the box.",
        page("Acme Bolt never ships with a charger in the box, the store confirmed."),
    )
    assert neg and neg["kind"] == "negation"
    myth = check_refutation(
        "Acme Bolt includes wireless charging.",
        page("The claim that Acme Bolt includes wireless charging is a myth and has been debunked."),
    )
    assert myth and myth["kind"] == "negation"
    assert check_refutation("Acme Bolt ships with a charger in the box.", page("Acme Bolt ships with a charger in the box.")) is None


# ----------------------------------------------------- gathering real (scripted) pages


@pytest.fixture
def pages(monkeypatch):
    table: dict[str, FetchedPage] = {}

    async def fake_fetch(url, *, timeout_s=30, max_chars=12000, browser_fetch=None):
        return table.get(url) or FetchedPage(url=url, ok=False, error="timeout")

    monkeypatch.setattr(sources, "fetch_page", fake_fetch)
    return table


async def test_counter_link_that_contradicts_becomes_refuting_evidence(pages):
    pages["https://store.example/bolt"] = page("The Acme Bolt launched at $549 and is sold only through the Acme store.")
    pages["https://blog.example/other"] = page("A long review of cameras and lenses that never mentions the product in question.", "https://blog.example/other")
    links = [
        {"href": "https://store.example/bolt", "claim_id": "c1", "claim_text": CLAIM_499, "counter": True},
        {"href": "https://blog.example/other", "claim_id": "c1", "claim_text": CLAIM_499, "counter": True},
    ]
    got = await gather_from_links("j", links, max_pages=5)
    assert [e.url for e in got] == ["https://store.example/bolt"], "an irrelevant counter-query hit is dropped, not recorded"
    ev = got[0]
    assert ev.polarity == "refute" and ev.check_status == SourceCheckStatus.CONFIRMED and ev.claim_id == "c1"
    assert "$549" in (ev.verbatim_excerpt or "")
    assert "contradicts the claim" in (ev.check_notes or "")


async def test_cited_page_that_contradicts_is_refuting_not_a_mere_citation_mismatch(pages):
    """A provider cites the store for $499; the store says $549. That is a refutation."""
    pages["https://store.example/bolt"] = page("The Acme Bolt is priced at $549 today.")
    pages["https://blog.example/vague"] = page("Acme makes the Bolt. Many people like the Bolt. The Bolt exists.", "https://blog.example/vague")
    got = await gather_from_links(
        "j",
        [
            {"href": "https://store.example/bolt", "claim_id": "c1", "claim_text": CLAIM_499},
            {"href": "https://blog.example/vague", "claim_id": "c1", "claim_text": CLAIM_499},
        ],
    )
    by_url = {e.url: e for e in got}
    assert by_url["https://store.example/bolt"].polarity == "refute"
    assert by_url["https://blog.example/vague"].polarity == "support"
    assert by_url["https://blog.example/vague"].check_status != SourceCheckStatus.CONFIRMED


# --------------------------------------------------------------------- the pool


def test_counter_query_targets_material_claims_and_carries_the_rebuttal_words():
    claim = Claim(job_id="j", claim=CLAIM_499, kind="statistic", provider_sources=["chatgpt", "gemini", "copilot"])
    opinion = Claim(job_id="j", claim="The Bolt is a delightful gadget.", kind="opinion")
    plain = Claim(job_id="j", claim="Acme makes many products across several categories worldwide.", kind="fact")
    assert is_material(claim) and not is_material(opinion) and not is_material(plain)
    query = counter_query(claim)
    for word in ("acme", "bolt", "correction", "rebuttal", "contradicts", "actually"):
        assert word in query.lower(), query
    targets = counter_targets([opinion, plain, claim], "question", limit=3)
    assert [c.id for c, _ in targets] == [claim.id]


async def test_pool_runs_a_counter_query_per_material_claim_and_marks_the_refuter(settings, pages, monkeypatch):
    pages["https://store.example/bolt"] = page("The Acme Bolt launched at $549 and is sold only through the Acme store.")
    claim = Claim(job_id="j", claim=CLAIM_499, kind="statistic", provider_sources=["chatgpt", "gemini"])
    asked: list[str] = []

    async def fake_search(query, *, engines=None, limit=8, timeout_s=25):
        asked.append(query)
        if "correction" in query:
            return [{"href": "https://store.example/bolt", "title": "Acme Bolt", "snippet": "price"}], {"used": "script", "tried": []}
        return [], {"used": None, "tried": []}

    monkeypatch.setattr(search_http, "search", fake_search)
    evidence, trace = await build_pool(
        job_id="j", question="How much is the Bolt?", claims=[claim], responses=[], mode=ResearchMode.STANDARD,
        settings=settings, engine=None,
    )
    assert any("correction rebuttal contradicts actually" in q for q in asked), asked
    refuting = [e for e in evidence if e.polarity == "refute"]
    assert len(refuting) == 1 and refuting[0].claim_id == claim.id
    assert trace["refuting"] == 1 and trace["counter_queries"][0]["claim_id"] == claim.id


async def test_pool_can_switch_refutation_search_off(pages, monkeypatch):
    settings = base_settings(search={"engines": ["google"], "max_results": 5, "refutation_queries": 0})
    claim = Claim(job_id="j", claim=CLAIM_499, kind="statistic")
    asked: list[str] = []

    async def fake_search(query, *, engines=None, limit=8, timeout_s=25):
        asked.append(query)
        return [], {"used": None, "tried": []}

    monkeypatch.setattr(search_http, "search", fake_search)
    _, trace = await build_pool(
        job_id="j", question="q", claims=[claim], responses=[], mode=ResearchMode.STANDARD, settings=settings, engine=None
    )
    assert not any("correction" in q for q in asked) and trace["counter_queries"] == []


# -------------------------------------------------------------------- the verdict


def _ev(claim: Claim, tier: SourceTier, polarity: str, domain: str, status=SourceCheckStatus.CONFIRMED) -> Evidence:
    return Evidence(
        job_id="j", claim_id=claim.id, url=f"https://{domain}/p", domain=domain, tier=tier, polarity=polarity,
        check_status=status, verbatim_excerpt="The Acme Bolt launched at $549." if polarity == "refute" else None,
    )


def _verifier() -> Verifier:
    return Verifier(Endpoint(provider="disabled", model="none", base_url=""), min_independent_sources=2)


def test_primary_source_contradiction_refutes_a_popular_claim():
    popular = Claim(job_id="j", claim=CLAIM_499, kind="statistic", provider_sources=["chatgpt", "gemini", "copilot", "qwen", "le_chat"])
    v = _verifier()
    alone = v._verdict_for(popular, [_ev(popular, SourceTier.PRIMARY_OFFICIAL, "refute", "store.example")])
    assert alone.verdict == ClaimStatus.REFUTED and "store.example" in " ".join(alone.strong_evidence) or alone.strong_evidence
    # even with a weak, unranked page backing it
    weak_support = v._verdict_for(
        popular,
        [_ev(popular, SourceTier.PRIMARY_OFFICIAL, "refute", "store.example"), _ev(popular, SourceTier.UNKNOWN, "support", "blog.example")],
    )
    assert weak_support.verdict == ClaimStatus.REFUTED
    assert "contradicted_by_source" in weak_support.problems


def test_contradiction_is_weighed_by_tier_not_counted():
    claim = Claim(job_id="j", claim=CLAIM_499, kind="statistic", provider_sources=["chatgpt"])
    v = _verifier()
    # comparable tiers disagree -> a genuine conflict
    both = v._verdict_for(claim, [_ev(claim, SourceTier.JOURNALISM, "support", "a.example"), _ev(claim, SourceTier.JOURNALISM, "refute", "b.example")])
    assert both.verdict == ClaimStatus.CONTESTED
    # a stray low-tier page does not overturn a primary source
    strong = v._verdict_for(
        claim,
        [
            _ev(claim, SourceTier.PRIMARY_OFFICIAL, "support", "a.example"),
            _ev(claim, SourceTier.GOVERNMENT, "support", "g.example"),
            _ev(claim, SourceTier.UNKNOWN, "refute", "random.example"),
        ],
    )
    assert strong.verdict == ClaimStatus.SUPPORTED and "contradicted_by_weaker_source" in strong.problems
    # a lone weak contradiction is not enough to call anything refuted
    weak = v._verdict_for(claim, [_ev(claim, SourceTier.UNKNOWN, "refute", "random.example")])
    assert weak.verdict == ClaimStatus.INSUFFICIENT_EVIDENCE and "weak_contradiction" in weak.problems


# ----------------------------------------------------------------------- end to end


async def test_popular_claim_contradicted_by_a_primary_source_is_refuted_end_to_end(settings, net):
    """Five providers say $499 and their citations do not hold up. The vendor's own
    page says $549. The popular claim is REFUTED, and the answer says so."""
    for url in ["https://blog.example/rumour", "https://forum.example/claims"]:
        net.fail(url, status=SourceCheckStatus.MISMATCH)
    net.confirm("https://store.example/bolt", tier=SourceTier.PRIMARY_OFFICIAL, polarity="refute",
                excerpt="The Acme Bolt launched at $549.")
    popular = Claim(job_id="j", claim=CLAIM_499, kind="statistic")
    net.add_results(counter_query(popular, "How much does the Acme Bolt cost?"), [
        {"href": "https://store.example/bolt", "title": "Acme Bolt - store", "snippet": "The Acme Bolt launched at $549."}
    ])

    wrong = "KEY CLAIMS\n1. " + CLAIM_499
    scripts = {
        name: {"answer": wrong, "citations": [{"url": "https://blog.example/rumour", "title": "Acme Bolt costs $499"}]}
        for name in ["chatgpt", "gemini", "copilot", "qwen", "le_chat"]
    }
    scripts["search"] = {"answer": "results", "citations": []}
    adapters = adapters_from(scripts, settings)
    job = await ResearchRunner(settings, adapters, engine=None).run(
        Job(question="How much does the Acme Bolt cost?", mode=ResearchMode.STANDARD, max_rounds=2)
    )

    claim = next(c for c in job.claims if "$499" in c.claim)
    assert len(claim.provider_sources) >= 3, "the wrong figure must be the popular one for this test to mean anything"
    verdict = next(v for v in job.reports[-1].verdicts if v.claim_id == claim.id)
    assert verdict.verdict == ClaimStatus.REFUTED, (verdict.verdict, verdict.problems)
    assert any(e.polarity == "refute" and e.claim_id == claim.id for e in job.evidence)
    assert job.final is not None and "$549" in job.final.answer and "store.example" in job.final.answer
    assert "$499" not in job.final.answer.replace("Acme Bolt costs $499", "")  # the false figure is not asserted
    assert any("store.example" in s.url for s in job.final.sources)
    # provider agreement was recorded, never counted
    assert "not counted" in verdict.reasoning.lower()


async def test_refuting_page_never_counts_as_establishing_the_claim(settings, net):
    """Regression guard: a CONFIRMED refuting page must not make a claim 'established'."""
    net.confirm("https://store.example/bolt", tier=SourceTier.PRIMARY_OFFICIAL, polarity="refute")
    popular = Claim(job_id="j", claim=CLAIM_499, kind="statistic")
    net.add_results(counter_query(popular, "q"), [{"href": "https://store.example/bolt", "title": "Acme", "snippet": "x"}])
    scripts = {"chatgpt": {"answer": "KEY CLAIMS\n1. " + CLAIM_499}, "search": {"answer": "r", "citations": []}}
    adapters = adapters_from(scripts, settings)
    runner = ResearchRunner(settings, adapters, engine=None)
    job = await runner.run(Job(question="How much does the Acme Bolt cost?", mode=ResearchMode.STANDARD, max_rounds=1))
    first = job.assessments[0]
    assert first.established == [], first.established
    assert first.sufficient is False
    assert "contradicted" in first.reason