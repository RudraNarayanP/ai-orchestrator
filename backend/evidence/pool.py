"""Evidence pool: turning citations and searches into checked evidence.

A provider's link list is a set of *allegations* about sources. This module makes
them into evidence by opening the pages and recording whether the claim actually
appears in them. Search results enter the same funnel, so a claim supported by a
page we read and a claim supported by "ChatGPT said Reuters" are visibly
different objects in the ledger.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Iterable

from backend.evidence.sources import TIER_WEIGHT, gather_from_links, tier_for
from backend.models import Claim, Evidence, ProviderResponse, ResearchMode
from backend.research.claims import signature
from backend.settings import Settings

PRIORITY_KINDS = {"statistic", "date", "legal", "scientific", "product", "ranking"}


def claim_for_link(link: dict[str, Any], claims: list[Claim]) -> Claim | None:
    """Attach a cited URL to the claim it was offered for.

    Two guards, because a wrong attachment is worse than none: a claim is only
    eligible if it shares distinctive words with the link, and every figure or
    year in the claim must actually appear in the link. Without that second rule a
    "$549" source gets filed under the "$499" claim simply because both sentences
    are about the same product -- which manufactures the very citation mismatch
    this layer is supposed to catch.
    """
    hay = f"{link.get('title') or ''} {link.get('snippet') or ''} {link.get('href') or ''}".lower()
    scored: list[tuple[float, Claim]] = []
    for claim in claims:
        sig = signature(claim.claim)
        tokens = [t for t in sig["tokens"][:10] if not t.replace(".", "").isdigit()]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in hay)
        score = hits / len(tokens)
        figures = set(sig["numbers"]) | set(sig["years"])
        digits = {re.sub(r"[^0-9]", "", f) for f in figures if f}
        digits = {d for d in digits if len(d) >= 2}
        # All distinctive figures must appear, not merely one of them. Two claims
        # about the same 2026 deadline differing only by day would otherwise both
        # "match" any page mentioning 2026, and the wrong one would collect the
        # evidence. An unattached source stays job-level, which is honest; a
        # misattached one becomes fabricated support for a claim it never mentioned.
        if digits and not all(d in hay.replace(",", "") for d in digits):
            continue
        if score >= 0.3:
            scored.append((score, claim))
    if not scored:
        return None
    scored.sort(key=lambda pair: -pair[0])
    return scored[0][1]


def _rank_claims(claims: list[Claim]) -> list[Claim]:
    return sorted(
        claims,
        key=lambda c: (c.kind in PRIORITY_KINDS, bool(signature(c.claim)["numbers"] or signature(c.claim)["years"]), len(c.provider_sources)),
        reverse=True,
    )


def _claim_query(claim: Claim, question: str) -> str:
    sig = signature(claim.claim)
    figures = list(dict.fromkeys((sig["numbers"] + sig["years"])[:2]))
    cleaned_numbers = {re.sub(r"[^0-9.]", "", f) for f in figures}
    words = [t for t in sig["tokens"][:8] if t not in cleaned_numbers and not t.replace(".", "").isdigit()]
    query = " ".join(words + figures).strip()
    if len(query) < 8:
        query = " ".join(words[:6]).strip() or question[:150]
    return query[:180]


def search_queries(claims: list[Claim], question: str, *, limit: int = 4) -> list[str]:
    """Queries aimed at the checkable parts, not at the whole question."""
    queries: list[str] = []
    for claim in _rank_claims(claims)[:limit]:
        query = _claim_query(claim, question)
        if query and query not in queries:
            queries.append(query)
    if not queries:
        queries.append(question[:180])
    return queries[:limit]


COUNTER_SUFFIX = " correction rebuttal contradicts actually"


def is_material(claim: Claim) -> bool:
    """Worth a hunt for refuting evidence: checkable, and not an opinion."""
    if claim.kind in {"opinion", "contradiction"}:
        return False
    sig = signature(claim.claim)
    return claim.kind in PRIORITY_KINDS or bool(sig["numbers"] or sig["years"] or sig["dates"])


def counter_query(claim: Claim, question: str = "") -> str:
    """The adversarial query for one claim: the claim's own words plus the words
    pages use when they are disputing something."""
    return (_claim_query(claim, question) + COUNTER_SUFFIX)[:200]


def counter_targets(claims: list[Claim], question: str, *, limit: int) -> list[tuple[Claim, str]]:
    """Material claims to try to refute, most-repeated and most-checkable first.

    Popularity ranks *which claim gets challenged first*; it is never evidence.
    """
    out: list[tuple[Claim, str]] = []
    seen: set[str] = set()
    for claim in _rank_claims([c for c in claims if is_material(c)]):
        query = counter_query(claim, question)
        if query in seen:
            continue
        seen.add(query)
        out.append((claim, query))
        if len(out) >= limit:
            break
    return out


def browser_fetch_factory(engine: Any, settings: Settings) -> Callable[..., Awaitable[Any]]:
    """Read a page through a dedicated OmniBrain window when plain HTTP fails.

    Some sources are JS-only. This uses the same isolated profile infrastructure,
    so it never reaches into the user's own browser.
    """

    async def browser_fetch(url: str, *, timeout_s: int = 30, max_chars: int = 12000):
        from backend.evidence.sources import FetchedPage, _html_to_text, extract_date

        try:
            page = await engine.open_research_page("fetch", url, key=f"fetch_{abs(hash(url)) % 61}")
            await page.wait_for_timeout(1200)
            html = await page.content()
            text = _html_to_text(await page.evaluate("() => document.body ? document.body.innerText : ''") or _html_to_text(html))
            title = await page.title()
            return FetchedPage(
                url=url,
                final_url=page.url,
                status=200,
                title=(title or "")[:220],
                text=text[:max_chars],
                html_head=html[:12000],
                published=extract_date(html[:40000], text[:6000]),
                ok=bool(text.strip()),
                transport="browser",
            )
        except Exception:  # noqa: BLE001
            return None

    return browser_fetch


async def build_pool(
    *,
    job_id: str,
    question: str,
    claims: list[Claim],
    responses: list[ProviderResponse],
    mode: ResearchMode,
    settings: Settings,
    engine: Any,
    search_adapter: Any = None,
    round_no: int = 1,
    emit: Callable[..., Awaitable[None]] | None = None,
    run_searches: bool = True,
    max_pages: int | None = None,
) -> tuple[list[Evidence], dict[str, Any]]:
    """Returns evidence records plus a small trace of what was attempted."""
    emit = emit or _noop
    budget = max_pages or {ResearchMode.QUICK: 6, ResearchMode.STANDARD: 12, ResearchMode.DEEP_RESEARCH: 20}[mode]
    browser_fetch = browser_fetch_factory(engine, settings) if engine is not None else None

    raw_links: list[dict[str, Any]] = []
    for response in responses:
        if response.status.value not in {"completed", "timeout"}:
            continue
        for citation in response.citations:
            claim = claim_for_link({"title": citation.title, "snippet": citation.snippet, "href": citation.url}, claims)
            raw_links.append(
                {
                    "href": citation.url,
                    "title": citation.title,
                    "snippet": citation.snippet,
                    "claim_id": claim.id if claim else None,
                    "claim_text": claim.claim if claim else "",
                    "polarity": "support",
                    "origin": "provider",
                    "cited_by": response.provider,
                }
            )

    searched: list[dict[str, Any]] = []
    queries = search_queries(claims, question, limit=2 if mode == ResearchMode.QUICK else 4) if run_searches else []
    discovery: list[dict[str, Any]] = []
    if queries:
        # HTTP discovery first: cheap, and it still works when a browser page
        # decides to show a robot check. The in-browser search window is the
        # fallback, not the default.
        from backend.evidence.search_http import search as http_search

        for query in queries:
            try:
                items, trace = await http_search(
                    query,
                    engines=settings.search.engines,
                    limit=settings.search.max_results,
                    timeout_s=settings.search.per_query_timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                discovery.append({"query": query[:70], "error": f"{type(exc).__name__}"})
                continue
            discovery.append({"query": query[:70], "used": trace.get("used"), "engines": [t.get("engine") for t in trace.get("tried", [])]})
            for item in items:
                claim = claim_for_link(item, claims)
                searched.append(
                    {
                        **item,
                        "claim_id": claim.id if claim else None,
                        "claim_text": claim.claim if claim else "",
                        "polarity": "support",
                        "origin": "search",
                        "cited_by": query,
                    }
                )

        if not searched and search_adapter is not None:
            await emit("evidence", "http discovery found nothing -- searching in the dedicated browser window", None, round_no)
            for query in queries:
                try:
                    response = await search_adapter.ask(job_id, query, round_no)
                except Exception as exc:  # noqa: BLE001
                    await emit("evidence", f"browser search failed: {type(exc).__name__}", None, round_no)
                    continue
                if response.status.value != "completed":
                    continue
                for citation in response.citations:
                    claim = claim_for_link({"title": citation.title, "snippet": citation.snippet, "href": citation.url}, claims)
                    searched.append(
                        {
                            "href": citation.url,
                            "title": citation.title,
                            "snippet": citation.snippet,
                            "claim_id": claim.id if claim else None,
                            "claim_text": claim.claim if claim else "",
                            "polarity": "support",
                            "origin": "search",
                            "cited_by": query,
                        }
                    )

    links = _rank_links(raw_links + searched, claims)[:budget]
    evidence = await gather_from_links(
        job_id,
        links,
        max_pages=budget,
        concurrency=min(6, max(2, settings.research.max_workers)),
        browser_fetch=browser_fetch,
        round_no=round_no,
    )

    # Look for the other side. Nothing above ever asks "who says this is wrong?",
    # so without this the REFUTED verdict could never come from real data.
    counter_trace: list[dict[str, Any]] = []
    counter_links = await _counter_links(
        claims, question, mode, settings, links, emit, round_no, counter_trace, enabled=run_searches
    )
    if counter_links:
        evidence.extend(
            await gather_from_links(
                job_id,
                counter_links,
                max_pages=len(counter_links),
                concurrency=min(6, max(2, settings.research.max_workers)),
                browser_fetch=browser_fetch,
                round_no=round_no,
                origin="search",
            )
        )

    # keep provenance the fetch layer does not carry
    by_url = {l.get("href"): l for l in links}
    for ev in evidence:
        link = by_url.get(ev.url) or {}
        if link.get("origin") and ev.origin == "provider":
            ev.origin = link["origin"]
        if link.get("cited_by") and ev.check_notes:
            ev.check_notes += f"; offered by {link['cited_by']}"

    trace = {
        "links_collected": len(raw_links),
        "links_from_search": len(searched),
        "pages_fetched": len(evidence),
        "confirmed": sum(1 for e in evidence if e.check_status.value == "confirmed"),
        "mismatched": sum(1 for e in evidence if e.check_status.value in {"mismatch", "hallucinated", "broken_url"}),
        "distinct_domains": len({e.domain for e in evidence if e.domain}),
        "queries": queries,
        "discovery": discovery,
        "counter_queries": counter_trace,
        "refuting": sum(1 for e in evidence if e.polarity == "refute"),
    }
    return evidence, trace


async def _counter_links(
    claims: list[Claim],
    question: str,
    mode: ResearchMode,
    settings: Settings,
    taken: list[dict[str, Any]],
    emit: Callable[..., Awaitable[None]],
    round_no: int,
    trace: list[dict[str, Any]],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    limit = int(settings.search.refutation_queries)
    if not enabled or limit <= 0 or not claims:
        return []
    limit = {ResearchMode.QUICK: min(1, limit), ResearchMode.STANDARD: limit, ResearchMode.DEEP_RESEARCH: limit + 2}[mode]
    from backend.evidence.search_http import search as http_search

    already = {l.get("href") for l in taken}
    out: list[dict[str, Any]] = []
    for claim, query in counter_targets(claims, question, limit=limit):
        try:
            items, info = await http_search(
                query,
                engines=settings.search.engines,
                limit=settings.search.max_results,
                timeout_s=settings.search.per_query_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            trace.append({"query": query[:80], "error": type(exc).__name__})
            continue
        trace.append({"query": query[:80], "claim_id": claim.id, "results": len(items), "used": info.get("used")})
        kept = 0
        for item in items:
            href = item.get("href") or ""
            if not href.startswith("http") or href in already:
                continue
            already.add(href)
            out.append(
                {
                    **item,
                    "claim_id": claim.id,
                    "claim_text": claim.claim,
                    "polarity": "support",  # decided after the page is read, never before
                    "counter": True,
                    "origin": "search",
                    "cited_by": query,
                }
            )
            kept += 1
            if kept >= 3:
                break
    if out:
        await emit("evidence", f"looked for sources that contradict {len(trace)} claim(s); reading {len(out)} page(s)", None, round_no)
    return out


def _rank_links(links: list[dict[str, Any]], claims: list[Claim]) -> list[dict[str, Any]]:
    """Spend the fetch budget where the claim actually needs checking."""
    claim_by_id = {c.id: c for c in claims}
    scored: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for link in links:
        href = link.get("href") or ""
        if not href.startswith("http") or href in seen:
            continue
        seen.add(href)
        score = 0.0
        claim = claim_by_id.get(link.get("claim_id") or "")
        if claim:
            sig = signature(claim.claim)
            score += 3.0 if (sig["numbers"] or sig["years"]) else 1.0
            score += 1.5 if claim.kind in PRIORITY_KINDS else 0.0
            score += min(len(claim.provider_sources), 4) * 0.25
        score += TIER_WEIGHT.get(tier_for(href), 0.3) * 2.0
        if link.get("origin") == "search":
            score += 0.6
        scored.append((score, link))
    scored.sort(key=lambda pair: -pair[0])
    return [link for _, link in scored]


async def _noop(*args: Any, **kwargs: Any) -> None:
    return None
