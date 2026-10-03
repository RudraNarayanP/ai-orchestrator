"""Product / service review behaviour (spec 8).

For a "is X any good?" question the facts (price, specs, release date) go through the
normal evidence ledger. What owners and users *say* about living with the thing is a
different kind of information, so it is gathered separately and kept separate:

* sources are review/community sites (Reddit, G2, Trustpilot, app stores, long-term
  user threads) and are always tagged ``SourceTier.COMMUNITY``;
* we mine them for complaints that RECUR across independent sources -- one angry post
  is an anecdote, the same complaint on three sites is worth a line;
* the result is a distinct, labelled caveat ("Owner reports (community sources, not
  verified facts)"). It never enters ``answer``, never attaches to a claim as
  evidence, and never moves a verdict. Community chatter is not a primary source.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

from backend.evidence import sources
from backend.evidence.sources import fetch_page, tier_for
from backend.models import Complaint, ReviewFindings, SourceTier
from backend.research import memory
from backend.settings import Settings

REVIEW_HOST_RE = re.compile(
    r"(reddit\.com|(?:^|\.)(?:g2|capterra|trustpilot|producthunt|softwareadvice|sitejabber)\.com|"
    r"apps\.apple\.com|play\.google\.com|news\.ycombinator\.com|"
    r"gartner\.com/reviews|bestbuy\.com|amazon\.[a-z.]+|stackexchange\.com|quora\.com)",
    re.I,
)
REVIEW_PATH_RE = re.compile(r"(review|complaint|problem|experience|long[- ]term|forum|thread|discussion)", re.I)

QUERY_TEMPLATES = [
    "{subject} long-term review problems reddit",
    "{subject} reviews complaints G2 OR Trustpilot OR Capterra",
    "{subject} app store reviews common complaints",
]

MIN_SOURCES = 2  # a complaint is "recurring" only when this many independent pages make it

THEMES: list[tuple[str, re.Pattern[str]]] = [
    ("battery life", re.compile(r"battery (?:life )?(?:drains?|dies|died|is (?:bad|poor|terrible|awful)|barely|lasts? (?:only|less))|poor battery|bad battery", re.I)),
    ("overheating", re.compile(r"overheat\w*|runs? (?:very |too )?hot|thermal throttl\w*", re.I)),
    ("reliability", re.compile(r"stopped working|died after|broke (?:after|within)|keeps? (?:breaking|disconnecting)|unreliable|failed after|dead after|defective", re.I)),
    ("crashes and bugs", re.compile(r"\bcrash(?:es|ed|ing)?\b|\bbuggy\b|\bbugs?\b|glitch\w*|freez\w+", re.I)),
    ("customer support", re.compile(r"(?:customer|tech(?:nical)?) support (?:is |was |has been )?(?:\w+ )?(?:bad|poor|terrible|useless|unresponsive|slow|awful)|support never|unhelpful support|no response from support", re.I)),
    ("price and value", re.compile(r"overpriced|not worth (?:the|it)|too expensive|price hike|hidden fees?|raised (?:the |their )?prices?", re.I)),
    ("billing and cancellation", re.compile(r"hard to cancel|can'?t cancel|cancel(?:l?ation)? (?:is )?(?:impossible|difficult)|charged me|billing (?:issue|problem|error)|refund (?:denied|refused|never)|unauthori[sz]ed charge", re.I)),
    ("slowness", re.compile(r"\blag(?:gy|s|ging)?\b|\bslow(?:s|ed)? down\b|sluggish|takes forever", re.I)),
    ("privacy and data", re.compile(r"privacy (?:concern|issue|problem)|sells? (?:your |my )?data", re.I)),
    ("build quality", re.compile(r"flimsy|cheap(?:ly)? made|feels? cheap|poor build|scratch(?:es|ed)? easily|cracked", re.I)),
]
_NEGATION_BEFORE = re.compile(r"(?:\b(?:no|never|not|without|zero|didn'?t|doesn'?t|haven'?t|hasn'?t|isn'?t|wasn'?t|nothing|free of)\b\W+(?:\w+\W+){0,2})$", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_LEAD = re.compile(
    r"^\s*(?:is|are|was|were|does|do|did|what|which|who|how|should|would|can|could|tell me about|"
    r"give me|show me|any good|worth it)\b[\s,]*",
    re.I,
)


def subject_for(question: str, entities: list[str] | None = None) -> str:
    """What the reviews are about: the named product, else the question without its question words."""
    own = memory.entities_in(question)
    inherited = [e for e in (entities or []) if e and len(e) >= 3 and e.lower() not in memory._NOT_ENTITY]
    names = own[:2] or inherited[:3]
    if names:
        return " ".join(dict.fromkeys(names))
    cleaned = question.strip().rstrip("?.! ")
    for _ in range(3):
        stripped = _LEAD.sub("", cleaned, count=1)
        if stripped == cleaned:
            break
        cleaned = stripped
    return cleaned[:90].strip() or question.strip()[:90]


def review_queries(subject: str, limit: int) -> list[str]:
    return [t.format(subject=subject) for t in QUERY_TEMPLATES[: max(0, limit)]]


def is_review_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    return bool(REVIEW_HOST_RE.search(host) or tier_for(url) == SourceTier.COMMUNITY or REVIEW_PATH_RE.search(parsed.path or ""))


def mine_complaints(pages: list[sources.FetchedPage]) -> list[Complaint]:
    """Recurring complaints across independent pages, most widespread first."""
    seen: dict[str, dict[str, Any]] = {}
    for page in pages:
        if not page.ok or not page.text:
            continue
        domain = (urlparse(page.final_url or page.url).netloc or "").lower().removeprefix("www.")
        counted_here: set[str] = set()
        sentences_done: set[str] = set()
        for sentence in _SENTENCE_SPLIT.split(page.text):
            sentence = " ".join(sentence.split())
            if not (12 <= len(sentence) <= 400) or sentence.lower() in sentences_done:
                continue
            sentences_done.add(sentence.lower())
            for theme, pattern in THEMES:
                match = pattern.search(sentence)
                if not match or _NEGATION_BEFORE.search(sentence[: match.start()]):
                    continue
                entry = seen.setdefault(theme, {"mentions": 0, "urls": set(), "domains": set(), "example": "", "example_url": ""})
                entry["mentions"] += 1
                entry["urls"].add(page.url)
                entry["domains"].add(domain)
                if not entry["example"]:
                    entry["example"], entry["example_url"] = sentence[:240], page.url
                counted_here.add(theme)
    out = [
        Complaint(
            theme=theme,
            mentions=data["mentions"],
            sources=len(data["urls"]),
            domains=sorted(data["domains"]),
            example=data["example"],
            example_url=data["example_url"],
        )
        for theme, data in seen.items()
        if len(data["urls"]) >= MIN_SOURCES
    ]
    out.sort(key=lambda c: (-c.sources, -c.mentions, c.theme))
    return out


async def gather_reviews(
    subject: str,
    settings: Settings,
    *,
    deep: bool = False,
    browser_fetch: Any | None = None,
    emit: Callable[..., Awaitable[None]] | None = None,
) -> ReviewFindings:
    """Find review pages, read them, and report what recurs. Never raises."""
    from backend.evidence import search_http

    findings = ReviewFindings(attempted=True, subject=subject)
    limit = int(settings.search.review_queries)
    queries = review_queries(subject, limit)
    findings.queries = queries
    budget = 10 if deep else 6
    candidates: list[str] = []
    for query in queries:
        try:
            items, _trace = await search_http.search(
                query,
                engines=settings.search.engines,
                limit=settings.search.max_results,
                timeout_s=settings.search.per_query_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            findings.note = f"review search failed: {type(exc).__name__}"
            continue
        for item in items:
            href = item.get("href") or ""
            if href.startswith("http") and href not in candidates and is_review_url(href):
                candidates.append(href)
    # review platforms first, then other community pages
    candidates.sort(key=lambda u: 0 if REVIEW_HOST_RE.search(urlparse(u).netloc or "") else 1)
    candidates = candidates[:budget]

    gate = asyncio.Semaphore(4)

    async def read(url: str) -> sources.FetchedPage:
        async with gate:
            try:
                return await fetch_page(url, browser_fetch=browser_fetch, max_chars=settings.search.fetch_body_chars)
            except Exception as exc:  # noqa: BLE001
                return sources.FetchedPage(url=url, ok=False, error=type(exc).__name__)

    pages = list(await asyncio.gather(*(read(u) for u in candidates))) if candidates else []
    for page in pages:
        findings.sources.append(
            {
                "url": page.url,
                "domain": (urlparse(page.url).netloc or "").lower(),
                "tier": SourceTier.COMMUNITY.value,
                "read": bool(page.ok),
                "error": None if page.ok else (page.error or f"http {page.status}"),
            }
        )
    findings.pages_read = sum(1 for p in pages if p.ok)
    findings.complaints = mine_complaints(pages)
    if emit:
        summary = (
            "recurring in owner reviews: " + ", ".join(c.theme for c in findings.complaints[:4])
            if findings.complaints
            else f"read {findings.pages_read} review page(s); no complaint recurred across sources"
        )
        await emit("evidence", summary, None, 1)
    return findings


def caveat_lines(findings: ReviewFindings | None) -> list[str]:
    """The distinct, labelled line for the final answer. Empty when nothing was attempted."""
    if findings is None or not findings.attempted:
        return []
    if findings.complaints:
        parts = [f"{c.theme} ({c.sources} sources)" for c in findings.complaints[:4]]
        return [
            "Owner reports (community sources, not verified facts): recurring complaints - "
            + "; ".join(parts)
            + "."
        ]
    if findings.pages_read:
        return [
            f"Owner reports: I read {findings.pages_read} review page(s) (community sources) and no complaint "
            "recurred across them, which is not the same as there being none."
        ]
    return ["Owner reports: I couldn't read any review pages, so I can't say what users complain about."]