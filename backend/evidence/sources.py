"""Source verification: actually open the pages the providers cited.

"According to Reuters..." is not evidence until Reuters has been read. This module
fetches a cited URL, pulls the body text and any publication date, classifies the
domain's tier, and then checks whether the page really contains the claim -- by
looking for the claim's distinctive tokens and, crucially, its numbers and dates.

A source that does not contain the figure it is credited with is a citation
mismatch, not a weak support. A domain that does not resolve is reported as
hallucinated rather than merely unreachable.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from backend.models import Evidence, SourceCheckStatus, SourceTier, new_id

GOV_RE = re.compile(r"\.(gov|mil)(\.[a-z]{2})?$|\.gov\.uk$|europa\.eu$|un\.org$|who\.int$|worldbank\.org$|imf\.org$|oecd\.org$")
EDU_RE = re.compile(r"\.edu(\.[a-z]{2})?$|\.ac\.[a-z]{2}$")
JOURNAL_RE = re.compile(
    r"(nature\.com|science\.org|nejm\.org|thelancet\.com|jama\.network|bmj\.com|pnas\.org|"
    r"arxiv\.org|pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov|ieee\.org|acm\.org|"
    r"sciencedirect\.com|link\.springer\.com|onlinelibrary\.wiley\.com|tandfonline\.com|"
    r"Cell\.com|papers\.arxiv\.org|openreview\.net|medrxiv\.org|bioRxiv\.org)",
    re.I,
)
PRIMARY_RE = re.compile(
    r"(whitehouse\.gov|prewhitehouse\.gov|sec\.gov|press\.openai\.com|openai\.com|blog\.google|"
    r"ai\.google|deepmind\.google|about\.google|research\.google|microsoft\.com/en-us/"
    r".*(blog|news|ai)|blogs\.microsoft\.com|news\.microsoft\.com|meta\.com|about\.meta|"
    r"anthropic\.com|mistral\.ai|deepseek\.com|qwen\.ai|changelog|documentation|"
    r"docs\.|developer\.|ir\.[a-z0-9-]+\.(com|net)|investors\.|press-release|newsroom|"
    r"products\.)",
    re.I,
)
NEWS_RE = re.compile(
    r"(reuters\.com|ap\.com|associatedpress\.com|bbc\.(com|co\.uk)|bloomberg\.com|ft\.com|"
    r"wsj\.com|nytimes\.com|washingtonpost\.com|theguardian\.com|arstechnica\.com|"
    r"theverge\.com|wired\.com|techcrunch\.com|axios\.com|nist\.gov|cnbc\.com|"
    r"economist\.com|newscientist\.com|scientificamerican\.com|npr\.org|aljazeera\.com|"
    r"france24\.com|straitstimes\.com|timesofindia\.indiatimes\.com|hindustantimes\.com)",
    re.I,
)
INDUSTRY_RE = re.compile(
    r"(infoq\.com|lwn\.net|phoronix\.com|tomshardware\.com|anandtech\.com|semianalysis\.com|"
    r"venturebeat\.com|zdnet\.com|theregister\.(com|co\.uk)|stackoverflow\.blog|github\.blog)",
    re.I,
)
COMMUNITY_RE = re.compile(
    r"(reddit\.com|news\.ycombinator\.com|hn\.a|"
    r"quora\.com|stackexchange\.com|discuss\.[a-z0-9-]+\.[a-z]{2,}|forum[s]?\.[a-z0-9-]+\.[a-z]{2,}|"
    r"github\.com/.*/discussions|lobste\.rs|"
    r"(?:^|\.)(?:g2|capterra|trustpilot|producthunt|softwareadvice|sitejabber)\.com|"
    r"apps\.apple\.com|play\.google\.com)",
    re.I,
)
SOCIAL_RE = re.compile(
    r"(x\.com|twitter\.com|facebook\.com|instagram\.com|tiktok\.com|youtube\.com|"
    r"linkedin\.com/posts|threads\.net|bsky\.app|mastodon\.social)",
    re.I,
)
AI_SOURCE_RE = re.compile(r"(chatgpt\.com|gemini\.google|copilot\.microsoft|meta\.ai|chat\.mistral|pi\.ai|perplexity\.ai)")

BLOCK_HINT_RE = re.compile(
    r"(are you a robot|unusual traffic|access denied|enable javascript and cookies|"
    r"verify you are a human|captcha|attention required|rate limit exceeded|403 forbidden)",
    re.I,
)
PAYWALL_RE = re.compile(r"(subscribe to read|sign in to read|this article is limited|paywall|metered access)", re.I)

DATE_META_RE = re.compile(
    r"(?:article:published_time|citation_publication_date|datePublished|published_time|pubDate|dc\.date)"
    r"""['"]?\s*(?::|=>|content=)\s*['"]?([0-9]{4}-[0-9]{2}-[0-9]{2}[T ][^\s'"<]*)""",
    re.I,
)
TIME_TAG_RE = re.compile(r"<time[^>]+datetime=[\"']([^\"']+)[\"']", re.I)
ISO_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
MONTH_DAY_YEAR_RE = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{1,2}),?\s+(20\d{2})",
    re.I,
)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
)


@dataclass
class FetchedPage:
    url: str
    final_url: str = ""
    status: int = 0
    title: str | None = None
    text: str = ""
    html_head: str = ""
    published: str | None = None
    ok: bool = False
    error: str | None = None
    transport: str = "http"
    notes: list[str] = field(default_factory=list)

    @property
    def words(self) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9']+", self.text.lower()) if len(w) > 2}


def tier_for(url: str) -> SourceTier:
    host = (urlparse(url).netloc or "").lower()
    if not host:
        return SourceTier.UNKNOWN
    if AI_SOURCE_RE.search(host):
        return SourceTier.AI_UNSOURCED
    if GOV_RE.search(host):
        # A regulator's own filing is primary evidence about the filing.
        return SourceTier.GOVERNMENT
    if JOURNAL_RE.search(host):
        return SourceTier.ORIGINAL_RESEARCH
    if PRIMARY_RE.search(host):
        return SourceTier.PRIMARY_OFFICIAL
    if NEWS_RE.search(host):
        return SourceTier.JOURNALISM
    if INDUSTRY_RE.search(host):
        return SourceTier.TECHNICAL
    if SOCIAL_RE.search(host):
        return SourceTier.SOCIAL
    if COMMUNITY_RE.search(host):
        return SourceTier.COMMUNITY
    if EDU_RE.search(host):
        return SourceTier.ORIGINAL_RESEARCH
    return SourceTier.UNKNOWN


TIER_WEIGHT = {
    SourceTier.PRIMARY_OFFICIAL: 1.0,
    SourceTier.ORIGINAL_RESEARCH: 0.95,
    SourceTier.GOVERNMENT: 0.9,
    SourceTier.JOURNALISM: 0.72,
    SourceTier.TECHNICAL: 0.6,
    SourceTier.UNKNOWN: 0.4,
    SourceTier.EXPERT: 0.45,
    SourceTier.COMMUNITY: 0.25,
    SourceTier.SOCIAL: 0.15,
    SourceTier.AI_UNSOURCED: 0.05,
}


def _looks_like_host(host: str) -> bool:
    if not host or "." not in host:
        return False
    if re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", host):
        return True
    return bool(re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", host.lower()))


def extract_date(page_html: str, text: str = "") -> str | None:
    for pattern in (DATE_META_RE, TIME_TAG_RE):
        match = pattern.search(page_html or "")
        if match:
            return match.group(1)[:32]
    ld = re.search(r'"datePublished"\s*:\s*"([^"]+)"', page_html or "")
    if ld:
        return ld.group(1)[:32]
    body = (text or page_html or "")[:8000]
    iso = ISO_RE.search(body)
    if iso:
        return iso.group(0)
    mdy = MONTH_DAY_YEAR_RE.search(body)
    if mdy:
        return f"{mdy.group(1)} {mdy.group(2)}, {mdy.group(3)}"
    return None


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", html or "")
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    html = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr)[^>]*>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#x27;", "'")
        .replace("&quot;", '"')
        .replace("&nbsp;", " ")
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


async def fetch_page(
    url: str,
    *,
    timeout_s: int = 30,
    max_chars: int = 12000,
    browser_fetch: Any | None = None,
) -> FetchedPage:
    """httpx first (cheap), the real browser second (JS-heavy pages)."""
    host = (urlparse(url).netloc or "").lower()
    if not url.startswith("http") or not _looks_like_host(host):
        return FetchedPage(url=url, ok=False, error="not a resolvable domain", transport="none")
    try:
        async with httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9", "Accept": "text/html,application/xhtml+xml"},
        ) as client:
            response = await client.get(url)
        html = response.text or ""
        status = response.status_code
        text = _html_to_text(html)
        page = FetchedPage(
            url=url,
            final_url=str(response.url),
            status=status,
            title=(re.search(r"(?is)<title[^>]*>(.*?)</title>", html) or [None, None])[1],
            text=text[:max_chars],
            html_head=html[:12000],
            published=extract_date(html[:40000], text[:6000]),
            ok=status < 400 and bool(text.strip()),
            transport="http",
        )
        page.title = _html_to_text(page.title or "")[:220] or None
        if BLOCK_HINT_RE.search(text[:3000]):
            page.notes.append("bot or consent gate")
        if PAYWALL_RE.search(text[:4000]):
            page.notes.append("possible paywall")
        if status in {404, 410}:
            page.error = f"HTTP {status}"
        if page.ok or status in {404, 410}:
            return page
    except httpx.TimeoutException:
        page = FetchedPage(url=url, ok=False, error="timeout", transport="http")
    except httpx.HTTPError as exc:
        page = FetchedPage(url=url, ok=False, error=f"{type(exc).__name__}", transport="http")
    except Exception as exc:  # noqa: BLE001
        page = FetchedPage(url=url, ok=False, error=f"{type(exc).__name__}: {exc}", transport="http")

    if browser_fetch is not None:
        try:
            got = await browser_fetch(url, timeout_s=timeout_s, max_chars=max_chars)
            if got and got.text:
                got.notes.append("http path failed, read through the dedicated browser window")
                return got
        except Exception as exc:  # noqa: BLE001
            page.notes.append(f"browser fetch failed: {type(exc).__name__}")
    return page


_NAME_STOP = {
    "university", "college", "section", "article", "articles", "chapter", "court", "parliament", "government", "regulation",
    "regulations", "constitution", "english", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december",
    "under", "which", "where", "while", "their", "after", "before", "state", "states", "united", "kingdom", "national",
    "official", "student", "students", "policy", "policies", "guidance", "definition", "department", "ministry", "office",
}


def _fold(text: str) -> str:
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(ch)).lower()


def subject_names(claim: str) -> list[str]:
    """Proper names a claim is about (Oxford, Manchester, Wiles), as 5-letter stems.

    Live eval defect: "The University of Oxford defines plagiarism as ..." was marked
    confirmed by a Nottingham Trent library page that defines plagiarism in the same
    words, and the answer then called it "Oxford's official guidance". A page about
    the right idea but the wrong subject is not evidence for this claim.
    """
    words = re.findall(r"[^\W\d_][\w'-]*", claim or "")
    names: list[str] = []
    for i, w in enumerate(words):
        if i == 0 or not w[0].isupper() or len(w) < 5 or w.lower() in _NAME_STOP or w.isupper():
            continue
        stem = _fold(w)[:5]
        if stem not in names:
            names.append(stem)
    return names[:6]


def check_support(claim: str, page: FetchedPage, *, min_coverage: float = 0.42) -> dict[str, Any]:
    """Does this page actually contain what the claim says it contains?

    Distinctive-token coverage plus a hard requirement that every number and
    year in the claim appears somewhere in the page. A page about the same topic
    that lacks the figure is a mismatch, which is exactly the failure mode models
    produce when they paraphrase a headline into a statistic.
    """
    text_lower = (page.text or "").lower()
    if not text_lower:
        return {"status": SourceCheckStatus.UNREACHABLE, "coverage": 0.0, "excerpt": None, "missing": []}

    from backend.research.claims import signature

    sig = signature(claim)
    tokens = sig["tokens"][:14]
    hit = [t for t in tokens if t in text_lower]
    coverage = (len(hit) / len(tokens)) if tokens else 0.0

    missing: list[str] = []
    for number in sig["numbers"][:6]:
        digits = re.sub(r"[^0-9.]", "", number)
        if not digits:
            continue
        # "3.2 billion" should match "3.2B" and "3,200,000,000", so compare
        # significant digits loosely rather than by exact string.
        core = digits.rstrip("0").rstrip(".") or digits
        if core not in re.sub(r"[,\s]", "", text_lower.replace(".", "")):
            if digits not in re.sub(r"[,\s]", "", text_lower):
                missing.append(number)
    for year in sig["years"][:4]:
        if year not in text_lower:
            missing.append(year)
    for date in sig["dates"][:3]:
        if not any(part in text_lower for part in re.split(r"[\s.]+", date) if len(part) > 3):
            missing.append(date)
    names = subject_names(claim)
    if names:
        folded = _fold((page.text or "") + " " + (page.title or "") + " " + (page.final_url or page.url or ""))
        if not any(n in folded for n in names):
            missing.append("subject: " + ", ".join(names[:3]))

    if BLOCK_HINT_RE.search((page.text or "")[:2000]) or "gate" in " ".join(page.notes):
        status = SourceCheckStatus.BLOCKED
    elif not page.ok:
        status = SourceCheckStatus.UNREACHABLE if page.status not in {404, 410} else SourceCheckStatus.BROKEN_URL
    elif missing and coverage >= min_coverage:
        status = SourceCheckStatus.MISMATCH
    elif coverage >= min_coverage:
        status = SourceCheckStatus.CONFIRMED
    elif coverage >= 0.25:
        status = SourceCheckStatus.IRRELEVANT if len(tokens) > 6 else SourceCheckStatus.MISMATCH
    else:
        status = SourceCheckStatus.IRRELEVANT

    excerpt = None
    if hit:
        anchor = hit[0]
        idx = text_lower.find(anchor)
        if idx >= 0:
            excerpt = re.sub(r"\s+", " ", page.text[max(0, idx - 200) : idx + 320]).strip()
    return {
        "status": status,
        "coverage": round(coverage, 3),
        "excerpt": excerpt,
        "missing": missing,
        "tokens_checked": len(tokens),
    }


_CURRENCY_RE = re.compile(r"[$\u20ac\u00a3\u20b9]|\b(?:usd|eur|gbp|inr|dollars?|euros?|pounds?|rupees?)\b", re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
_STRONG_CUE_RE = re.compile(
    r"\b(false(?:ly)?|incorrect(?:ly)?|untrue|not true|myth|debunk(?:ed|s)?|misreport(?:ed)?|mistaken(?:ly)?|"
    r"misconception|correction|corrected|retract(?:ed|ion)?|rebut(?:ted|tal)?|erroneous(?:ly)?|inaccurate)\b",
    re.I,
)
_NEGATOR_BEFORE = r"(?:\bnot|rather than|instead of|contrary to|unlike|\bnever)\s+(?:[\w$%.,-]+\s+){0,2}"


def _figure_value(raw: str) -> tuple[str, float] | None:
    """(kind, value) for a figure string such as '$549', '12%', '1.2 billion'."""
    text = (raw or "").strip().lower()
    digits = re.sub(r"[^0-9.]", "", text)
    if not digits or digits == ".":
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    unit = re.search(r"(thousand|million|billion|bn|k|m|b)\s*$", text)
    if unit:
        value *= _MULT[unit.group(1)]
    if "%" in text or "percent" in text:
        kind = "pct"
    elif _CURRENCY_RE.search(raw or ""):
        kind = "cur"
    else:
        kind = "plain"
    return kind, value


def _figures(sig: dict[str, Any], context: str = "") -> dict[str, set[float]]:
    out: dict[str, set[float]] = {}
    years = set(sig.get("years") or [])
    for raw in sig.get("numbers") or []:
        if re.sub(r"[^0-9]", "", raw) in years:
            continue  # a year is compared as a year, not as a quantity
        parsed = _figure_value(raw)
        if parsed:
            kind, value = parsed
            out.setdefault(kind, set()).add(value)
    if context and _CURRENCY_RE.search(context) and "plain" in out and "cur" not in out:
        out["cur"] = out.pop("plain")  # "549 dollars" -> currency
    return out


def check_refutation(claim: str, page: "FetchedPage") -> dict[str, Any] | None:
    """Does this page document the *opposite* of the claim?

    Deliberately conservative, because a false refutation is as damaging as a false
    confirmation. A page only counts if one sentence of it is about the same
    subject (most of the claim's distinctive words) AND does one of:

    * explicit correction -- negates the claim's own figure ("not $499") or pairs it
      with a retraction word ("the $499 price is false");
    * conflicting figure -- gives a different figure of the same kind (currency,
      percentage, plain count) or a different year for the same subject;
    * negation -- flips the claim's polarity, or calls it a myth/false/debunked.

    Returns the evidence sentence and why it counts, or None.
    """
    from backend.research.claims import signature, split_sentences

    text = page.text or ""
    if not text.strip():
        return None
    sig = signature(claim)
    topical = [t for t in sig["tokens"] if not re.fullmatch(r"[\d.,$%]+", t)][:10]
    if len(topical) < 2:
        return None
    need = max(2, -(-len(topical) * 6 // 10))
    claim_figs = _figures(sig, claim)
    claim_years = set(sig["years"])
    raw_figures = [n.strip().rstrip(".,") for n in sig["numbers"] if re.sub(r"[^0-9]", "", n) not in claim_years]

    for sentence in split_sentences(text)[:400]:
        low = sentence.lower()
        if sum(1 for t in topical if t in low) < need:
            continue
        ssig = signature(sentence)
        sent_figs = _figures(ssig, sentence)
        sent_years = set(ssig["years"])

        def hit(kind: str, why: str) -> dict[str, Any]:
            return {"kind": kind, "excerpt": re.sub(r"\s+", " ", sentence)[:420], "why": why}

        for raw in raw_figures:
            if re.search(_NEGATOR_BEFORE + re.escape(raw), sentence, re.I):
                return hit("explicit_correction", f"page says it is not {raw}")
        for raw in raw_figures:
            value = _figure_value(raw)
            if value and value[1] in sent_figs.get(value[0], set()) and _STRONG_CUE_RE.search(sentence):
                return hit("explicit_correction", f"page calls the {raw} figure wrong")
        for kind, values in claim_figs.items():
            theirs = sent_figs.get(kind)
            if not theirs or values & theirs:
                continue
            if kind == "plain" and (len(values) != 1 or len(theirs) != 1):
                continue  # bare counts are too ambiguous unless exactly one on each side
            return hit("conflicting_figure", f"page gives {sorted(theirs)[:2]} where the claim has {sorted(values)[:2]}")
        if claim_years and sent_years and not (claim_years & sent_years):
            return hit("conflicting_figure", f"page gives {sorted(sent_years)} where the claim has {sorted(claim_years)}")
        strict = max(3, -(-len(topical) * 7 // 10))
        if sum(1 for t in topical if t in low) >= strict:
            if sig["polarity"] != ssig["polarity"] and not (claim_figs or claim_years):
                return hit("negation", f"page states the opposite ({ssig['polarity']} vs claim {sig['polarity']})")
            if _STRONG_CUE_RE.search(sentence) and not (claim_figs or claim_years):
                return hit("negation", "page calls this claim false or a myth")
    return None


def evidence_from_page(job_id: str, claim_id: str | None, page: FetchedPage, check: dict[str, Any], *, round_no: int = 1, polarity: str = "support", origin: str = "provider") -> Evidence:
    return Evidence(
        job_id=job_id,
        round=round_no,
        claim_id=claim_id,
        url=page.final_url or page.url,
        title=page.title,
        domain=(urlparse(page.final_url or page.url).netloc or "").lower() or None,
        snippet=(page.text or "")[:400] or None,
        published=page.published,
        tier=tier_for(page.final_url or page.url),
        polarity=polarity,
        check_status=check["status"],
        check_notes=_notes(check, page),
        verbatim_excerpt=check.get("excerpt"),
        origin=origin,
    )


def _notes(check: dict[str, Any], page: FetchedPage) -> str:
    bits: list[str] = []
    if check.get("missing"):
        bits.append("claim figures absent from page: " + ", ".join(map(str, check["missing"][:4])))
    bits.append(f"token coverage {check.get('coverage')}")
    if page.error:
        bits.append(f"fetch error: {page.error}")
    if page.notes:
        bits.extend(page.notes)
    if page.published:
        bits.append(f"published {page.published}")
    return "; ".join(bits)[:400]


def freshness(published: str | None, *, now: datetime | None = None) -> dict[str, Any]:
    """Age of a source, and whether that age undermines the claim."""
    if not published:
        return {"known": False, "age_days": None, "verdict": "unknown"}
    now = now or datetime.now(timezone.utc)
    parsed: datetime | None = None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed = datetime.strptime(published[: len(fmt) + 6].strip(), fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        match = ISO_RE.search(published)
        if match:
            try:
                parsed = datetime.strptime(match.group(0), "%Y-%m-%d")
            except ValueError:
                parsed = None
    if parsed is None:
        return {"known": False, "age_days": None, "verdict": "unparsable"}
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age = (now - parsed).days
    verdict = "fresh" if age <= 365 else ("aging" if age <= 3 * 365 else "outdated")
    return {"known": True, "age_days": age, "verdict": verdict, "parsed": parsed.isoformat()}


async def gather_from_links(
    job_id: str,
    links: list[dict[str, Any]],
    *,
    max_pages: int = 14,
    concurrency: int = 5,
    browser_fetch: Any | None = None,
    round_no: int = 1,
    origin: str = "provider",
    max_chars: int = 12000,
    attribute_to: list[tuple[str, str]] | None = None,
) -> list[Evidence]:
    """Fetch many cited pages without letting one slow server serialise us.

    ``attribute_to`` is a list of (claim_id, claim_text). Once a page is read, it is
    checked against each of those claims and filed under every claim whose figures
    and wording it actually contains. A link's own title and snippet are too thin to
    decide that (a legislation.gov.uk section that settles the answer used to end up
    attached to no claim at all), but the full text can.

    ``max_chars`` is ``search.fetch_body_chars``: how much of each page is read and kept.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    unique: dict[str, dict[str, Any]] = {}
    for link in links:
        url = (link.get("href") or link.get("url") or "").strip()
        if url.startswith("http") and url not in unique:
            unique[url] = link
    items = list(unique.values())[:max_pages]

    async def one(link: dict[str, Any]) -> list[Evidence]:
        url = link.get("href") or link.get("url")
        async with sem:
            page = await fetch_page(url, browser_fetch=browser_fetch, max_chars=max_chars)
        claim = link.get("claim_text")
        polarity = link.get("polarity", "support")
        refutation: dict[str, Any] | None = None
        if claim:
            check = check_support(claim, page)
            counter = bool(link.get("counter"))
            # A page that failed to back the claim may be documenting the opposite;
            # a page found by a counter-query is only worth keeping if it does.
            if page.ok and (counter or check["status"] == SourceCheckStatus.MISMATCH):
                refutation = check_refutation(claim, page)
            if refutation:
                polarity = "refute"
                check = {
                    "status": SourceCheckStatus.CONFIRMED,
                    "coverage": check.get("coverage"),
                    "excerpt": refutation["excerpt"],
                    "missing": [],
                    "refutation": refutation,
                }
            elif counter and check["status"] != SourceCheckStatus.CONFIRMED:
                return []
        else:
            check = {
                "status": SourceCheckStatus.CONFIRMED if page.ok else SourceCheckStatus.UNREACHABLE,
                "coverage": None,
                "excerpt": (page.text or "")[:400] or None,
                "missing": [],
            }
        ev = evidence_from_page(
            job_id,
            link.get("claim_id"),
            page,
            check,
            round_no=round_no,
            polarity=polarity,
            origin=origin,
        )
        if refutation:
            ev.check_notes = f"contradicts the claim ({refutation['kind']}): {refutation['why']}; " + (ev.check_notes or "")
            ev.check_notes = ev.check_notes[:400]
        if link.get("title") and not ev.title:
            ev.title = str(link["title"])[:220]
        if link.get("snippet") and not ev.snippet:
            ev.snippet = str(link["snippet"])[:400]
        fresh = freshness(ev.published)
        if fresh["verdict"] == "outdated" and ev.check_status == SourceCheckStatus.CONFIRMED:
            ev.check_status = SourceCheckStatus.OUTDATED
            ev.check_notes = (ev.check_notes or "") + f"; {fresh['age_days']} days old"
        if ev.check_status == SourceCheckStatus.UNREACHABLE and not _looks_like_host(ev.domain or ""):
            ev.check_status = SourceCheckStatus.HALLUCINATED
            ev.check_notes = (ev.check_notes or "") + "; domain does not exist"
        out = [ev]
        if attribute_to and page.ok and not link.get("counter"):
            out.extend(_attributed_copies(ev, page, attribute_to, skip=link.get("claim_id")))
        return out

    gathered = await asyncio.gather(*(one(link) for link in items)) if items else []
    return [ev for group in gathered for ev in group]


def _attributed_copies(base: Evidence, page: FetchedPage, targets: list[tuple[str, str]], *, skip: str | None = None) -> list[Evidence]:
    """One extra evidence row per claim this page's text really supports."""
    copies: list[Evidence] = []
    if base.check_status not in {SourceCheckStatus.CONFIRMED, SourceCheckStatus.OUTDATED, SourceCheckStatus.NOT_CHECKED, SourceCheckStatus.IRRELEVANT, SourceCheckStatus.MISMATCH}:
        return copies
    for claim_id, claim_text in targets:
        if not claim_id or claim_id == skip or claim_id == base.claim_id:
            continue
        check = check_support(claim_text, page)
        if check["status"] != SourceCheckStatus.CONFIRMED:
            continue
        copy = base.model_copy(deep=True)
        copy.id = new_id("ev")
        copy.claim_id = claim_id
        copy.polarity = "support"
        copy.check_status = SourceCheckStatus.OUTDATED if base.check_status == SourceCheckStatus.OUTDATED else SourceCheckStatus.CONFIRMED
        copy.verbatim_excerpt = check.get("excerpt") or copy.verbatim_excerpt
        copy.check_notes = (f"token coverage {check.get('coverage')}; read in full and matched to this claim")[:400]
        copies.append(copy)
    return copies


def summarise(evidence: list[Evidence]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for ev in evidence:
        counts[ev.check_status.value] = counts.get(ev.check_status.value, 0) + 1
    return {
        "total": len(evidence),
        "by_status": counts,
        "strongest": sorted(evidence, key=lambda e: -TIER_WEIGHT.get(e.tier, 0.3))[:3],
        "distinct_domains": len({e.domain for e in evidence if e.domain}),
    }
