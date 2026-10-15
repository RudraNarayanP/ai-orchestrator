"""Official text of a law the question names, fetched alongside the AIs' citations (2026-10-15).

Live legal runs kept ending "no primary source cited": the AIs cited commentary, so the ledger had nothing official to rank first.
When a question names a law, a provision or a case, OmniBrain now runs a few ``site:``-restricted searches against the official
publishers for that jurisdiction, in parallel with the normal source checks, and reads the top hits like any cited page.
A page only counts if its text confirms a claim -- an official page that does not is kept as unconfirmed, never promoted.

Gates are the site's to set: legislation.gov.uk answers bots with an AWS WAF challenge. That is recorded as a gate by
``fetch_page`` and never worked around; the search simply also covers other official publishers (gov.uk guidance, the ICO).
"""

from __future__ import annotations

import re
from typing import Any

# jurisdiction -> official publishers searched for the named law
OFFICIAL_SITES: dict[str, list[str]] = {
    "eu": ["eur-lex.europa.eu", "commission.europa.eu", "ec.europa.eu", "edpb.europa.eu", "curia.europa.eu"],
    "india": ["indiacode.nic.in", "egazette.gov.in", "sci.gov.in", "meity.gov.in", "legislative.gov.in"],
    "us_case": ["supreme.justia.com", "courtlistener.com", "law.cornell.edu", "supremecourt.gov"],
    "us": ["govinfo.gov", "ecfr.gov", "uscode.house.gov", "supremecourt.gov", "law.cornell.edu", "supreme.justia.com", "courtlistener.com"],
    "uk": ["legislation.gov.uk", "gov.uk", "ico.org.uk", "supremecourt.uk"],
}

# official publishers of legal text: a page from one of these is primary evidence of what the law says
OFFICIAL_LAW_RE = re.compile(
    r"(^|\.)(eur-lex\.europa\.eu|curia\.europa\.eu|legislation\.gov\.uk|indiacode\.nic\.in|egazette\.gov\.in|egazette\.nic\.in|"
    r"sci\.gov\.in|legislative\.gov\.in|govinfo\.gov|law\.cornell\.edu|ecfr\.gov|uscode\.house\.gov|supremecourt\.gov|supremecourt\.uk)$"
)

_LAW_RE = re.compile(
    r"(?i:\b(?:section|sec\.|article|art\.|regulation|directive|rule|schedule|clause)\s*\(?\d+)|"
    r"\b[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)*\s+(?:Act|Code|Sanhita|Regulation|Directive|Constitution|Framework)\b|"
    r"\b(?:GDPR|DPDP|IPC|BNS|CrPC|BNSS|U\.?S\.?C\.?|C\.?F\.?R\.?)\b|"
    r"\b[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*\s+v\.?\s+[A-Z][\w.&'-]+",
)

_JURISDICTION = [
    ("india", re.compile(r"\bIndia|Indian\b|\bIPC\b|\bBNS\b|Sanhita|Article 368|DPDP|Lok Sabha|Parliament of India", re.I)),
    ("eu", re.compile(r"\bEU\b|European|GDPR|Data Privacy Framework|\bCJEU\b|Directive|Schrems", re.I)),
    ("uk", re.compile(r"\bUK\b|United Kingdom|England|Wales|Scotland|Data Protection Act 2018|\bICO\b", re.I)),
    ("us", re.compile(r"\bUS\b|U\.S\.|United States|federal court|Supreme Court of the United States|U\.S\.C|C\.F\.R|\bv\.\s", re.I)),
]


def names_a_law(question: str) -> bool:
    return bool(_LAW_RE.search(question or ""))


def jurisdictions(question: str) -> list[str]:
    return [name for name, rx in _JURISDICTION if rx.search(question or "")]


def _subject(question: str) -> str:
    """The law/provision words of the question, for the search query."""
    hits = [m.group(0).strip() for m in _LAW_RE.finditer(question or "")]
    hits = [re.sub(r"^(?:Under|In|Per|By)\s+", "", h) for h in hits]
    text = " ".join(dict.fromkeys(hits)) or question
    return re.sub(r"\s+", " ", text)[:120]


def official_queries(question: str, *, per_jurisdiction: int = 2, limit: int = 2) -> list[str]:
    """``site:`` queries for the official publishers of the jurisdiction(s) the question is about. Empty when no law is named."""
    if not names_a_law(question):
        return []
    subject = _subject(question)
    # plus the few content words the question asks about ("theft", "consent"), so the hit is the provision, not a homepage
    stop = {"which", "what", "does", "under", "currently", "still", "would", "could", "should", "there", "their", "about", "company",
            "court", "must", "section", "punishes", "federal", "apply", "nder", "hears", "claim", "transfer", "personal", "data", "amend", "parliament", "constitution", "indian", "that", "this", "with", "without", "also", "using", "certified"}
    words = [w for w in re.findall(r"\b[a-z]{4,}\b", (question or "").lower()) if w not in stop and w.lower() not in subject.lower()][:3]
    subject = f"{subject} {' '.join(words)}".strip()
    out: list[str] = []
    case = bool(re.search(r"\bv\.?\s+[A-Z]", question or ""))
    for j in jurisdictions(question):
        key = "us_case" if j == "us" and case else j
        for site in OFFICIAL_SITES[key][:per_jurisdiction]:
            out.append(f"{subject} site:{site}")  # one site per query: OR-ed site: filters return nothing on some engines
    return out[:limit]  # a burst of searches trips the engine's bot check


def is_official_host(host: str) -> bool:
    return bool(OFFICIAL_LAW_RE.search((host or "").lower()))


def official_link(item: dict[str, Any], question: str) -> bool:
    from urllib.parse import urlparse  # noqa: PLC0415

    parsed = urlparse(item.get("href") or "")
    host = (parsed.netloc or "").lower()
    if len((parsed.path or "").strip("/")) < 3 or re.search(r"/(?:default|index)\.\w+$", parsed.path or "", re.I):
        return False  # a publisher's homepage says nothing about the provision
    sites = {s for j in jurisdictions(question) for s in OFFICIAL_SITES[j] + (OFFICIAL_SITES["us_case"] if j == "us" else [])}
    return any(host == s or host.endswith("." + s) for s in sites)


_US_REPORTS_RE = re.compile(r"\b(\d{1,3})\s+U\.\s?S\.\s+(\d{1,4})\b")
_EU_ACT_RE = re.compile(r"\((?:EU|EC|EEC)\)\s*(?:No\s*)?(\d{4})/(\d{1,4})|\b(\d{4})/(\d{1,4})/(?:EU|EC)\b")


def direct_official_urls(texts: list[str], question: str = "") -> list[str]:
    """Official URLs built from citations the AIs already gave (no search engine needed -- DuckDuckGo answers bursts with a
    202 challenge, which is a gate we respect). "304 U.S. 64" -> the US Reports text; "(EU) 2023/1795" -> its EUR-Lex record.
    A guessed URL that does not resolve is dropped by the caller, never counted against anyone."""
    blob = " ".join([question or ""] + list(texts or []))
    out: list[str] = []
    for vol, page in _US_REPORTS_RE.findall(blob)[:2]:
        out.append(f"https://supreme.justia.com/cases/federal/us/{int(vol)}/{int(page)}/")
        out.append(f"https://www.law.cornell.edu/supremecourt/text/{int(vol)}/{int(page)}")
    for m in list(_EU_ACT_RE.finditer(blob))[:2]:
        year, num = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        for kind in ("R", "D", "L"):
            out.append(f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:3{year}{kind}{int(num):04d}")
    return list(dict.fromkeys(out))
