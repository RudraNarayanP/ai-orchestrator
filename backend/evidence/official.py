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
    r"sci\.gov\.in|legislative\.gov\.in|govinfo\.gov|ecfr\.gov|uscode\.house\.gov|supremecourt\.gov|supremecourt\.uk)$"
)

_LAW_RE = re.compile(
    r"(?i:\b(?:section|sec\.|article|art\.|regulation|directive|rule|schedule|clause)\s*\(?\d+)|"
    r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Act|Code|Sanhita|Regulation|Directive|Constitution|Framework)\b|"
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


def official_queries(question: str, *, per_jurisdiction: int = 2) -> list[str]:
    """``site:`` queries for the official publishers of the jurisdiction(s) the question is about. Empty when no law is named."""
    if not names_a_law(question):
        return []
    subject = _subject(question)
    out: list[str] = []
    case = bool(re.search(r"\bv\.?\s+[A-Z]", question or ""))
    for j in jurisdictions(question):
        key = "us_case" if j == "us" and case else j
        sites = OFFICIAL_SITES[key][:per_jurisdiction + 1]
        out.append(f"{subject} " + " OR ".join(f"site:{s}" for s in sites))
    return out


def is_official_host(host: str) -> bool:
    return bool(OFFICIAL_LAW_RE.search((host or "").lower()))


def official_link(item: dict[str, Any], question: str) -> bool:
    from urllib.parse import urlparse  # noqa: PLC0415

    host = (urlparse(item.get("href") or "").netloc or "").lower()
    sites = {s for j in jurisdictions(question) for s in OFFICIAL_SITES[j] + (OFFICIAL_SITES["us_case"] if j == "us" else [])}
    return any(host == s or host.endswith("." + s) for s in sites)
