"""Which cited pages did the AI actually open, and which did it only mention?

A chat UI shows source chips whether or not the model read the page. The prompt asks
the AI to label each URL OPENED or MENTIONED ONLY; that label is kept on the citation
and carried to the evidence record and the final source. No label means "cited by the
AI, opening not confirmed" (None) -- never silently promoted to opened.
"""

from __future__ import annotations

import re

from backend.models import Citation, ProviderResponse

URL_RE = re.compile(r"https?://[^\s<>\")\]]+", re.I)
_MENTIONED = re.compile(r"\b(?:mentioned only|not opened|did not open|didn'?t open|unopened|only seen in results|not read)\b", re.I)
_OPENED = re.compile(r"\bopened\b|\bread the page\b|\bvisited\b", re.I)


def normalise(url: str) -> str:
    return re.sub(r"[.,;:]+$", "", (url or "").strip()).rstrip("/").lower()


_HEAD_MENTIONED = re.compile(r"\b(?:mentioned only|not opened|did not open|didn'?t open|unopened|only seen|only mentioned|not read)\b", re.I)
_HEAD_OPENED = re.compile(r"\b(?:pages?|sources?|links?|urls?|documents?)?\s*(?:i|we)?\s*(?:have\s+)?(?:actually\s+)?(?:opened|read|visited)\b|\bopened (?:pages?|sources?|links?)\b", re.I)


def labels_from_text(text: str) -> dict[str, bool]:
    """url -> opened?, from lines like 'https://x/y - OPENED' or 'https://x/z (MENTIONED ONLY)'.

    A heading without a URL ("Pages I opened:" / "Mentioned only:") labels the bare URLs listed under it, up to the
    next blank line or heading -- the AI said it opened them, even if it did not repeat the word on every line.
    """
    out: dict[str, bool] = {}
    section: bool | None = None
    for line in (text or "").splitlines():
        urls = URL_RE.findall(line)
        if not urls:
            if not line.strip():
                section = None
            elif _HEAD_MENTIONED.search(line):
                section = False
            elif _HEAD_OPENED.search(line) and len(line) < 140:
                section = True
            elif len(line) < 60 and line.rstrip().endswith(":"):
                section = None
            continue
        if _MENTIONED.search(line):
            flag = False
        elif _OPENED.search(line):
            flag = True
        elif section is not None:
            flag = section
        else:
            continue
        for url in urls:
            out[normalise(url)] = flag
    return out


def mark_opened(response: ProviderResponse) -> None:
    labels = labels_from_text(response.raw_text or response.answer_text or "")
    labels.update(labels_from_text(response.answer_text or ""))
    for citation in response.citations:
        key = normalise(citation.url)
        if key in labels:
            citation.ai_opened = labels[key]
        else:
            hit = next((flag for url, flag in labels.items() if url.startswith(key) or key.startswith(url)), None)
            citation.ai_opened = hit
        if not citation.provider:
            citation.provider = response.provider

# ------------------------------------------------------------- the AI's own citations from its text

_FIRST_PARTY = re.compile(
    r"^(?:[\w-]+\.)*(?:chatgpt\.com|openai\.com|gemini\.google\.com|copilot\.microsoft\.com|meta\.ai|chat\.mistral\.ai|pi\.ai|"
    r"chat\.qwen\.ai|chat\.deepseek\.com|bard\.google\.com|accounts\.google\.com|support\.google\.com)$",
    re.I,
)


def _clean_url(url: str) -> str:
    return re.sub(r"[.,;:*_)\]]+$", "", url.strip())


def text_citations(response: ProviderResponse) -> int:
    """Add a Citation for every URL the AI wrote into its answer and the site did not render as a link.

    Logged-out ChatGPT shows its source chips as bare text, so its real citations arrive only as the URLs it
    was told to list. These are the AI's own words, kept as the AI's citations (origin="text"); nothing is
    searched for or invented. Returns how many were added.
    """
    text = response.answer_text or response.raw_text or ""
    have = {normalise(c.url) for c in response.citations}
    added = 0
    for line in text.splitlines():
        for raw in URL_RE.findall(line):
            url = _clean_url(raw)
            key = normalise(url)
            host = (re.sub(r"^https?://", "", url).split("/")[0]).lower()
            if not key or key in have or _FIRST_PARTY.match(host) or len(response.citations) >= 40:
                continue
            have.add(key)
            label = re.split(r"\s+[-\u2014\u2013]\s+|\s*\(", line.replace(raw, " ", 1), maxsplit=1)[0].strip(" *-\u2022:[]\"'")
            response.citations.append(
                Citation(url=url, title=(label[:200] or host), snippet=line.strip()[:400], provider=response.provider, origin="text")
            )
            added += 1
    return added


ORDER = ["MENTIONED", "OPENED", "INSPECTED", "CITED", "CLAIM_SUPPORTED"]


def provenance(*, cited_by: list[str], ai_opened: bool | None, omnibrain_opened: bool, claim_attached: bool, supported: bool) -> str | None:
    """Highest state reached. OmniBrain opening a page never raises it: only what the AI did counts."""
    if not cited_by:
        return None
    if ai_opened is not True:
        return "MENTIONED"
    state = "OPENED"
    if omnibrain_opened:
        state = "INSPECTED"
    if claim_attached:
        state = "CITED"
        if supported:
            state = "CLAIM_SUPPORTED"
    return state


def aggregate_opened(responses: list[ProviderResponse], url: str) -> bool | None:
    """True if any AI said it opened the URL; False if every AI that listed it said mentioned only."""
    key = normalise(url)
    flags = [c.ai_opened for r in responses for c in r.citations if normalise(c.url) == key]
    if True in flags:
        return True
    if flags and all(f is False for f in flags):
        return False
    return None
