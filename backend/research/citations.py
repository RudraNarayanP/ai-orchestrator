"""Which cited pages did the AI actually open, and which did it only mention?

A chat UI shows source chips whether or not the model read the page. The prompt asks
the AI to label each URL OPENED or MENTIONED ONLY; that label is kept on the citation
and carried to the evidence record and the final source. No label means "cited by the
AI, opening not confirmed" (None) -- never silently promoted to opened.
"""

from __future__ import annotations

import re

from backend.models import ProviderResponse

URL_RE = re.compile(r"https?://[^\s<>\")\]]+", re.I)
_MENTIONED = re.compile(r"\b(?:mentioned only|not opened|did not open|didn'?t open|unopened|only seen in results|not read)\b", re.I)
_OPENED = re.compile(r"\bopened\b|\bread the page\b|\bvisited\b", re.I)


def normalise(url: str) -> str:
    return re.sub(r"[.,;:]+$", "", (url or "").strip()).rstrip("/").lower()


def labels_from_text(text: str) -> dict[str, bool]:
    """url -> opened?, from lines like 'https://x/y - OPENED' or 'https://x/z (MENTIONED ONLY)'."""
    out: dict[str, bool] = {}
    for line in (text or "").splitlines():
        urls = URL_RE.findall(line)
        if not urls:
            continue
        if _MENTIONED.search(line):
            flag = False
        elif _OPENED.search(line):
            flag = True
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