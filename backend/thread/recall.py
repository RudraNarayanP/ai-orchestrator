"""Recall helpers: what to search for, and when ("8 months ago")."""

from __future__ import annotations

import re
import time

from backend.memory.embed import tokens, _STOP

DAY = 86400.0
RECALL_CUES = re.compile(
    r"\b(remember|recall|that thing|what did (we|i|you)|we (talked|discussed|spoke)|(talked|spoke|discussed) about|"
    r"i (told|mentioned|said)|you (told|said|mentioned|suggested)|(months?|weeks?|years?|days?) ago|last (week|month|year|time)|"
    r"earlier|a while back|back when|the other day|you gave me|that (idea|plan|decision|conversation|discussion))\b",
    re.I,
)
# words that describe the act of remembering, not the topic being remembered
CUE_WORDS = frozenset(
    "remember recall thing things about ago month months week weeks year years day days earlier before talked talk spoke speak discussed discuss "
    "said told mentioned mention suggested back while other last time again please can could tell remind reminded what whats which that when "
    "conversation idea plan the".split()
)
_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "eleven": 11, "twelve": 12, "couple": 2, "few": 3}
_UNIT = {"day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}
_AGO = re.compile(r"\b(\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(?:a )?couple(?: of)?|(?:a )?few)\s+(day|week|month|year)s?\s+ago\b", re.I)
_LAST = re.compile(r"\blast\s+(week|month|year)\b", re.I)


def is_recall_request(text: str) -> bool:
    return bool(RECALL_CUES.search(text or ""))


def query_terms(text: str, limit: int = 8) -> list[str]:
    """Distinctive words to search for: no stopwords, none of the 'do you remember' scaffolding, longest first."""
    seen: list[str] = []
    for w in tokens(text):
        w = w.replace("'s", "")
        if len(w) < 3 or w in _STOP or w in CUE_WORDS or w in seen:
            continue
        seen.append(w)
    seen.sort(key=lambda w: -len(w))
    return seen[:limit]


def time_window(text: str, now: float | None = None) -> tuple[float, float] | None:
    """A soft (lo, hi) timestamp window for "N months ago" style hints. It is a ranking boost, never a filter."""
    now = time.time() if now is None else now
    m = _AGO.search(text or "")
    if m:
        raw = m.group(1).lower().replace("a couple of", "couple").replace("a couple", "couple").replace("a few", "few")
        n = int(raw) if raw.isdigit() else _NUM_WORDS.get(raw, 1)
        span = n * _UNIT[m.group(2).lower()] * DAY
        half = max(3 * DAY, span * 0.3)
        return now - span - half, now - span + half
    m = _LAST.search(text or "")
    if m:
        span = _UNIT[m.group(1).lower()] * DAY
        return now - 2 * span, now
    if re.search(r"\byesterday\b", text or "", re.I):
        return now - 2 * DAY, now
    return None