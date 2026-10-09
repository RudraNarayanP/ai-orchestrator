"""Which event a date belongs to, decided from the page's own words.

A page that says "Construction began in January 1887 and was finished on 31 March 1889"
contains both years, and matching a claim against the page as a whole cannot tell a start
from a completion -- so "completed in 1887" read as supported (live: the Eiffel Tower runs,
2026-10-09). This module binds a date to the event cue in the same clause, on a small and
explicit vocabulary, and says "unknown" rather than guessing: an unbound claim is left for
the curator instead of being promoted here.
"""

from __future__ import annotations

import re
from typing import Any

from backend.research.claims import date_spans

_DATE_ONLY_YEAR = re.compile(r"\b(1[0-9]{3}|2[0-9]{3})\b")


def dated(text: str) -> list[tuple[int, int, str]]:
    """Every date written in the text, including a bare year.

    "completed in 1887" carries no month, and a year on its own still has an event role to
    check -- leaving it out made the reversed-date claim look not-applicable.
    """
    spans = list(date_spans(text))
    covered = [(start, end) for start, end, _ in spans]
    for match in _DATE_ONLY_YEAR.finditer(text or ""):
        if any(start <= match.start() < end for start, end in covered):
            continue
        spans.append((match.start(), match.end(), match.group(1)))
    return sorted(spans)

# Explicit cue -> event kind. Kept short and readable on purpose: every word here is a
# reason a claim is accepted or rejected, so it can be audited. Anything not listed is
# "unknown", which leaves the claim unverified rather than silently promoting it.
EVENT_CUES: dict[str, tuple[str, ...]] = {
    # Verb-led cues only: a noun such as "construction" is a claim's subject, and marking
    # it as a cue would strip it out of the subject the clause has to agree with.
    "start": (
        "began", "begun", "started", "commenced", "commencement", "was laid",
        "laid the foundation", "foundation stone", "first introduced",
    ),
    "end": (
        "completed", "completion", "finished", "finish", "topped out", "structurally complete",
        "handed over", "handover", "delivered", "substantially complete",
    ),
    "open": (
        "opened to the public", "public opening", "opened", "opening", "inaugurated",
        "inauguration", "unveiled",
    ),
    "force": (
        "came into force", "enters into force", "entered into force", "coming into force",
        "came into effect", "took effect", "effective date", "in force on", "in force from",
    ),
    "enacted": (
        "received royal assent", "royal assent", "enacted", "signed into law",
    ),
    "published": (
        "published", "publication date", "released",
    ),
}

_CUE_RX = re.compile(
    r"\b(" + "|".join(re.escape(c) for group in EVENT_CUES.values() for c in sorted(group, key=len, reverse=True)) + r")\b",
    re.I,
)
_KIND_BY_CUE = {cue: kind for kind, group in EVENT_CUES.items() for cue in group}

# A date is bound to a cue inside one clause. Coordinating conjunctions split too, so that
# "began in 1887 and was finished in 1889" becomes two clauses with one cue each.
_CLAUSE_SPLIT = re.compile(r"[.;:!?—–\n]|,\s+|\s+\b(?:and|but|while|whereas|then|after|followed by)\b", re.I)


_CUE_WORDS = {w for group in EVENT_CUES.values() for cue in group for w in re.findall(r"[a-z']+", cue.lower())}
_STOP = {
    "the", "a", "an", "it", "its", "this", "that", "these", "those", "there", "was", "were", "is", "are",
    "and", "of", "in", "on", "at", "by", "for", "to", "from", "with", "as", "which", "who", "most", "all",
    "new", "than", "then", "has", "have", "had", "been", "while", "after", "before", "during", "between",
}


def _compatible(claim_key: str, page_key: str) -> bool:
    """The page states the claim's date at least as precisely as the claim does.

    A bare year on the page is not a match for a claim that names a day: "The Data
    Protection Act 2018 received Royal Assent on 23 May 2018" must not evidence
    "Royal Assent on 25 May 2018" just because 2018 is in the title.
    """
    if claim_key.startswith("MD-") or page_key.startswith("MD-"):
        return claim_key == page_key
    return page_key == claim_key or page_key.startswith(claim_key + "-")


def _words(text: str, limit: int = 6) -> list[str]:
    return [w for w in re.findall(r"[a-z']+", (text or "").lower())[:limit]]


def _claim_subject(text: str) -> set[str]:
    """The nouns the claim puts before its first event cue -- who the claim is about.

    "Construction of the Eiffel Tower began in January 1887" is about the construction,
    so a clause headed "Renovations were completed on 24 June 1985" cannot evidence it.
    """
    lower = (text or "").lower()
    cues = _cues(lower)
    head = lower[: cues[0][0]] if cues else lower
    return {w for w in re.findall(r"[a-z']+", head) if len(w) > 3 and w not in _STOP and w not in _CUE_WORDS}


def _strip_dates(segment: str) -> str:
    """The clause without its dates: a month is not the subject of anything."""
    out, last = [], 0
    for start, end, _ in dated(segment):
        out.append(segment[last:start])
        last = end
    out.append(segment[last:])
    return " ".join(out)


def _clause_about_the_claim(segment: str, subject: set[str]) -> bool:
    """Does this clause still talk about the claim's subject?

    A clause with no noun of its own ("was finished on 31 March 1889") continues the
    subject of the sentence before it, so it counts; a clause that opens with a different
    noun is a different event.
    """
    if not subject:
        return True
    words = _words(_strip_dates(segment), 8)
    candidates = [w for w in words if len(w) > 3 and w not in _STOP and w not in _CUE_WORDS]
    if not candidates:
        return True
    first = candidates[0]
    if any(first == s or first.startswith(s[:4]) or s.startswith(first[:4]) for s in subject):
        return True
    return bool(set(words[:5]) & subject)


def _cues(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), _KIND_BY_CUE[m.group(1).lower()]) for m in _CUE_RX.finditer(text)]


def _claim_pairs(text: str) -> list[tuple[str, str, int]]:
    """Each date in the claim with the event kind written nearest to it."""
    spans = dated(text)
    cues = _cues((text or "").lower())
    pairs: list[tuple[str, str, int]] = []
    for start, end, key in spans:
        if not cues:
            continue
        mid = (start + end) / 2
        pos, kind = min(((abs((s + e) / 2 - mid), kind) for s, e, kind in cues), key=lambda p: p[0])
        pairs.append((kind, key, int(pos)))
    return pairs


def _clauses(text: str) -> list[tuple[str, list[str]]]:
    """Page text cut into clauses, each with the canonical keys of the dates inside it.

    Splitting happens everywhere *except* inside a written date, so "31 March, 1889" is
    not torn in half, and the clause is returned as the page's own words: the recorded
    passage has to be quotable, not a canonicalised key.
    """
    text = text or ""
    spans = dated(text)
    protected = [False] * len(text)
    for start, end, _ in spans:
        for i in range(start, min(end, len(text))):
            protected[i] = True
    cuts: list[int] = []
    i = 0
    while i < len(text):
        if not protected[i]:
            match = _CLAUSE_SPLIT.match(text, i)
            if match and match.end() > match.start():
                cuts.append(match.end())
                i = match.end()
                continue
        i += 1
    bounds = [0, *cuts, len(text)]
    out: list[tuple[str, list[str]]] = []
    for start, end in zip(bounds, bounds[1:]):
        piece = re.sub(r"\s+", " ", text[start:end]).strip(" ,;:")
        if not piece:
            continue
        keys = [key for s, e, key in spans if s >= start and e <= end]
        out.append((piece, keys))
    return out


def bind_event_dates(claim: str, page_text: str) -> dict[str, Any] | None:
    """Decide, clause by clause, whether the page states the claim's dates as its events.

    Returns None when the question of event roles does not arise (no dated event in the
    claim), so callers keep the plain figure-and-wording checks. Otherwise:
      bound    -- every date sits in a clause with the claim's own event cue;
      excluded -- a date sits only in a clause about a different event, and the claim's
                  event is dated elsewhere on the page;
      unknown  -- the page gives the date but never says which event it belongs to.
    """
    pairs = _claim_pairs(claim)
    if not pairs:
        return None
    subject = _claim_subject(claim)
    clauses = _clauses(page_text or "")
    verdicts: list[dict[str, Any]] = []
    for kind, key, _ in pairs:
        supportive = next(
            (
                {"segment": segment, "keys": keys}
                for segment, keys in clauses
                if kind in {k for _, _, k in _cues(segment.lower())}
                and any(_compatible(key, k) for k in keys)
                and _clause_about_the_claim(segment, subject)
            ),
            None,
        )
        if supportive:
            verdicts.append({"verdict": "bound", "kind": kind, "date": key, **supportive})
            continue
        same_date = [(segment, keys) for segment, keys in clauses if any(_compatible(key, k) for k in keys)]
        on_subject = [(segment, keys) for segment, keys in same_date if _clause_about_the_claim(segment, subject)]
        other_kinds = {k for segment, _ in on_subject for _, _, k in _cues(segment.lower())} - {kind}
        kind_elsewhere = {
            k
            for segment, keys in clauses
            for _, _, k in _cues(segment.lower())
            if k == kind and any(not _compatible(key, x) for x in keys)
        }
        if same_date and not on_subject and (kind_elsewhere or other_kinds):
            head = _words(_strip_dates(same_date[0][0]), 1)
            verdicts.append(
                {
                    "verdict": "excluded",
                    "kind": kind,
                    "date": key,
                    "segment": same_date[0][0],
                    "why": f"the page dates {key} to {head[0] if head else 'another subject'}, not to the subject of this claim",
                }
            )
        elif same_date and other_kinds and kind_elsewhere:
            verdicts.append(
                {
                    "verdict": "excluded",
                    "kind": kind,
                    "date": key,
                    "segment": same_date[0][0],
                    "why": f"the page dates {key} to {'/'.join(sorted(other_kinds))}, not to {kind}",
                }
            )
        elif same_date or kind_elsewhere:
            why = (
                f"the page gives {key} but never states which event it belongs to"
                if same_date
                else f"the page dates {kind} to another date and never gives {key}"
            )
            verdicts.append(
                {"verdict": "unknown", "kind": kind, "date": key, "segment": same_date[0][0] if same_date else "", "why": why}
            )
        else:
            verdicts.append({"verdict": "unknown", "kind": kind, "date": key, "segment": "", "why": f"{key} is not dated to any event on this page"})
    if any(v["verdict"] == "excluded" for v in verdicts):
        return next(v for v in verdicts if v["verdict"] == "excluded")
    if all(v["verdict"] == "bound" for v in verdicts):
        return {**verdicts[0], "verdict": "bound"}
    return next(v for v in verdicts if v["verdict"] == "unknown")
