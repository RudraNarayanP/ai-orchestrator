"""Temporal framing for UK legislation answers when evidence separates current law from history.

Assembly-only: does not adjudicate claims or open pages. Today this reads omission signals from
curator prose (caveats and why). Longer-term, prefer structured evidence metadata per authority
(effective_from, effective_until, legal_status, superseded_by) so reconciliation does not depend
on a particular wording upstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

_UK_ISS_AGE_Q = re.compile(
    r"data protection act 2018|uk gdpr|information society services|iss consent",
    re.I,
)
_S9_OMITTED = re.compile(r"section\s*9.{0,160}omitted|omitted.{0,120}section\s*9", re.I | re.S)
_ART8_CURRENT = re.compile(r"article\s*8\s*\(\s*1\s*\).{0,60}uk\s*gdpr", re.I | re.S)
_S9_AS_OPERATIVE = re.compile(
    r"operative provision is section\s*9|section\s*9\b.{0,80}currently in force|as of.{0,40}section\s*9\b",
    re.I | re.S,
)
_HAS_13 = re.compile(r"\b13\b")
_DENIES_16_DEFAULT = re.compile(
    r"\b16\b.{0,40}(gdpr|default)|not.{0,20}\b16\b|does not apply the gdpr default",
    re.I,
)
_AUTHORITY_UNRESOLVED = "The sources I found don't say which provision currently carries that rule in force."


@dataclass(frozen=True)
class LegalAuthorityMeta:
    """Structured authority row for future reconciliation (not yet wired from evidence)."""

    citation: str
    effective_from: str | None = None
    effective_until: str | None = None
    legal_status: Literal["in_force", "omitted", "repealed", "historical"] = "in_force"
    superseded_by: str | None = None


def _omission_mentioned(caveats: Iterable[str], why: str = "") -> bool:
    text = " ".join(p for p in [why, *list(caveats or ())] if p)
    return bool(_S9_OMITTED.search(text))


def apply_uk_iss_age_temporal_frame(
    answer: str,
    caveats: Iterable[str],
    question: str,
    *,
    why: str = "",
) -> str:
    """Frame UK GDPR Art 8(1) as in-force and s.9 DPA 2018 as historical when omission is evidenced.

    If the answer calls section 9 operative but nothing in caveats or why records omission, do not
    invent current authority — leave the age substance and mark the provision as unresolved.
    """
    text = (answer or "").strip()
    if not text or not _UK_ISS_AGE_Q.search(question or ""):
        return text
    omission = _omission_mentioned(caveats, why)
    if not omission:
        if _S9_AS_OPERATIVE.search(text):
            if _HAS_13.search(text):
                age = (
                    "The minimum age is 13, not 16."
                    if _DENIES_16_DEFAULT.search(text) or re.search(r"substitut\w+\s+13\s+for\s+16", text, re.I)
                    else "The minimum age is 13."
                )
                return f"{age} {_AUTHORITY_UNRESOLVED}"
            return _AUTHORITY_UNRESOLVED
        return text
    if _ART8_CURRENT.search(text):
        return text
    frame = (
        "The minimum age is 13, not 16. As of the law now in force, Article 8(1) of the UK GDPR sets that age; "
        "section 9 of the Data Protection Act 2018 originally substituted 13 for 16 in Article 8(1) but was omitted "
        "from 31 December 2020."
    )
    if _S9_AS_OPERATIVE.search(text):
        return frame
    if not (_HAS_13.search(text) and _DENIES_16_DEFAULT.search(text)):
        return text
    return f"{frame} {text}"


def temporal_authority_checks(answer: str, caveats: Iterable[str]) -> dict[str, bool]:
    """Whether the visible answer states the current/historical split (for tests)."""
    blob = f"{answer} {' '.join(caveats or ())}"
    low = blob.lower()
    return {
        "age_13_not_16": bool(_HAS_13.search(answer))
        and ("not 16" in low or "does not apply" in low or "not the gdpr default" in low),
        "current_art8": bool(_ART8_CURRENT.search(answer))
        or ("article 8(1)" in answer.lower() and "uk gdpr" in answer.lower()),
        "s9_historical": bool(
            re.search(r"section\s*9.{0,80}(originally|omitted|historical)", answer, re.I)
            or re.search(r"originally.{0,80}section\s*9", answer, re.I)
        ),
        "s9_not_operative_now": not bool(_S9_AS_OPERATIVE.search(answer)),
        "authority_unresolved": _AUTHORITY_UNRESOLVED.lower() in answer.lower(),
    }
