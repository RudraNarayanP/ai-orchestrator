"""Self-corrections: an AI revising itself under follow-up is evidence, not failure.

The first escalation re-asks the SAME AI in the SAME conversation. What comes back
is classified (confirmed / corrected / cannot establish) and kept as a record --
initial claim, follow-up result, reason, final position -- so a correction X -> Y
is visible all the way to the curator instead of being averaged away.
"""

from __future__ import annotations

import re

from backend.models import ProviderResponse, SelfCorrection
from backend.research import claims as claim_ops
from backend.research.prompts import VERDICT_RE

_CORRECTED = re.compile(
    r"\b(?:i was (?:wrong|mistaken|incorrect)|my (?:earlier|previous|original) (?:answer|claim|statement) was (?:wrong|incorrect|mistaken)|"
    r"correction:|i(?:'ve| have) corrected|the correct (?:answer|figure|value|date) is)\b",
    re.I,
)
_CANNOT = re.compile(r"\bi cannot establish this\b|\bcannot establish\b", re.I)
_REASON = re.compile(r"[^.\n]*\b(?:because|since|changed|corrected|correction|wrong|mistaken|should (?:be|read)|actually)\b[^.\n]*\.?", re.I)


def headline(response: ProviderResponse, limit: int = 300) -> str:
    text = response.answer_text or response.raw_text or ""
    sections = claim_ops.sectionise(text)
    body = (sections.get("direct_answer") or "").strip() or text.strip()
    body = re.sub(r"\s+", " ", body)
    return body[:limit]


def classify(follow: ProviderResponse) -> str:
    text = follow.answer_text or follow.raw_text or ""
    m = re.search(VERDICT_RE, text, re.I)
    if m:
        return {"confirmed": "confirmed", "corrected": "corrected", "cannot establish": "cannot_establish"}[m.group(1).lower()]
    if _CANNOT.search(text):
        return "cannot_establish"
    if _CORRECTED.search(text):
        return "corrected"
    return "unclear"


def build_record(first: ProviderResponse, follow: ProviderResponse, *, round_no: int) -> SelfCorrection:
    verdict = classify(follow)
    initial = headline(first)
    result = headline(follow)
    reason_match = _REASON.search(follow.answer_text or "") if verdict == "corrected" else None
    reason = (reason_match.group(0).strip()[:300] if reason_match else "")
    if verdict == "corrected":
        final = result
    elif verdict == "confirmed":
        final = initial
    else:
        final = ""
    return SelfCorrection(
        job_id=first.job_id,
        provider=first.provider,
        thread_id=follow.thread_id,
        round=round_no,
        initial_claim=initial,
        follow_up_result=result,
        verdict=verdict,
        correction_reason=reason,
        final_position=final,
        first_response_id=first.id,
        follow_up_response_id=follow.id,
    )