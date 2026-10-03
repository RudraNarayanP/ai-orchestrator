"""Conversation memory (spec 8: use past context when it is genuinely relevant).

A job used to be an island, so "and how did they do it?" had nothing to resolve
against. A thread now carries the earlier turns: the question, the answer we gave,
and the claims OUR LEDGER confirmed (never what a provider merely said).

Two rules keep this honest:

* context is attached only when the new question actually depends on it -- a
  standalone question in the same thread is researched as itself;
* a follow-up inherits the stakes and time-sensitivity of what it follows, so a
  medical thread does not become casual just because the second question is short.
"""

from __future__ import annotations

import re
from typing import Iterable

from backend.models import ClaimStatus, ConversationTurn, Job

STRONG_REFERENT_RE = re.compile(r"\b(they|them|their|theirs|he|him|his|she|her|hers|its|those|these|the same|such)\b", re.I)
WEAK_REFERENT_RE = re.compile(r"\b(it|that|this|there|then|one)\b", re.I)
CONTINUER_RE = re.compile(r"^\s*(and|also|but|so|then|what about|how about|wait)\b", re.I)
_NOT_ENTITY = {
    "and", "also", "but", "so", "then", "what", "how", "why", "when", "where", "who", "which", "whose", "does", "did",
    "is", "was", "are", "were", "can", "could", "would", "should", "will", "the", "this", "that", "these", "those",
    "there", "they", "their", "it", "its", "his", "her", "he", "she", "you", "your", "we", "our", "i", "if", "do",
    "has", "have", "had", "tell", "give", "show", "explain", "compare", "list", "according", "yes", "no", "sure",
    "however", "any", "some", "my", "please", "anything", "everything", "while", "after", "before", "in", "on", "at", "for", "to", "of",
}
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9.'-]{1,}(?:\s+[A-Z][A-Za-z0-9.'-]{1,})*\b")
IDK_RE = re.compile(r"^\s*i (don'?t know|couldn'?t verify)", re.I)


def entities_in(text: str, *, skip_first: bool = False) -> list[str]:
    """Capitalised names/phrases, minus sentence-opening function words."""
    out: list[str] = []
    for match in _ENTITY_RE.finditer(text or ""):
        phrase = match.group(0).strip(".'-")
        words = phrase.split()
        while words and words[0].lower() in _NOT_ENTITY:
            words.pop(0)
        if not words:
            continue
        if skip_first and match.start() <= 1 and len(words) == len(phrase.split()) and len(words) == 1:
            continue
        name = " ".join(words)
        if len(name) >= 3 and name not in out:
            out.append(name)
    return out


def is_follow_up(question: str, prior: ConversationTurn | None) -> bool:
    """Does this question lean on the previous turn?"""
    if prior is None:
        return False
    text = (question or "").strip()
    words = text.split()
    if not words:
        return False
    own = entities_in(text, skip_first=True)
    continuer = bool(CONTINUER_RE.match(text))
    if own:
        # It names its own subject. Only an explicit "and what about X?" is a follow-up.
        return continuer and len(words) <= 8
    if continuer or STRONG_REFERENT_RE.search(text):
        return True
    if WEAK_REFERENT_RE.search(text) and len(words) <= 8:
        return True
    return len(words) <= 3


def _shorten(text: str, limit: int) -> str:
    clean = " ".join((text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "\u2026"


def inherited_entities(prior: ConversationTurn) -> list[str]:
    names = entities_in(prior.question, skip_first=True) + entities_in(prior.answer)
    for claim in prior.confirmed_claims:
        names += entities_in(claim)
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen[:6]


def standalone_question(question: str, prior: ConversationTurn) -> str:
    """A heuristic rewrite that can be researched on its own."""
    subject = _shorten(prior.question.rstrip("?.! "), 160)
    return f'{question.strip()} (in the context of: "{subject}")'


def history_block(history: Iterable[ConversationTurn], *, max_turns: int = 3) -> str:
    """Prompt text that puts the earlier turns in front of a provider."""
    turns = list(history)[-max_turns:]
    if not turns:
        return ""
    lines = ["Earlier in this conversation (context only; do not repeat it back):"]
    for turn in turns:
        lines.append(f'- The user asked: "{_shorten(turn.question, 200)}"')
        if turn.answer and not IDK_RE.match(turn.answer):
            lines.append(f'  Answered: "{_shorten(turn.answer, 500)}"')
        elif turn.answer:
            lines.append("  That question was not settled.")
        if turn.confirmed_claims:
            lines.append("  Confirmed so far: " + "; ".join(_shorten(c, 160) for c in turn.confirmed_claims[:5]))
    lines.append(
        'The new question below may refer to these ("they", "it", "that"). Resolve such references against '
        "the above, then answer the new question itself."
    )
    return "\n".join(lines)


def turn_from_job(job: Job) -> ConversationTurn | None:
    """What a finished job contributes to the thread. Unfinished jobs contribute nothing."""
    if job.final is None:
        return None
    confirmed: list[str] = []
    if job.reports:
        for verdict in job.reports[-1].verdicts:
            if verdict.verdict in {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}:
                confirmed.append(verdict.claim)
    return ConversationTurn(
        job_id=job.id,
        question=job.question,
        answer=job.final.answer or "",
        confirmed_claims=confirmed[:8],
        confidence=job.final.confidence.value if job.final.confidence else "",
        at=job.finished_at or job.updated_at,
    )


REWRITE_PROMPT = """You rewrite a follow-up question so it can be researched without the conversation.

Earlier question: {prior_q}
Answer given: {prior_a}
Confirmed facts: {facts}

Follow-up: {question}

Resolve every pronoun and reference ("they", "it", "that", "the same") using ONLY the earlier turn. Do not
answer the question and do not add facts. If the follow-up does not depend on the earlier turn, return it
unchanged. Reply with strict JSON: {{"standalone": "..."}}"""


def rewrite_messages(question: str, prior: ConversationTurn) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": REWRITE_PROMPT.format(
                prior_q=_shorten(prior.question, 300),
                prior_a=_shorten(prior.answer, 600) or "(none)",
                facts="; ".join(_shorten(c, 160) for c in prior.confirmed_claims[:5]) or "(none)",
                question=question.strip(),
            ),
        }
    ]


def accept_rewrite(text: object, question: str) -> str | None:
    """Guard against a small model's malformed or runaway rewrite."""
    if not isinstance(text, str):
        return None
    clean = " ".join(text.split())
    if not (8 <= len(clean) <= 500) or clean.lower().startswith(("{", "[", "```")):
        return None
    return clean