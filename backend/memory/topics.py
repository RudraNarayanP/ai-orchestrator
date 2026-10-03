"""Topic tags, sensitivity and the fact/interpretation split.

Topics are coarse life-domain tags (finance, family, education, ...) assigned by word lists. They do three jobs:
  * group related memories into a context cluster (family finances -> education spending -> the current decision),
  * decide whether a memory is sensitive (family / money / health / relationships / legal / identity),
  * let a vague follow-up ("why can't my parents fund this?") pull the connected cluster, and nothing unrelated.
They are heuristics - cheap, deterministic, inspectable - not a claim about meaning.
"""

from __future__ import annotations

import re

TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {k: re.compile(v, re.I) for k, v in {
    "finance": r"\b(?:inr|rs\.?|rupees?|lakhs?|lacs?|crores?|usd|eur|gbp|dollars?|euros?|pounds?|funds?|funded|funding|afford\w*|tuition|fees?|loans?|savings?|saved|salary|income|budget\w*|spent|spend\w*|paid|pays?|cost\w*|expens\w*|debts?|mortgage|invest\w*|scholarships?|bank|money|bought|buy|sold|land|house|property|flat|plot|assets?|wealth|rich|poor)\b|[\u20b9$\u20ac\u00a3]",
    "education": r"\b(?:coaching|universit\w*|college|school|degrees?|masters?|bachelors?|phd|courses?|exams?|admissions?|scholarships?|stud(?:y|ies|ying|ied)|abroad|allen|jee|neet|gre|ielts|education|campus|semester|professor)\b",
    "family": r"\b(?:parents?|mother|father|mom|dad|mum|mama|papa|sisters?|brothers?|family|grandparents?|grandmother|grandfather|uncle|aunt|cousins?|siblings?|son|daughter)\b",
    "decision": r"\b(?:consider\w*|decid\w*|decision|reject\w*|declin\w*|accept\w*|turned down|choos\w*|chose|weigh\w*|apply\w*|applied|application|offers?|shortlist\w*|option|options)\b",
    "health": r"\b(?:doctors?|medical|illness|diagnos\w*|medication|therapy|therapist|anxiety|depress\w*|allerg\w*|surgery|hospital|disease|disabilit\w*|pregnan\w*)\b",
    "relationship": r"\b(?:girlfriend|boyfriend|partner|dating|wife|husband|breakup|broke up|crush|fianc\w*|marriage|married)\b",
    "legal": r"\b(?:lawyer|visa|court|contract|lawsuit|passport|immigration|citizenship)\b",
    "identity": r"\b(?:religio\w*|caste|politic\w*|orientation|gay|lesbian|transgender|atheist|muslim|hindu|christian|jewish)\b",
    "fitness": r"\b(?:gym|workouts?|lifting|running|marathon|cardio|protein|squats?|bench press)\b",
    "gaming": r"\b(?:gaming|games?|valorant|steam|console|fps|esports|playstation|xbox|minecraft)\b",
    "tech": r"\b(?:laptop|pc|monitor|keyboard|cpu|gpu|ram|ssd|rtx|motherboard|build)\b",
}.items()}

SENSITIVE_TOPICS = frozenset({"finance", "family", "health", "relationship", "legal", "identity"})

# the user's feelings, beliefs and readings of other people's motives. Never an objective fact.
INTERPRETATION_RE = re.compile(
    r"\bi(?:'m| am)?\s*(?:may|might|could)?\s*(?:read|see|take|interpret)\b[^.]{0,60}\bas\b"
    r"|\bi\s+(?:really\s+|kind of\s+|sort of\s+|just\s+)?(?:feel|felt|resent|suspect|fear|worry|worried|guess|assume|imagine|sense)\b"
    r"|\bi(?:'m| am)\s+(?:so\s+|very\s+|really\s+|quite\s+)?(?:worried|afraid|scared|anxious|upset|sad|angry|hurt|disappointed|frustrated|ashamed|jealous|stressed|hopeless|guilty|bitter)\b"
    r"|\bi\s+(?:think|believe|reckon|bet|doubt)\b[^.]{0,80}\b(?:care|cares|value|values|valuing|love|loves|respect|trust|support|supports|fair|unfair|blame|judg\w+|favou?r\w*|want|wants|mean|meant|intend\w*|don'?t believe in me|give up on|abandon\w*)\b"
    r"|\b(?:it\s+(?:feels|seems|looks)\s+(?:like|as if|as though)|as if they|seems to me|makes me feel|doesn'?t value|don'?t value|not valuing|doesn'?t care|don'?t care about me|doesn'?t believe in me|don'?t believe in me)\b",
    re.I,
)


def topics_for(text: str) -> list[str]:
    return sorted(t for t, p in TOPIC_PATTERNS.items() if p.search(text or ""))


def is_sensitive(topics: list[str] | set[str]) -> bool:
    return bool(set(topics) & SENSITIVE_TOPICS)


def is_interpretation(text: str) -> bool:
    """A feeling, belief or reading of someone's motives ("I may read refusal to fund as not valuing my education")."""
    return bool(INTERPRETATION_RE.search(text or ""))


VIEW_PREFIX = "User's view: "


def view_text(content: str) -> str:
    """Stored form of an interpretation keeps the user's own words, quoted, behind a fixed prefix."""
    c = (content or "").strip()
    if c.startswith(VIEW_PREFIX):
        return c
    return f'{VIEW_PREFIX}"{c.rstrip(".")}"'


def view_quote(content: str) -> str:
    c = (content or "").strip()
    return c[len(VIEW_PREFIX):] if c.startswith(VIEW_PREFIX) else f'"{c}"'