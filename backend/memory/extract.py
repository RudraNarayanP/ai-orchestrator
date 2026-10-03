"""Turning what the USER said into memories. Never reads an AI's answer: only the user's own messages.

Rule-based on purpose (deterministic, testable, no model call, no data leaves the machine). A usefulness threshold
keeps small talk out; a privacy filter keeps secrets out; "don't remember this" and "forget X" are honoured first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.memory.embed import content_tokens
from backend.memory.schema import MemoryType, Source

NO_REMEMBER_RE = re.compile(
    r"\b(?:do(?:n'?t| not)\s+(?:remember|save|store|keep|record)\s+(?:this|that|it|any of this)|off the record|"
    r"(?:no need|not necessary) to remember|just between us|"
    r"don'?t\s+(?:add|put)\s+(?:this|that|it)\s+(?:to|in|into)\s+(?:your\s+)?memory)", re.I)
FORGET_RE = re.compile(r"\b(?:forget|delete|remove|erase|stop remembering)\s+(?:(?:that|about|the memory (?:that|about|of)|my memory (?:that|about|of))\s+)?(?P<what>[^.?!\n]{3,120})", re.I)
REMEMBER_RE = re.compile(r"^\s*(?:please\s+)?(?:remember|note|keep in mind|save|store)(?:\s+(?:this|that))?\s*[:,-]?\s*(?:that\s+)?(?P<what>.+)$", re.I | re.S)
FUTURE_REF_RE = re.compile(r"^\s*for future reference\s*[,:]?\s*(?P<what>.+)$", re.I | re.S)
CORRECTION_RE = re.compile(r"^\s*(?:actually|correction|update|no,?\s+(?:it'?s|it is|i meant|i said|that'?s wrong)|that'?s (?:wrong|not right|outdated)|scratch that|i was wrong)\s*[:,-]?\s*(?P<what>.*)$", re.I | re.S)
SECRET_RE = re.compile(r"(api[_ -]?key|secret|password|passwd|passcode|token|sk-[a-z0-9]{10,}|ghp_[a-z0-9]{10,}|\b(?:\d[ -]*?){13,19}\b|private key|\bcvv\b|\bpin\b\s*(?:is|:)?\s*\d)", re.I)
QUESTION_START = re.compile(r"^\s*(what|which|who|whom|whose|when|where|why|how|is|are|was|were|do|does|did|can|could|would|should|will|tell me|explain|list|show)\b", re.I)
HYPOTHETICAL_RE = re.compile(r"\b(?:if i|suppose i|imagine i|what if i|let'?s say i|would i)\b", re.I)

# (regex, type, slot builder, base score, neutral rewrite)
PATTERNS: list[tuple[re.Pattern[str], MemoryType, str, float]] = [
    (re.compile(r"\bi (?:currently )?live (?:in|at|near) (?P<v>[^.,;!?]+)", re.I), MemoryType.FACT, "residence", 0.8),
    (re.compile(r"\bi(?:'ve| have)? (?:just )?moved (?:to|into) (?P<v>[^.,;!?]+)", re.I), MemoryType.FACT, "residence", 0.8),
    (re.compile(r"\bi(?:'m| am) (?:from|based in|living in) (?P<v>[^.,;!?]+)", re.I), MemoryType.FACT, "residence", 0.75),
    (re.compile(r"\bmy name is (?P<v>[^.,;!?]+)", re.I), MemoryType.FACT, "name", 0.9),
    (re.compile(r"\bi work (?:at|for) (?P<v>[^.,;!?]+)", re.I), MemoryType.FACT, "employer", 0.75),
    (re.compile(r"\bi work as (?:an? )?(?P<v>[^.,;!?]+)", re.I), MemoryType.FACT, "occupation", 0.75),
    (re.compile(r"\bi(?:'m| am) (?:studying|a student of|majoring in|doing (?:a|my) (?:phd|masters?|degree) in) (?P<v>[^.,;!?]+)", re.I), MemoryType.FACT, "education", 0.75),
    (re.compile(r"\bi(?:'m| am) (?:an? )(?P<v>[a-z][a-z -]{2,40}(?:developer|engineer|student|researcher|designer|teacher|doctor|analyst|writer|scientist))", re.I), MemoryType.FACT, "occupation", 0.7),
    (re.compile(r"\bi (?:speak|write in) (?P<v>[^.,;!?]+)", re.I), MemoryType.FACT, "languages", 0.65),
    (re.compile(r"\bmy (?P<k>[a-z ]{2,24}) is (?P<v>[^.,;!?]+)", re.I), MemoryType.FACT, "attr", 0.6),
    (re.compile(r"\bi (?:really )?(?P<verb>prefer|like|love|enjoy|hate|dislike|can'?t stand) (?P<v>[^.;!?]+)", re.I), MemoryType.PREFERENCE, "pref", 0.7),
    (re.compile(r"\b(?:please )?always (?P<v>[^.;!?]+)", re.I), MemoryType.PREFERENCE, "pref", 0.7),
    (re.compile(r"\b(?:please )?never (?P<v>[^.;!?]+)", re.I), MemoryType.PREFERENCE, "pref", 0.7),
    (re.compile(r"\bi want (?:my )?(?:answers?|replies|responses?) (?:to be )?(?P<v>[^.;!?]+)", re.I), MemoryType.PREFERENCE, "pref:answer_style", 0.8),
    (re.compile(r"\bmy fav(?:ou?rite)? (?P<k>[a-z ]{2,24}) is (?P<v>[^.,;!?]+)", re.I), MemoryType.PREFERENCE, "attr", 0.75),
    (re.compile(r"\b(?:my goal is(?: to)?|i(?:'m| am) trying to|i want to|i plan to|i(?:'m| am) planning to|i need to|i(?:'m| am) aiming to) (?P<v>[^.;!?]+)", re.I), MemoryType.GOAL, "goal", 0.7),
    (re.compile(r"\bi(?:'m| am) (?:currently )?(?:working on|building|developing|writing) (?P<v>[^.;!?]+)", re.I), MemoryType.GOAL, "goal", 0.65),
    (re.compile(r"\b(?:yesterday|last (?:week|month|year|night)|earlier today|this morning|on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)),? i (?P<v>[^.;!?]+)", re.I), MemoryType.EPISODIC, "", 0.5),
]
GOAL_DONE_RE = re.compile(r"\bi(?:'ve| have)? (?:finished|completed|done|submitted|shipped|given up on|abandoned|dropped|stopped) (?P<v>[^.;!?]+)|\bi no longer (?:want|plan) to (?P<w>[^.;!?]+)", re.I)
PROJECT_RE = re.compile(r"\b(?:my |our |the )?(?:project|repo|codebase|app|thesis|paper)\s+(?:called |named |is )?[\"']?(?P<p>[A-Z][\w-]{2,30})[\"']?|\b(?P<q>[A-Z][\w-]{2,30}) (?:project|repo|codebase)\b")
ENTITY_RE = re.compile(r"\b[A-Z][\w'-]{1,}(?:\s+[A-Z][\w'-]{1,})*\b")
_NOT_ENT = {"I", "I'm", "I've", "My", "The", "A", "An", "Please", "Actually", "Yesterday", "Also", "And", "But", "So", "Remember", "Note", "Always", "Never", "It", "This", "That", "What", "How"}

TEMPLATES = {"residence": "Lives in {v}", "name": "Name is {v}", "employer": "Works at {v}", "education": "Studies {v}", "languages": "Speaks {v}"}
_VERB3 = {"prefer": "Prefers", "like": "Likes", "love": "Loves", "enjoy": "Enjoys", "hate": "Hates", "dislike": "Dislikes", "can't stand": "Can't stand", "cant stand": "Can't stand"}


@dataclass
class Candidate:
    content: str
    memory_type: MemoryType
    slot: str | None
    source: Source
    score: float
    entities: list[str] = field(default_factory=list)
    project: str | None = None
    correction: bool = False
    deactivate_goal: str | None = None
    goal_active: bool | None = None


@dataclass
class Extraction:
    candidates: list[Candidate] = field(default_factory=list)
    forget: list[str] = field(default_factory=list)
    skipped: str = ""  # why nothing was kept ("no_remember", "secret", "question", "low_value")


def entities_in(text: str) -> list[str]:
    out = []
    for m in ENTITY_RE.finditer(text or ""):
        e = m.group(0).strip()
        if e not in _NOT_ENT and len(e) > 1 and e.split()[0] not in _NOT_ENT:
            out.append(e)
    return sorted(set(out))


def neutralise(sentence: str) -> str:
    """'I live in Kyiv' -> 'Lives in Kyiv' (memory reads as a fact about the user, not as the user talking)."""
    s = " ".join(sentence.strip().rstrip(".!").split())
    s = re.sub(r"^please\s+", "", s, flags=re.I)
    rules = [
        (r"^i(?:'m| am) ", "Is "), (r"^i(?:'ve| have) ", "Has "), (r"^i (?:currently )?live ", "Lives "), (r"^i work ", "Works "),
        (r"^i (?:really )?(prefer|like|love|enjoy|hate|dislike) ", lambda m: _VERB3[m.group(1).lower()] + " "),
        (r"^i want to ", "Wants to "), (r"^i want ", "Wants "), (r"^i need to ", "Needs to "), (r"^i plan to ", "Plans to "), (r"^i speak ", "Speaks "),
        (r"^i write in ", "Writes in "), (r"^i study ", "Studies "), (r"^my ", "The user's "), (r"^i ", ""),
    ]
    for pat, rep in rules:
        if re.match(pat, s, re.I):
            s = re.sub(pat, rep, s, count=1, flags=re.I)
            break
    s = re.sub(r"\bmy\b", "their", s, flags=re.I)
    s = re.sub(r"\bi(?:'m| am)\b", "is", s, flags=re.I)
    s = re.sub(r"\bi\b", "they", s, flags=re.I)
    return s[:1].upper() + s[1:] if s else s


def _slot(base: str, m: re.Match[str], value: str) -> str:
    if base == "attr":
        return "attr:" + "_".join(re.findall(r"[a-z0-9]+", m.group("k").lower())[:3])
    if base == "pref":
        verbless = " ".join(re.findall(r"[a-z0-9]+", value.lower())[:4])
        return "pref:" + (verbless or "general")
    return base


def extract(message: str, *, project: str | None = None, conversation: str | None = None) -> Extraction:
    """Candidates from ONE user message. Usefulness threshold 0.55; nothing from questions, hypotheticals or secrets."""
    text = " ".join((message or "").split())
    out = Extraction()
    if not text:
        return out
    if NO_REMEMBER_RE.search(text):
        out.skipped = "no_remember"
        return out
    fm = FORGET_RE.search(text)
    if fm and not re.search(r"\bforget (?:it|that|this)\b", text, re.I) is None or (fm and fm.group("what")):
        what = fm.group("what").strip(" .") if fm else ""
        if what and not re.match(r"^(?:it|this|that|the (?:question|previous|last))\b", what, re.I) and len(what) > 3:
            out.forget.append(what)
            return out
    if SECRET_RE.search(text):
        out.skipped = "secret"
        return out
    explicit = False
    body = text
    m = REMEMBER_RE.match(text) or FUTURE_REF_RE.match(text)
    if m:
        explicit, body = True, m.group("what").strip()
    corr = CORRECTION_RE.match(body)
    correction = False
    if corr and corr.group("what").strip():
        correction, body = True, corr.group("what").strip()
    elif corr:
        out.skipped = "empty_correction"
        return out
    if not explicit and not correction and (QUESTION_START.match(body) or body.rstrip().endswith("?")):
        out.skipped = "question"
        return out
    if HYPOTHETICAL_RE.search(body):
        out.skipped = "hypothetical"
        return out
    pm = PROJECT_RE.search(body)
    proj = project or ((pm.group("p") or pm.group("q")) if pm else None)
    for sentence in re.split(r"(?<=[.!?])\s+|;\s+|,?\s+(?:and|but|also)\s+(?=i(?:'m|'ve| am| have| live| work| prefer| like| love| want| need| plan)?\b)", body, flags=re.I):
        sentence = sentence.strip()
        if len(sentence.split()) < 3:
            continue
        done = None if re.match(r"(?:yesterday|last |earlier today|this morning|on (?:mon|tues|wednes|thurs|fri|satur|sun)day)", sentence, re.I) else GOAL_DONE_RE.search(sentence)
        if done:
            out.candidates.append(Candidate(content=neutralise(sentence), memory_type=MemoryType.GOAL, slot=None, source=Source.USER_EXPLICIT, score=0.7,
                                            deactivate_goal=(done.group("v") or done.group("w") or "").strip(), goal_active=False, project=proj))
            continue
        hit = None
        for pat, mtype, slot_base, score in PATTERNS:
            pm2 = pat.search(sentence)
            if pm2:
                hit = (pat, mtype, slot_base, score, pm2)
                break
        if hit is None and explicit:
            # "remember that the deadline is Friday": the user asked, so it is kept even without a known pattern
            hit = (None, MemoryType.PROJECT if proj else MemoryType.FACT, "", 0.7, None)
        if hit is None:
            continue
        _, mtype, slot_base, score, pm2 = hit
        value = (pm2.group("v") if pm2 is not None and "v" in pm2.re.groupindex else sentence) or sentence
        slot = _slot(slot_base, pm2, value) if (pm2 is not None and slot_base) else (slot_base or None)
        if explicit:
            score += 0.25
        if correction:
            score += 0.2
        score += min(0.1, 0.02 * len(entities_in(sentence)))
        if mtype == MemoryType.PROJECT or (proj and mtype in (MemoryType.FACT, MemoryType.GOAL, MemoryType.PREFERENCE) and explicit):
            slot = slot if mtype != MemoryType.PROJECT else None
        if score < 0.55:
            out.skipped = out.skipped or "low_value"
            continue
        # Explicit = the user stated it as a fact or rule about themselves (or asked to remember / corrected it).
        # Inferred = soft signals: casual likes, plans, work-in-progress, one-off events. Inferred never overrides explicit.
        soft = mtype == MemoryType.EPISODIC or slot_base == "goal" and not re.search(r"my goal is", sentence, re.I) or (
            pm2 is not None and pm2.groupdict().get("verb", "").lower() in ("like", "enjoy", "dislike")) or slot_base == "pref" and False
        src = Source.USER_EXPLICIT if (explicit or correction or not soft) else Source.MODEL_INFERRED
        templ = TEMPLATES.get(slot_base or "")
        if templ and pm2 is not None and "v" in pm2.re.groupindex:
            text_out = templ.format(v=pm2.group("v").strip())
        elif not explicit or pm2 is not None:
            text_out = neutralise(sentence)
        else:
            text_out = sentence.rstrip(".!")
        out.candidates.append(Candidate(
            content=text_out, memory_type=mtype, slot=slot, source=src,
            score=min(1.0, score), entities=entities_in(sentence), project=proj if (proj and (mtype in (MemoryType.PROJECT, MemoryType.GOAL) or explicit)) else None,
            correction=correction, goal_active=True if mtype == MemoryType.GOAL else None))
    if not out.candidates and not out.skipped:
        out.skipped = "low_value"
    return out