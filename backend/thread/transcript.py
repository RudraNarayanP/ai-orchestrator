"""Compress: the whole conversation, read completely, summarised by OmniBrain's own model, handed to fresh provider chats.

1. read   -- each provider tab's chat is read top to bottom (the page script scrolls the chat container in the background tab,
             expands "show more", collects every user/assistant turn); dedupe_turns() merges what several scroll positions saw.
2. check  -- page_complete() compares the page's turns with OmniBrain's own stored log of that chat. A page that lazy-loaded
             or virtualised turns away is not trusted: the stored log is used for that chat instead (merge_sources()).
3. sum    -- chunked_summary(): the transcript is cut into chunks the curator model can take; each chunk gets the structured
             summary (compress.curator_summary: OmniBrain's OpenRouter model, reasoning off, free-model fallback + cooldown,
             deterministic floor) and the chunks are folded together with compress.consolidate().
4. hand   -- the caller opens a NEW chat per provider and sends opening_context() as its first message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from backend.thread.compress import Curator, consolidate, curator_summary, empty_state

ROLES = {"user": "user", "human": "user", "you": "user", "assistant": "assistant", "model": "assistant", "bot": "assistant", "ai": "assistant"}


@dataclass
class Turn:
    role: str
    text: str
    id: str = ""
    provider: str = ""
    seq: int = 0  # position in the merged transcript (stored seq when it came from the log)
    source: str = "page"  # page | log


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _role(raw: str) -> str:
    return ROLES.get((raw or "").strip().lower(), "")


def dedupe_turns(raw: Sequence[dict[str, Any]], provider: str = "") -> list[Turn]:
    """Turns seen at several scroll positions, in page order, each kept once.

    Same message id -> same turn (the longest text wins: a turn read while half-rendered or collapsed is shorter). Without an
    id, the same role and text at the same place in the order is the same turn; the same text said again LATER (after other
    turns) is a new turn, so a user who typed "ok" twice keeps both."""
    out: list[Turn] = []
    by_id: dict[str, Turn] = {}
    for item in raw or []:
        role = _role(str(item.get("role") or ""))
        text = _norm(str(item.get("text") or ""))
        if not role or not text:
            continue
        mid = str(item.get("id") or "").strip()
        if mid and mid in by_id:
            if len(text) > len(by_id[mid].text):
                by_id[mid].text = text
            continue
        last = out[-1] if out else None
        if not mid and last is not None and last.role == role and (last.text == text or text.startswith(last.text) or last.text.startswith(text)):
            if len(text) > len(last.text):
                last.text = text
            continue
        if not mid and any(t.role == role and t.text == text and not t.id for t in out[-3:]):
            continue  # an overlapping scroll window saw it again
        turn = Turn(role=role, text=text, id=mid, provider=provider)
        out.append(turn)
        if mid:
            by_id[mid] = turn
    return out


def merge_windows(windows: Sequence[Sequence[dict[str, Any]]], near_px: int = 400) -> list[dict[str, Any]]:
    """The turns read at successive scroll positions, as one list in page order.

    The same turn shows up in overlapping windows: the same site id, or the same element (mark), is one turn; a virtualised
    list re-creates the element, so the same role and text at nearly the same position is one turn too. The same words much
    further down are a different turn. The longest text seen wins (a turn read half-rendered or still collapsed is shorter)."""
    kept: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    by_mark: dict[str, dict[str, Any]] = {}
    for win in windows:
        for raw in win or []:
            text = _norm(str(raw.get("text") or ""))
            if not text:
                continue
            mid, mark, top = str(raw.get("id") or ""), str(raw.get("mark") or ""), int(raw.get("top") or 0)
            prev = (by_id.get(mid) if mid else None) or (by_mark.get(mark) if mark else None)
            if prev is None:
                prev = next((k for k in kept if k["role"] == raw.get("role") and k["text"][:200] == text[:200] and abs(k["top"] - top) < near_px), None)
            if prev is None:
                prev = {"role": raw.get("role") or "", "text": text, "id": mid, "top": top}
                kept.append(prev)
            elif len(text) > len(prev["text"]):
                prev["text"] = text
            if mid:
                by_id[mid] = prev
            if mark:
                by_mark[mark] = prev
    kept.sort(key=lambda k: k["top"])
    return kept


def _key(text: str, n: int = 80) -> str:
    return _norm(text).lower()[:n]


def page_complete(page: Sequence[Turn], stored: Sequence[Any]) -> tuple[bool, str]:
    """Is the page's transcript at least everything OmniBrain itself logged for this chat, in the same order?

    The stored user line may be shorter than what the page shows (the first message of a chat carried a context packet around
    it), so a stored turn matches a page turn of the same role that CONTAINS its opening words."""
    logged = [m for m in stored if getattr(m, "role", "") in ("user", "assistant") and _norm(getattr(m, "content", ""))]
    if not page:
        return False, "nothing could be read from the page"
    if len(page) < len(logged):
        return False, f"the page showed {len(page)} turn(s), the log has {len(logged)}: some were not loaded"
    i = 0
    for m in logged:
        needle = _key(m.content, 60)
        while i < len(page) and not (page[i].role == m.role and needle in _norm(page[i].text).lower()):
            i += 1
        if i >= len(page):
            return False, f"a logged {m.role} turn (seq {getattr(m, 'seq', '?')}) is missing from the page"
        i += 1
    return True, f"{len(page)} turn(s), every logged turn present in order"


@dataclass
class Gathered:
    turns: list[Turn]
    sources: dict[str, str] = field(default_factory=dict)  # provider -> "page (...)" | "log (...)"


def merge_sources(stored_thread: Sequence[Any], pages: dict[str, list[Turn] | None]) -> Gathered:
    """The whole conversation, oldest first. OmniBrain's log is the backbone (it holds every chat the thread ever used, and
    research turns); a provider chat whose page was read COMPLETELY replaces its logged turns with the page's (full text,
    expanded parts, and anything typed straight into that tab). A page that was not complete is ignored, with the reason kept."""
    by_provider: dict[str, list[Any]] = {}
    for m in stored_thread:
        by_provider.setdefault(getattr(m, "provider", "") or "", []).append(m)
    use_page: dict[str, list[Turn]] = {}
    sources: dict[str, str] = {}
    for provider, turns in pages.items():
        # the page shows only the provider's CURRENT chat: compare it with that chat's own logged turns
        logged = [m for m in by_provider.get(provider, []) if getattr(m, "segment_id", None) == _last_segment(by_provider.get(provider, []))]
        ok, why = page_complete(turns or [], logged) if turns is not None else (False, "the tab could not be read")
        sources[provider] = ("page: " if ok else "log: ") + why
        if ok:
            use_page[provider] = list(turns or [])
    out: list[Turn] = []
    replaced: set[str] = set()
    for m in stored_thread:
        provider = getattr(m, "provider", "") or ""
        seg = getattr(m, "segment_id", None)
        if provider in use_page and seg == _last_segment(by_provider.get(provider, [])):
            if provider not in replaced:  # the page's turns go where that chat's first logged turn was
                replaced.add(provider)
                out.extend(Turn(t.role, t.text, t.id, provider, getattr(m, "seq", 0), "page") for t in use_page[provider])
            continue
        out.append(Turn(m.role, _norm(m.content), "", provider, getattr(m, "seq", 0), "log"))
    for provider, turns in use_page.items():
        if provider not in replaced:  # a chat OmniBrain never logged (typed straight into the tab): appended, not lost
            out.extend(Turn(t.role, t.text, t.id, provider, 0, "page") for t in turns)
    for provider in by_provider:
        sources.setdefault(provider or "omnibrain", "log: no tab to read")
    for n, t in enumerate(out, 1):
        t.seq = n
    return Gathered(out, sources)


def _last_segment(msgs: Sequence[Any]) -> Any:
    segs = [getattr(m, "segment_id", None) for m in msgs if getattr(m, "segment_id", None)]
    return segs[-1] if segs else None


@dataclass
class _Msg:  # the shape compress.curator_summary reads
    seq: int
    role: str
    content: str


def chunks(turns: Sequence[Turn], max_chars: int = 18000) -> list[list[Turn]]:
    """Consecutive turns, each chunk under max_chars; a single over-long turn is cut, never dropped."""
    out: list[list[Turn]] = []
    cur: list[Turn] = []
    size = 0
    for t in turns:
        pieces = [t.text[i : i + max_chars] for i in range(0, len(t.text), max_chars)] or [""]
        for p in pieces:
            if cur and size + len(p) > max_chars:
                out.append(cur)
                cur, size = [], 0
            cur.append(Turn(t.role, p, t.id, t.provider, t.seq, t.source))
            size += len(p) + 20
    if cur:
        out.append(cur)
    return out


def chunked_summary(turns: Sequence[Turn], curator: Curator | None, *, max_chars: int = 18000) -> tuple[dict[str, Any], list[str]]:
    """(thread state, method per chunk). Each chunk gets the structured summary; chunks fold in order, so a later correction
    retires what it corrects (consolidate)."""
    state = empty_state()
    methods: list[str] = []
    for n, part in enumerate(chunks(turns, max_chars), 1):
        summary, method = curator_summary([_Msg(t.seq, t.role, t.text) for t in part], curator)
        methods.append(method)
        state = consolidate(state, summary, label=f"part {n}", idx=n)
    return state, methods


OPENING = (
    "We are continuing a long conversation in a fresh chat. Below is a structured summary of everything so far "
    "(decisions, facts, corrections, open questions, preferences, the latest turns). Treat it as our shared context. "
    "Reply with just: Ready."
)


def opening_context(packet_text: str) -> str:
    return f"{OPENING}\n\n{packet_text.strip()}"
