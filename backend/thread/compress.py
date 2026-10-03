"""Hierarchical compression. Not "summarise the chat": a structured record of what the conversation established.

L1  recent turns, verbatim (packet.py)
L2  per-segment continuation summary (this module): decisions, facts, arguments, open questions, preferences, key terms,
    project state, entities, corrections, conclusions, rejected things, current direction
L3  thread state: the segment summaries consolidated, corrections superseding what they correct (consolidate())
L4  only genuinely durable items go on to long-term memory (promote())

The curator model (OpenRouter / local) does L2 when available; a deterministic extractor is the fallback and also the
safety net: every LLM item must be grounded in the segment's own words, and the high-precision categories (preferences,
corrections, decisions, rejections) are always unioned in from the deterministic pass.
Raw messages are never touched.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Sequence

from backend.memory.embed import content_tokens

FIELDS = ("decisions", "facts", "arguments", "open_questions", "preferences", "key_terms", "project_state", "entities", "corrections",
          "conclusions", "rejected", "direction")
CAPS = {"decisions": 8, "facts": 10, "arguments": 6, "open_questions": 6, "preferences": 8, "key_terms": 12, "project_state": 6, "entities": 12,
        "corrections": 6, "conclusions": 6, "rejected": 6, "direction": 2}
STATE_CAPS = {k: int(v * 2.5) for k, v in CAPS.items()}

_S = re.IGNORECASE
PAT = {
    "preferences": re.compile(r"\b(i prefer|i'd prefer|i would prefer|i like|i love|i hate|i don'?t like|i do not like|i want you to|please (always|never)|from now on|"
                              r"always (use|give|keep|answer|reply|write)|never (use|give|add|write)|keep (it|answers|replies|responses)|don'?t (use|give|include|add|explain))\b", _S),
    "decisions": re.compile(r"\b(let'?s (go with|use|do|pick|stick with)|we('ll| will) (use|go|do|pick)|we decided|i('ve| have)? decided|i('ll| will) (use|go with|pick)|"
                            r"i('m| am) going (with|to use)|going with|settled on|the (final )?(choice|decision) is|we('re| are) (using|going with))\b", _S),
    "corrections": re.compile(r"\b(actually|that'?s (wrong|not right|incorrect|not what)|i meant|correction|i misspoke|scratch that|no,? (it|that|the)\b|not .{1,40}, (but|rather))", _S),
    "rejected": re.compile(r"\b(not going to|won'?t (use|do|go)|rejected|ruled out|rule out|scrap(ped)?|dropp?(ed|ing)|don'?t want|no longer|abandon(ed)?|instead of|decided against|off the table)\b", _S),
    "project_state": re.compile(r"\b(currently|already|so far|is (running|deployed|live|done|built)|are (running|using)|deployed|implemented|working on|status:|we have|version \d|v\d+(\.\d+)*)\b", _S),
    "conclusions": re.compile(r"\b(in (summary|short|conclusion)|overall|therefore|bottom line|the (best|short|simple) answer|i recommend|we recommend|so the answer|conclusion)\b", _S),
    "arguments": re.compile(r"\b(because|trade-?offs?|however|whereas|pros|cons|advantages?|downsides?|on the other hand|the reason|which means|the catch)\b", _S),
    "open_hint": re.compile(r"\b(still (unsure|not sure|undecided)|need to (figure|decide|find|check)|todo|to be decided|haven'?t decided|not sure (yet|whether|if))\b", _S),
}
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")
_CODE = re.compile(r"```.*?```", re.S)
_QUOTED = re.compile(r"[\"\u201c]([^\"\u201d]{4,60})[\"\u201d]|`([^`]{2,40})`")
_ENTITY = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,2})\b")
_IDENT = re.compile(r"\b([A-Za-z]+_[A-Za-z_0-9]+|[a-z]+[A-Z][A-Za-z0-9]+|[A-Z]{2,}[0-9]*|\d+(?:\.\d+)?\s?(?:GB|MB|ms|%|km|USD|EUR|INR|GBP|k|M))\b")
_NOT_ENTITY = frozenset("The This That These Those It I We You He She They There Here What When Where Why How Which Who If But And Or So As In On At For To Of By With Yes No Not "
                        "Please Thanks Thank Hello Hi Let Lets Also Then Now Just My Our Your Can Could Would Should Do Does Did Is Are Was Were Be Have Has Had User Assistant".split())


def sentences(text: str) -> list[str]:
    text = _CODE.sub(" ", text or "")
    out = []
    for s in _SPLIT.split(text):
        s = re.sub(r"^[\s\-*\u2022>#\d.)]+", "", s).strip()
        if 12 <= len(s) <= 300:
            out.append(s)
        elif len(s) > 300:
            out.append(s[:297].rstrip() + "...")
    return out


def _toks(s: str) -> set[str]:
    return set(content_tokens(s))


def similar(a: str, b: str, thr: float = 0.6) -> bool:
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    return len(ta & tb) / len(ta | tb) >= thr


def overlap(a: str, b: str) -> float:
    ta, tb = _toks(a), _toks(b)
    return len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0


def _add(items: dict[str, list[dict[str, Any]]], key: str, text: str, seq: int, score: float) -> None:
    text = text.strip()
    if not text:
        return
    lst = items.setdefault(key, [])
    for it in lst:
        if similar(it["t"], text):
            if seq >= it["seq"]:
                it["t"], it["seq"] = text, seq
            it["w"] = max(it["w"], score)
            return
    lst.append({"t": text, "seq": seq, "w": round(score, 2)})


def _cap(items: dict[str, list[dict[str, Any]]], caps: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for k in FIELDS:
        lst = items.get(k, [])
        if len(lst) > caps[k]:
            keep = sorted(lst, key=lambda i: (-i["w"], -i["seq"]))[: caps[k]]
            lst = sorted(keep, key=lambda i: i["seq"])
        out[k] = [{"t": i["t"], "seq": i["seq"]} for i in lst]
    return out


def deterministic_summary(messages: Sequence[Any]) -> dict[str, Any]:
    """Rule-based L2. messages need .role, .content, .seq. Cheap, predictable, always available."""
    items: dict[str, list[dict[str, Any]]] = {}
    ent_count: Counter[str] = Counter()
    ent_user: set[str] = set()
    term_count: Counter[str] = Counter()
    users = [m for m in messages if m.role == "user"]
    assistants = [m for m in messages if m.role == "assistant"]
    for m in messages:
        is_user = m.role == "user"
        for t in content_tokens(m.content):
            term_count[t] += 1
        for g in _QUOTED.finditer(m.content or ""):
            _add(items, "key_terms", (g.group(1) or g.group(2)).strip(), m.seq, 1.0)
        for g in _IDENT.finditer(_CODE.sub(" ", m.content or "")):
            _add(items, "key_terms", g.group(1), m.seq, 0.5)
        for sent in sentences(m.content):
            body = sent
            for g in _ENTITY.finditer(sent):
                e = g.group(1)
                if e.split()[0] in _NOT_ENTITY or len(e) < 3:
                    continue
                ent_count[e] += 1
                if is_user:
                    ent_user.add(e)
            q = sent.rstrip().endswith("?")
            if is_user:
                if PAT["preferences"].search(body):
                    _add(items, "preferences", body, m.seq, 2.0)
                if PAT["decisions"].search(body):
                    _add(items, "decisions", body, m.seq, 2.0)
                if PAT["corrections"].search(body) and not q:
                    _add(items, "corrections", body, m.seq, 2.0)
                if PAT["rejected"].search(body):
                    _add(items, "rejected", body, m.seq, 1.5)
                if PAT["project_state"].search(body) and not q:
                    _add(items, "project_state", body, m.seq, 1.2)
                if q or PAT["open_hint"].search(body):
                    _add(items, "open_questions", body, m.seq, 1.0 + (0.5 if m is users[-1] else 0))
                if PAT["arguments"].search(body):
                    _add(items, "arguments", body, m.seq, 0.8)
            else:
                if PAT["conclusions"].search(body):
                    _add(items, "conclusions", body, m.seq, 1.5)
                if PAT["arguments"].search(body):
                    _add(items, "arguments", body, m.seq, 1.0)
                if PAT["decisions"].search(body) and not q:
                    _add(items, "decisions", body, m.seq, 0.9)
                if PAT["project_state"].search(body) and not q:
                    _add(items, "project_state", body, m.seq, 0.7)
                if not q and re.search(r"\d|\b[A-Z][a-z]+ [A-Z][a-z]+|\b(is|are|was|were|has|have)\b", body) and not re.search(r"\b(might|may|could|perhaps|maybe|I think)\b", body, _S):
                    _add(items, "facts", body, m.seq, 0.5 + (0.4 if re.search(r"\d", body) else 0) + (0.3 if m is assistants[-1] else 0))
                if q and m is assistants[-1]:
                    _add(items, "open_questions", body, m.seq, 1.2)
    for e, n in ent_count.most_common(30):
        if n >= 2 or e in ent_user:
            _add(items, "entities", e, 0, n + (2 if e in ent_user else 0))
    if users:
        _add(items, "direction", users[-1].content.strip().replace("\n", " ")[:240], users[-1].seq, 3.0)
    if assistants:
        last = sentences(assistants[-1].content)
        if last:
            _add(items, "direction", last[-1], assistants[-1].seq, 2.0)
    for k, v in items.items():  # ranked within field by weight
        v.sort(key=lambda i: (-i["w"], -i["seq"]))
    capped = _cap(items, CAPS)
    top_terms = [t for t, _ in term_count.most_common(8)]
    first = users[0].content.strip().replace("\n", " ")[:140] if users else ""
    last_u = users[-1].content.strip().replace("\n", " ")[:140] if users else ""
    narrative = f"Topics: {', '.join(top_terms)}. Started with: \"{first}\"." + (f" Latest: \"{last_u}\"." if last_u and last_u != first else "") if top_terms else (first or "")
    return {"narrative": narrative, "items": capped, "n_messages": len(messages), "first_seq": messages[0].seq if messages else 0, "last_seq": messages[-1].seq if messages else 0}


# ---------------------------------------------------------------------------------------------- curator (LLM) pass
CURATOR_SYSTEM = (
    "You compress a chat transcript into a structured continuation record so another assistant can carry on the conversation without the user repeating anything. "
    "Use ONLY what the transcript says; never add facts. Reply with JSON: {\"narrative\": str, " + ", ".join(f"\"{k}\": [str]" for k in FIELDS) + "}. "
    "decisions = things the user/assistant settled; facts = established facts stated in the chat; arguments = reasoning and trade-offs; open_questions = unresolved; "
    "preferences = how the user wants things done; key_terms = names, numbers, quoted phrases worth keeping exactly; project_state = where the work stands; "
    "entities = people, places, products, projects; corrections = things the user corrected; conclusions = what was concluded; rejected = options explicitly ruled out; "
    "direction = where the conversation is heading now. Short items, quote exact wording where it matters. Empty list if none."
)
Curator = Callable[[list[dict[str, str]]], "dict[str, Any] | None"]


def _transcript(messages: Sequence[Any], limit_chars: int = 24000) -> str:
    lines = [f"[{m.seq}] {m.role}: {m.content.strip()}" for m in messages]
    text = "\n".join(lines)
    if len(text) <= limit_chars:
        return text
    head, tail = text[: limit_chars // 2], text[-limit_chars // 2 :]
    return head + "\n[... middle of the transcript omitted ...]\n" + tail


def _grounded(item: str, corpus: set[str]) -> bool:
    toks = _toks(item)
    return bool(toks) and len(toks & corpus) / len(toks) >= 0.5


def curator_summary(messages: Sequence[Any], llm: Curator | None) -> tuple[dict[str, Any], str]:
    """(summary, method). method says which path produced it; the deterministic record is the floor."""
    det = deterministic_summary(messages)
    if llm is None or not messages:
        return det, "deterministic"
    try:
        parsed = llm([{"role": "system", "content": CURATOR_SYSTEM}, {"role": "user", "content": _transcript(messages)}])
    except Exception:  # noqa: BLE001 -- the curator being down must never lose or block the thread
        return det, "deterministic (curator unavailable)"
    if not isinstance(parsed, dict) or sum(1 for k in FIELDS if isinstance(parsed.get(k), list) and parsed.get(k)) < 2:
        return det, "deterministic (curator reply unusable)"
    corpus: set[str] = set()
    for m in messages:
        corpus |= _toks(m.content)
    by_seq = {m.seq: m for m in messages}
    items: dict[str, list[dict[str, Any]]] = {}
    for k in FIELDS:
        for n, raw in enumerate(parsed.get(k) or []):
            text = str(raw).strip()[:300]
            if not text or (k not in ("entities", "key_terms") and not _grounded(text, corpus)):
                continue
            seq = max(by_seq, key=lambda s: overlap(text, by_seq[s].content)) if by_seq else 0
            _add(items, k, text, seq, 3.0 - n * 0.1)
    for k in ("preferences", "corrections", "decisions", "rejected"):  # high-precision rules always count
        for it in det["items"].get(k, []):
            _add(items, k, it["t"], it["seq"], 1.0)
    narrative = str(parsed.get("narrative") or "").strip()[:500] or det["narrative"]
    out = {"narrative": narrative, "items": _cap(items, CAPS), "n_messages": det["n_messages"], "first_seq": det["first_seq"], "last_seq": det["last_seq"]}
    return out, "curator"


# ---------------------------------------------------------------------------------------------- L3 thread state
def empty_state() -> dict[str, Any]:
    return {"items": {k: [] for k in FIELDS}, "history": [], "superseded": [], "n_segments": 0}


def consolidate(state: dict[str, Any], summary: dict[str, Any], *, label: str, idx: int, span: str = "") -> dict[str, Any]:
    """Fold one closed segment into the thread state. A later correction retires the earlier fact/decision it corrects (kept under 'superseded')."""
    state = json.loads(json.dumps(state)) if state else empty_state()
    items = state.setdefault("items", {k: [] for k in FIELDS})
    for k in FIELDS:
        items.setdefault(k, [])
    new = summary.get("items", {})
    for k in FIELDS:
        for it in new.get(k, []):
            if k == "direction":
                continue
            _add_state(items[k], it)
    items["direction"] = list(new.get("direction", []))[:2]  # the current direction is the latest one
    for corr in new.get("corrections", []):
        for k in ("facts", "decisions", "project_state"):
            keep = []
            for it in items[k]:
                if it["seq"] < corr["seq"] and not similar(it["t"], corr["t"], 0.9) and overlap(it["t"], corr["t"]) >= 0.5:
                    state.setdefault("superseded", []).append({"t": it["t"], "by": corr["t"]})
                else:
                    keep.append(it)
            items[k] = keep
    state["superseded"] = state.get("superseded", [])[-12:]
    for k in FIELDS:
        cap = STATE_CAPS[k]
        if len(items[k]) > cap:
            items[k] = items[k][-cap:]
    state["history"].append({"idx": idx, "label": label, "span": span, "line": (summary.get("narrative") or "")[:220]})
    state["n_segments"] = state.get("n_segments", 0) + 1
    return state


def _add_state(lst: list[dict[str, Any]], it: dict[str, Any]) -> None:
    for old in lst:
        if similar(old["t"], it["t"]):
            old["t"], old["seq"] = it["t"], max(old["seq"], it["seq"])
            return
    lst.append({"t": it["t"], "seq": it["seq"]})


# ---------------------------------------------------------------------------------------------- L4 promotion
def durable_candidates(summary: dict[str, Any]) -> list[str]:
    """Only things the USER said that are standing instructions or facts about them. Chatter, arguments and project decisions stay in the thread."""
    out = []
    for it in summary.get("items", {}).get("preferences", []):
        out.append(it["t"])
    return out


def promote(summary: dict[str, Any], memory: Any, *, project: str | None = None, conversation: str | None = None) -> list[str]:
    """Hand the durable candidates to long-term memory, whose own extractor decides what is worth keeping. Returns what was stored."""
    if memory is None:
        return []
    stored: list[str] = []
    for text in durable_candidates(summary):
        try:
            learned = memory.learn(text, project=project, conversation=conversation)
        except Exception:  # noqa: BLE001
            continue
        stored += [r.memory.content for r in learned.added if r.memory is not None and r.action in {"created", "superseded"}]
    return stored