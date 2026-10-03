"""The continuation packet: what a fresh provider chat receives so the user never has to repeat anything.

Facts in it are context from the conversation, NOT evidence for research verification. The packet stays inside a token
budget: each section has its own share, the newest material wins, nothing is dropped silently without a marker, and the raw
messages stay in the store.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Sequence

from backend.thread.compress import FIELDS, render_summary
from backend.thread.store import approx_tokens

HEADER = "OMNIBRAIN CONTINUATION CONTEXT"
NOTE = ("This is the running conversation so far, carried over from an earlier chat. Everything below is context from that conversation "
        "(including things the user said); it is not verified evidence.")
JOB_INSTRUCTIONS = (
    "Use this as background for the question that follows. Don't mention that it was supplied, and don't ask the user to repeat anything already covered above."
)
INSTRUCTIONS = (
    "Continue the conversation naturally from where it left off. Don't restart or re-introduce yourself, don't ask the user to repeat anything "
    "already covered above, and don't mention this context, the earlier chat or any transfer. Treat the user's latest message as the thing to answer now."
)
SHARES = {"user": 0.08, "project": 0.08, "history": 0.17, "recent": 0.30, "facts": 0.10, "decisions": 0.07, "open": 0.05, "corrections": 0.04,
          "prefs": 0.05, "earlier": 0.16}
ORDER = ["user", "project", "history", "recent", "earlier", "facts", "decisions", "open", "corrections", "prefs"]
TITLES = {
    "user": "CURRENT USER", "project": "PROJECT", "history": "CONVERSATION HISTORY (compressed)", "recent": "RECENT DISCUSSION",
    "earlier": "RELEVANT EARLIER DISCUSSION", "facts": "ESTABLISHED FACTS", "decisions": "DECISIONS", "open": "OPEN QUESTIONS",
    "corrections": "CORRECTIONS", "prefs": "IMPORTANT USER PREFERENCES",
}


def clip(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split()) if len(text or "") < 400 else (text or "").strip()
    if len(text) <= max_chars:
        return text
    keep = max(20, max_chars - 40)
    head, tail = text[: int(keep * 0.65)], text[-int(keep * 0.35) :]
    return f"{head} [... {len(text) - keep} characters omitted ...] {tail}"


def _tail(task_text: str, include_task: bool) -> list[str]:
    if include_task:
        return ["INSTRUCTIONS: " + INSTRUCTIONS, "", "CURRENT TASK (the user's latest message):", task_text]
    return ["INSTRUCTIONS: " + JOB_INSTRUCTIONS]


def _fit(lines: list[str], budget_tokens: int) -> list[str]:
    out, used = [], 0
    for ln in lines:
        t = approx_tokens(ln) + 1
        if used + t > budget_tokens:
            break
        out.append(ln)
        used += t
    return out


def _day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%d %b %Y")


@dataclass
class Packet:
    text: str
    tokens: int
    sections: dict[str, int] = field(default_factory=dict)  # lines kept per section
    omitted: dict[str, int] = field(default_factory=dict)  # lines that did not fit, per section
    budget: int = 0

    @property
    def within_budget(self) -> bool:
        return self.tokens <= self.budget


def build_packet(
    *,
    state: dict[str, Any] | None,
    recent: Sequence[Any],
    task: str,
    budget_tokens: int = 6000,
    user_lines: Sequence[str] = (),
    project: str | None = None,
    open_summary: dict[str, Any] | None = None,
    earlier: Sequence[Any] = (),
    summary_hits: Sequence[Any] = (),
    now_label: str = "",
    include_task: bool = True,
) -> Packet:
    """state = L3 thread state; open_summary = L2 of the open segment's older messages (research threads); recent = L1 verbatim."""
    items: dict[str, list[dict[str, Any]]] = {k: list((state or {}).get("items", {}).get(k, [])) for k in FIELDS}
    if open_summary:  # the open segment's own older material counts as thread state too
        for k in FIELDS:
            for it in open_summary.get("items", {}).get(k, []):
                if all(it["t"] != o["t"] for o in items[k]):
                    items[k].append(it)
    budget = max(600, budget_tokens)
    task_text = clip(task, int(budget * 0.25) * 4) if include_task else ""
    avail = budget - approx_tokens(task_text) - approx_tokens(HEADER + NOTE + INSTRUCTIONS) - 60
    cap = {k: max(40, int(avail * v)) for k, v in SHARES.items()}
    sections: dict[str, list[str]] = {}

    sections["user"] = [f"- {u}" for u in user_lines]
    proj: list[str] = ([f"Project: {project}"] if project else []) + [f"- {i['t']}" for i in reversed(items["project_state"])]
    sections["project"] = proj
    hist: list[str] = []
    h = (state or {}).get("history", [])
    arch = (state or {}).get("archive") or {}
    if len(h) > 6 or arch:
        older = h[:-6]
        n_old = len(older) + int(arch.get("segments", 0))
        sample = list(arch.get("sample", [])) + [x["line"][:90] for x in older]
        hist.append(f"- Earlier ({n_old} earlier parts of the conversation): " + "; ".join(sample[-3:]))
    hist += [f"- {x['label']}{(' (' + x['span'] + ')') if x.get('span') else ''}: {x['line']}" for x in h[-6:]]
    if open_summary and open_summary.get("narrative"):
        hist.append(f"- Most recent part (before the verbatim excerpt below): {open_summary['narrative']}")
    if items["conclusions"]:
        hist += [f"- Concluded: {i['t']}" for i in items["conclusions"][-3:]]
    if items["rejected"]:
        hist += [f"- Ruled out: {i['t']}" for i in items["rejected"][-4:]]
    if items["arguments"]:
        hist += [f"- Reasoning: {i['t']}" for i in items["arguments"][-3:]]
    kt = [i["t"] for i in items["key_terms"][-10:]] + [i["t"] for i in items["entities"][-8:]]
    if kt:
        hist.append("- Names and terms in play: " + ", ".join(dict.fromkeys(kt)))
    if items["direction"]:
        hist.append(f"- Current direction: {items['direction'][-1]['t']}")
    sections["history"] = hist
    rec: list[str] = []
    per_msg = max(300, cap["recent"] * 4 // 6)
    for m in reversed(list(recent)):  # newest first so the budget cuts the OLD end
        rec.append(f"{'User' if m.role == 'user' else 'Assistant'}: {clip(m.content, per_msg)}")
    sections["recent"] = rec  # newest first for fitting; reversed after
    earl: list[str] = []
    for s in summary_hits:
        earl.append(f"- (summary of an earlier part) {render_summary(s.summary, 500) if s.summary else clip(s.text, 500)}")
    for f in sorted(earlier, key=lambda f: f.message.seq):
        m = f.message
        earl.append(f"- [{_day(m.ts)}] {'User' if m.role == 'user' else 'Assistant'}: {clip(m.content, 420)}")
    sections["earlier"] = earl
    sections["facts"] = [f"- {i['t']}" for i in reversed(items["facts"])]
    sections["decisions"] = [f"- {i['t']}" for i in reversed(items["decisions"])]
    sections["open"] = [f"- {i['t']}" for i in reversed(items["open_questions"])]
    sections["corrections"] = [f"- {i['t']}" for i in reversed(items["corrections"])]
    sections["prefs"] = [f"- {i['t']}" for i in reversed(items["preferences"])]

    kept: dict[str, list[str]] = {}
    omitted: dict[str, int] = {}
    for k in ORDER:
        lines = sections.get(k, [])
        fit = _fit(lines, cap[k])
        omitted[k] = len(lines) - len(fit)
        kept[k] = list(reversed(fit)) if k == "recent" else fit
    # facts/decisions/open/corrections/prefs were built newest-first: show oldest-to-newest
    for k in ("facts", "decisions", "open", "corrections", "prefs"):
        kept[k] = list(reversed(kept[k]))

    out = [HEADER, NOTE, ""]
    if now_label:
        out += [now_label, ""]
    for k in ORDER:
        if kept[k]:
            out.append(TITLES[k] + ":")
            out += kept[k]
            if omitted[k] and k != "recent":
                out.append(f"- (+{omitted[k]} more kept in the full record)")
            out.append("")
    out += _tail(task_text, include_task)
    text = "\n".join(out)
    # last resort: never exceed the budget -- shrink the verbatim excerpt first, then history
    guard = 0
    while approx_tokens(text) > budget and guard < 6:
        guard += 1
        for k in ("recent", "history", "earlier", "facts"):
            if len(kept[k]) > 1:
                if k == "recent":
                    kept[k] = kept[k][1:]
                else:
                    kept[k] = kept[k][:-1] if k != "facts" else kept[k][1:]
                break
        else:
            break
        out = [HEADER, NOTE, ""] + ([now_label, ""] if now_label else [])
        for k in ORDER:
            if kept[k]:
                out.append(TITLES[k] + ":")
                out += kept[k]
                out.append("")
        out += _tail(task_text, include_task)
        text = "\n".join(out)
    return Packet(text=text, tokens=approx_tokens(text), sections={k: len(v) for k, v in kept.items()}, omitted=omitted, budget=budget)

RECALL_NOTE = ("Earlier parts of this same conversation that match what the user is asking about (oldest first). Use them if relevant; "
               "they are context from the conversation, not verified evidence. Don't mention that they were supplied.")


def recall_block(found: Sequence[Any], summaries: Sequence[Any], task: str, budget_tokens: int = 2500) -> tuple[str, int]:
    """For a message that stays in the current chat but asks about something far back: just the relevant lines + the task."""
    lines: list[str] = []
    for s in summaries[:2]:
        lines.append(f"- (summary of an earlier part) {render_summary(s.summary, 500) if s.summary else clip(s.text, 500)}")
    for f in sorted(found, key=lambda f: f.message.seq):
        m = f.message
        lines.append(f"- [{_day(m.ts)}] {'User' if m.role == 'user' else 'Assistant'}: {clip(m.content, 420)}")
    kept = _fit(lines, budget_tokens)
    if not kept:
        return task, 0
    text = "RELEVANT EARLIER DISCUSSION:\n" + RECALL_NOTE + "\n" + "\n".join(kept) + "\n\nCURRENT TASK (the user's latest message):\n" + task
    return text, len(kept)