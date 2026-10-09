"""Export a finished job as Markdown or JSON.

The Markdown is a readable record of the answer *and* its evidence: what was
claimed, which pages we opened and what they said, where the sources conflicted,
and what is still unverified. Text that came from the web or from a chat site is
untrusted, so it is flattened to single-line inline text and has link/markup
characters escaped; a hostile page title cannot inject structure into the export.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

_SPECIAL = re.compile(r"([\\`*_\[\]<>|])")
_WS = re.compile(r"\s+")


def inline(value: Any, limit: int = 300) -> str:
    """One line of plain text: whitespace collapsed, markdown-significant characters escaped."""
    text = _WS.sub(" ", str(value or "")).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "\u2026"
    return _SPECIAL.sub(r"\\\1", text)


def _url(value: Any) -> str | None:
    text = str(value or "").strip()
    if re.match(r"https?://[^\s<>()\[\]]+$", text, re.I):
        return text
    return None


def _when(ts: Any) -> str:
    try:
        stamp = float(ts)
    except (TypeError, ValueError):
        return ""
    if stamp <= 0:
        return ""
    return _dt.datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")


def _link(title: Any, url: Any) -> str:
    safe = _url(url)
    label = inline(title or url, 160) or "source"
    return f"[{label}]({safe})" if safe else label


def export_filename(job_id: str, fmt: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", job_id)[:64] or "job"
    return f"omnibrain-{safe}.{'md' if fmt == 'md' else 'json'}"


def job_markdown(snap: dict[str, Any]) -> str:
    final = snap.get("final") or {}
    lines: list[str] = [f"# {inline(snap.get('question'), 400) or 'Untitled question'}", ""]
    meta = [
        f"mode: {inline(snap.get('mode'))}" if snap.get("mode") else "",
        f"status: {inline(snap.get('status'))}" if snap.get("status") else "",
        f"asked: {_when(snap.get('created_at'))}" if _when(snap.get("created_at")) else "",
        f"rounds: {final.get('rounds_run', snap.get('rounds_run'))}" if final.get("rounds_run", snap.get("rounds_run")) else "",
        f"review: {inline(final.get('reviewer_status'))}/{inline(final.get('synthesis_status'))}"
        if final.get("reviewer_status") or final.get("synthesis_status")
        else "",
    ]
    meta = [m for m in meta if m]
    if meta:
        lines += ["_" + " \u00b7 ".join(meta) + "_", ""]

    answer = (final.get("answer") or snap.get("answer_text") or "").strip()
    lines += ["## Answer", "", _block(answer) if answer else "_No answer was produced._", ""]
    label = final.get("confidence_label") or snap.get("confidence")
    if label:
        lines += [f"**Confidence:** {inline(label)}", ""]
    # "status: completed" alone would make a job whose review was cut off read as a finished one.
    review = final.get("reviewer_status") or ""
    if review not in {"", "COMPLETED", "NOT_RUN"}:
        reason = final.get("fallback_reason") or "the reviewer never finished"
        lines += ["## Review status", "", _block(f"{review}: {reason}"), ""]
    if final.get("why"):
        lines += ["## Why", "", _block(final["why"]), ""]
    if final.get("important_disagreement"):
        lines += ["## Where sources disagree", "", _block(final["important_disagreement"]), ""]

    sources = final.get("sources") or []
    if sources:
        lines += ["## Sources", ""]
        for s in sources:
            extra = " \u2014 " + inline(s.get("published"), 40) if s.get("published") else ""
            lines.append(f"- {_link(s.get('title') or s.get('provider'), s.get('url'))}{extra}")
        lines.append("")

    caveats = final.get("caveats") or []
    if caveats:
        lines += ["## Caveats", ""] + [f"- {inline(c, 600)}" for c in caveats] + [""]

    claims = snap.get("claims") or []
    if claims:
        lines += ["## Claim ledger", "", "| Claim | Verdict | Asserted by |", "|---|---|---|"]
        for c in claims:
            # Confidence.NONE is spelled "insufficient_evidence", the same word as the
            # status, so a bare `insufficient_evidence / insufficient_evidence` says
            # nothing twice over.
            confidence = c.get("confidence")
            verdict = inline(c.get("status")) + (f" / {inline(confidence)}" if confidence and confidence != c.get("status") else "")
            by = ", ".join(inline(p, 40) for p in (c.get("providers") or c.get("providers_json") or []))
            lines.append(f"| {inline(c.get('claim'), 400)} | {verdict} | {by} |")
        lines.append("")

    evidence = snap.get("evidence") or []
    opened = [e for e in evidence if e.get("origin") != "provider_quote"]
    quoted = [e for e in evidence if e.get("origin") == "provider_quote"]
    if opened:
        lines += [f"## Evidence we opened ({len(opened)})", ""]
        for e in opened:
            bits = [inline(e.get("check_status"), 30)]
            if e.get("tier") and e.get("tier") != "unknown":
                bits.append(inline(e["tier"], 30))
            if e.get("polarity") and e.get("polarity") != "support":
                bits.append(inline(e["polarity"], 30))
            if e.get("published"):
                bits.append("published " + inline(e["published"], 30))
            row = f"- {_link(e.get('title') or e.get('domain'), e.get('url'))} ({', '.join(b for b in bits if b)})"
            if e.get("verbatim_excerpt"):
                row += f' \u2014 found on page: "{inline(e["verbatim_excerpt"], 200)}"'
            lines.append(row)
        lines.append("")

    if quoted:
        # Its own heading: we never opened these pages, so the audit trail must not say we did.
        lines += [f"## Quoted by a provider, not opened by us ({len(quoted)})", ""]
        for e in quoted:
            lines.append(
                f"- {_link(e.get('title') or e.get('domain'), e.get('url'))} ({inline(e.get('check_status'), 30)})"
                + (f' \u2014 provider displayed: "{inline(e["verbatim_excerpt"], 200)}"' if e.get("verbatim_excerpt") else "")
            )
        lines.append("")

    conflicts = snap.get("disagreements") or []
    if conflicts:
        lines += ["## Conflicts detected", ""] + [f"- [{inline(d.get('severity'), 20)}] {inline(d.get('description'), 500)}" for d in conflicts] + [""]

    reviews = snap.get("reviews") or {}
    complaints = reviews.get("complaints") if isinstance(reviews, dict) else None
    if complaints:
        lines += ["## Owner reports (community sources, not verified facts)", ""]
        for c in complaints:
            lines.append(f"- {inline(c.get('theme'), 120)} \u2014 {c.get('mentions', 0)} mentions across {c.get('sources', 0)} pages")
        lines.append("")

    responses = snap.get("responses") or []
    if responses:
        lines += ["## Researchers", ""]
        for r in responses:
            note = f" \u2014 {inline(r.get('error'), 160)}" if r.get("error") else ""
            lines.append(f"- {inline(r.get('provider'), 40)}, round {r.get('round')}: {inline(r.get('status'), 30)}{note}")
        lines.append("")

    if snap.get("stop_reason"):
        lines += [f"_Why it stopped: {inline(snap['stop_reason'], 300)}_", ""]
    lines.append("_Exported from OmniBrain. Generated answers are claims to check against the sources above, not facts._")
    return "\n".join(lines).rstrip() + "\n"


_BLOCK_START = re.compile(r"^(#{1,6}\s|[-+>=~]|\d+[.)]\s)")


def _no_block_start(text: str) -> str:
    """Stop a paragraph from being read as a heading, list item, quote or rule."""
    return "\\" + text if _BLOCK_START.match(text) else text


def _block(text: str) -> str:
    """A paragraph of untrusted text: keep paragraph breaks, escape markup, never emit raw HTML or headings."""
    paragraphs = [_no_block_start(inline(p, 4000)) for p in re.split(r"\n\s*\n", str(text).strip())]
    return "\n\n".join(p for p in paragraphs if p)