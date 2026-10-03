"""MemoryService: what the research pipeline talks to. Learn from the USER's message, offer relevant context to prompts.

The context block is the ONLY way memory reaches a provider, and it never touches claims, evidence, verification or
citations (see tests/test_memory_isolation.py)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from backend.memory.extract import Candidate, extract
from backend.memory.retrieve import Retrieval, Retriever
from backend.memory.schema import Memory, MemoryType, Source, Status
from backend.memory.store import AddResult, MemoryStore

CONTEXT_HEADER = "Relevant context about the user:"
CONTEXT_FOOTER = (
    "Use when relevant. Don't mention it was supplied. Don't assume irrelevant memories apply. "
    "The user's current instructions override this context."
)


def format_context(hits: list[Any]) -> str:
    """The same block for every provider. Empty when nothing is relevant (then nothing is injected at all)."""
    if not hits:
        return ""
    lines = []
    for h in hits:
        m = h.memory if hasattr(h, "memory") else h
        suffix = " (earlier, since changed)" if getattr(h, "history", False) else ""
        lines.append(f"- {m.content.rstrip('.')}{suffix}")
    return CONTEXT_HEADER + "\n" + "\n".join(lines) + "\n" + CONTEXT_FOOTER


def approx_tokens(text: str) -> int:
    return max(0, int(len(text) / 4))


@dataclass
class Learned:
    added: list[AddResult] = field(default_factory=list)
    forgotten: list[str] = field(default_factory=list)
    skipped: str = ""


class MemoryService:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.retriever = Retriever(store)

    # --------------------------------------------------------------- read side
    def context_for(self, question: str, *, project: str | None = None, conversation: str | None = None) -> tuple[str, Retrieval]:
        res = self.retriever.retrieve(question, project=project, conversation=conversation)
        return format_context(res.hits), res

    # --------------------------------------------------------------- write side
    def learn(self, message: str, *, project: str | None = None, conversation: str | None = None) -> Learned:
        out = Learned()
        if not self.store.capture_enabled:
            out.skipped = "capture is switched off"
            return out
        ex = extract(message, project=project, conversation=conversation)
        out.skipped = ex.skipped
        for what in ex.forget:
            out.forgotten += self.forget_matching(what)
        for c in ex.candidates:
            if c.deactivate_goal is not None:
                if self._finish_goal(c.deactivate_goal):
                    continue
                if not c.content:
                    continue
                c = Candidate(content=c.content, memory_type=MemoryType.EPISODIC, slot=None, source=Source.MODEL_INFERRED, score=c.score, project=c.project)
            res = self.store.add(
                c.content, memory_type=c.memory_type, source=c.source, project=c.project, entities=c.entities, slot=c.slot,
                source_conversation=conversation, goal_active=c.goal_active, importance=0.8 if c.source == Source.USER_EXPLICIT else None,
                metadata={"extracted_score": round(c.score, 2), "correction": c.correction},
            )
            out.added.append(res)
        return out

    def remember(self, content: str, **kw: Any) -> AddResult:
        """The user typed this into the memory panel: explicit by definition."""
        kw.setdefault("source", Source.USER_EXPLICIT)
        return self.store.add(content, **kw)

    def forget_matching(self, text: str, limit: int = 3) -> list[str]:
        """'Forget that I live in Kyiv': delete the best-matching memories (physically)."""
        from backend.memory.embed import content_tokens

        terms = content_tokens(text)
        if not terms:
            return []
        found = self.store.fts_search(terms, 10, statuses=("ACTIVE", "SUPERSEDED", "ARCHIVED"))
        need = max(1, min(2, len(terms)))
        gone = []
        for mid, _s, _ in found[:limit]:
            m = self.store.get(mid)
            if m and sum(1 for t in terms if t in set(content_tokens(m.content))) >= need and self.store.delete(mid):
                gone.append(mid)
        return gone

    def _finish_goal(self, what: str) -> bool:
        from backend.memory.embed import content_tokens

        terms = content_tokens(what)
        if not terms:
            return False
        for mid, _s, _ in self.store.fts_search(terms, 10):
            m = self.store.get(mid)
            if m and m.memory_type == MemoryType.GOAL and m.status == Status.ACTIVE and sum(1 for t in terms if t in set(content_tokens(m.content))) >= 1:
                self.store.update(mid, goal_active=False, status=Status.ARCHIVED, by_user=False)
                return True
        return False

    # --------------------------------------------------------------- conversation memory (kept apart from long-term)
    def note_turn(self, conversation: str, text: str) -> AddResult:
        return self.store.add(text[:400], memory_type=MemoryType.CONVERSATION, source=Source.MODEL_INFERRED, source_conversation=conversation,
                              importance=0.3, metadata={"expires_at": time.time() + 2 * 86400})

    def prune_conversation_memory(self, older_than_s: float = 2 * 86400) -> int:
        rows = self.store.db.execute("select memory_id from memories where memory_type='conversation' and created_at < ?", (time.time() - older_than_s,)).fetchall()
        return sum(1 for r in rows if self.store.delete(r["memory_id"]))