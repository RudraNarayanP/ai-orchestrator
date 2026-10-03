"""Retrieval: decide whether memory should be used at all, then a hybrid pipeline down to a handful of memories.

query understanding -> candidates (keyword FTS5 + exact vector + entity/project graph) -> dedupe -> temporal filter ->
multi-signal rerank -> a small final set whose size depends on the question. A general-knowledge question
("capital of France") retrieves nothing, however many memories exist.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from backend.memory.embed import content_tokens
from backend.memory.schema import Memory, MemoryType, Status
from backend.memory.store import MemoryStore

PERSONAL_RE = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself|we|our|us)\b", re.I)
ADVICE_RE = re.compile(r"\b(recommend|suggest|should i|what should|best (?:way|option|choice) for|for me|help me|advise|which (?:one )?(?:is|would be) (?:best|better) for)\b", re.I)
WORK_RE = re.compile(r"\b(plan|design|build|implement|write|draft|compare|summari[sz]e|review|debug|refactor|architect|outline|prepare|analy[sz]e|continue|extend|fix)\b", re.I)
HISTORY_RE = re.compile(r"\b(used to|before|previously|earlier|originally|in the past|did i (?:say|tell|mention)|what was my|history of)\b", re.I)
GENERIC_Q_RE = re.compile(r"^\s*(what|who|when|where|how (?:many|much|old|tall|far|long)|which|define|is|are|does|do|did|why)\b", re.I)


@dataclass
class Hit:
    memory: Memory
    score: float
    relevance: float
    signals: dict[str, float] = field(default_factory=dict)
    why: list[str] = field(default_factory=list)
    history: bool = False

    def public(self) -> dict[str, Any]:
        return {"memory_id": self.memory.memory_id, "content": self.memory.content, "memory_type": self.memory.memory_type.value,
                "source": self.memory.source.value, "confidence": self.memory.confidence, "status": self.memory.status.value,
                "project": self.memory.project, "created_at": self.memory.created_at, "updated_at": self.memory.updated_at,
                "score": round(self.score, 3), "why": self.why}


@dataclass
class Retrieval:
    hits: list[Hit] = field(default_factory=list)
    kind: str = "general"  # general | personal | project | complex
    budget: int = 0
    candidates: int = 0
    project: str | None = None
    reason: str = ""
    elapsed_ms: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)


BUDGETS = {"general": 0, "personal": 2, "project": 5, "complex": 8}
HALF_LIFE_DAYS = {MemoryType.EPISODIC: 30.0, MemoryType.GOAL: 120.0, MemoryType.CONVERSATION: 2.0}
W = {"rel": 0.46, "importance": 0.14, "confidence": 0.08, "recency": 0.08, "source": 0.08, "project": 0.10, "graph": 0.06}


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


class Retriever:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    # ------------------------------------------------------------ 1. query understanding
    def understand(self, query: str, project: str | None = None) -> tuple[str, str | None, list[str], bool]:
        q = query or ""
        ql = q.lower()
        projects = self.store.known_projects()
        mentioned = next((p for p in sorted(projects, key=len, reverse=True) if re.search(r"(?<![\w-])" + re.escape(p) + r"(?![\w-])", ql)), None)
        proj = project or mentioned
        ents = self.store.known_entities()
        ent_hits = sorted(e for e in ents if len(e) > 2 and re.search(r"(?<![\w-])" + re.escape(e) + r"(?![\w-])", ql))
        personal = bool(PERSONAL_RE.search(q))
        advice = bool(ADVICE_RE.search(q))
        work = bool(WORK_RE.search(q))
        history = bool(HISTORY_RE.search(q))
        if proj:
            kind = "complex" if (work or len(q.split()) > 25) else "project"
        elif personal and (work or advice) and len(q.split()) > 12:
            kind = "complex"
        elif personal or advice:
            kind = "personal"
        elif ent_hits and (work or advice):
            kind = "project"
        else:
            kind = "general"
        return kind, proj, ent_hits, history

    # ------------------------------------------------------------ the pipeline
    def retrieve(self, query: str, *, project: str | None = None, conversation: str | None = None, now: float | None = None,
                 budget: int | None = None, include_history: bool | None = None, min_relevance: float | None = None) -> Retrieval:
        t0 = time.perf_counter()
        now = now or time.time()
        kind, proj, ent_hits, history = self.understand(query, project)
        res = Retrieval(kind=kind, project=proj)
        res.stages["understand_ms"] = (time.perf_counter() - t0) * 1000
        if not self.store.inject_enabled:
            res.reason = "memory injection is switched off"
            return res
        cap = BUDGETS[kind] if budget is None else budget
        res.budget = cap
        if cap <= 0:
            res.reason = "general question: nothing about the user is relevant"
            res.elapsed_ms = (time.perf_counter() - t0) * 1000
            return res
        include_history = history if include_history is None else include_history
        emb = self.store.embedder
        floor = emb.floor if min_relevance is None else min_relevance
        qterms = list(dict.fromkeys(content_tokens(query)))
        # --- candidates
        t1 = time.perf_counter()
        qvec = emb.embed([query])[0]
        vec = self.store.vector_search(qvec, 60)
        res.stages["vector_ms"] = (time.perf_counter() - t1) * 1000
        t1 = time.perf_counter()
        statuses = ("ACTIVE", "SUPERSEDED") if include_history else ("ACTIVE",)
        kw = self.store.fts_search(qterms, 60, statuses=statuses)
        res.stages["fts_ms"] = (time.perf_counter() - t1) * 1000
        cos: dict[str, float] = {i: s for i, s in vec}
        bm: dict[str, float] = {i: s for i, s, _ in kw}
        bmax = max(bm.values(), default=1.0) or 1.0
        t1 = time.perf_counter()
        ent_map = self.store.entity_memories(ent_hits) if ent_hits else {}
        graph_ids: set[str] = set().union(*ent_map.values()) if ent_map else set()
        ids = set(cos) | set(bm) | graph_ids
        if proj:  # the project namespace is always a candidate source
            ids |= {m.memory_id for m in self.store.list(project=proj, status="ACTIVE", limit=40)}
        res.candidates = len(ids)
        mems = self.store.get_many(ids)
        missing = [i for i in ids if i not in cos and i in mems]
        if missing:
            for i, s in self.store.vector_search(qvec, len(missing), allow=set(missing)):
                cos[i] = s
        res.stages["hydrate_ms"] = (time.perf_counter() - t1) * 1000
        # --- vector / keyword ranks (for RRF)
        v_rank = {i: r for r, (i, _) in enumerate(sorted(((i, cos.get(i, -1)) for i in ids), key=lambda x: -x[1]))}
        k_rank = {i: r for r, i in enumerate(sorted(bm, key=lambda x: -bm[x]))}
        # --- score
        t1 = time.perf_counter()
        hits: list[Hit] = []
        for mid in ids:
            m = mems.get(mid)
            if m is None:
                continue
            # temporal / lifecycle filters
            if m.memory_type == MemoryType.CONVERSATION and (not conversation or m.source_conversation != conversation):
                continue
            is_history = m.status == Status.SUPERSEDED
            if m.status not in (Status.ACTIVE,) and not (include_history and is_history):
                continue
            if m.memory_type == MemoryType.GOAL and m.goal_active is False:
                continue
            exp = m.metadata.get("expires_at")
            if exp and exp < now:
                continue
            if m.scope == "project" and m.project and proj and m.project.lower() != proj.lower():
                continue  # project isolation: another project's memory never leaks in
            if m.scope == "project" and m.project and not proj:
                continue  # a project-scoped memory is only for that project's questions
            c = cos.get(mid, 0.0)
            vec_rel = max(0.0, (c - floor) / (1.0 - floor)) if c >= floor else 0.0
            matched = sum(1 for t in qterms if t in set(content_tokens(m.content + " " + " ".join(m.entities) + " " + " ".join(m.topics))))
            kw_rel = 0.0
            if mid in bm:
                coverage = matched / max(1, min(3, len(qterms)))
                kw_rel = min(1.0, coverage) * (0.55 + 0.45 * (bm[mid] / bmax))
                if matched < 2 and not any(mid in s for s in ent_map.values()):
                    kw_rel *= 0.45  # a single shared common word is not relevance
            ent_hit = any(mid in s for s in ent_map.values())
            ent_rel = 0.7 if ent_hit else 0.0
            rel = max(vec_rel, kw_rel, ent_rel)
            fused = _rrf(v_rank.get(mid, 999)) + (_rrf(k_rank[mid]) if mid in k_rank else 0.0)
            in_proj = bool(proj and m.project and m.project.lower() == proj.lower())
            if rel < 0.18 and not in_proj:
                continue
            if rel < 0.30 and in_proj and kind != "complex" and c < floor:
                continue
            age_days = max(0.0, (now - max(m.updated_at, m.created_at)) / 86400.0)
            hl = HALF_LIFE_DAYS.get(m.memory_type)
            recency = 0.5 ** (age_days / hl) if hl else 1.0 / (1.0 + age_days / 730.0)
            sig = {
                "rel": rel, "importance": m.importance, "confidence": m.confidence, "recency": recency,
                "source": 1.0 if m.source.value == "user_explicit" else 0.6, "project": 1.0 if in_proj else 0.0,
                "graph": 1.0 if (ent_hit and mid not in bm and c < floor) else 0.0,
            }
            score = sum(W[k] * v for k, v in sig.items())
            if is_history:
                score *= 0.8
            why = []
            if c >= floor:
                why.append(f"meaning match {c:.2f}")
            if mid in bm and matched:
                why.append(f"{matched} keyword(s) match")
            if ent_hit:
                why.append("mentions " + ", ".join(sorted({e for e, s in ent_map.items() if mid in s})[:3]))
            if in_proj:
                why.append(f"project {m.project}")
            why.append("you said this" if m.source.value == "user_explicit" else "inferred from earlier chats")
            if is_history:
                why.append("earlier version (history asked)")
            hits.append(Hit(m, score, rel, sig, why, history=is_history))
        res.stages["score_ms"] = (time.perf_counter() - t1) * 1000
        hits.sort(key=lambda h: -h.score)
        # --- dedupe near-identical memories, then cut to the budget
        out: list[Hit] = []
        for h in hits:
            if any(h.memory.slot and h.memory.slot == o.memory.slot and not h.history == o.history for o in out) and False:
                continue
            if any(h.memory.content.lower() == o.memory.content.lower() for o in out):
                continue
            if out:
                hv = self.store.embedder.embed([h.memory.content])[0] if False else None
            out.append(h)
            if len(out) >= cap:
                break
        # an inferred memory never rides in beside an explicit one about the same slot
        explicit_slots = {h.memory.slot for h in out if h.memory.source.value == "user_explicit" and h.memory.slot}
        out = [h for h in out if not (h.memory.source.value == "model_inferred" and h.memory.slot in explicit_slots)]
        res.hits = out
        if out:
            self.store.touch([h.memory.memory_id for h in out], now)
        else:
            res.reason = "nothing stored is relevant to this question"
        res.elapsed_ms = (time.perf_counter() - t0) * 1000
        return res