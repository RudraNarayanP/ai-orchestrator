"""The memory store: SQLite is the source of truth (rows + FTS5 + entity/link tables); embeddings live in the row as
float32 BLOBs and are searched exactly through an in-RAM numpy matrix (measured: data/MEMORY_DESIGN.md).

Invariants this class enforces (and `check_invariants` verifies):
  * at most one ACTIVE memory per (slot, scope, project) -- a correction supersedes, it never sits beside the old one;
  * model_inferred never overrides user_explicit;
  * DELETE is physical: row, FTS entry, entities and links are gone (only a content-free audit line stays);
  * CONVERSATION memories never leak into long-term retrieval.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backend.memory.embed import Embedder, HashEmbedder, content_tokens
from backend.memory.topics import is_interpretation, is_sensitive, topics_for, view_text
from backend.memory.schema import Memory, MemoryType, Source, Status

DDL = """
create table if not exists memories (
  memory_id text primary key, content text not null, memory_type text not null, scope text not null default 'global',
  project text, entities text not null default '[]', topics text not null default '[]', source_conversation text,
  created_at real not null, updated_at real not null, last_accessed_at real, access_count integer not null default 0,
  importance real not null default 0.5, confidence real not null default 0.8, status text not null default 'ACTIVE',
  supersedes text, superseded_by text, embedding blob, metadata text not null default '{}', source text not null,
  slot text, valid_from real, valid_to real, goal_active integer, sensitivity text not null default 'normal'
);
create index if not exists ix_mem_status on memories(status, memory_type);
create index if not exists ix_mem_project on memories(project, status);
create index if not exists ix_mem_slot on memories(slot, scope, project, status);
create table if not exists memory_fts_map (memory_id text primary key, rid integer not null);
create virtual table if not exists memory_fts using fts5(memory_id unindexed, content, entities, topics, tokenize='porter unicode61');
create virtual table if not exists memory_fts_vocab using fts5vocab(memory_fts, 'row');
create table if not exists memory_entities (memory_id text not null, entity text not null, primary key (memory_id, entity));
create index if not exists ix_ent on memory_entities(entity);
create table if not exists memory_topics (memory_id text not null, topic text not null, primary key (memory_id, topic));
create index if not exists ix_topic on memory_topics(topic);
create table if not exists memory_links (src text not null, dst text not null, kind text not null, weight real default 1.0, primary key (src, dst, kind));
create table if not exists memory_events (id integer primary key autoincrement, memory_id text, action text not null, at real not null, detail text);
create table if not exists memory_settings (key text primary key, value text not null);
"""

COLS = ("memory_id content memory_type scope project entities topics source_conversation created_at updated_at last_accessed_at access_count "
        "importance confidence status supersedes superseded_by embedding metadata source slot valid_from valid_to goal_active sensitivity").split()


def norm_text(text: str) -> str:
    return " ".join(content_tokens(text))


@dataclass
class AddResult:
    action: str  # created | merged | superseded | rejected
    memory: Memory | None = None
    superseded: Memory | None = None
    reason: str = ""
    notes: list[str] = field(default_factory=list)


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:", embedder: Embedder | None = None) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.embedder: Embedder = embedder or HashEmbedder()
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(DDL)
        if "sensitivity" not in {r[1] for r in self.db.execute("pragma table_info(memories)")}:
            self.db.execute("alter table memories add column sensitivity text not null default 'normal'")
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._row: dict[str, int] = {}
        self._live: np.ndarray | None = None  # bool mask: row is a searchable (ACTIVE or history) memory
        self._loaded = False
        self._gen, self._proj_gen, self._projects = 0, -1, set()
        self._ndocs, self._ndocs_gen = 0, -1
        if self.db.execute("select count(*) from memory_fts_map").fetchone()[0] == 0 and self.count() > 0:
            self.db.execute("delete from memory_fts")
            for row in self.db.execute("select * from memories").fetchall():
                self._fts_put(self._from_row(row), replace=False)
            self.db.commit()
        if self.db.execute("select count(*) from memory_topics").fetchone()[0] == 0 and self.count() > 0:
            for row in self.db.execute("select * from memories").fetchall():
                mm = self._from_row(row)
                mm.topics = sorted(set(mm.topics) | set(topics_for(mm.content)))
                if is_sensitive(mm.topics) and mm.sensitivity == "normal":
                    mm.sensitivity = "sensitive"
                self._write(mm, insert=False)
            self.db.commit()
        model = self.get_setting("embed_model")
        if model and model != self.embedder.name:
            self.reindex()
        elif not model:
            self.set_setting("embed_model", self.embedder.name)

    # ---------------------------------------------------------------- settings
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        r = self.db.execute("select value from memory_settings where key=?", (key,)).fetchone()
        return r["value"] if r else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self.db.execute("insert into memory_settings(key,value) values (?,?) on conflict(key) do update set value=excluded.value", (key, str(value)))
            self.db.commit()

    @property
    def inject_enabled(self) -> bool:
        return self.get_setting("inject", "1") == "1"

    @property
    def capture_enabled(self) -> bool:
        return self.get_setting("capture", "1") == "1"

    # ---------------------------------------------------------------- row <-> model
    @staticmethod
    def _from_row(r: sqlite3.Row) -> Memory:
        d = dict(r)
        d["entities"] = json.loads(d["entities"] or "[]")
        d["topics"] = json.loads(d["topics"] or "[]")
        d["metadata"] = json.loads(d["metadata"] or "{}")
        d["goal_active"] = None if d["goal_active"] is None else bool(d["goal_active"])
        return Memory.model_validate(d)

    def _event(self, memory_id: str | None, action: str, detail: str = "") -> None:
        self.db.execute("insert into memory_events(memory_id, action, at, detail) values (?,?,?,?)", (memory_id, action, time.time(), detail[:300]))

    def _write(self, m: Memory, *, insert: bool) -> None:
        self._gen += 1
        vals = (
            m.memory_id, m.content, m.memory_type.value, m.scope, m.project, json.dumps(m.entities), json.dumps(m.topics), m.source_conversation,
            m.created_at, m.updated_at, m.last_accessed_at, m.access_count, m.importance, m.confidence, m.status.value, m.supersedes,
            m.superseded_by, m.embedding, json.dumps(m.metadata), m.source.value, m.slot, m.valid_from, m.valid_to,
            None if m.goal_active is None else int(m.goal_active), m.sensitivity,
        )
        if insert:
            self.db.execute(f"insert into memories({','.join(COLS)}) values ({','.join('?' * len(COLS))})", vals)
        else:
            self.db.execute(f"update memories set {','.join(c + '=?' for c in COLS[1:])} where memory_id=?", vals[1:] + (m.memory_id,))
        self._fts_put(m, replace=not insert)
        self.db.execute("delete from memory_topics where memory_id=?", (m.memory_id,))
        for tp in set(m.topics):
            self.db.execute("insert or ignore into memory_topics(memory_id, topic) values (?,?)", (m.memory_id, tp))
        self.db.execute("delete from memory_entities where memory_id=?", (m.memory_id,))
        for e in {x.lower() for x in m.entities}:
            self.db.execute("insert or ignore into memory_entities(memory_id, entity) values (?,?)", (m.memory_id, e))

    def _fts_put(self, m: Memory, *, replace: bool) -> None:
        """Keyword index entry. Deleting by the unindexed memory_id column would scan the whole FTS table (quadratic
        bulk loads), so every entry's rowid is remembered in memory_fts_map and removed by rowid."""
        if replace:
            self._fts_drop(m.memory_id)
        cur = self.db.execute("insert into memory_fts(memory_id, content, entities, topics) values (?,?,?,?)", (m.memory_id, m.content, " ".join(m.entities), " ".join(m.topics)))
        self.db.execute("insert or replace into memory_fts_map(memory_id, rid) values (?,?)", (m.memory_id, cur.lastrowid))

    def _fts_drop(self, memory_id: str) -> None:
        r = self.db.execute("select rid from memory_fts_map where memory_id=?", (memory_id,)).fetchone()
        if r:
            self.db.execute("delete from memory_fts where rowid=?", (r["rid"],))
            self.db.execute("delete from memory_fts_map where memory_id=?", (memory_id,))

    # ---------------------------------------------------------------- vector cache
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            rows = self.db.execute("select memory_id, embedding, status, memory_type from memories where embedding is not null").fetchall()
            dim = self.embedder.dim
            n = len(rows)
            cap = max(1024, int(n * 1.25))
            self._mat = np.zeros((cap, dim), dtype=np.float32)
            self._live = np.zeros(cap, dtype=bool)
            self._ids = []
            self._row = {}
            for i, r in enumerate(rows):
                self._mat[i] = np.frombuffer(r["embedding"], dtype=np.float32)
                self._ids.append(r["memory_id"])
                self._row[r["memory_id"]] = i
                self._live[i] = r["status"] in ("ACTIVE", "SUPERSEDED")
            self._loaded = True

    def _cache_put(self, mid: str, vec: np.ndarray, live: bool) -> None:
        if not self._loaded:
            return
        i = self._row.get(mid)
        if i is None:
            i = len(self._ids)
            if i >= self._mat.shape[0]:
                new = np.zeros((self._mat.shape[0] * 2, self._mat.shape[1]), dtype=np.float32)
                new[:i] = self._mat[:i]
                self._mat = new
                live_new = np.zeros(self._mat.shape[0], dtype=bool)
                live_new[:i] = self._live[:i]
                self._live = live_new
            self._ids.append(mid)
            self._row[mid] = i
        self._mat[i] = vec
        self._live[i] = live

    def _cache_drop(self, mid: str) -> None:
        if self._loaded and mid in self._row:
            self._live[self._row[mid]] = False

    def vector_search(self, qvec: np.ndarray, k: int = 50, *, allow: set[str] | None = None) -> list[tuple[str, float]]:
        """Exact cosine top-k over live rows (optionally restricted to a candidate id set)."""
        self._ensure_loaded()
        n = len(self._ids)
        if n == 0:
            return []
        if allow is not None:
            idx = np.array([self._row[i] for i in allow if i in self._row], dtype=np.int64)  # lifecycle is judged by the caller (history lookups need superseded rows)
            if idx.size == 0:
                return []
            sims = self._mat[idx] @ qvec
            top = np.argsort(-sims)[:k]
            return [(self._ids[idx[t]], float(sims[t])) for t in top]
        sims = self._mat[:n] @ qvec
        sims = np.where(self._live[:n], sims, -2.0)
        k = min(k, n)
        part = np.argpartition(-sims, k - 1)[:k]
        part = part[np.argsort(-sims[part])]
        return [(self._ids[i], float(sims[i])) for i in part if sims[i] > -1.5]

    # ---------------------------------------------------------------- reads
    def get(self, memory_id: str) -> Memory | None:
        r = self.db.execute("select * from memories where memory_id=?", (memory_id,)).fetchone()
        return self._from_row(r) if r else None

    def get_many(self, ids: Iterable[str]) -> dict[str, Memory]:
        ids = list(ids)
        out: dict[str, Memory] = {}
        for s in range(0, len(ids), 500):
            chunk = ids[s : s + 500]
            for r in self.db.execute(f"select * from memories where memory_id in ({','.join('?' * len(chunk))})", chunk):
                out[r["memory_id"]] = self._from_row(r)
        return out

    def list(self, *, status: str | None = None, memory_type: str | None = None, source: str | None = None, project: str | None = None,
             q: str | None = None, limit: int = 100, offset: int = 0, sensitivity: str | None = None) -> list[Memory]:
        where, args = [], []
        for col, v in (("status", status), ("memory_type", memory_type), ("source", source), ("project", project), ("sensitivity", sensitivity)):
            if v:
                where.append(f"{col}=?")
                args.append(v)
        if q:
            where.append("content like ?")
            args.append(f"%{q}%")
        sql = "select * from memories" + (" where " + " and ".join(where) if where else "") + " order by updated_at desc limit ? offset ?"
        return [self._from_row(r) for r in self.db.execute(sql, (*args, limit, offset))]

    def count(self, **kw: str) -> int:
        where = " and ".join(f"{k}=?" for k in kw)
        return self.db.execute("select count(*) c from memories" + (" where " + where if where else ""), tuple(kw.values())).fetchone()["c"]

    def entity_memories(self, entities: Iterable[str]) -> dict[str, set[str]]:
        ents = [e.lower() for e in entities]
        out: dict[str, set[str]] = {}
        for s in range(0, len(ents), 500):
            chunk = ents[s : s + 500]
            for r in self.db.execute(f"select entity, memory_id from memory_entities where entity in ({','.join('?' * len(chunk))})", chunk):
                out.setdefault(r["entity"], set()).add(r["memory_id"])
        return out

    def topic_memories(self, topics: Iterable[str], limit: int = 300) -> list[Memory]:
        """ACTIVE memories carrying any of these topic tags (indexed), most important first - the candidates of a context cluster."""
        ts = sorted(set(topics))
        if not ts:
            return []
        rows = self.db.execute(
            f"select m.* from memories m where m.status='ACTIVE' and m.memory_id in (select memory_id from memory_topics where topic in ({','.join('?' * len(ts))})) "
            "order by m.importance desc, m.updated_at desc limit ?", (*ts, limit)).fetchall()
        return [self._from_row(r) for r in rows]

    @property
    def sensitive_enabled(self) -> bool:
        return self.get_setting("sensitive", "1") == "1"

    def match_entities(self, candidates: Iterable[str]) -> set[str]:
        """Which of these strings are stored entities? Indexed lookup - cost does not grow with the number of memories."""
        cands = sorted({c.lower() for c in candidates if c})
        found: set[str] = set()
        for s in range(0, len(cands), 500):
            chunk = cands[s : s + 500]
            found |= {r["entity"] for r in self.db.execute(f"select distinct entity from memory_entities where entity in ({','.join('?' * len(chunk))})", chunk)}
        return found

    def known_entities(self, kind: str | None = None) -> set[str]:
        return {r["entity"] for r in self.db.execute("select distinct entity from memory_entities")}

    def known_projects(self) -> set[str]:
        """Distinct project names (cached; invalidated by any write - the set is tiny, the scan is not)."""
        if self._proj_gen != self._gen:
            self._projects = {r["project"].lower() for r in self.db.execute("select distinct project from memories where project is not null and status='ACTIVE'")}
            self._proj_gen = self._gen
        return self._projects

    COMMON_DF = 0.03  # a term in more than 3% of a large store carries no signal and makes OR-queries scan most of the index
    COMMON_MIN_DOCS = 20000

    def _drop_ubiquitous(self, terms: list[str]) -> list[str]:
        """On big stores skip query terms that appear in a large share of memories (never if that would leave nothing)."""
        if self._ndocs_gen != self._gen:
            self._ndocs = self.db.execute("select count(*) from memory_fts_map").fetchone()[0]
            self._ndocs_gen = self._gen
        if self._ndocs < self.COMMON_MIN_DOCS:
            return terms
        keep = []
        for t in terms:
            r = self.db.execute("select doc from memory_fts_vocab where term=?", (t,)).fetchone()
            if r is None or r["doc"] <= self._ndocs * self.COMMON_DF:
                keep.append(t)
        return keep or terms

    def fts_search(self, query_terms: list[str], k: int = 50, *, statuses: tuple[str, ...] = ("ACTIVE",)) -> list[tuple[str, float, int]]:
        """BM25 over content+entities+topics. Returns (id, score>0, matched_term_count)."""
        terms = [t for t in dict.fromkeys(query_terms) if t and re.fullmatch(r"[\w']+", t)]
        if not terms:
            return []
        terms = self._drop_ubiquitous(terms)
        match = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self.db.execute(
                "select f.memory_id, bm25(memory_fts) b, m.status from memory_fts f join memories m on m.memory_id=f.memory_id "
                "where memory_fts match ? order by b limit ?", (match, k * 3 if len(statuses) else k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        for r in rows:
            if r["status"] in statuses:
                out.append((r["memory_id"], -float(r["b"]), 0))
            if len(out) >= k:
                break
        return out

    # ---------------------------------------------------------------- writes
    def add(self, content: str, *, memory_type: MemoryType | str = MemoryType.FACT, source: Source | str = Source.USER_EXPLICIT,
            scope: str | None = None, project: str | None = None, entities: list[str] | None = None, topics: list[str] | None = None,
            importance: float | None = None, confidence: float | None = None, source_conversation: str | None = None,
            metadata: dict[str, Any] | None = None, slot: str | None = None, supersedes: str | None = None,
            goal_active: bool | None = None, now: float | None = None, embed: np.ndarray | None = None, sensitivity: str | None = None) -> AddResult:
        content = " ".join((content or "").split())
        if not content:
            return AddResult("rejected", reason="empty")
        mtype = MemoryType(memory_type)
        src = Source(source)
        now = now or time.time()
        notes: list[str] = []
        if mtype != MemoryType.INTERPRETATION and mtype != MemoryType.CONVERSATION and is_interpretation(content):
            mtype = MemoryType.INTERPRETATION  # a feeling or a reading of motives is never stored as an objective fact
            slot = None
            supersedes = None
            notes.append("stored as the user's view, not as a fact")
        if mtype == MemoryType.INTERPRETATION:
            content = view_text(content)
        tags = sorted(set(topics or []) | set(topics_for(content)))
        if sensitivity is None:
            sensitivity = "sensitive" if (is_sensitive(tags) or mtype == MemoryType.INTERPRETATION) else "normal"
        if scope is None:
            scope = "conversation" if mtype == MemoryType.CONVERSATION else ("project" if project else "global")
        if mtype == MemoryType.GOAL and goal_active is None:
            goal_active = True
        vec = embed if embed is not None else self.embedder.embed([content])[0]
        m = Memory(
            content=content, memory_type=mtype, scope=scope, project=project, entities=sorted({e.strip() for e in (entities or []) if e.strip()}),
            topics=tags, sensitivity=sensitivity, source_conversation=source_conversation, created_at=now, updated_at=now,
            importance=importance if importance is not None else (0.8 if src == Source.USER_EXPLICIT else 0.4),
            confidence=confidence if confidence is not None else (0.95 if src == Source.USER_EXPLICIT else 0.6),
            source=src, slot=slot, valid_from=now, goal_active=goal_active, metadata=metadata or {}, embedding=vec.astype(np.float32).tobytes(),
        )
        with self._lock:
            # 1. an explicit correction names what it replaces
            target = self.get(supersedes) if supersedes else None
            # 2. the same slot already holds a different ACTIVE memory
            if target is None and slot and mtype != MemoryType.CONVERSATION:
                r = self.db.execute(
                    "select * from memories where slot=? and scope=? and coalesce(project,'')=coalesce(?, '') and status='ACTIVE' limit 1", (slot, scope, project)
                ).fetchone()
                target = self._from_row(r) if r else None
            # 3. same text / near-duplicate -> merge, do not store twice
            dup = self._find_duplicate(m, vec) if mtype != MemoryType.CONVERSATION else None
            if dup is not None and (target is None or dup.memory_id == target.memory_id):
                return self._merge(dup, m, vec)
            if target is not None:
                if target.source == Source.USER_EXPLICIT and src == Source.MODEL_INFERRED:
                    self._event(None, "rejected_inferred", f"conflicts with explicit {target.memory_id}")
                    self.db.commit()
                    return AddResult("rejected", None, target, "inferred memory may not override an explicit one")
                m.supersedes = target.memory_id
                target.status = Status.SUPERSEDED
                target.superseded_by = m.memory_id
                target.valid_to = now
                target.updated_at = now
                if target.memory_type == MemoryType.GOAL:
                    target.goal_active = False
                self._write(target, insert=False)
                self._cache_put(target.memory_id, np.frombuffer(target.embedding, dtype=np.float32), True)
                self.db.execute("insert or ignore into memory_links(src,dst,kind) values (?,?,?)", (m.memory_id, target.memory_id, "supersedes"))
                self._event(target.memory_id, "superseded", f"by {m.memory_id}")
            self._write(m, insert=True)
            self._cache_put(m.memory_id, vec, True)
            self._link_entities(m)
            self._event(m.memory_id, "created", f"{src.value}/{mtype.value}")
            self.db.commit()
        return AddResult("superseded" if target is not None else "created", m, target, notes=notes)

    def _find_duplicate(self, m: Memory, vec: np.ndarray) -> Memory | None:
        r = self.db.execute("select * from memories where status='ACTIVE' and memory_type=? and scope=? and coalesce(project,'')=coalesce(?, '') and lower(content)=lower(?) limit 1",
                            (m.memory_type.value, m.scope, m.project, m.content)).fetchone()
        if r:
            return self._from_row(r)
        if not self._loaded and self.count() > 20000:
            self._ensure_loaded()
        if self._loaded or self.count() <= 20000:
            self._ensure_loaded()
            for mid, sim in self.vector_search(vec, 3):
                if sim >= self.embedder.dup:
                    o = self.get(mid)
                    if o and o.status == Status.ACTIVE and o.memory_type == m.memory_type and o.scope == m.scope and (o.project or "") == (m.project or "") and (o.slot or None) == (m.slot or None):
                        return o
        return None

    def _merge(self, old: Memory, new: Memory, vec: np.ndarray) -> AddResult:
        old.updated_at = new.updated_at
        old.confidence = min(1.0, max(old.confidence, new.confidence) + 0.02)
        old.importance = min(1.0, max(old.importance, new.importance) + 0.02)
        if new.source == Source.USER_EXPLICIT and old.source != Source.USER_EXPLICIT:
            old.source = Source.USER_EXPLICIT
            old.confidence = max(old.confidence, 0.95)
        old.entities = sorted(set(old.entities) | set(new.entities))
        old.topics = sorted(set(old.topics) | set(new.topics))
        self._write(old, insert=False)
        self._event(old.memory_id, "merged", "duplicate said again")
        self.db.commit()
        return AddResult("merged", old, None, "same memory said again")

    def consolidate(self, threshold: float | None = None) -> int:
        """Merge near-duplicate ACTIVE memories (same type, scope, project; cosine >= the embedder's duplicate threshold).

        The stronger one survives (explicit beats inferred, then importance, then newer); the weaker one is deleted
        after its entities/topics/usage are folded in. Different slots are never merged (a slot is a fact about one thing).
        """
        thr = self.embedder.dup if threshold is None else threshold
        merged = 0
        with self._lock:
            groups: dict[tuple, list[Memory]] = {}
            for m in self.list(status="ACTIVE", limit=100000):
                groups.setdefault((m.memory_type, m.scope, m.project or "", m.slot or ""), []).append(m)
            for items in groups.values():
                if len(items) < 2:
                    continue
                self._ensure_loaded()
                vecs = np.stack([self._mat[self._row[m.memory_id]] for m in items])
                sims = vecs @ vecs.T
                dead: set[str] = set()
                order = sorted(range(len(items)), key=lambda i: (items[i].source != Source.USER_EXPLICIT, -items[i].importance, -items[i].updated_at))
                for a in order:
                    if items[a].memory_id in dead:
                        continue
                    for b in order:
                        if b == a or items[b].memory_id in dead or items[a].memory_id in dead or sims[a, b] < thr:
                            continue
                        if order.index(b) < order.index(a):
                            continue
                        keep, drop = items[a], items[b]
                        keep.entities = sorted(set(keep.entities) | set(drop.entities))
                        keep.topics = sorted(set(keep.topics) | set(drop.topics))
                        keep.access_count += drop.access_count
                        keep.confidence = min(1.0, max(keep.confidence, drop.confidence) + 0.02)
                        self._write(keep, insert=False)
                        self._event(keep.memory_id, "consolidated", "absorbed a near-duplicate")
                        self.db.commit()
                        self.delete(drop.memory_id)
                        dead.add(drop.memory_id)
                        merged += 1
        return merged

    def _link_entities(self, m: Memory) -> None:
        if not m.entities:
            return
        near = self.entity_memories(m.entities)
        seen: set[str] = set()
        for ids in near.values():
            for o in list(ids)[:25]:
                if o != m.memory_id and o not in seen:
                    seen.add(o)
                    self.db.execute("insert or ignore into memory_links(src,dst,kind,weight) values (?,?,?,?)", (m.memory_id, o, "related", 0.5))
                    self.db.execute("insert or ignore into memory_links(src,dst,kind,weight) values (?,?,?,?)", (o, m.memory_id, "related", 0.5))

    def update(self, memory_id: str, *, content: str | None = None, importance: float | None = None, confidence: float | None = None,
               status: Status | str | None = None, project: str | None = None, entities: list[str] | None = None, topics: list[str] | None = None,
               goal_active: bool | None = None, expires_at: float | None = None, sensitivity: str | None = None, by_user: bool = True) -> Memory | None:
        """Edit in place (the user fixing their own memory). Editing is explicit by definition."""
        with self._lock:
            m = self.get(memory_id)
            if m is None:
                return None
            if content is not None and " ".join(content.split()) != m.content:
                m.content = view_text(" ".join(content.split())) if m.memory_type == MemoryType.INTERPRETATION else " ".join(content.split())
                m.topics = sorted(set(m.topics) | set(topics_for(m.content)))
                if is_sensitive(m.topics):
                    m.sensitivity = "sensitive"
                vec = self.embedder.embed([m.content])[0]
                m.embedding = vec.astype(np.float32).tobytes()
                self._cache_put(m.memory_id, vec, m.status in (Status.ACTIVE, Status.SUPERSEDED))
            if importance is not None:
                m.importance = float(min(1, max(0, importance)))
            if confidence is not None:
                m.confidence = float(min(1, max(0, confidence)))
            if project is not None:
                m.project = project or None
                m.scope = "project" if project else "global"
            if entities is not None:
                m.entities = sorted(set(entities))
            if topics is not None:
                m.topics = sorted(set(topics))
            if goal_active is not None:
                m.goal_active = goal_active
            if sensitivity in ("normal", "sensitive"):
                m.sensitivity = sensitivity
            if expires_at is not None:
                m.metadata = {**m.metadata, "expires_at": expires_at}
            if status is not None:
                st = Status(status)
                if st == Status.DELETED:
                    return self.delete(memory_id) and None
                if st == Status.ACTIVE and m.slot:  # re-activating must not create two ACTIVE in a slot
                    self.db.execute("update memories set status='SUPERSEDED', valid_to=? where slot=? and scope=? and coalesce(project,'')=coalesce(?,'') and status='ACTIVE' and memory_id<>?",
                                    (time.time(), m.slot, m.scope, m.project, m.memory_id))
                m.status = st
                self._cache_put(m.memory_id, np.frombuffer(m.embedding, dtype=np.float32), st in (Status.ACTIVE, Status.SUPERSEDED))
            if by_user:
                m.source = Source.USER_EXPLICIT
            m.updated_at = time.time()
            self._write(m, insert=False)
            self._event(memory_id, "edited", "by user" if by_user else "")
            self.db.commit()
            return m

    def delete(self, memory_id: str) -> bool:
        """Physically remove the memory everywhere. Only a content-free audit line remains."""
        with self._lock:
            if not self.db.execute("select 1 from memories where memory_id=?", (memory_id,)).fetchone():
                return False
            self._gen += 1
            # what this one had replaced is not resurrected (the user may have deleted it on purpose) -- it is archived
            for r in self.db.execute("select memory_id from memories where superseded_by=?", (memory_id,)).fetchall():
                self._cache_drop(r["memory_id"])
            self.db.execute("update memories set status='ARCHIVED' where superseded_by=? and status='SUPERSEDED'", (memory_id,))
            self.db.execute("delete from memories where memory_id=?", (memory_id,))
            self._fts_drop(memory_id)
            self.db.execute("delete from memory_entities where memory_id=?", (memory_id,))
            self.db.execute("delete from memory_topics where memory_id=?", (memory_id,))
            self.db.execute("delete from memory_links where src=? or dst=?", (memory_id, memory_id))
            self.db.execute("update memories set supersedes=null where supersedes=?", (memory_id,))
            self.db.execute("update memories set superseded_by=null where superseded_by=?", (memory_id,))
            self._cache_drop(memory_id)
            self._event(memory_id, "deleted", "")
            self.db.commit()
            return True

    def delete_all(self) -> int:
        with self._lock:
            n = self.count()
            self._gen += 1
            for t in ("memories", "memory_fts", "memory_fts_map", "memory_entities", "memory_topics", "memory_links"):
                self.db.execute(f"delete from {t}")
            self._event(None, "delete_all", f"{n} memories")
            self.db.commit()
            self._loaded = False
            self._ids, self._row, self._mat, self._live = [], {}, None, None
            return n

    def touch(self, ids: Iterable[str], now: float | None = None) -> None:
        now = now or time.time()
        with self._lock:
            self.db.executemany("update memories set last_accessed_at=?, access_count=access_count+1 where memory_id=?", [(now, i) for i in ids])
            self.db.commit()

    def reindex(self) -> int:
        """Re-embed every memory (the embedding model changed)."""
        with self._lock:
            rows = self.db.execute("select memory_id, content from memories").fetchall()
            for s in range(0, len(rows), 256):
                chunk = rows[s : s + 256]
                vecs = self.embedder.embed([r["content"] for r in chunk])
                self.db.executemany("update memories set embedding=? where memory_id=?", [(vecs[i].astype(np.float32).tobytes(), chunk[i]["memory_id"]) for i in range(len(chunk))])
            self.set_setting("embed_model", self.embedder.name)
            self._loaded = False
            self.db.commit()
            return len(rows)

    def bulk_insert(self, mems: list[Memory]) -> None:
        """Fast path for imports and benchmarks: rows + FTS + entities in one transaction (no dedupe/supersede logic)."""
        with self._lock:
            for m in mems:
                self._write(m, insert=True)
            self.db.commit()
            self._loaded = False

    # ---------------------------------------------------------------- audit
    def stats(self) -> dict[str, Any]:
        def grp(col: str) -> dict[str, int]:
            return {r[0]: r[1] for r in self.db.execute(f"select {col}, count(*) from memories group by {col}")}
        size = Path(self.path).stat().st_size if self.path != ":memory:" and Path(self.path).exists() else 0
        return {"total": self.count(), "by_type": grp("memory_type"), "by_status": grp("status"), "by_source": grp("source"),
                "projects": sorted(self.known_projects()), "db_bytes": size, "embedder": self.embedder.name,
                "sensitive": self.count(sensitivity="sensitive"), "inject_enabled": self.inject_enabled, "capture_enabled": self.capture_enabled, "sensitive_enabled": self.sensitive_enabled}

    def check_invariants(self) -> list[str]:
        problems = []
        for r in self.db.execute("select slot, scope, coalesce(project,'') p, count(*) c from memories where status='ACTIVE' and slot is not null group by 1,2,3 having c>1"):
            problems.append(f"two ACTIVE memories in slot {r['slot']} ({r['scope']}/{r['p']})")
        for r in self.db.execute("select memory_id from memories where status='SUPERSEDED' and superseded_by is null"):
            problems.append(f"{r['memory_id']} superseded by nothing")
        for r in self.db.execute("select m.memory_id from memories m join memories n on n.memory_id=m.superseded_by where m.status='SUPERSEDED' and n.status<>'ACTIVE' and n.status<>'SUPERSEDED'"):
            problems.append(f"{r['memory_id']} superseded by a non-live memory")
        return problems

    def events(self, memory_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql, args = "select * from memory_events", ()
        if memory_id:
            sql, args = sql + " where memory_id=?", (memory_id,)
        return [dict(r) for r in self.db.execute(sql + " order by id desc limit ?", (*args, limit))]

    def export(self) -> list[dict[str, Any]]:
        return [m.public() for m in self.list(limit=10**9)]

    def close(self) -> None:
        with self._lock:
            self.db.close()