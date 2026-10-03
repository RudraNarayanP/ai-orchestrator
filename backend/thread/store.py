"""ThreadStore: SQLite source of truth for threads, segments and raw messages (+ FTS5 and exact vectors).

Raw messages are append-only and indexed (keywords + vectors); summaries live beside them and never replace them.
Every query is scoped by thread_id, so one thread can never see another's messages.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backend.memory.embed import Embedder, HashEmbedder, content_tokens
from backend.thread.recall import query_terms

SCHEMA = """
create table if not exists threads(thread_id text primary key, title text, project text, created_at real, updated_at real,
  state_json text default '{}', active_segment text, embedder text, next_seq integer default 1);
create table if not exists segments(segment_id text primary key, thread_id text not null, idx integer, provider text, provider_key text,
  opened_at real, closed_at real, first_seq integer, last_seq integer, tokens integer default 0, summary_json text, method text, reason text);
create index if not exists seg_thread on segments(thread_id, idx);
create table if not exists messages(msg_id integer primary key autoincrement, thread_id text not null, seq integer not null, segment_id text,
  role text, content text, provider text, ts real, tokens integer, job_id text, emb blob);
create index if not exists msg_thread_seq on messages(thread_id, seq);
create index if not exists msg_seg on messages(segment_id);
create virtual table if not exists messages_fts using fts5(tid, content, tokenize='porter unicode61');
create virtual table if not exists summaries_fts using fts5(tid, segment, content, tokenize='porter unicode61');
"""


def approx_tokens(text: str) -> int:
    return max(0, int(len(text or "") / 4))


@dataclass
class Message:
    msg_id: int
    thread_id: str
    seq: int
    segment_id: str | None
    role: str
    content: str
    provider: str
    ts: float
    tokens: int
    job_id: str = ""

    def public(self) -> dict[str, Any]:
        return {"seq": self.seq, "role": self.role, "content": self.content, "provider": self.provider, "ts": self.ts}


@dataclass
class Segment:
    segment_id: str
    thread_id: str
    idx: int
    provider: str
    provider_key: str
    opened_at: float
    closed_at: float | None
    first_seq: int | None
    last_seq: int | None
    tokens: int
    summary: dict[str, Any] | None
    method: str
    reason: str

    @property
    def open(self) -> bool:
        return self.closed_at is None


@dataclass
class Found:
    message: Message
    score: float
    why: list[str] = field(default_factory=list)


@dataclass
class SummaryFound:
    segment_idx: int
    segment_id: str
    text: str
    score: float
    summary: dict[str, Any] | None = None


class _VecIndex:
    def __init__(self, dim: int) -> None:
        self.ids: list[int] = []
        self.mat = np.zeros((256, dim), dtype=np.float32)
        self.n = 0

    def add(self, mid: int, vec: np.ndarray) -> None:
        if self.n >= self.mat.shape[0]:
            self.mat = np.vstack([self.mat, np.zeros_like(self.mat)])
        self.mat[self.n] = vec
        self.ids.append(mid)
        self.n += 1


class ThreadStore:
    def __init__(self, path: str | Path = ":memory:", embedder: Embedder | None = None) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.embedder: Embedder = embedder or HashEmbedder()
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("pragma journal_mode=wal")
        self.db.execute("pragma synchronous=normal")
        self.db.executescript(SCHEMA)
        self._vec: dict[str, _VecIndex] = {}

    def close(self) -> None:
        with self._lock:
            self.db.close()

    # ------------------------------------------------------------------ threads
    def create_thread(self, title: str = "", project: str | None = None) -> str:
        tid = "thr" + uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self.db.execute("insert into threads(thread_id,title,project,created_at,updated_at,embedder) values(?,?,?,?,?,?)",
                            (tid, title, project, now, now, self.embedder.name))
            self.db.commit()
        return tid

    def get_thread(self, tid: str) -> dict[str, Any] | None:
        r = self.db.execute("select * from threads where thread_id=?", (tid,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["state"] = json.loads(d.pop("state_json") or "{}")
        return d

    def list_threads(self) -> list[dict[str, Any]]:
        rows = self.db.execute("select thread_id,title,project,created_at,updated_at,next_seq-1 as messages from threads order by updated_at desc").fetchall()
        return [dict(r) for r in rows]

    def set_state(self, tid: str, state: dict[str, Any]) -> None:
        with self._lock:
            self.db.execute("update threads set state_json=?, updated_at=? where thread_id=?", (json.dumps(state), time.time(), tid))
            self.db.commit()

    def delete_thread(self, tid: str) -> bool:
        """Physical delete of everything in the thread (messages, vectors, summaries, index rows)."""
        with self._lock:
            if self.get_thread(tid) is None:
                return False
            self.db.execute("delete from messages_fts where rowid in (select msg_id from messages where thread_id=?)", (tid,))
            self.db.execute("delete from summaries_fts where tid=?", (tid,))
            for t in ("messages", "segments"):
                self.db.execute(f"delete from {t} where thread_id=?", (tid,))
            self.db.execute("delete from threads where thread_id=?", (tid,))
            self.db.commit()
            self._vec.pop(tid, None)
            return True

    # ------------------------------------------------------------------ segments
    def open_segment(self, tid: str, provider: str, reason: str = "start") -> Segment:
        with self._lock:
            idx = (self.db.execute("select coalesce(max(idx),0)+1 from segments where thread_id=?", (tid,)).fetchone()[0])
            sid = f"{tid}s{idx}"
            self.db.execute("insert into segments(segment_id,thread_id,idx,provider,provider_key,opened_at,reason) values(?,?,?,?,?,?,?)",
                            (sid, tid, idx, provider, f"{tid}:{idx}", time.time(), reason))
            self.db.execute("update threads set active_segment=? where thread_id=?", (sid, tid))
            self.db.commit()
        return self.segment(sid)  # type: ignore[return-value]

    def segment(self, sid: str) -> Segment | None:
        r = self.db.execute("select * from segments where segment_id=?", (sid,)).fetchone()
        return self._seg(r) if r else None

    def segments(self, tid: str) -> list[Segment]:
        return [self._seg(r) for r in self.db.execute("select * from segments where thread_id=? order by idx", (tid,))]

    def active_segment(self, tid: str) -> Segment | None:
        r = self.db.execute("select active_segment from threads where thread_id=?", (tid,)).fetchone()
        if r is None or not r[0]:
            return None
        s = self.segment(r[0])
        return s if s is not None and s.open else None

    @staticmethod
    def _seg(r: sqlite3.Row) -> Segment:
        return Segment(r["segment_id"], r["thread_id"], r["idx"], r["provider"], r["provider_key"], r["opened_at"], r["closed_at"], r["first_seq"],
                       r["last_seq"], r["tokens"] or 0, json.loads(r["summary_json"]) if r["summary_json"] else None, r["method"] or "", r["reason"] or "")

    def close_segment(self, sid: str, summary: dict[str, Any], method: str) -> None:
        seg = self.segment(sid)
        if seg is None:
            return
        flat = summary_text(summary)
        with self._lock:
            self.db.execute("update segments set closed_at=?, summary_json=?, method=? where segment_id=?", (time.time(), json.dumps(summary), method, sid))
            self.db.execute("update threads set active_segment=NULL where thread_id=? and active_segment=?", (seg.thread_id, sid))
            self.db.execute("delete from summaries_fts where tid=? and segment=?", (seg.thread_id, str(seg.idx)))
            self.db.execute("insert into summaries_fts(tid,segment,content) values(?,?,?)", (seg.thread_id, str(seg.idx), flat))
            self.db.commit()

    # ------------------------------------------------------------------ messages
    def add_message(self, tid: str, role: str, content: str, *, provider: str = "", segment_id: str | None = None, ts: float | None = None,
                    job_id: str = "") -> Message:
        return self.add_messages(tid, [dict(role=role, content=content, provider=provider, segment_id=segment_id, ts=ts, job_id=job_id)])[0]

    def add_messages(self, tid: str, items: Iterable[dict[str, Any]], *, embed: bool = True) -> list[Message]:
        items = list(items)
        if not items:
            return []
        with self._lock:
            row = self.db.execute("select next_seq from threads where thread_id=?", (tid,)).fetchone()
            if row is None:
                raise KeyError(f"unknown thread {tid}")
            seq = row[0]
            vecs = self.embedder.embed([i["content"] for i in items]) if embed else None
            out: list[Message] = []
            now = time.time()
            for n, it in enumerate(items):
                text = it["content"] or ""
                toks = approx_tokens(text)
                ts = it.get("ts") or now
                emb = vecs[n].astype(np.float32).tobytes() if vecs is not None else None
                cur = self.db.execute(
                    "insert into messages(thread_id,seq,segment_id,role,content,provider,ts,tokens,job_id,emb) values(?,?,?,?,?,?,?,?,?,?)",
                    (tid, seq, it.get("segment_id"), it["role"], text, it.get("provider") or "", ts, toks, it.get("job_id") or "", emb))
                mid = cur.lastrowid
                self.db.execute("insert into messages_fts(rowid,tid,content) values(?,?,?)", (mid, tid, text))
                msg = Message(mid, tid, seq, it.get("segment_id"), it["role"], text, it.get("provider") or "", ts, toks, it.get("job_id") or "")
                out.append(msg)
                if vecs is not None and tid in self._vec:
                    self._vec[tid].add(mid, vecs[n])
                sid = it.get("segment_id")
                if sid:
                    self.db.execute("update segments set tokens=tokens+?, last_seq=?, first_seq=coalesce(first_seq,?) where segment_id=?", (toks, seq, seq, sid))
                seq += 1
            self.db.execute("update threads set next_seq=?, updated_at=? where thread_id=?", (seq, now, tid))
            self.db.commit()
        return out

    @staticmethod
    def _msg(r: sqlite3.Row) -> Message:
        return Message(r["msg_id"], r["thread_id"], r["seq"], r["segment_id"], r["role"], r["content"], r["provider"] or "", r["ts"], r["tokens"] or 0, r["job_id"] or "")

    _COLS = "msg_id,thread_id,seq,segment_id,role,content,provider,ts,tokens,job_id"

    def messages(self, tid: str, *, last: int | None = None, segment_id: str | None = None, after_seq: int | None = None) -> list[Message]:
        q, args = f"select {self._COLS} from messages where thread_id=?", [tid]
        if segment_id:
            q += " and segment_id=?"
            args.append(segment_id)
        if after_seq is not None:
            q += " and seq>?"
            args.append(after_seq)
        if last is not None:
            rows = self.db.execute(q + " order by seq desc limit ?", (*args, last)).fetchall()
            return [self._msg(r) for r in reversed(rows)]
        return [self._msg(r) for r in self.db.execute(q + " order by seq", args)]

    def add_segment_tokens(self, sid: str, n: int) -> None:
        with self._lock:
            self.db.execute("update segments set tokens=max(0,tokens+?) where segment_id=?", (n, sid))
            self.db.commit()

    def count(self, tid: str) -> int:
        return self.db.execute("select count(*) from messages where thread_id=?", (tid,)).fetchone()[0]

    def get_by_ids(self, ids: list[int], tid: str) -> dict[int, Message]:
        out: dict[int, Message] = {}
        for i in range(0, len(ids), 400):
            chunk = ids[i : i + 400]
            rows = self.db.execute(f"select {self._COLS} from messages where thread_id=? and msg_id in ({','.join('?' * len(chunk))})", (tid, *chunk))
            out.update({r["msg_id"]: self._msg(r) for r in rows})
        return out

    # ------------------------------------------------------------------ search (hybrid)
    def _index(self, tid: str) -> _VecIndex | None:
        th = self.get_thread(tid)
        if th is None or th["embedder"] != self.embedder.name:
            return None  # vectors from another embedder are not comparable; keywords still work
        if tid not in self._vec:
            idx = _VecIndex(self.embedder.dim)
            for r in self.db.execute("select msg_id, emb from messages where thread_id=? and emb is not null order by seq", (tid,)):
                idx.add(r[0], np.frombuffer(r[1], dtype=np.float32))
            self._vec[tid] = idx
        return self._vec[tid]

    def fts_ids(self, tid: str, terms: list[str], limit: int = 60) -> list[int]:
        if not terms:
            return []
        expr = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
        try:
            rows = self.db.execute(
                "select rowid from messages_fts where messages_fts match ? order by bm25(messages_fts, 0.0, 1.0) limit ?",
                (f'tid : "{tid}" AND content : ({expr})', limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows]

    def vector_ids(self, tid: str, query: str, limit: int = 60) -> list[tuple[int, float]]:
        idx = self._index(tid)
        if idx is None or idx.n == 0:
            return []
        qv = self.embedder.embed([query])[0]
        sims = idx.mat[: idx.n] @ qv
        k = min(limit, idx.n)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        floor = getattr(self.embedder, "floor", 0.2)
        return [(idx.ids[i], float(sims[i])) for i in top if sims[i] >= floor]

    def search_summaries(self, tid: str, terms: list[str], limit: int = 5) -> list[SummaryFound]:
        if not terms:
            return []
        expr = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
        try:
            rows = self.db.execute("select segment, content, bm25(summaries_fts, 0.0, 0.0, 1.0) as b from summaries_fts where summaries_fts match ? order by b limit ?",
                                   (f'tid : "{tid}" AND content : ({expr})', limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        for n, r in enumerate(rows):
            seg = self.db.execute("select segment_id from segments where thread_id=? and idx=?", (tid, int(r["segment"]))).fetchone()
            full = self.segment(seg[0]) if seg else None
            out.append(SummaryFound(int(r["segment"]), seg[0] if seg else "", r["content"], 1.0 / (30 + n), full.summary if full else None))
        return out

    def search(self, tid: str, query: str, *, k: int = 12, window: tuple[float, float] | None = None) -> tuple[list[Found], list[SummaryFound]]:
        """Hybrid recall over ONE thread: keywords + vectors + summaries, fused by rank, with a soft time boost."""
        terms = query_terms(query)
        fts = self.fts_ids(tid, terms)
        vec = self.vector_ids(tid, query)
        sums = self.search_summaries(tid, terms)
        score: dict[int, float] = {}
        why: dict[int, list[str]] = {}
        for n, mid in enumerate(fts):
            score[mid] = score.get(mid, 0.0) + 1.0 / (30 + n)
            why.setdefault(mid, []).append("keywords")
        for n, (mid, _sim) in enumerate(vec):
            score[mid] = score.get(mid, 0.0) + 1.0 / (30 + n)
            why.setdefault(mid, []).append("meaning")
        msgs = self.get_by_ids(list(score), tid)
        seg_boost = {s.segment_id for s in sums}
        qset = set(terms)
        found: list[Found] = []
        for mid, s in score.items():
            m = msgs.get(mid)
            if m is None:
                continue
            if qset:
                have = set(re.findall(r"[a-z0-9']+", m.content.lower()))
                cov = sum(1 for t in qset if t in have or any(h.startswith(t[:5]) for h in have if len(h) >= 5 and len(t) >= 5)) / len(qset)
                s += 0.03 * cov
            if window and window[0] <= m.ts <= window[1]:
                s += 0.03
                why[mid].append("time")
            if m.segment_id in seg_boost:
                s += 0.012
                why[mid].append("summary")
            found.append(Found(m, s, why[mid]))
        found.sort(key=lambda f: -f.score)
        seen: set[str] = set()
        out: list[Found] = []
        for f in found:
            key = f.message.content[:160]
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
            if len(out) >= k:
                break
        return out, sums

    def stats(self) -> dict[str, Any]:
        q = lambda s: self.db.execute(s).fetchone()[0]  # noqa: E731
        return {"threads": q("select count(*) from threads"), "segments": q("select count(*) from segments"), "messages": q("select count(*) from messages")}


def summary_text(summary: dict[str, Any]) -> str:
    parts = [summary.get("narrative", "")]
    for k, items in (summary.get("items") or {}).items():
        parts += [i["t"] if isinstance(i, dict) else str(i) for i in items]
    return " . ".join(p for p in parts if p)