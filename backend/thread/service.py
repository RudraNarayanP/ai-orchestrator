"""ThreadService: one canonical thread over replaceable provider chats.

plan_turn() decides, for each user message, whether the provider's current chat keeps going or a new chat (same tab slot)
opens with a continuation packet; chat() drives that against any provider-agnostic `ask`.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from backend.thread.compress import Curator, consolidate, curator_summary, deterministic_summary, empty_state, promote
from backend.thread.packet import Packet, build_packet, recall_block
from backend.thread.recall import is_recall_request, time_window
from backend.thread.store import Found, Message, Segment, SummaryFound, ThreadStore, approx_tokens

# Approximate usable context per web chat, in tokens. These are conservative defaults, configurable per provider
# (settings: threads.limits); the point is to rotate BEFORE a chat degrades or hits its cap.
DEFAULT_LIMITS = {"chatgpt": 32000, "gemini": 100000, "copilot": 16000, "le_chat": 32000, "deepseek": 64000, "qwen": 32000, "meta_ai": 16000,
                  "pi": 8000, "google_ai_mode": 8000, "omnibrain": 12000}
DEFAULT_LIMIT = 24000
DISPLAY = {"chatgpt": "ChatGPT", "gemini": "Gemini", "copilot": "Copilot", "le_chat": "Le Chat", "deepseek": "DeepSeek", "qwen": "Qwen", "meta_ai": "Meta AI",
           "pi": "Pi", "google_ai_mode": "Google AI Mode", "omnibrain": "OmniBrain"}
VIRTUAL = "omnibrain"  # research jobs write their turns into a virtual segment (their own chats are per-job)


class ContextManager:
    """Approximate token accounting per provider chat. Tokens ~ characters / 4: deliberately rough, deliberately early."""

    def __init__(self, limits: dict[str, int] | None = None, rotate_at: float = 0.8, reply_reserve: int = 1500) -> None:
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self.rotate_at = rotate_at
        self.reply_reserve = reply_reserve

    def limit(self, provider: str) -> int:
        return int(self.limits.get(provider, DEFAULT_LIMIT))

    def threshold(self, provider: str) -> int:
        return int(self.limit(provider) * self.rotate_at)

    def should_rotate(self, seg: Segment, incoming_tokens: int, provider: str) -> bool:
        # a chat holding fewer than two exchanges gains nothing from rotating (it would only re-send the same packet)
        if seg.first_seq is None or seg.last_seq is None or seg.last_seq - seg.first_seq < 3:
            return False
        return seg.tokens + incoming_tokens + self.reply_reserve > self.threshold(provider)


@dataclass
class TurnPlan:
    thread_id: str
    segment_id: str
    provider: str
    provider_key: str  # the key the browser adapter uses to find/open the chat: a new segment is a new chat in the same tab slot
    prompt: str  # what to send to the provider
    continue_thread: bool
    rotated: bool = False
    reason: str = ""  # start | context_limit | provider_switch | resume | ""
    packet: Packet | None = None
    recalled: list[int] = field(default_factory=list)  # seqs of earlier messages injected
    segment_label: str = ""


@dataclass
class ChatTurn:
    plan: TurnPlan
    reply: str


class ThreadService:
    def __init__(self, store: ThreadStore, *, curator: Curator | None = None, memory: Any = None, context: ContextManager | None = None,
                 packet_budget: int = 6000, recent_messages: int = 10, recall_k: int = 12) -> None:
        self.store = store
        self.curator = curator
        self.memory = memory
        self.ctx = context or ContextManager()
        self.packet_budget = packet_budget
        self.recent_messages = recent_messages
        self.recall_k = recall_k
        self._open_cache: dict[tuple[str, int], dict[str, Any]] = {}

    # ------------------------------------------------------------------ basics
    def create_thread(self, title: str = "", project: str | None = None) -> str:
        return self.store.create_thread(title, project)

    def label(self, seg: Segment) -> str:
        letter = chr(ord("A") + (self.store.provider_ordinal(seg) - 1) % 26)
        return f"{DISPLAY.get(seg.provider, seg.provider)} {letter}"

    def view(self, tid: str, last: int | None = None) -> list[dict[str, Any]]:
        """What the user sees: one thread. Segments are an implementation detail and are not shown."""
        return [m.public() for m in self.store.messages(tid, last=last)]

    # ------------------------------------------------------------------ segments
    def close_segment(self, seg: Segment, reason: str = "") -> dict[str, Any]:
        th = self.store.get_thread(seg.thread_id) or {}
        msgs = self.store.messages(seg.thread_id, segment_id=seg.segment_id)
        summary, method = curator_summary(msgs, self.curator)
        span = ""
        if msgs:
            span = _span(msgs[0].ts, msgs[-1].ts)
        state = consolidate(th.get("state") or empty_state(), summary, label=self.label(seg), idx=seg.idx, span=span)
        self.store.close_segment(seg.segment_id, summary, method)
        self.store.set_state(seg.thread_id, state)
        promote(summary, self.memory, project=th.get("project"), conversation=seg.thread_id)
        self._open_cache.pop((seg.thread_id, seg.idx), None)
        return summary

    def _segment_for(self, tid: str, provider: str, incoming: int) -> tuple[Segment, bool, str]:
        """(segment to use, rotated?, reason). Closes the old segment when rotating."""
        seg = self.store.active_segment(tid)
        if seg is None:
            reason = "resume" if self.store.has_segments(tid) else "start"
            return self.store.open_segment(tid, provider, reason), reason != "start" or False, reason
        if seg.provider != provider:
            self.close_segment(seg, "provider_switch")
            return self.store.open_segment(tid, provider, "provider_switch"), True, "provider_switch"
        if self.ctx.should_rotate(seg, incoming, provider):
            self.close_segment(seg, "context_limit")
            return self.store.open_segment(tid, provider, "context_limit"), True, "context_limit"
        return seg, False, ""

    # ------------------------------------------------------------------ the main entry point
    def plan_turn(self, tid: str, text: str, provider: str, *, now: float | None = None, ts: float | None = None) -> TurnPlan:
        """Decide what to send. The user's message is stored (raw, forever) before anything can go wrong with a provider."""
        if self.store.get_thread(tid) is None:
            raise KeyError(f"unknown thread {tid}")
        prior = self.store.count(tid)
        before = self.store.active_segment(tid)
        incoming = approx_tokens(text)
        seg, rotated, reason = self._segment_for(tid, provider, incoming)
        fresh = seg.segment_id != (before.segment_id if before else None)  # a newly opened chat has no memory of the conversation
        recent = self.store.messages(tid, last=self.recent_messages) if (fresh and prior) else []
        recall = is_recall_request(text) and prior > 0
        found: list[Found] = []
        sums: list[SummaryFound] = []
        if recall:
            found, sums = self.store.search(tid, text, k=self.recall_k, window=time_window(text, now))
        packet: Packet | None = None
        prompt = text
        if fresh and prior:
            th = self.store.get_thread(tid) or {}
            packet = build_packet(
                state=th.get("state"), recent=recent, task=text, budget_tokens=self.packet_budget, user_lines=self._user_lines(text),
                project=th.get("project"), earlier=[f for f in found if f.message.seq not in {m.seq for m in recent}], summary_hits=sums[:2],
            )
            prompt = packet.text
            self.store.add_segment_tokens(seg.segment_id, packet.tokens)
        elif recall and (found or sums):
            prompt, n = recall_block(found, sums, text)
            self.store.add_segment_tokens(seg.segment_id, approx_tokens(prompt) - incoming)
        self.store.add_messages(tid, [dict(role="user", content=text, provider=provider, segment_id=seg.segment_id, ts=ts)])
        return TurnPlan(
            thread_id=tid, segment_id=seg.segment_id, provider=provider, provider_key=seg.provider_key, prompt=prompt,
            continue_thread=not (fresh) and seg.tokens > 0, rotated=rotated, reason=reason or ("start" if fresh and not prior else ""),
            packet=packet, recalled=[f.message.seq for f in found] if (packet or recall) else [], segment_label=self.label(seg),
        )

    def record_reply(self, tid: str, text: str, provider: str, *, segment_id: str | None = None, ts: float | None = None) -> Message:
        seg = self.store.segment(segment_id) if segment_id else self.store.active_segment(tid)
        return self.store.add_message(tid, "assistant", text, provider=provider, segment_id=seg.segment_id if seg else None, ts=ts)

    def _user_lines(self, text: str) -> list[str]:
        """Relevant personal memory only (retrieved for THIS message) -- context for the new chat, never evidence."""
        if self.memory is None:
            return []
        try:
            res = self.memory.retriever.retrieve(text)
        except Exception:  # noqa: BLE001
            return []
        out = []
        for h in res.hits:
            m = h.memory
            out.append(f"{m.content.rstrip('.')}" + (" (the user's own view, not an established fact)" if m.memory_type.value == "interpretation" else ""))
        return out

    # ------------------------------------------------------------------ research jobs inside a thread
    def context_for_job(self, tid: str, question: str, *, now: float | None = None, with_user_lines: bool = True, include_task: bool = True) -> Packet | None:
        """A packet for a research job's FRESH provider chats. Context only: it is never added to claims, evidence or the verifier."""
        prior = self.store.count(tid)
        if prior == 0:
            return None
        th = self.store.get_thread(tid) or {}
        recent = self.store.messages(tid, last=self.recent_messages)
        open_summary = None
        seg = self.store.active_segment(tid)
        if seg is not None and recent:
            older = [m for m in self.store.messages(tid, segment_id=seg.segment_id) if m.seq < recent[0].seq]
            if older:
                key = (tid, seg.idx)
                cached = self._open_cache.get(key)
                if not cached or cached["last_seq"] != older[-1].seq:
                    cached = deterministic_summary(older)
                    self._open_cache[key] = cached
                open_summary = cached
        found, sums = ([], [])
        if is_recall_request(question):
            found, sums = self.store.search(tid, question, k=self.recall_k, window=time_window(question, now))
        have = {m.seq for m in recent}
        return build_packet(state=th.get("state"), recent=recent, task=question, budget_tokens=self.packet_budget,
                            user_lines=self._user_lines(question) if with_user_lines else [], project=th.get("project"), open_summary=open_summary,
                            earlier=[f for f in found if f.message.seq not in have], summary_hits=sums[:2], include_task=include_task)

    def record_job(self, tid: str, question: str, answer: str, *, job_id: str = "", ts: float | None = None) -> None:
        """A finished research turn becomes part of the canonical thread (user message + the final answer). Rotates the virtual segment by size."""
        seg, _rot, _why = self._segment_for(tid, VIRTUAL, approx_tokens(question) + approx_tokens(answer))
        rows = [dict(role="user", content=question, provider=VIRTUAL, segment_id=seg.segment_id, ts=ts, job_id=job_id)]
        if (answer or "").strip():  # the question is kept even when the job produced no answer: raw messages are never lost
            rows.append(dict(role="assistant", content=answer, provider=VIRTUAL, segment_id=seg.segment_id, ts=ts, job_id=job_id))
        self.store.add_messages(tid, rows)

    # ------------------------------------------------------------------ recall
    def recall(self, tid: str, query: str, *, k: int | None = None, now: float | None = None) -> tuple[list[Found], list[SummaryFound]]:
        return self.store.search(tid, query, k=k or self.recall_k, window=time_window(query, now))

    # ------------------------------------------------------------------ driving a real provider
    async def chat(self, tid: str, text: str, provider: str, ask: "Ask") -> ChatTurn:
        plan = await asyncio.to_thread(self.plan_turn, tid, text, provider)
        reply = await ask(provider, plan.provider_key, plan.prompt, plan.continue_thread)
        if reply and reply.strip():
            await asyncio.to_thread(self.record_reply, tid, reply, provider, segment_id=plan.segment_id)
        return ChatTurn(plan, reply or "")


Ask = Callable[[str, str, str, bool], Awaitable[str]]


def adapter_ask(adapters: dict[str, Any]) -> Ask:
    """Bind the real browser adapters. A segment's provider_key is the adapter's conversation key: a new key is a new chat in the same tab."""

    async def ask(provider: str, key: str, prompt: str, continue_thread: bool) -> str:
        adapter = adapters.get(provider)
        if adapter is None:
            raise RuntimeError(f"{provider} is not available")
        extra = {"continue_thread": True} if continue_thread else {}
        resp = await adapter.ask(key, prompt, 1, emit=None, **extra)
        if resp.status.value != "completed":
            raise RuntimeError(f"{provider}: {resp.status.value}" + (f" -- {resp.error}" if resp.error else ""))
        return resp.answer_text or ""

    return ask


def endpoint_curator(endpoint: Any) -> Curator | None:
    """Wrap the configured curator model (OpenRouter / local) as a synchronous callable; None when it is off."""
    if endpoint is None or not getattr(endpoint, "enabled", False):
        return None

    def run(messages: list[dict[str, str]]) -> dict[str, Any] | None:
        from backend.verification.llm import LLMClient  # noqa: PLC0415

        async def go() -> dict[str, Any] | None:
            parsed, _reply = await LLMClient(endpoint).complete_json(messages, temperature=0.0)
            return parsed

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(go())
        with ThreadPoolExecutor(1) as ex:
            return ex.submit(lambda: asyncio.run(go())).result(timeout=180)

    return run


def _span(a: float, b: float) -> str:
    f = lambda t: time.strftime("%d %b %Y", time.localtime(t))  # noqa: E731
    return f(a) if f(a) == f(b) else f"{f(a)} - {f(b)}"