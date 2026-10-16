"""Compress: complete transcript reading (dedup, order, completeness), chunked summary, fresh primed chats."""

from __future__ import annotations

import asyncio

from backend.thread.service import ThreadService
from backend.thread.store import ThreadStore
from backend.thread.transcript import Turn, chunked_summary, chunks, dedupe_turns, merge_sources, merge_windows, page_complete


def w(*turns):
    return [dict(role=r, text=t, top=top, **({"id": i} if i else {}), **({"mark": m} if m else {})) for r, t, top, i, m in turns]


# ------------------------------------------------------------------ dedup + ordering of scroll windows
def test_overlapping_scroll_windows_keep_each_turn_once_in_page_order():
    a = w(("user", "first question", 0, "", "r-1"), ("assistant", "first answer", 120, "", "r-2"))
    b = w(("assistant", "first answer", 120, "", "r-2"), ("user", "second question", 900, "", "r-3"))
    c = w(("assistant", "second answer", 1100, "", "r-4"), ("user", "second question", 900, "", "r-3"))  # read out of order
    got = merge_windows([c, a, b])
    assert [g["text"] for g in got] == ["first question", "first answer", "second question", "second answer"]


def test_a_virtualised_turn_recreated_with_a_new_element_is_still_one_turn_but_the_same_words_later_are_a_new_turn():
    a = w(("user", "ok", 0, "", "r-1"), ("assistant", "answer one", 100, "", "r-2"))
    b = w(("user", "ok", 10, "", "r-9"))  # same turn, element re-created, slightly moved
    c = w(("user", "ok", 3000, "", "r-10"))  # the user said "ok" again much later
    got = merge_windows([a, b, c])
    assert [(g["role"], g["text"]) for g in got] == [("user", "ok"), ("assistant", "answer one"), ("user", "ok")]


def test_the_same_message_id_is_one_turn_and_the_longest_text_wins_after_show_more():
    a = w(("assistant", "A long answer that was coll", 100, "msg-7", ""))
    b = w(("assistant", "A long answer that was collapsed and is now expanded", 140, "msg-7", ""))
    got = merge_windows([a, b])
    assert len(got) == 1 and got[0]["text"].endswith("now expanded")


def test_dedupe_turns_drops_empty_and_unknown_roles_and_merges_adjacent_partial_reads():
    raw = [dict(role="user", text="Hello  there"), dict(role="assistant", text="Hi"), dict(role="assistant", text="Hi, how can I help?"),
           dict(role="system", text="x"), dict(role="user", text="  ")]
    got = dedupe_turns(raw)
    assert [(t.role, t.text) for t in got] == [("user", "Hello there"), ("assistant", "Hi, how can I help?")]


# ------------------------------------------------------------------ completeness
class M:
    def __init__(self, seq, role, content, provider="chatgpt", segment_id="s1"):
        self.seq, self.role, self.content, self.provider, self.segment_id = seq, role, content, provider, segment_id


def T(role, text):
    return Turn(role, text)


def test_a_page_missing_lazy_loaded_turns_is_not_complete():
    log = [M(1, "user", "q1"), M(2, "assistant", "a1 is long"), M(3, "user", "q2"), M(4, "assistant", "a2")]
    ok, why = page_complete([T("user", "q2"), T("assistant", "a2")], log)
    assert not ok and "not loaded" in why


def test_a_page_with_every_logged_turn_in_order_is_complete_even_when_the_first_message_carried_a_context_packet():
    log = [M(1, "user", "what is the DPDP Act"), M(2, "assistant", "It is India's data law."), M(3, "user", "and children?")]
    page = [T("user", "We are continuing... [context] what is the DPDP Act"), T("assistant", "It is India's data law."), T("user", "and children?"),
            T("assistant", "typed straight in the tab")]
    ok, _ = page_complete(page, log)
    assert ok


def test_a_page_with_a_turn_out_of_order_or_missing_is_not_complete():
    log = [M(1, "user", "alpha question"), M(2, "assistant", "alpha answer"), M(3, "user", "beta question")]
    ok, why = page_complete([T("user", "beta question"), T("assistant", "alpha answer"), T("user", "alpha question")], log)
    assert not ok and "missing" in why


def test_merge_uses_a_complete_page_in_place_of_its_chat_and_falls_back_to_the_log_otherwise():
    log = [M(1, "user", "r1", "omnibrain", "v1"), M(2, "assistant", "ra1", "omnibrain", "v1"),
           M(3, "user", "c1", "chatgpt", "s2"), M(4, "assistant", "ca1 short", "chatgpt", "s2"),
           M(5, "user", "d1", "deepseek", "s3"), M(6, "assistant", "da1", "deepseek", "s3")]
    pages = {"chatgpt": [T("user", "c1"), T("assistant", "ca1 short, and the rest the log never had")],
             "deepseek": [T("assistant", "da1")]}  # incomplete: the user turn did not load
    g = merge_sources(log, pages)
    assert [t.text for t in g.turns] == ["r1", "ra1", "c1", "ca1 short, and the rest the log never had", "d1", "da1"]
    assert [t.seq for t in g.turns] == [1, 2, 3, 4, 5, 6]
    assert g.sources["chatgpt"].startswith("page") and g.sources["deepseek"].startswith("log")
    assert [t.source for t in g.turns][2:] == ["page", "page", "log", "log"]


def test_an_unreadable_tab_falls_back_to_the_log():
    log = [M(1, "user", "q"), M(2, "assistant", "a")]
    g = merge_sources(log, {"chatgpt": None})
    assert [t.text for t in g.turns] == ["q", "a"] and g.sources["chatgpt"].startswith("log")


# ------------------------------------------------------------------ chunked summary
def test_long_transcripts_are_chunked_without_dropping_text_and_every_chunk_is_summarised_and_merged():
    turns = [Turn("user" if i % 2 == 0 else "assistant", f"We decided to use Postgres number {i}. " * 40, seq=i) for i in range(30)]
    parts = chunks(turns, max_chars=4000)
    assert len(parts) > 1 and sum(len(t.text) for p in parts for t in p) == sum(len(t.text) for t in turns)
    calls = []

    def curator(messages):
        calls.append(messages)
        return None  # unusable -> deterministic floor, still per chunk

    state, methods = chunked_summary(turns, curator, max_chars=4000)
    assert len(calls) == len(parts) == len(methods)
    assert state["items"]["decisions"]


def test_a_single_huge_turn_is_cut_into_chunks_not_dropped():
    parts = chunks([Turn("assistant", "x" * 10000)], max_chars=4000)
    assert [len(p[0].text) for p in parts] == [4000, 4000, 2000]


# ------------------------------------------------------------------ the service: compress, then the next message continues the primed chat
def test_compress_summarises_the_whole_thread_and_the_next_message_continues_the_fresh_chat():
    svc = ThreadService(ThreadStore(":memory:"))
    tid = svc.create_thread("t")
    sent = []

    async def ask(provider, key, prompt, cont):
        sent.append((provider, key, prompt, cont))
        return "Ready." if prompt.startswith("We are continuing") else f"reply to {prompt[-20:]}"

    async def go():
        for i in range(3):
            await svc.chat(tid, f"We decided to use Rust for part {i}. What next?", "chatgpt", ask)

        async def read(p):
            return {"turns": [{"role": "user", "text": "only the last turn loaded"}]}  # incomplete: falls back to the log

        out = await svc.compress(tid, ["chatgpt", "deepseek"], ask, read)
        n_before = len(sent)
        await svc.chat(tid, "and the tests?", "chatgpt", ask)
        await svc.chat(tid, "and docs?", "deepseek", ask)
        return out, sent[n_before:]

    out, after = asyncio.run(go())
    assert out["providers"]["chatgpt"]["ok"] and out["providers"]["deepseek"]["ok"]
    assert out["sources"]["chatgpt"].startswith("log") and out["turns_summarised"] == 6
    openings = [s for s in sent if s[2].startswith("We are continuing")]
    assert {s[0] for s in openings} == {"chatgpt", "deepseek"} and all(not s[3] for s in openings)
    assert "Rust" in openings[0][2]
    # the next messages go into those primed chats: same key, continued, and no second copy of the thread
    keys = {s[0]: s[1] for s in openings}
    assert after[0] == ("chatgpt", keys["chatgpt"], "and the tests?", True)
    assert after[1] == ("deepseek", keys["deepseek"], "and docs?", True)


def test_a_provider_that_fails_to_open_its_fresh_chat_does_not_stop_the_others():
    svc = ThreadService(ThreadStore(":memory:"))
    tid = svc.create_thread("t")

    async def ask(provider, key, prompt, cont):
        if provider == "qwen" and prompt.startswith("We are continuing"):
            raise RuntimeError("qwen: rate_limited")
        return "ok"

    async def go():
        await svc.chat(tid, "hello there friend", "chatgpt", ask)
        return await svc.compress(tid, ["qwen", "chatgpt"], ask)

    out = asyncio.run(go())
    assert not out["providers"]["qwen"]["ok"] and out["providers"]["chatgpt"]["ok"]
    assert svc.store.active_segment(tid).segment_id == out["providers"]["chatgpt"]["segment_id"]


def test_a_segment_that_vanished_is_reported_not_deferenced():
    """Reproduced under CPU load 2026-10-16 (300 repeats of this same compress): open_segment inserts the
    row inside the store lock but reads it back outside it, on one shared sqlite connection, so a second
    provider's transaction boundary can leave the first with nothing to read. prime() then did
    `s.provider_key` on None, the except handler did `s.segment_id` on the same None, and the AttributeError
    escaped compress -- one provider's lost row stopped every provider."""
    svc = ThreadService(ThreadStore(":memory:"))
    tid = svc.create_thread("t")
    real = svc.store.open_segment

    def flaky(t, provider, reason="start"):
        return None if provider == "deepseek" else real(t, provider, reason)

    async def ask(provider, key, prompt, cont):
        return "Ready." if prompt.startswith("We are continuing") else "ok"

    async def go():
        await svc.chat(tid, "we decided to use Rust", "chatgpt", ask)
        svc.store.open_segment = flaky
        try:
            return await svc.compress(tid, ["deepseek", "chatgpt"], ask)
        finally:
            svc.store.open_segment = real

    out = asyncio.run(go())
    assert out["providers"]["deepseek"]["ok"] is False, "the vanished row is a reported failure, not a crash"
    assert "store race" in out["providers"]["deepseek"]["error"]
    assert out["providers"]["chatgpt"]["ok"] is True
    assert svc.store.active_segment(tid).segment_id == out["providers"]["chatgpt"]["segment_id"]
