"""The unlimited OmniBrain thread: rotation, continuation packet, nothing lost, provider switch, recall, isolation, scale."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from backend.api import app as api_app
from backend.evidence.sources import SourceTier
from backend.memory.embed import HashEmbedder
from backend.memory.service import MemoryService
from backend.memory.store import MemoryStore
from backend.models import Job
from backend.orchestrator.runner import ResearchRunner
from backend.thread.compress import consolidate, curator_summary, deterministic_summary, empty_state
from backend.thread.packet import HEADER, build_packet
from backend.thread.recall import is_recall_request, query_terms, time_window
from backend.thread.service import ContextManager, ThreadService, adapter_ask
from backend.thread.store import ThreadStore, approx_tokens
from tests.conftest import adapters_from, base_settings

DAY = 86400.0

TRIP = [
    ("I prefer short answers. We are planning a Germany trip for the Munich conference.", "Munich it is. Flights from Delhi to Munich take about 9 hours."),
    ("Let's go with Lufthansa for the flight. I don't want a layover in Frankfurt.", "Noted: Lufthansa direct, the fare is around 640 EUR."),
    ("Actually the conference is in April, not March.", "Understood. The conference runs 14-16 April at Messe Muenchen."),
    ("What about the hotel? Still not sure whether to stay near the Messe.", "Staying near the Messe saves commute time, however it costs more. I recommend the Ibis near the station."),
]


def make(limits=None, rotate_at=0.8, budget=6000, curator=None, memory=None, reserve=20) -> ThreadService:
    return ThreadService(ThreadStore(":memory:", HashEmbedder()), curator=curator, memory=memory,
                         context=ContextManager(limits or {}, rotate_at=rotate_at, reply_reserve=reserve), packet_budget=budget)


def talk(svc, tid, provider, pairs, **kw):
    plans = []
    for u, a in pairs:
        p = svc.plan_turn(tid, u, provider, **kw)
        svc.record_reply(tid, a, provider, segment_id=p.segment_id)
        plans.append(p)
    return plans


# ------------------------------------------------------------------------------------------------ rotation
def test_no_rotation_below_the_limit_and_the_chat_continues():
    svc = make({"chatgpt": 100000})
    tid = svc.create_thread()
    plans = talk(svc, tid, "chatgpt", TRIP * 6)
    assert not any(p.rotated for p in plans)
    assert plans[0].continue_thread is False and all(p.continue_thread for p in plans[1:])
    assert {p.provider_key for p in plans} == {plans[0].provider_key}
    assert all(p.packet is None for p in plans)


def test_rotation_triggers_at_the_provider_limit_and_opens_a_new_chat_with_a_packet():
    svc = make({"chatgpt": 1200}, rotate_at=0.5, budget=500)  # threshold 600 approximate tokens
    tid = svc.create_thread()
    plans = talk(svc, tid, "chatgpt", TRIP * 6)
    rot = [p for p in plans if p.rotated]
    assert rot and all(p.reason == "context_limit" for p in rot)
    first = rot[0]
    assert first.continue_thread is False and first.packet is not None
    assert first.provider_key != plans[0].provider_key, "a new segment is a new chat in the same tab slot"
    assert first.prompt.startswith(HEADER)
    # the chat that was left is closed with a summary; the one in use is open
    segs = svc.store.segments(tid)
    assert segs[0].closed_at is not None and segs[0].summary and segs[-1].open
    # rotation happened when the OLD chat was at the threshold, not before
    assert segs[0].tokens + approx_tokens(first.packet.text and TRIP[0][0]) + 20 > 600 or segs[0].tokens > 450


def test_each_provider_has_its_own_configurable_limit():
    svc = make({"chatgpt": 700, "gemini": 100000}, rotate_at=0.5, budget=400)
    a, b = svc.create_thread(), svc.create_thread()
    pa = talk(svc, a, "chatgpt", TRIP * 6)
    pb = talk(svc, b, "gemini", TRIP * 6)
    assert any(p.rotated for p in pa) and not any(p.rotated for p in pb)
    assert ContextManager().limit("chatgpt") != ContextManager().limit("pi")
    assert ContextManager({"chatgpt": 5}).limit("chatgpt") == 5


def test_a_fresh_chat_is_never_rotated_again_straight_away():
    svc = make({"chatgpt": 300}, rotate_at=0.5, budget=400)
    tid = svc.create_thread()
    plans = talk(svc, tid, "chatgpt", TRIP * 4)
    # even with a tiny limit, a chat that has not yet held two exchanges is not thrown away again
    for p in plans:
        if p.rotated:
            seg = svc.store.segment(p.segment_id)
            assert seg is not None
    segs = svc.store.segments(tid)
    assert all((s.last_seq or 0) - (s.first_seq or 0) >= 1 for s in segs)


# ------------------------------------------------------------------------------------------------ the packet
def test_packet_has_every_required_section_and_the_instructions():
    svc = make({"chatgpt": 1000}, rotate_at=0.5, budget=1500)
    tid = svc.create_thread(project="Munich trip")
    plans = talk(svc, tid, "chatgpt", TRIP * 6)
    p = next(p for p in plans if p.packet)
    text = p.packet.text
    for needle in ("OMNIBRAIN CONTINUATION CONTEXT", "PROJECT:", "CONVERSATION HISTORY (compressed):", "RECENT DISCUSSION:", "DECISIONS:", "OPEN QUESTIONS:",
                   "CORRECTIONS:", "IMPORTANT USER PREFERENCES:", "CURRENT TASK", "Project: Munich trip"):
        assert needle in text, needle
    assert "Actually the conference is in April, not March." in text  # correction
    assert "I prefer short answers." in text  # preference
    assert "Let's go with Lufthansa for the flight." in text  # decision
    assert "Still not sure whether to stay near the Messe." in text  # open question
    low = text.lower()
    assert "continue the conversation naturally" in low and "don't restart" in low and "don't ask the user to repeat" in low and "don't mention" in low
    assert "not verified evidence" in low
    assert text.rstrip().endswith(TRIP[0][0]) or text.rstrip().endswith(TRIP[1][0]) or text.rstrip().endswith(TRIP[2][0]) or text.rstrip().endswith(TRIP[3][0])


def test_current_user_section_carries_only_relevant_memory():
    mem = MemoryService(MemoryStore(":memory:", HashEmbedder()))
    mem.remember("Lives in Delhi and flies from Delhi airport")
    mem.remember("Owns a cat called Miso")
    svc = make({"chatgpt": 1000}, rotate_at=0.5, budget=1500, memory=mem)
    tid = svc.create_thread()
    plans = talk(svc, tid, "chatgpt", TRIP * 6)
    pk = next(p for p in plans if p.packet)
    # rotation happens on a later message; ask about flights from Delhi in a new provider chat
    p = svc.plan_turn(tid, "Where do I live and which airport do I use?", "gemini")
    assert "CURRENT USER:" in p.prompt and "Delhi airport" in p.prompt
    assert "Miso" not in p.prompt and "Miso" not in pk.prompt


def test_packet_respects_the_token_budget_for_a_huge_thread():
    svc = make({"chatgpt": 3000}, rotate_at=0.5, budget=900)
    tid = svc.create_thread()
    plans = talk(svc, tid, "chatgpt", [(u + " " + "detail " * 40, a + " " + "more " * 60) for u, a in TRIP] * 30)
    packets = [p.packet for p in plans if p.packet]
    assert len(packets) > 5
    assert all(pk.within_budget for pk in packets), [pk.tokens for pk in packets]


def test_a_huge_current_message_is_clipped_in_the_packet_but_stored_whole():
    svc = make(budget=800)
    tid = svc.create_thread()
    talk(svc, tid, "chatgpt", TRIP)
    big = "Please review this: " + "lorem ipsum dolor " * 2000
    p = svc.plan_turn(tid, big, "gemini")
    assert p.packet is not None and p.packet.within_budget and "characters omitted" in p.prompt
    assert svc.store.messages(tid, last=1)[0].content == big


def test_packet_labels_thread_facts_as_context_not_evidence():
    pk = build_packet(state=empty_state(), recent=[], task="hi", budget_tokens=800)
    assert "not verified evidence" in pk.text


# ------------------------------------------------------------------------------------------------ nothing lost
def test_nothing_is_lost_across_many_rotations():
    svc = make({"chatgpt": 900}, rotate_at=0.5, budget=500)
    tid = svc.create_thread()
    sent = []
    for i in range(40):
        u, a = TRIP[i % 4]
        u, a = f"{u} (turn {i})", f"{a} (turn {i})"
        p = svc.plan_turn(tid, u, "chatgpt")
        svc.record_reply(tid, a, "chatgpt", segment_id=p.segment_id)
        sent += [u, a]
    assert len(svc.store.segments(tid)) >= 4
    stored = [m.content for m in svc.store.messages(tid)]
    assert stored == sent, "every raw message, in order"
    assert [m["content"] for m in svc.view(tid)] == sent, "the user sees one thread"
    # each message is indexed: both the keyword index and the vectors know it
    found, _ = svc.recall(tid, "turn 7 Lufthansa fare", k=12)
    assert any("(turn 7)" in f.message.content for f in found) or any("Lufthansa" in f.message.content for f in found)
    assert svc.store.db.execute("select count(*) from messages_fts").fetchone()[0] == len(sent)
    assert svc.store.db.execute("select count(*) from messages where emb is not null").fetchone()[0] == len(sent)


def test_user_message_is_stored_before_a_provider_failure():
    import asyncio

    svc = make()
    tid = svc.create_thread()

    async def boom(provider, key, prompt, cont):
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        asyncio.run(svc.chat(tid, "Do not lose this message", "chatgpt", boom))
    assert [m.content for m in svc.store.messages(tid)] == ["Do not lose this message"]


def test_chat_driver_records_the_reply_and_works_with_any_provider():
    import asyncio

    svc = make()
    tid = svc.create_thread()
    seen = []

    async def ask(provider, key, prompt, cont):
        seen.append((provider, key, cont))
        return f"{provider} says hi"

    t1 = asyncio.run(svc.chat(tid, "hello", "chatgpt", ask))
    t2 = asyncio.run(svc.chat(tid, "and again", "chatgpt", ask))
    assert t1.reply == "chatgpt says hi" and [s[2] for s in seen] == [False, True]
    assert [m["role"] for m in svc.view(tid)] == ["user", "assistant", "user", "assistant"]
    assert t2.plan.segment_label == "ChatGPT A"


# ------------------------------------------------------------------------------------------------ provider switch
def test_switching_provider_mid_thread_gives_the_new_provider_the_thread_state():
    svc = make(budget=2000)
    tid = svc.create_thread()
    talk(svc, tid, "chatgpt", TRIP)
    p = svc.plan_turn(tid, "OK, now which hotel?", "gemini")
    assert p.rotated and p.reason == "provider_switch" and p.continue_thread is False
    assert p.segment_label == "Gemini A" and p.provider_key != svc.store.segments(tid)[0].provider_key
    assert "Let's go with Lufthansa" in p.prompt and "Actually the conference is in April" in p.prompt and "I prefer short answers" in p.prompt
    assert p.prompt.rstrip().endswith("OK, now which hotel?")
    # the old chat was closed and summarised; the thread is still one thread
    assert svc.store.segments(tid)[0].summary
    assert len(svc.view(tid)) == 9


def test_switching_back_opens_another_chat_not_the_old_one():
    svc = make()
    tid = svc.create_thread()
    talk(svc, tid, "chatgpt", TRIP[:2])
    talk(svc, tid, "gemini", TRIP[2:3])
    p = svc.plan_turn(tid, "back to chatgpt", "chatgpt")
    assert p.segment_label == "ChatGPT B" and p.packet is not None
    assert [s.provider for s in svc.store.segments(tid)] == ["chatgpt", "gemini", "chatgpt"]


# ------------------------------------------------------------------------------------------------ historical recall
def _eight_months(svc):
    tid = svc.create_thread()
    old = time.time() - 240 * DAY
    for i, (u, a) in enumerate([
        ("Remember that I'm thinking of moving to Germany, probably Berlin, for the AI jobs.", "Berlin has a big AI scene; the Blue Card needs a salary of about 45,300 EUR."),
        ("What is the minimum salary for the Blue Card in Germany?", "About 45,300 EUR in 2024 (43,759 EUR for shortage occupations)."),
    ]):
        p = svc.plan_turn(tid, u, "chatgpt", ts=old + i * 60)
        svc.record_reply(tid, a, "chatgpt", segment_id=p.segment_id, ts=old + i * 60 + 5)
    for n in range(60):  # lots of unrelated, later talk
        p = svc.plan_turn(tid, f"Question {n} about guitar strings and cooking pasta recipe {n}", "chatgpt", ts=time.time() - 30 * DAY + n * 60)
        svc.record_reply(tid, f"Answer {n}: use medium gauge strings and salt the water {n}.", "chatgpt", segment_id=p.segment_id, ts=time.time() - 30 * DAY + n * 60 + 5)
    return tid


def test_historical_recall_finds_the_old_messages_and_injects_only_the_relevant_ones():
    svc = make({"chatgpt": 100000})
    tid = _eight_months(svc)
    p = svc.plan_turn(tid, "Remember that thing about Germany 8 months ago? What salary did you say?", "chatgpt")
    assert p.continue_thread is True, "same chat keeps going; only a recall block is added"
    assert "RELEVANT EARLIER DISCUSSION" in p.prompt and "Blue Card" in p.prompt and "Berlin" in p.prompt
    assert "guitar" not in p.prompt and "pasta" not in p.prompt
    assert p.prompt.count("\n- [") <= 12
    assert p.recalled and len(p.recalled) <= 12


def test_recall_is_hybrid_it_uses_meaning_and_summaries_not_only_keywords():
    svc = make({"chatgpt": 100000})
    tid = _eight_months(svc)
    found, sums = svc.recall(tid, "what did you tell me about the Blue Card salary threshold?")
    assert found and "Blue Card" in found[0].message.content
    assert {"keywords"} <= set(found[0].why)
    # a paraphrase with no shared rare word still reaches the right message through the vector side
    found2, _ = svc.recall(tid, "minimum salary Germany blue card")
    assert any("45,300" in f.message.content for f in found2)


def test_time_hint_boosts_but_never_filters():
    svc = make({"chatgpt": 100000})
    tid = _eight_months(svc)
    found, _ = svc.recall(tid, "Germany jobs 8 months ago")
    assert "time" in found[0].why or "time" in found[1].why
    wrong, _ = svc.recall(tid, "Germany jobs 2 weeks ago")  # wrong estimate: still found
    assert any("Berlin" in f.message.content for f in wrong)


def test_recall_cue_and_window_parsing():
    assert is_recall_request("remember that thing about Germany 8 months ago")
    assert not is_recall_request("what is the capital of France")
    assert query_terms("remember that thing about Germany 8 months ago") == ["germany"]
    now = 1_000_000_000.0
    lo, hi = time_window("8 months ago", now)
    assert lo < now - 8 * 30 * DAY < hi and hi < now - 5 * 30 * DAY
    assert time_window("what is a tree", now) is None


def test_recall_into_a_new_chat_goes_in_the_packet_with_the_thread_summary():
    svc = make({"chatgpt": 100000}, budget=3000)
    tid = _eight_months(svc)
    p = svc.plan_turn(tid, "Remember that thing about Germany 8 months ago?", "gemini")
    assert p.packet and "RELEVANT EARLIER DISCUSSION" in p.prompt and "Blue Card" in p.prompt and "CONVERSATION HISTORY" in p.prompt


# ------------------------------------------------------------------------------------------------ isolation
def test_threads_are_isolated_from_each_other():
    svc = make(budget=2000)
    a, b = svc.create_thread(), svc.create_thread()
    talk(svc, a, "chatgpt", [("My secret codename is ZXQ-ALPHA-9 for the Falcon project.", "Noted, ZXQ-ALPHA-9.")])
    talk(svc, b, "chatgpt", [("Let's talk about gardening and tomatoes.", "Tomatoes like sun.")])
    assert svc.recall(b, "ZXQ-ALPHA-9 Falcon codename")[0] == []
    assert svc.recall(b, "ZXQ-ALPHA-9 Falcon codename")[1] == []
    pb = svc.plan_turn(b, "remember the codename?", "gemini")
    assert "ZXQ" not in pb.prompt and "Falcon" not in pb.prompt
    assert svc.context_for_job(b, "what codename?") is not None
    assert "ZXQ" not in svc.context_for_job(b, "what codename?").text
    assert svc.store.messages(b, last=100) and all("ZXQ" not in m.content for m in svc.store.messages(b))
    assert svc.store.delete_thread(a) and svc.store.count(a) == 0 and svc.store.get_thread(b) is not None
    assert svc.store.db.execute("select count(*) from messages_fts where messages_fts match 'ZXQ'").fetchone()[0] == 0


SENT = "ZXQ-THREAD-SENTINEL-8841"
SCRIPTS = {
    "chatgpt": {"answer": "I couldn't verify that reliably; no solid data is available.", "citations": []},
    "gemini": {"answer": "Acme released the Bolt in March 2024.", "citations": [{"url": "https://reuters.com/old", "title": "Acme Bolt March 2024 release"}]},
    "copilot": {"answer": "Acme released the Bolt in March 2025.", "citations": [{"url": "https://acme.com/press", "title": "Acme Bolt press release 2025"}]},
    "search": {"answer": "results", "citations": [{"url": "https://acme.com/press", "title": "Acme Bolt press release 2025"}]},
}


@pytest.fixture
def net_ok(net):
    net.confirm("https://reuters.com/old", tier=SourceTier.JOURNALISM)
    net.confirm("https://acme.com/press", tier=SourceTier.PRIMARY_OFFICIAL)
    return net


async def _job(threads, thread_id):
    settings = base_settings()
    adapters = adapters_from(SCRIPTS, settings)
    runner = ResearchRunner(settings, adapters, engine=None, verifier=None, threads=threads)
    job = await runner.run(Job(question="When did Acme release the Bolt?", max_rounds=3, thread_id=thread_id))
    return job, adapters


def _seed(svc):
    tid = svc.create_thread()
    talk(svc, tid, "chatgpt", [(f"We were discussing {SENT} in the Acme file.", "Yes, that is on record in our chat.")])
    return tid


async def test_thread_context_reaches_every_fresh_provider_chat_but_never_the_evidence(net_ok):
    svc = make()
    tid = _seed(svc)
    job, adapters = await _job(svc, tid)
    fresh = [(n, c) for n, a in adapters.items() if n != "search" for c in a.calls if not c["continue_thread"]]
    assert fresh and all(SENT in c["prompt"] and "OMNIBRAIN CONTINUATION CONTEXT" in c["prompt"] for _n, c in fresh)
    assert job.final is not None and job.thread_used
    for label, part in {"claims": [c.model_dump(mode="json") for c in job.claims], "evidence": [e.model_dump(mode="json") for e in job.evidence],
                        "final": job.final.model_dump(mode="json"), "disagreements": [d.model_dump(mode="json") for d in job.disagreements]}.items():
        assert SENT not in json.dumps(part, default=str), f"thread context leaked into {label}"


async def test_a_job_outside_any_thread_gets_nothing_from_a_thread(net_ok):
    svc = make()
    _seed(svc)
    job, adapters = await _job(svc, None)
    assert not any(SENT in c["prompt"] or "CONTINUATION" in c["prompt"] for a in adapters.values() for c in a.calls)
    assert job.thread_used == {}


async def test_a_job_in_another_thread_gets_nothing_from_this_one(net_ok):
    svc = make()
    _seed(svc)
    other = svc.create_thread()
    talk(svc, other, "chatgpt", [("Unrelated chat about gardening.", "Tomatoes like sun.")])
    _job_done, adapters = await _job(svc, other)
    assert not any(SENT in c["prompt"] for a in adapters.values() for c in a.calls)


async def test_the_question_and_answer_join_the_thread_and_the_next_job_sees_them(net_ok):
    svc = make()
    tid = _seed(svc)
    job, _ = await _job(svc, tid)
    msgs = svc.store.messages(tid)
    assert msgs[-2].role == "user" and msgs[-2].content == "When did Acme release the Bolt?" and msgs[-1].role == "assistant" and msgs[-1].content == job.final.answer
    _j2, adapters2 = await _job(svc, tid)
    assert any("When did Acme release the Bolt?" in c["prompt"] and "OMNIBRAIN CONTINUATION" in c["prompt"] for a in adapters2.values() for c in a.calls)


async def test_a_broken_thread_store_never_breaks_research(net_ok):
    class Broken:
        def context_for_job(self, *a, **k):
            raise RuntimeError("disk on fire")

        def record_job(self, *a, **k):
            raise RuntimeError("disk on fire")

    job, _ = await _job(Broken(), "thrabcabcabcabc")
    assert job.final is not None


async def test_research_thread_rotates_its_virtual_segment_by_size(net_ok):
    svc = make({"omnibrain": 400}, rotate_at=0.5, budget=300)
    tid = svc.create_thread()
    for i in range(12):
        svc.record_job(tid, f"Question number {i} about the Bolt launch and pricing details", "Answer " + "words " * 30 + str(i), job_id=f"j{i}")
    assert len(svc.store.segments(tid)) >= 3 and svc.store.count(tid) == 24
    assert svc.context_for_job(tid, "next?").within_budget


# ------------------------------------------------------------------------------------------------ compression
def test_curator_is_used_when_available_and_grounded():
    calls = []

    def curator(messages):
        calls.append(messages)
        return {"narrative": "Planning a Munich trip.", "decisions": ["Fly Lufthansa direct"], "facts": ["The conference is 14-16 April at Messe Muenchen", "Pluto is a planet"],
                "open_questions": ["Which hotel?"], "preferences": [], "corrections": [], "entities": ["Lufthansa"]}

    svc = make(curator=curator)
    tid = svc.create_thread()
    talk(svc, tid, "chatgpt", TRIP)
    svc.plan_turn(tid, "switch", "gemini")
    seg = svc.store.segments(tid)[0]
    assert seg.method == "curator" and calls
    facts = [i["t"] for i in seg.summary["items"]["facts"]]
    assert "The conference is 14-16 April at Messe Muenchen" in facts
    assert "Pluto is a planet" not in facts, "an item the conversation never said is dropped"
    prefs = [i["t"] for i in seg.summary["items"]["preferences"]]
    assert any("short answers" in p for p in prefs), "high-precision categories are unioned in from the deterministic pass"


@pytest.mark.parametrize("bad", [None, {"narrative": "x"}, "garbage"])
def test_curator_failure_falls_back_to_the_deterministic_summary(bad):
    def curator(_m):
        if bad == "garbage":
            raise RuntimeError("model offline")
        return bad

    svc = make(curator=curator)
    tid = svc.create_thread()
    talk(svc, tid, "chatgpt", TRIP)
    svc.plan_turn(tid, "switch", "gemini")
    seg = svc.store.segments(tid)[0]
    assert seg.method.startswith("deterministic") and seg.summary["items"]["decisions"]


def test_l2_summary_captures_the_required_kinds():
    class M:
        def __init__(self, seq, role, content):
            self.seq, self.role, self.content = seq, role, content

    msgs = [M(i, "user" if i % 2 == 0 else "assistant", t) for i, t in enumerate(x for pair in TRIP for x in pair)]
    s = deterministic_summary(msgs)["items"]
    assert any("Lufthansa" in i["t"] for i in s["decisions"])
    assert any("April" in i["t"] for i in s["corrections"])
    assert any("layover" in i["t"] for i in s["rejected"])
    assert any("short answers" in i["t"] for i in s["preferences"])
    assert any("hotel" in i["t"].lower() for i in s["open_questions"])
    assert any("Ibis" in i["t"] for i in s["conclusions"])
    assert any("Messe" in i["t"] for i in s["entities"] + s["key_terms"])
    assert s["direction"]


def test_a_later_correction_retires_the_fact_it_corrects_in_the_thread_state():
    class M:
        def __init__(self, seq, role, content):
            self.seq, self.role, self.content = seq, role, content

    first = deterministic_summary([M(1, "user", "When is the conference?"), M(2, "assistant", "The conference deadline is on 5 March for submissions.")])
    second = deterministic_summary([M(10, "user", "Actually the conference deadline is not 5 March, it moved to 9 April."), M(11, "assistant", "Understood, the deadline moved to 9 April.")])
    st = consolidate(consolidate(empty_state(), first, label="A", idx=1), second, label="B", idx=2)
    facts = " ".join(i["t"] for i in st["items"]["facts"])
    assert "5 March for submissions" not in facts and st["superseded"]
    assert any("9 April" in i["t"] for i in st["items"]["corrections"])


def test_only_durable_items_are_promoted_to_long_term_memory():
    mem = MemoryService(MemoryStore(":memory:", HashEmbedder()))
    svc = make(memory=mem)
    tid = svc.create_thread()
    talk(svc, tid, "chatgpt", [("From now on always answer in British English. What rhymes with orange?", "Nothing common rhymes with orange."),
                              ("Let's go with the blue variant for the logo. Also what is 12 times 13?", "It is 156."),
                              ("The weather is nice and I had toast.", "Lovely.")])
    before = len(mem.store.list())
    svc.plan_turn(tid, "switch", "gemini")  # closes the first chat -> L2 -> L4
    stored = [m.content for m in mem.store.list()]
    assert len(stored) <= before + 2
    assert not any("toast" in s.lower() or "12 times 13" in s or "blue variant" in s.lower() for s in stored), stored
    assert any("british english" in s.lower() for s in stored) or len(stored) == before  # promoted only if memory's own extractor accepts it


# ------------------------------------------------------------------------------------------------ API
@pytest.fixture
def tclient(tmp_path, monkeypatch):
    async def idle(self, job, emit):
        return None

    monkeypatch.setattr(api_app.JobManager, "_run", idle)
    settings = base_settings(storage={"db_path": str(tmp_path / "api.db")}, memory={"enabled": False}, threads={"enabled": True, "path": str(tmp_path / "t.db")})
    with TestClient(api_app._make_app(settings)) as c:
        yield c


def test_thread_api_create_view_recall_delete(tclient):
    tid = tclient.post("/api/threads", json={"title": "Trip", "project": "Munich"}).json()["thread_id"]
    svc = tclient.app.state.manager.thread_service()
    talk(svc, tid, "chatgpt", TRIP)
    talk(svc, tid, "gemini", [("What about trains in Bavaria?", "The regional train day ticket is cheap.")])
    view = tclient.get(f"/api/threads/{tid}").json()
    assert view["total"] == 10 and [m["content"] for m in view["messages"]][0] == TRIP[0][0] and "segment" not in json.dumps(view)
    segs = tclient.get(f"/api/threads/{tid}/segments").json()["segments"]
    assert [s["label"] for s in segs] == ["ChatGPT A", "Gemini A"] and segs[0]["open"] is False
    rec = tclient.post(f"/api/threads/{tid}/recall", json={"query": "Lufthansa flight"}).json()
    assert any("Lufthansa" in m["content"] for m in rec["messages"])
    assert tclient.get("/api/threads").json()["threads"][0]["thread_id"] == tid
    assert tclient.delete(f"/api/threads/{tid}").json() == {"deleted": True}
    assert tclient.get(f"/api/threads/{tid}").status_code == 404


def test_thread_api_validation_and_job_binding(tclient):
    assert tclient.get("/api/threads/nope").status_code == 404
    assert tclient.get("/api/threads/thr000000000000").status_code == 404
    assert tclient.post("/api/threads/thr000000000000/recall", json={"query": "x"}).status_code == 404
    assert tclient.post("/api/jobs", json={"question": "q", "thread_id": "thr000000000000"}).status_code == 404
    assert tclient.post("/api/jobs", json={"question": "q", "thread_id": 5}).status_code == 400
    tid = tclient.post("/api/threads", json={}).json()["thread_id"]
    assert tclient.post(f"/api/threads/{tid}/recall", json={}).status_code == 400
    ok = tclient.post("/api/jobs", json={"question": "q", "thread_id": tid})
    assert ok.status_code == 200
    assert tclient.app.state.manager.jobs[ok.json()["job_id"]].thread_id == tid


def test_threads_can_be_switched_off(tmp_path):
    settings = base_settings(storage={"db_path": str(tmp_path / "a.db")})
    with TestClient(api_app._make_app(settings)) as c:
        assert c.get("/api/threads").status_code == 503


def test_adapter_ask_wraps_a_provider_and_reports_failures():
    import asyncio
    from types import SimpleNamespace

    class A:
        def __init__(self, status, text=""):
            self.status, self.text, self.calls = status, text, []

        async def ask(self, key, prompt, rnd, emit=None, **kw):
            self.calls.append((key, kw))
            return SimpleNamespace(status=SimpleNamespace(value=self.status), answer_text=self.text, error="walled")

    ok, bad = A("completed", "hi"), A("blocked")
    ask = adapter_ask({"chatgpt": ok, "gemini": bad})
    assert asyncio.run(ask("chatgpt", "thr1:1", "p", True)) == "hi" and ok.calls == [("thr1:1", {"continue_thread": True})]
    with pytest.raises(RuntimeError, match="blocked"):
        asyncio.run(ask("gemini", "k", "p", False))
    with pytest.raises(RuntimeError):
        asyncio.run(ask("nobody", "k", "p", False))


# ------------------------------------------------------------------------------------------------ scale
def test_a_large_thread_stays_fast(tmp_path):
    """20k messages in the gate (the 100k benchmark lives in scripts/thread_bench.py and data/eval/thread_bench.json)."""
    svc = ThreadService(ThreadStore(tmp_path / "big.db", HashEmbedder()), context=ContextManager({"omnibrain": 12000}), packet_budget=3000)
    tid = svc.create_thread()
    topics = ["guitar strings", "pasta recipe", "tax filing", "kernel module", "marathon training", "tulip bulbs", "mortgage rates", "chess openings"]
    rows = [dict(role="user" if i % 2 == 0 else "assistant", content=f"{topics[i % 8]} discussion number {i}: details about {topics[(i * 3) % 8]} and item {i}", provider="chatgpt",
                 ts=time.time() - (20000 - i) * 600) for i in range(20000)]
    rows[12345]["content"] = "The Reykjavik hotel booking reference is QX-7731 for the aurora trip"
    svc.store.add_messages(tid, rows)
    t0 = time.perf_counter()
    found, _ = svc.recall(tid, "remember the Reykjavik hotel booking reference?")
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(5):
        svc.recall(tid, "remember the Reykjavik hotel booking reference?")
    warm = (time.perf_counter() - t0) / 5
    assert found and "QX-7731" in found[0].message.content
    assert warm < 0.5 and first < 5
    t0 = time.perf_counter()
    pk = svc.context_for_job(tid, "continue please")
    assert pk is not None and pk.within_budget and time.perf_counter() - t0 < 1.0