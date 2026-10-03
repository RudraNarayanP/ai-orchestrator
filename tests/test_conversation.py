"""Conversation memory (backlog item 3).

A thread of questions is no longer a set of islands: "and how did they build it?"
is resolved against the turn before it, inherits that turn's stakes, reaches the
providers with the earlier question and answer in front of it, and survives a
reload through the store and the API.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend.api import app as api_app
from backend.models import (
    Claim,
    ConversationTurn,
    EscalationLevel,
    Job,
    JobStatus,
    ResearchMode,
    SourceTier,
    StakesDomain,
)
from backend.orchestrator.runner import ResearchRunner
from backend.research import memory, router
from backend.research.prompts import research_prompt
from backend.storage.db import Store
from backend.verification.llm import Endpoint
from tests.conftest import adapters_from, base_settings

PRIOR = ConversationTurn(
    job_id="job_1",
    question="Who built the Acme Bolt?",
    answer="Acme Corp built the Acme Bolt in its Austin factory.",
    confirmed_claims=["Acme Corp built the Acme Bolt in March 2026."],
    confidence="moderate",
)


# --------------------------------------------------------------------- detection


@pytest.mark.parametrize(
    "question",
    [
        "And how did they build it?",
        "what about the price?",
        "Why did that happen?",
        "how much is it?",
        "and Tesla?",
        "What about Tesla's version?",
        "Really?",
    ],
)
def test_short_dependent_questions_are_follow_ups(question):
    assert memory.is_follow_up(question, PRIOR), question


@pytest.mark.parametrize(
    "question",
    [
        "Who won the 2018 World Cup?",
        "How does photosynthesis work in desert plants?",
        "Why is the sky blue?",
        "What is the capital of Australia and when was it founded?",
        "Is Mount Everest taller than K2?",
    ],
)
def test_self_contained_questions_in_the_same_thread_are_not_follow_ups(question):
    assert not memory.is_follow_up(question, PRIOR), question


def test_no_history_means_never_a_follow_up():
    assert not memory.is_follow_up("And how did they build it?", None)
    analysis = router.classify("And how did they build it?")
    assert analysis.follow_up is False and analysis.standalone_question is None


def test_classify_resolves_a_follow_up_against_the_previous_turn():
    analysis = router.classify("And how did they build it?", history=[PRIOR])
    assert analysis.follow_up is True
    assert "Acme Corp" in analysis.inherited_entities and "Acme Bolt" in analysis.inherited_entities
    assert "Who built the Acme Bolt" in analysis.standalone_question
    assert "How did they build it" in analysis.standalone_question.replace("And h", "H")
    assert "Acme Bolt" in analysis.key_entities


def test_a_follow_up_inherits_stakes_and_time_sensitivity():
    prior = ConversationTurn(job_id="j", question="What dose of ibuprofen is safe for children right now?", answer="x")
    followed = router.classify("and for adults?", history=[prior])
    assert followed.follow_up and followed.high_stakes and followed.stakes == StakesDomain.MEDICAL
    assert followed.starting_level == EscalationLevel.DEEP
    assert followed.needs_current_data
    alone = router.classify("and for adults?")
    assert not alone.high_stakes, "without the thread the same words are casual"


def test_a_follow_up_never_takes_the_level_zero_shortcut():
    prior = ConversationTurn(job_id="j", question="Who built the Acme Bolt?", answer="Acme Corp.")
    analysis = router.classify("What is the boiling point of water?", history=[prior], llm_stable_answer="100 C")
    assert not analysis.follow_up  # self-contained -> normal routing, shortcut allowed
    follow = router.classify("and what is it for?", history=[prior], llm_stable_answer="Something.")
    assert follow.follow_up and not follow.can_answer_directly


def test_arithmetic_is_still_computed_inside_a_thread():
    analysis = router.classify("what is 12 * 12?", history=[PRIOR])
    assert analysis.can_answer_directly and analysis.direct_answer == "144"


# ----------------------------------------------------------------------- prompts


def test_history_block_carries_question_answer_and_confirmed_claims():
    block = memory.history_block([PRIOR])
    assert "Earlier in this conversation" in block
    assert "Who built the Acme Bolt?" in block and "Austin factory" in block
    assert "Confirmed so far: Acme Corp built the Acme Bolt in March 2026." in block


def test_history_block_does_not_present_an_unsettled_answer_as_one():
    idk = ConversationTurn(job_id="j", question="What is the Zorb price?", answer="I don't know. I couldn't verify it.")
    block = memory.history_block([idk])
    assert "I don't know" not in block and "not settled" in block


def test_history_block_is_bounded():
    turns = [ConversationTurn(job_id=f"j{i}", question="q" * 900, answer="a" * 900) for i in range(8)]
    assert len(memory.history_block(turns)) < 3500


def test_research_prompt_puts_the_history_before_the_question_and_keeps_the_guard():
    prompt = research_prompt("And how did they build it?", provider="chatgpt", history=[PRIOR])
    assert prompt.index("Earlier in this conversation") < prompt.index("And how did they build it?")
    assert "Austin factory" in prompt
    assert "untrusted" in prompt.lower() or "evidence" in prompt.lower()
    assert "Earlier in this conversation" not in research_prompt("Who built it?", provider="chatgpt")


# ------------------------------------------------------------------- end to end


async def _run_turn(settings, net, question, scripts, history=(), conversation_id="conv_a"):
    adapters = adapters_from(scripts, settings)
    job = Job(question=question, mode=ResearchMode.STANDARD, max_rounds=1, conversation_id=conversation_id)
    job.history = list(history)
    job = await ResearchRunner(settings, adapters, engine=None).run(job)
    return job, adapters


async def test_two_turn_thread_second_question_sees_the_first(settings, net):
    first = await _finished_job(settings, net, "Who built the Acme Bolt?", "conv_a")
    assert "Acme Corp built the Acme Bolt in March 2026" in first.final.answer, first.final.answer
    turn = memory.turn_from_job(first)
    assert turn.confirmed_claims == ["Acme Corp built the Acme Bolt in March 2026."]
    assert turn is not None and turn.question == "Who built the Acme Bolt?"
    assert turn.answer == first.final.answer and turn.job_id == first.id

    second_scripts = {
        "chatgpt": {"answer": "KEY CLAIMS\n1. Acme Corp built it by hand.", "citations": []},
        "search": {"answer": "r", "citations": []},
    }
    second, adapters = await _run_turn(settings, net, "And how did they build it?", second_scripts, history=[turn])

    assert second.analysis.follow_up is True
    assert "Acme" in " ".join(second.analysis.inherited_entities)
    sent = adapters["chatgpt"].calls[0]["prompt"]
    assert "Who built the Acme Bolt?" in sent, "the earlier question must reach the provider"
    assert first.final.answer.splitlines()[0][:40] in sent, "so must the answer we gave"
    assert "Confirmed so far: Acme Corp built the Acme Bolt in March 2026." in sent
    assert sent.index("Earlier in this conversation") < sent.index("And how did they build it?")


async def test_self_contained_question_in_a_thread_is_researched_as_itself(settings, net):
    scripts = {"chatgpt": {"answer": "KEY CLAIMS\n1. Canberra is the capital of Australia.", "citations": []}, "search": {"answer": "r", "citations": []}}
    job, adapters = await _run_turn(settings, net, "What is the capital of Australia?", scripts, history=[PRIOR])
    assert job.analysis.follow_up is False
    assert "Earlier in this conversation" not in adapters["chatgpt"].calls[0]["prompt"]


async def test_turn_from_unfinished_job_is_nothing():
    assert memory.turn_from_job(Job(question="x")) is None


async def test_analysis_model_rewrites_the_follow_up(fake_openai, settings):
    fake_openai.script('{"standalone": "How did Acme Corp build the Acme Bolt?"}')
    endpoint = Endpoint(provider="openai_compatible", model="m", base_url=fake_openai.base_url)
    runner = ResearchRunner(settings, adapters_from({"chatgpt": {"answer": ""}}, settings), engine=None, analysis_endpoint=endpoint)
    analysis = await runner._analyze("And how did they build it?", ResearchMode.STANDARD, ["chatgpt"], history=[PRIOR])
    assert analysis.follow_up and analysis.standalone_question == "How did Acme Corp build the Acme Bolt?"
    # only the rewrite was asked -- no "can you answer from memory" shortcut for a follow-up
    assert len(fake_openai.requests) == 1


async def test_a_bad_rewrite_falls_back_to_the_heuristic(fake_openai, settings):
    fake_openai.script("not json at all")
    endpoint = Endpoint(provider="openai_compatible", model="m", base_url=fake_openai.base_url)
    runner = ResearchRunner(settings, adapters_from({"chatgpt": {"answer": ""}}, settings), engine=None, analysis_endpoint=endpoint)
    analysis = await runner._analyze("And how did they build it?", ResearchMode.STANDARD, ["chatgpt"], history=[PRIOR])
    assert analysis.follow_up and "Who built the Acme Bolt" in analysis.standalone_question


# ------------------------------------------------------------------------- store


def _settings_in(tmp_path):
    return base_settings(storage={"db_path": str(tmp_path / "omni.db")})


async def _finished_job(settings, net, question, conversation_id):
    net.confirm("https://acme.example/about", tier=SourceTier.PRIMARY_OFFICIAL, excerpt="Acme Corp built the Acme Bolt in March 2026.")
    net.confirm("https://news.example/bolt", tier=SourceTier.JOURNALISM)
    scripts = {
        "chatgpt": {
            "answer": "DIRECT ANSWER\nAcme Corp built the Acme Bolt in March 2026.\n\nKEY CLAIMS\n1. Acme Corp built the Acme Bolt in March 2026.",
            "citations": [
                {"url": "https://acme.example/about", "title": "Acme Corp built the Acme Bolt in March 2026"},
                {"url": "https://news.example/bolt", "title": "Acme Corp built the Acme Bolt in March 2026"},
            ],
        },
        "search": {"answer": "r", "citations": []},
    }
    job, _ = await _run_turn(settings, net, question, scripts, conversation_id=conversation_id)
    return job


async def test_store_round_trips_a_thread_and_isolates_threads(tmp_path, net):
    settings = _settings_in(tmp_path)
    store = Store(settings)
    a1 = await _finished_job(settings, net, "Who built the Acme Bolt?", "conv_a")
    a2 = await _finished_job(settings, net, "And how?", "conv_a")
    b1 = await _finished_job(settings, net, "Something else entirely", "conv_b")
    for job in (a1, a2, b1):
        store.save_job(job)
    turns = store.conversation_turns("conv_a")
    assert [t.job_id for t in turns] == [a1.id, a2.id]
    assert turns[0].question == "Who built the Acme Bolt?" and turns[0].answer == a1.final.answer
    assert [t.job_id for t in store.conversation_turns("conv_a", exclude_job_id=a2.id)] == [a1.id]
    assert [t.job_id for t in store.conversation_turns("conv_b")] == [b1.id]
    assert store.conversation_turns("") == [] and store.conversation_turns("nope") == []
    assert {j["conversation_id"] for j in store.list_jobs()} == {"conv_a", "conv_b"}
    assert len(store.conversation_turns("conv_a", limit=1)) == 1


async def test_unfinished_jobs_are_not_part_of_the_thread(tmp_path):
    store = Store(_settings_in(tmp_path))
    job = Job(question="still running", conversation_id="conv_a")
    job.status = JobStatus.RESEARCHING if hasattr(JobStatus, "RESEARCHING") else JobStatus.PENDING
    store.save_job(job)
    assert store.conversation_turns("conv_a") == []


async def test_saving_a_job_twice_does_not_duplicate_its_reports(tmp_path, net):
    """Regression: reports were INSERTed on every save, so each save added duplicate rows."""
    settings = _settings_in(tmp_path)
    store = Store(settings)
    job = await _finished_job(settings, net, "Who built the Acme Bolt?", "conv_a")
    assert job.reports
    for _ in range(3):
        store.save_job(job)
    count = store._conn.execute("SELECT COUNT(*) FROM reports WHERE job_id=?", (job.id,)).fetchone()[0]
    assert count == len(job.reports)


async def test_evidence_polarity_is_updated_on_resave(tmp_path, net):
    """Regression: evidence rows kept their first polarity forever."""
    from backend.models import Evidence, SourceCheckStatus

    store = Store(_settings_in(tmp_path))
    job = Job(question="q", conversation_id="conv_a")
    ev = Evidence(job_id=job.id, url="https://a.example/x", domain="a.example", tier=SourceTier.JOURNALISM,
                  polarity="support", check_status=SourceCheckStatus.CONFIRMED)
    job.evidence.append(ev)
    store.save_job(job)
    ev.polarity = "refute"
    store.save_job(job)
    row = store._conn.execute("SELECT polarity FROM evidence WHERE id=?", (ev.id,)).fetchone()
    assert row[0] == "refute"


def test_an_old_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, question TEXT, mode TEXT, status TEXT, created_at REAL, "
                 "updated_at REAL, finished_at REAL, level INTEGER, rounds_run INTEGER, max_rounds INTEGER, stop_reason TEXT, "
                 "browser_sessions INTEGER, verifier_calls INTEGER, final_json TEXT, analysis_json TEXT, plan_json TEXT, "
                 "error TEXT, answer_text TEXT, confidence TEXT)")
    conn.execute("INSERT INTO jobs (id, question) VALUES ('job_old', 'old question')")
    conn.commit()
    conn.close()
    store = Store(base_settings(storage={"db_path": str(path)}))
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(jobs)").fetchall()}
    assert {"conversation_id", "confirmed_json"} <= cols
    assert store.list_jobs()[0]["id"] == "job_old"
    Store(base_settings(storage={"db_path": str(path)}))  # idempotent


# --------------------------------------------------------------------------- API


@pytest.fixture
def client(tmp_path, monkeypatch):
    async def idle(self, job, emit):  # never start a browser
        return None

    monkeypatch.setattr(api_app.JobManager, "_run", idle)
    app = api_app._make_app(_settings_in(tmp_path))
    with TestClient(app) as c:
        c.app_ = app
        yield c


def test_api_assigns_a_conversation_id_and_reuses_a_given_one(client):
    first = client.post("/api/jobs", json={"question": "Who built the Acme Bolt?"}).json()
    assert first["conversation_id"].startswith("conv_")
    second = client.post("/api/jobs", json={"question": "And how?", "conversation_id": first["conversation_id"]}).json()
    assert second["conversation_id"] == first["conversation_id"]
    assert client.post("/api/jobs", json={"question": "q"}).json()["conversation_id"] != first["conversation_id"]


def test_api_rejects_a_malformed_conversation_id(client):
    res = client.post("/api/jobs", json={"question": "q", "conversation_id": "../../etc"})
    assert res.status_code == 400
    assert client.get("/api/conversations/bad id").status_code in (400, 404)


async def test_api_loads_the_thread_history_into_the_job(client, net):
    store = client.app_.state.store
    settings = client.app_.state.settings
    done = await _finished_job(settings, net, "Who built the Acme Bolt?", "conv_x")
    store.save_job(done)
    res = client.post("/api/jobs", json={"question": "And how did they build it?", "conversation_id": "conv_x"}).json()
    job = client.app_.state.manager.jobs[res["job_id"]]
    assert [t.job_id for t in job.history] == [done.id]
    thread = client.get("/api/conversations/conv_x").json()
    assert thread["conversation_id"] == "conv_x"
    assert [t["question"] for t in thread["turns"]] == ["Who built the Acme Bolt?"]
    assert client.get("/api/conversations/conv_unknown").json()["turns"] == []