"""The thread API over real HTTP (a uvicorn server on a socket, httpx client), including POST /api/threads/{id}/chat with fake provider adapters."""

from __future__ import annotations

import socket
import sqlite3
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from backend.api import app as api_app
from backend.memory.embed import HashEmbedder
from backend.thread.store import ThreadStore
from tests.conftest import base_settings


class FakeAdapter:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, []

    async def ask(self, key, prompt, round_no, emit=None, **kw):
        self.calls.append({"key": key, "prompt": prompt, "continue": bool(kw.get("continue_thread"))})
        if self.fail:
            return SimpleNamespace(status=SimpleNamespace(value="blocked"), answer_text="", error="needs sign-in")
        return SimpleNamespace(status=SimpleNamespace(value="completed"), answer_text=f"{self.name} reply {len(self.calls)}: noted.", error=None)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def adapters():
    return {"chatgpt": FakeAdapter("chatgpt"), "gemini": FakeAdapter("gemini"), "copilot": FakeAdapter("copilot", fail=True)}


def start_server(tmp_path, monkeypatch, adapters, **extra):
    async def idle(self, job, emit):
        return None

    async def no_engine(self):
        return None

    class Catalog:
        def __init__(self, settings, engine):
            pass

        def all(self):
            return adapters

    monkeypatch.setattr(api_app.JobManager, "_run", idle)
    monkeypatch.setattr(api_app.JobManager, "engine_get", no_engine)
    monkeypatch.setattr(api_app, "ProviderCatalog", Catalog)
    cfg = dict(storage={"db_path": str(tmp_path / "h.db")}, threads={"enabled": True, "path": str(tmp_path / "t.db"), "rotate_at": 0.5, "packet_budget_tokens": 400,
                                                                     "limits": {"chatgpt": 300, "gemini": 100000}}, **extra)
    app = api_app._make_app(base_settings(**cfg))
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    deadline = time.time() + 15
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    assert srv.started
    return app, srv, th, f"http://127.0.0.1:{port}"


@pytest.fixture
def server(tmp_path, monkeypatch, adapters):
    app, srv, th, base = start_server(tmp_path, monkeypatch, adapters)
    with httpx.Client(base_url=base, timeout=20) as c:
        yield c
    srv.should_exit = True
    th.join(timeout=10)


def make(c, **kw):
    r = c.post("/api/threads", json=kw)
    assert r.status_code == 200, r.text
    return r.json()["thread_id"]


def chat(c, tid, text, provider="chatgpt"):
    return c.post(f"/api/threads/{tid}/chat", json={"text": text, "provider": provider})


def test_chat_over_http_keeps_one_thread_and_continues_the_same_chat(server, adapters):
    tid = make(server, title="Trip", project="Munich")
    a = chat(server, tid, "We are planning a Munich trip in April.").json()
    b = chat(server, tid, "I prefer window seats.").json()
    assert a["reply"].startswith("chatgpt reply 1") and a["chat"] == "ChatGPT A" and a["rotated"] is False and a["provider"] == "chatgpt"
    assert b["context"]["continued_chat"] is True and b["rotated"] is False
    calls = adapters["chatgpt"].calls
    assert calls[0]["key"] == calls[1]["key"] and calls[0]["continue"] is False and calls[1]["continue"] is True
    view = server.get(f"/api/threads/{tid}").json()
    assert [m["role"] for m in view["messages"]] == ["user", "assistant", "user", "assistant"] and view["total"] == 4
    assert view["messages"][1]["provider"] == "chatgpt" and view["messages"][0]["context"]["chat"] == "ChatGPT A"
    segs = server.get(f"/api/threads/{tid}/segments").json()
    assert [s["label"] for s in segs["segments"]] == ["ChatGPT A"] and segs["segments"][0]["open"] and segs["segments"][0]["limit"] == 300


def test_rotation_is_visible_over_http_with_its_context(server, adapters):
    tid = make(server)
    long = "Let's keep going about the Munich trip details: hotels near the Messe, trains from the airport, and the conference schedule. "
    rotated = None
    for i in range(12):
        out = chat(server, tid, f"{long}(message {i})").json()
        if out["rotated"]:
            rotated = out
            break
    assert rotated is not None, "a tiny limit must rotate within a dozen messages"
    assert rotated["reason"] == "context_limit" and rotated["chat"] == "ChatGPT B" and rotated["context"]["packet_tokens"] > 0 and rotated["context"]["continued_chat"] is False
    calls = adapters["chatgpt"].calls
    assert calls[-1]["key"] != calls[0]["key"] and calls[-1]["prompt"].startswith("OMNIBRAIN CONTINUATION CONTEXT") and calls[-1]["continue"] is False
    segs = server.get(f"/api/threads/{tid}/segments").json()["segments"]
    assert segs[0]["open"] is False and segs[0]["method"].startswith("deterministic") and segs[-1]["open"] and segs[-1]["reason"] == "context_limit"
    msgs = server.get(f"/api/threads/{tid}").json()["messages"]
    flagged = [m for m in msgs if m["role"] == "user" and m["context"]["rotated"]]
    assert len(flagged) == 1 and flagged[0]["context"]["reason"] == "context_limit"
    assert len(msgs) == 2 * (i + 1), "every message is still in the one thread"


def test_provider_switch_over_http_gives_the_new_provider_the_thread(server, adapters):
    tid = make(server)
    chat(server, tid, "My codeword for this thread is TANGERINE-42 and I prefer window seats.")
    out = chat(server, tid, "Which codeword did I give you?", provider="gemini").json()
    assert out["rotated"] and out["reason"] == "provider_switch" and out["chat"] == "Gemini A"
    sent = adapters["gemini"].calls[0]["prompt"]
    assert "TANGERINE-42" in sent and "OMNIBRAIN CONTINUATION CONTEXT" in sent and sent.rstrip().endswith("Which codeword did I give you?")
    labels = [s["label"] for s in server.get(f"/api/threads/{tid}/segments").json()["segments"]]
    assert labels == ["ChatGPT A", "Gemini A"]


def test_chat_validation_and_failures_over_http(server, adapters):
    tid = make(server)
    assert chat(server, "thr000000000000", "hi").status_code == 404
    assert server.post(f"/api/threads/{tid}/chat", json={"provider": "chatgpt"}).status_code == 400
    assert server.post(f"/api/threads/{tid}/chat", json={"text": "  ", "provider": "chatgpt"}).status_code == 400
    assert server.post(f"/api/threads/{tid}/chat", json={"text": "hi"}).status_code == 400
    assert chat(server, tid, "hi", provider="nobody").status_code == 400
    assert chat(server, tid, "x" * 5000).status_code == 400
    bad = chat(server, tid, "Please do not lose this message", provider="copilot")
    assert bad.status_code == 502 and "needs sign-in" in bad.json()["detail"]
    msgs = server.get(f"/api/threads/{tid}").json()["messages"]
    assert [m["content"] for m in msgs] == ["Please do not lose this message"], "the user's message is saved even when the provider fails"
    assert "needs sign-in" in msgs[0]["context"]["unanswered"], "and it is marked as not answered, with the reason"


def test_personal_memory_used_is_reported_in_the_turn_context(tmp_path, monkeypatch, adapters):
    app, srv, th, base = start_server(tmp_path, monkeypatch, adapters, memory={"enabled": True, "path": str(tmp_path / "m.db"), "embedder": "hash"})
    try:
        with httpx.Client(base_url=base, timeout=20) as c:
            assert c.post("/api/memory", json={"content": "Lives in Delhi and flies from Delhi airport"}).status_code == 200
            tid = make(c)
            chat(c, tid, "We are planning a trip to Munich.")
            out = chat(c, tid, "Where do I live and which airport do I use?", provider="gemini").json()
            assert out["context"]["memory_used"] == ["Lives in Delhi and flies from Delhi airport"]
            msgs = c.get(f"/api/threads/{tid}").json()["messages"]
            assert msgs[2]["context"]["memory_used"] == ["Lives in Delhi and flies from Delhi airport"]
            assert "Delhi airport" in adapters["gemini"].calls[0]["prompt"]
    finally:
        srv.should_exit = True
        th.join(timeout=10)


def test_threads_do_not_see_each_other_over_http(server, adapters):
    a, b = make(server, title="A"), make(server, title="B")
    chat(server, a, "The secret codename is ZXQ-ALPHA-9.")
    chat(server, b, "Let's talk about tomatoes.")
    chat(server, b, "And what did we say before?", provider="gemini")
    assert "ZXQ" not in adapters["gemini"].calls[0]["prompt"]
    assert server.post(f"/api/threads/{b}/recall", json={"query": "ZXQ-ALPHA-9 codename"}).json()["messages"] == []
    titles = {t["title"]: t["messages"] for t in server.get("/api/threads").json()["threads"]}
    assert titles == {"A": 2, "B": 4}


def test_a_thread_file_from_before_context_was_stored_still_opens(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript(
        "create table threads(thread_id text primary key, title text, project text, created_at real, updated_at real, state_json text default '{}', active_segment text, embedder text, next_seq integer default 1);"
        "create table segments(segment_id text primary key, thread_id text not null, idx integer, provider text, provider_key text, opened_at real, closed_at real, first_seq integer, last_seq integer, tokens integer default 0, summary_json text, method text, reason text);"
        "create table messages(msg_id integer primary key autoincrement, thread_id text not null, seq integer not null, segment_id text, role text, content text, provider text, ts real, tokens integer, job_id text, emb blob);")
    db.execute("insert into threads(thread_id,title,created_at,updated_at,embedder,next_seq) values('throld',?,1,1,'hash-v1',2)", ("old",))
    db.execute("insert into messages(thread_id,seq,role,content,provider,ts,tokens) values('throld',1,'user','hello from before',?,1,3)", ("chatgpt",))
    db.commit()
    db.close()
    store = ThreadStore(path, HashEmbedder())
    msgs = store.messages("throld")
    assert msgs[0].content == "hello from before" and msgs[0].meta == {} and "context" not in msgs[0].public()
    store.add_message("throld", "user", "and after", meta={"chat": "ChatGPT A"})
    assert store.messages("throld")[-1].public()["context"]["chat"] == "ChatGPT A"