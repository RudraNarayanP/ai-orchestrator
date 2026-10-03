"""HTTP API coverage (backlog item 6): masking, validation, SSE, audit shape, doctor.

Nothing here starts a browser: JobManager._run is stubbed, settings and the SQLite
store live in a tmp dir, and config saves are redirected so the user's own
config/settings.yaml is never written.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.api import app as api_app
from backend.models import JobEvent
from backend.settings import Settings
from tests.conftest import base_settings
from tests.test_conversation import _finished_job

SECRET = "sk-live-1234567890-SECRET"


@pytest.fixture
def saved(monkeypatch):
    """Capture config saves instead of writing config/settings.yaml."""
    box: dict[str, Settings] = {}
    def fake_save(settings, path=None):
        box["last"] = settings

    monkeypatch.setattr(api_app, "save_settings", fake_save)
    monkeypatch.setattr(api_app, "reload_settings", lambda: None)
    return box


@pytest.fixture
def client(tmp_path, monkeypatch, saved):
    async def idle(self, job, emit):
        return None

    monkeypatch.setattr(api_app.JobManager, "_run", idle)
    settings = base_settings(
        storage={"db_path": str(tmp_path / "api.db")},
        verifier={"provider": "openai_compatible", "model": "qwen", "base_url": "http://127.0.0.1:9/v1", "api_key": SECRET},
        analysis={"provider": "openai_compatible", "model": "small", "base_url": "http://127.0.0.1:9/v1", "api_key": SECRET},
        vision={"provider": "openai_compatible", "model": "vis", "base_url": "http://127.0.0.1:9/v1", "api_key": SECRET},
    )
    app = api_app._make_app(settings)
    with TestClient(app) as c:
        c.app_ = app
        yield c


# ------------------------------------------------------------------ /api/config


def test_get_config_never_contains_an_api_key(client):
    res = client.get("/api/config")
    assert res.status_code == 200
    assert SECRET not in res.text
    data = res.json()
    for section in ("verifier", "analysis", "vision"):
        assert data[section]["api_key"] == "***configured***", section
    assert "STANDARD" in data["modes"] and "DIRECT" in data["levels"]


def test_patch_config_keeps_the_stored_key_for_masked_or_empty_input_and_never_echoes_it(client, saved):
    for incoming in ("***configured***", "", None):
        body = {"verifier": {"model": "other-model"}}
        if incoming is not None:
            body["verifier"]["api_key"] = incoming
        res = client.post("/api/config", json=body)
        assert res.status_code == 200 and SECRET not in res.text, incoming
        assert saved["last"].verifier.api_key == SECRET, f"stored key lost for {incoming!r}"
        assert saved["last"].verifier.model == "other-model"
        assert res.json()["config"]["verifier"]["api_key"] == "***configured***"


def test_patch_config_masks_the_vision_and_analysis_keys_too(client, saved):
    """Regression: only verifier/analysis were protected, so saving the form wiped the vision key."""
    res = client.post("/api/config", json={"vision": {"model": "vis-2", "api_key": "***configured***"}, "analysis": {"api_key": ""}})
    assert res.status_code == 200
    assert saved["last"].vision.api_key == SECRET and saved["last"].vision.model == "vis-2"
    assert saved["last"].analysis.api_key == SECRET
    assert SECRET not in res.text


def test_patch_config_accepts_a_new_key_but_still_does_not_echo_it(client, saved):
    res = client.post("/api/config", json={"verifier": {"api_key": "sk-NEW-KEY-999"}})
    assert saved["last"].verifier.api_key == "sk-NEW-KEY-999"
    assert "sk-NEW-KEY-999" not in res.text
    assert client.get("/api/config").json()["verifier"]["api_key"] == "***configured***"


def test_patch_config_rejects_bad_input_without_saving(client, saved):
    bad = client.post("/api/config", json={"research": {"max_rounds": "lots"}})
    assert bad.status_code == 400
    assert client.post("/api/config", json=[1, 2]).status_code == 400
    assert client.post("/api/config", content=b"not json", headers={"content-type": "application/json"}).status_code == 400
    assert client.post("/api/config", json={"verifier": "oops"}).status_code == 400
    assert "last" not in saved


# ---------------------------------------------------------------- job validation


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({}, "question is required"),
        ({"question": "   "}, "question is required"),
        ({"question": 42}, "must be a string"),
        ({"question": "q" * 4001}, "too long"),
        ({"question": "q", "mode": "TURBO"}, "unknown mode"),
        ({"question": "q", "mode": 3}, "mode must be a string"),
        ({"question": "q", "max_rounds": "abc"}, "max_rounds"),
        ({"question": "q", "max_rounds": 0}, "max_rounds"),
        ({"question": "q", "max_rounds": 99}, "max_rounds"),
        ({"question": "q", "max_rounds": True}, "max_rounds"),
        ({"question": "q", "conversation_id": "../x"}, "conversation_id"),
        ({"question": "q", "conversation_id": 7}, "conversation_id"),
    ],
)
def test_job_validation_errors_are_400s_with_a_reason(client, body, fragment):
    res = client.post("/api/jobs", json=body)
    assert res.status_code == 400, (body, res.status_code, res.text)
    assert fragment in res.json()["detail"]


def test_malformed_job_bodies_are_400_not_500(client):
    assert client.post("/api/jobs", content=b"{broken", headers={"content-type": "application/json"}).status_code == 400
    assert client.post("/api/jobs", json=["question"]).status_code == 400


def test_a_valid_job_is_accepted_and_listed(client):
    res = client.post("/api/jobs", json={"question": "Who built the Acme Bolt?", "mode": "quick", "max_rounds": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["job_id"].startswith("job_") and data["status"] == "pending" and data["conversation_id"].startswith("conv_")
    job = client.app_.state.manager.jobs[data["job_id"]]
    assert job.mode.value == "QUICK" and job.max_rounds == 2
    listed = client.get("/api/jobs").json()
    assert [j["id"] for j in listed] == [data["job_id"]]


# ------------------------------------------------------------------- SSE + replay


def _publish(client, job_id, kinds):
    broker = client.app_.state.broker
    return [broker.publish(job_id, JobEvent(kind=k, message=f"{k} {i}")) for i, k in enumerate(kinds)]


def _sse(client, job_id):
    records = []
    with client.stream("GET", f"/api/stream/{job_id}") as res:
        assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
        for line in res.iter_lines():
            if line.startswith("data: "):
                records.append(json.loads(line[6:]))
    return records


def test_sse_replays_a_finished_job_in_order_and_closes(client):
    """Regression: replaying a job whose 'done' was already published used to hang the stream open."""
    _publish(client, "job_s", ["status", "provider", "evidence", "final", "done"])
    records = _sse(client, "job_s")
    assert [r["kind"] for r in records] == ["status", "provider", "evidence", "final", "done"]
    assert [r["seq"] for r in records] == [1, 2, 3, 4, 5]


def test_sse_delivers_live_events_after_the_replay_until_done(client):
    import threading
    import time

    _publish(client, "job_live", ["status"])
    got: list[dict] = []
    t = threading.Thread(target=lambda: got.extend(_sse(client, "job_live")))
    t.start()
    time.sleep(0.3)
    broker = client.app_.state.broker
    # publish from the app's own loop so the subscriber queue is woken correctly
    async def late():
        broker.publish("job_live", JobEvent(kind="provider", message="late"))
        broker.publish("job_live", JobEvent(kind="done", message="completed"))

    client.portal.call(late)
    t.join(timeout=10)
    assert not t.is_alive(), "stream did not end after done"
    assert [r["kind"] for r in got] == ["status", "provider", "done"]
    assert [r["seq"] for r in got] == [1, 2, 3]


def test_events_endpoint_replays_after_a_cursor(client):
    _publish(client, "job_e", ["status", "provider", "evidence", "done"])
    everything = client.get("/api/jobs/job_e/events").json()
    assert [e["seq"] for e in everything["events"]] == [1, 2, 3, 4] and "server_time" in everything
    tail = client.get("/api/jobs/job_e/events", params={"after": 2}).json()["events"]
    assert [e["seq"] for e in tail] == [3, 4] and [e["kind"] for e in tail] == ["evidence", "done"]
    assert client.get("/api/jobs/job_e/events", params={"after": 4}).json()["events"] == []
    assert client.get("/api/jobs/job_nope/events").json()["events"] == []


# ------------------------------------------------------------------- audit shape


async def test_raw_audit_has_the_whole_trail_for_a_live_job(client, net):
    settings = client.app_.state.settings
    job = await _finished_job(settings, net, "Who built the Acme Bolt?", "conv_raw")
    client.app_.state.manager.jobs[job.id] = job
    res = client.get(f"/api/jobs/{job.id}/raw")
    assert res.status_code == 200
    raw = res.json()
    for key in ("id", "question", "status", "analysis", "responses", "claims", "evidence", "reports", "final", "escalation_log", "assessments"):
        assert key in raw, key
    response = raw["responses"][0]
    assert response["provider"] == "chatgpt" and "prompt" in response and response["raw_text"]
    ev = raw["evidence"][0]
    assert {"url", "domain", "tier", "polarity", "check_status", "origin"} <= set(ev)
    assert raw["claims"][0]["claim"] and raw["final"]["answer"] == job.final.answer
    assert raw["reports"][0]["verdicts"][0]["verdict"] in {"supported", "partially_supported"}


async def test_raw_audit_falls_back_to_the_store_for_a_finished_job_not_in_memory(client, net):
    settings = client.app_.state.settings
    job = await _finished_job(settings, net, "Who built the Acme Bolt?", "conv_raw2")
    client.app_.state.store.save_job(job)  # not in manager.jobs
    raw = client.get(f"/api/jobs/{job.id}/raw").json()
    assert raw["id"] == job.id and raw["final"]["answer"] == job.final.answer
    assert raw["responses"] and raw["evidence"] and raw["claims"] and raw["reports"]
    assert {"prompt", "raw_text", "citations"} <= set(raw["responses"][0])
    assert len(raw["reports"]) == len(job.reports), "reports must not duplicate across saves"
    snapshot = client.get(f"/api/jobs/{job.id}").json()
    assert snapshot["id"] == job.id and snapshot["final"]["answer"] == job.final.answer


def test_unknown_job_is_404_everywhere(client):
    assert client.get("/api/jobs/job_nope").status_code == 404
    assert client.get("/api/jobs/job_nope/raw").status_code == 404


def test_cancel_of_an_unknown_job_is_a_clean_false(client):
    assert client.post("/api/jobs/job_nope/cancel").json() == {"cancelled": False}


# ---------------------------------------------------------------------- doctor


def test_doctor_shape_with_nothing_reachable(client, monkeypatch):
    async def nothing(self):
        return []

    monkeypatch.setattr(api_app.LLMClient, "list_models", nothing)  # skip the real connect-retry backoff
    res = client.get("/api/doctor")
    assert res.status_code == 200
    d = res.json()
    assert {"verifier", "analysis", "vision", "providers", "browser", "storage"} <= set(d)
    assert d["verifier"]["ok"] is False and d["verifier"]["state"] == "unreachable"
    assert d["vision"] == {"ok": True, "state": "configured", "detail": "openai_compatible / vis"}
    assert SECRET not in res.text
    chatgpt = d["providers"]["chatgpt"]
    assert set(chatgpt) == {"enabled", "label", "url", "requires_login", "last_status"} and chatgpt["last_status"] == "unknown"
    browser = d["browser"]
    assert browser["automation_visible"] is True and "never attached to" in browser["isolation"]
    assert {"channel", "window_mode", "reuse_tabs", "cdp_url", "profiles_dir"} <= set(browser)
    assert d["storage"]["db"].endswith("api.db")


def test_doctor_reports_a_reachable_model(tmp_path, monkeypatch, fake_openai, saved):
    monkeypatch.setattr(api_app.JobManager, "_run", lambda *a, **k: None)
    fake_openai.models = ["qwen:14b"]
    settings = base_settings(
        storage={"db_path": str(tmp_path / "d.db")},
        verifier={"provider": "openai_compatible", "model": "qwen", "base_url": fake_openai.base_url},
        vision={"provider": "disabled"},
    )
    with TestClient(api_app._make_app(settings)) as c:
        d = c.get("/api/doctor").json()
    assert d["verifier"]["ok"] is True and d["verifier"]["state"] == "ready" and "qwen:14b" in d["verifier"]["models"]
    assert d["analysis"]["state"] == "unconfigured"
    assert d["vision"]["state"] == "disabled"