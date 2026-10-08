"""Regression (e2e 2026-10-08): stopping the server left OmniBrain's provider tabs open in the user's browser, jobs running
and the stores open. Shutdown now cancels jobs, closes the engine (tabs), and a windowless server can be stopped cleanly."""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from backend.api import app as api_app
from backend.browser.live_chrome import LiveChromeEngine
from tests.conftest import base_settings
from tests.test_live_chrome import Fake, live_settings


class RecordingEngine:
    live = True

    def __init__(self) -> None:
        self.stopped: list[object] = []

    async def stop(self, keep_windows=None) -> None:
        self.stopped.append(keep_windows)


def _app(tmp_path):
    return api_app._make_app(base_settings(storage={"db_path": str(tmp_path / "h.db")}))


def test_app_shutdown_stops_the_engine_and_closes_tabs(tmp_path):
    app = _app(tmp_path)
    engine = RecordingEngine()
    with TestClient(app):
        app.state.manager.engine = engine
    assert engine.stopped == [False], "tabs are closed even with keep_windows_open (they are in the user's own browser)"
    assert app.state.manager.engine is None


def test_shutdown_cancels_running_jobs_and_closes_stores(tmp_path):
    manager = _app(tmp_path).state.manager

    async def go():
        started = asyncio.Event()

        async def forever():
            started.set()
            await asyncio.sleep(3600)

        task = asyncio.create_task(forever())
        manager.tasks["j1"] = task
        await started.wait()
        closed = []
        manager._threads = type("S", (), {"store": type("St", (), {"close": lambda self: closed.append("threads")})()})()
        t0 = time.monotonic()
        await manager.shutdown()
        return task, closed, time.monotonic() - t0

    task, closed, took = asyncio.run(go())
    assert task.cancelled() and closed == ["threads"] and took < 5 and manager._threads is None


def test_shutdown_closes_the_live_tabs_in_the_users_browser(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CU_DIR", str(tmp_path))
    fake = Fake(tmp_path)
    fake.update(last_tab_guard=True)  # like real chrome-use: a session's last tab is never closed by 'tab close'
    app = _app(tmp_path)
    engine = LiveChromeEngine(live_settings(str(tmp_path / "artifacts")), runner=fake.runner)

    with TestClient(app):
        app.state.manager.engine = engine
        asyncio.run(engine.open_research_page("chatgpt", "https://chatgpt.com/"))
        assert any(tab != "t1" for tab in fake.read()["tabs"])
    assert list(fake.read()["tabs"]) == ["t1"], "only the user's own tab is left"


def test_shutdown_endpoint_only_from_this_computer_and_only_as_json(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(api_app, "_request_exit", lambda: calls.append("exit"))
    app = _app(tmp_path)
    with TestClient(app) as remote:  # TestClient's client host is "testclient", i.e. not this computer
        assert remote.post("/api/shutdown", json={"confirm": True}).status_code == 403
    with TestClient(app, client=("127.0.0.1", 50000)) as local:
        assert local.post("/api/shutdown", data={"confirm": "true"}).status_code == 415, "a cross-site form post cannot stop it"
        assert local.post("/api/shutdown", json={}).status_code == 400
        r = local.post("/api/shutdown", json={"confirm": True})
        assert r.status_code == 200 and r.json() == {"stopping": True}
        deadline = time.time() + 3
        while not calls and time.time() < deadline:
            time.sleep(0.05)
    assert calls == ["exit"]


def test_run_py_stop_reports_when_no_server_answers(capsys):
    import argparse
    import socket

    import run

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert run.cmd_stop(argparse.Namespace(port=port)) == 1
    assert "no OmniBrain server answered" in capsys.readouterr().out
