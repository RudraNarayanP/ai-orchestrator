"""One-click launcher (backlog item 15) and the pieces of it that can be tested offline."""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest

import run as run_cli

ROOT = Path(__file__).resolve().parent.parent


def test_the_launcher_builds_the_venv_then_serves_and_opens_the_ui():
    bat = (ROOT / "start.bat").read_bytes().decode("utf-8")
    assert "\r\n" in bat, "a .bat file needs CRLF line endings"
    assert "-m venv .venv" in bat and "pip install -r requirements.txt" in bat
    assert 'run.py serve --open %*' in bat, "extra arguments (e.g. --port) must pass through"
    assert bat.index("venv") < bat.index("run.py serve"), "the environment is created before the app starts"


def test_gitattributes_keeps_bat_files_crlf():
    assert "*.bat text eol=crlf" in (ROOT / ".gitattributes").read_text()


def test_serve_accepts_open_and_it_is_off_by_default():
    parser = run_cli.build_parser()
    assert parser.parse_args(["serve"]).open is False
    assert parser.parse_args(["serve", "--open", "--port", "9000"]).open is True


def test_the_browser_is_opened_only_once_the_server_is_accepting_connections():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    opened: list[str] = []
    ticks = {"n": 0}

    def sleep(_s):
        ticks["n"] += 1
        if ticks["n"] == 3:  # the server comes up after a few polls
            listener.bind(("127.0.0.1", port))
            listener.listen()

    listener = socket.socket()
    try:
        assert run_cli.open_when_ready(f"http://127.0.0.1:{port}", "127.0.0.1", port, opener=opened.append, sleep=sleep, interval_s=0)
    finally:
        listener.close()
    assert opened == [f"http://127.0.0.1:{port}"] and ticks["n"] >= 3


def test_it_gives_up_quietly_when_the_server_never_starts():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    now = {"t": 0.0}

    def clock():
        return now["t"]

    def sleep(seconds):
        now["t"] += 1.0

    opened: list[str] = []
    assert run_cli.open_when_ready("http://x", "127.0.0.1", port, opener=opened.append, clock=clock, sleep=sleep, timeout_s=5, interval_s=1) is False
    assert opened == []

def test_pyproject_and_requirements_pin_the_same_playwright():
    """Regression: pyproject said 1.55.0 while requirements.txt (what the launcher installs) said 1.63.0."""
    import re

    pinned = lambda text: re.search(r"playwright==([\d.]+)", text).group(1)  # noqa: E731
    assert pinned((ROOT / "pyproject.toml").read_text()) == pinned((ROOT / "requirements.txt").read_text())

def test_ci_workflow_runs_the_offline_gate_on_push():
    """The workflow is unexercised (no remote), so at least keep it parseable and pointed at the right command."""
    import yaml

    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    triggers = wf.get("on") or wf.get(True)  # YAML 1.1 reads a bare `on` as True
    assert "push" in triggers
    offline = wf["jobs"]["offline"]
    runs = [s["run"] for s in offline["steps"] if "run" in s]
    assert 'python -m pytest -q -m "not browser"' in runs
    assert any("requirements.txt" in r for r in runs)
    assert not offline.get("continue-on-error"), "the offline gate must be able to fail the build"
    assert wf["jobs"]["browser"].get("continue-on-error") is True, "the browser job is advisory until it has run on a runner"