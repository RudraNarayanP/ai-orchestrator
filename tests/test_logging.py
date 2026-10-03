"""File logging (backlog item 7): a rotating log that never records credentials."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import pytest

from backend import logs
from backend.api import app as api_app
from backend.logs import get_logger, redact, setup_logging, uvicorn_log_config
from tests.conftest import base_settings

import run as run_cli


@pytest.fixture(autouse=True)
def clean_logging():
    logs.close_logging()
    yield
    logs.close_logging()


def read(path: Path) -> str:
    for handler in logging.getLogger(logs.LOGGER_NAME).handlers:
        handler.flush()
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_setup_creates_the_file_and_records_lines(tmp_path):
    target = tmp_path / "sub" / "omni.log"
    assert setup_logging(target) == target
    get_logger("jobs").info("hello from a job")
    text = read(target)
    assert "hello from a job" in text and "omnibrain.jobs" in text and "INFO" in text


def test_setup_is_idempotent_and_does_not_double_log(tmp_path):
    target = tmp_path / "omni.log"
    setup_logging(target)
    setup_logging(target)
    get_logger().info("once")
    assert read(target).count("once") == 1
    assert len(logging.getLogger(logs.LOGGER_NAME).handlers) == 1


def test_switching_the_log_file_moves_the_output(tmp_path):
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    setup_logging(a)
    get_logger().info("first")
    setup_logging(b)
    get_logger().info("second")
    assert "first" in read(a) and "second" not in read(a)
    assert "second" in read(b)


@pytest.mark.parametrize(
    "line, secret",
    [
        ("calling with key sk-abcdef1234567890XYZ now", "sk-abcdef1234567890XYZ"),
        ("Authorization: Bearer eyJhbGciOi.abc-DEF_123", "eyJhbGciOi.abc-DEF_123"),
        ("settings api_key=hunter2hunter2 saved", "hunter2hunter2"),
        ('{"api_key": "plain-secret-value", "model": "x"}', "plain-secret-value"),
        ("GET https://api.example/v1?key=AIzaSyDUMMY123&q=1", "AIzaSyDUMMY123"),
        ("password: correct-horse-battery", "correct-horse-battery"),
    ],
)
def test_redact_strips_credentials_but_keeps_the_rest(line, secret):
    clean = redact(line)
    assert secret not in clean and "***" in clean


def test_redact_leaves_ordinary_text_alone():
    text = "chatgpt: completed -- 3 sources opened; model qwen3:14b ready"
    assert redact(text) == text


def test_the_file_never_contains_a_credential_even_in_a_traceback(tmp_path):
    target = tmp_path / "omni.log"
    setup_logging(target)
    log = get_logger("jobs")
    log.info("config saved with api_key=SUPERSECRETVALUE1 and sk-ZZZZZZZZZZZZZZZZ")
    log.info("formatted %s", "Bearer abcdefghijklmnop")
    try:
        raise RuntimeError("401 for key sk-LEAKYLEAKYLEAKY1234")
    except RuntimeError:
        log.exception("job failed")
    text = read(target)
    for leaked in ("SUPERSECRETVALUE1", "sk-ZZZZZZZZZZZZZZZZ", "abcdefghijklmnop", "sk-LEAKYLEAKYLEAKY1234"):
        assert leaked not in text, leaked
    assert "Traceback" in text and "job failed" in text


def test_the_log_rotates_instead_of_growing_forever(tmp_path, monkeypatch):
    monkeypatch.setattr(logs, "MAX_BYTES", 2000)
    target = tmp_path / "omni.log"
    setup_logging(target)
    for i in range(200):
        get_logger().info("line %03d %s", i, "x" * 60)
    read(target)
    rotated = sorted(p.name for p in tmp_path.glob("omni.log*"))
    assert "omni.log.1" in rotated, rotated
    assert max(p.stat().st_size for p in tmp_path.glob("omni.log*")) < 4000


async def test_job_events_are_logged_with_their_job_id(tmp_path):
    settings = base_settings(storage={"db_path": str(tmp_path / "j.db")})
    target = tmp_path / "omni.log"
    setup_logging(target)
    manager = api_app.JobManager(settings, api_app.Store(settings), api_app.EventBroker())
    emit = manager.emit_factory("job_abc")
    await emit("provider", "answering", provider="chatgpt")
    await emit("error", "boom key=sk-1234567890abcdef")
    text = read(target)
    assert "[job_abc] chatgpt provider: answering" in text and "[job_abc] error: boom" in text
    assert "sk-1234567890abcdef" not in text


def test_engine_notes_reach_the_file(tmp_path):
    from backend.browser.engine import BrowserEngine

    target = tmp_path / "omni.log"
    setup_logging(target)
    engine = BrowserEngine(base_settings())
    engine._note("adopted the window's blank tab for chatgpt")
    assert "adopted the window's blank tab" in read(target) and engine.log


def test_log_file_flag_on_serve_and_ask_and_the_precedence(tmp_path):
    parser = run_cli.build_parser()
    assert parser.parse_args(["serve", "--log-file", "x.log"]).log_file == "x.log"
    assert parser.parse_args(["ask", "q", "--log-file", "y.log"]).log_file == "y.log"
    assert parser.parse_args(["serve"]).log_file is None
    settings = base_settings(storage={"log_path": str(tmp_path / "from_settings.log")})
    default = run_cli.setup_file_logging(parser.parse_args(["serve"]), settings)
    assert default == tmp_path / "from_settings.log"
    flagged = run_cli.setup_file_logging(parser.parse_args(["serve", "--log-file", str(tmp_path / "flag.log")]), settings)
    assert flagged == tmp_path / "flag.log"


def test_default_log_path_is_data_omnibrain_log():
    assert base_settings().storage.log_path.replace("\\", "/").endswith("data/omnibrain.log")


def test_uvicorn_output_lands_in_the_same_file(tmp_path):
    target = tmp_path / "omni.log"
    setup_logging(target)
    config = uvicorn_log_config(target)
    assert "file" in config["handlers"] and all("file" in config["loggers"][n]["handlers"] for n in ("uvicorn", "uvicorn.access"))
    logging.config.dictConfig(config)
    logging.getLogger("uvicorn.error").info("Uvicorn running on http://127.0.0.1:8730")
    logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET /api/doctor HTTP/1.1" 200')
    text = read(target)
    assert text.count("Uvicorn running") == 1, "uvicorn.error propagates to uvicorn; each line must be written once"
    assert text.count("GET /api/doctor") == 1
    assert len(logging.getLogger("uvicorn.error").handlers) >= 1