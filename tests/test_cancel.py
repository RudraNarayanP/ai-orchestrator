"""Cooperative cancellation (backlog item 5).

Task.cancel() abandons a job at its next await, but a browser poll loop and a
gather that swallows a child's CancelledError can keep going. The CancelToken makes
a stop visible to the adapter (mid-wait), the runner (between stages) and the tab.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.api import app as api_app
from backend.cancel import CancelToken, JobCancelled
from backend.models import Job, JobStatus, ProviderStatus, ResearchMode
from backend.orchestrator.runner import ResearchRunner
from backend.settings import ProviderConfig
from browser.adapters.base import ChatAdapter
from tests.conftest import FakeAdapter, adapters_from, base_settings


# ------------------------------------------------------------------ the token


async def test_sleep_ends_early_by_raising_when_cancelled():
    token = CancelToken()
    asyncio.get_running_loop().call_later(0.1, token.cancel)
    started = time.monotonic()
    with pytest.raises(JobCancelled):
        await token.sleep(5)
    assert time.monotonic() - started < 1.0


async def test_an_uncancelled_token_just_sleeps_and_a_cancelled_one_refuses():
    token = CancelToken()
    await token.sleep(0.05)
    token.cancel("because")
    with pytest.raises(JobCancelled, match="because"):
        token.raise_if_cancelled()
    with pytest.raises(JobCancelled):
        await token.sleep(0)


def test_job_cancelled_is_a_cancelled_error_so_existing_handlers_let_it_through():
    assert issubclass(JobCancelled, asyncio.CancelledError)
    assert not issubclass(JobCancelled, Exception)


# ------------------------------------------------------------------ the adapter


class StubEngine:
    def __init__(self) -> None:
        self.closed: list[str] = []

    async def close_tab(self, provider, key=None):
        self.closed.append(provider)
        return True

    async def focus_tab(self, page):
        return None


class StreamingForever(ChatAdapter):
    """A DOM that is still generating (busy, text growing) and never finishes."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.polls = 0

    async def _call(self, page, fn, *args):
        self.polls += 1
        text = "answer " * self.polls
        return {"text": text, "plainLength": len(text), "blockCount": 1, "busy": ["stop-button"], "visibility": "visible", "items": []}


def streaming_adapter(settings):
    cfg = ProviderConfig(enabled=True, label="X", url="https://x.test/", max_retries=0)
    adapter = StreamingForever(StubEngine(), settings, "chatgpt", cfg)
    adapter.sel.hard_timeout_ms = 600_000
    adapter.sel.force_capture_ms = 600_000
    adapter.sel.stuck_ms = 600_000
    adapter.sel.never_started_ms = 600_000
    return adapter


async def test_cancel_interrupts_a_mid_stream_wait_within_a_poll_interval():
    settings = base_settings()
    adapter = streaming_adapter(settings)
    token = adapter.cancel_token = CancelToken()
    from backend.models import ProviderResponse

    response = ProviderResponse(job_id="j", round=1, provider="chatgpt", prompt="p")

    async def emit(*a, **k):
        return None

    asyncio.get_running_loop().call_later(0.5, token.cancel)
    started = time.monotonic()
    with pytest.raises(JobCancelled):
        await adapter._await_completion(object(), {"count": 0, "lastText": ""}, response, emit, 1)
    assert time.monotonic() - started < 2.5, "the wait must end at the next poll, not at the hard timeout"
    assert adapter.polls >= 1


async def test_cancel_closes_the_provider_tab_and_marks_the_response():
    settings = base_settings()
    adapter = streaming_adapter(settings)
    token = adapter.cancel_token = CancelToken()

    async def attempt(**kwargs):
        token.cancel()
        adapter._check_cancel()

    adapter._attempt = attempt
    with pytest.raises(JobCancelled):
        await adapter.ask("j", "question", 1)
    assert adapter.engine.closed == ["chatgpt"], "the half-answered tab must be closed"


async def test_a_job_that_was_not_cancelled_leaves_its_tab_alone():
    settings = base_settings()
    adapter = streaming_adapter(settings)

    async def attempt(**kwargs):
        raise RuntimeError("ordinary failure")

    adapter._attempt = attempt
    response = await adapter.ask("j", "question", 1)
    assert response.status == ProviderStatus.FAILED and adapter.engine.closed == []


# ------------------------------------------------------------------ the runner


async def test_runner_hands_its_token_to_every_chat_adapter(settings):
    cfg = ProviderConfig(enabled=True, label="X", url="https://x.test/")
    adapter = ChatAdapter(StubEngine(), settings, "chatgpt", cfg)
    runner = ResearchRunner(settings, {"chatgpt": adapter}, engine=None)
    assert adapter.cancel_token is runner.cancel


async def test_cancelled_before_start_asks_nobody(settings):
    scripts = {"chatgpt": {"answer": "x"}, "gemini": {"answer": "y"}}
    adapters = adapters_from(scripts, settings)
    token = CancelToken()
    token.cancel()
    job = Job(question="When did Acme release the Bolt?", mode=ResearchMode.STANDARD)
    with pytest.raises(JobCancelled):
        await ResearchRunner(settings, adapters, engine=None, cancel=token).run(job)
    assert job.status == JobStatus.CANCELLED
    assert sum(len(a.calls) for a in adapters.values()) == 0


async def test_cancel_during_the_primary_stops_before_escalation_and_verification(settings, net):
    """The child finishes 'normally' after the flag is set; the runner must still stop."""
    token = CancelToken()

    class CancelsWhileAnswering(FakeAdapter):
        async def ask(self, job_id, prompt, round_no=1, emit=None):
            token.cancel()
            return await super().ask(job_id, prompt, round_no, emit)

    scripts = {"chatgpt": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026."}, "gemini": {"answer": "z"}, "search": {"answer": "r"}}
    adapters = adapters_from(scripts, settings)
    adapters["chatgpt"] = CancelsWhileAnswering("chatgpt", scripts["chatgpt"], settings)
    runner = ResearchRunner(settings, adapters, engine=None, cancel=token)
    job = Job(question="When did Acme release the Bolt?", mode=ResearchMode.STANDARD, max_rounds=3)
    with pytest.raises(JobCancelled):
        await runner.run(job)
    assert job.status == JobStatus.CANCELLED
    assert job.final is None and job.reports == []
    assert adapters["gemini"].calls == [] and adapters["search"].calls == []


# ------------------------------------------------------------------ the manager


async def test_manager_cancel_sets_the_token_then_cancels_the_task(tmp_path):
    settings = base_settings(storage={"db_path": str(tmp_path / "c.db")})
    manager = api_app.JobManager(settings, api_app.Store(settings), api_app.EventBroker())
    token = manager.cancels["job_x"] = CancelToken()
    task = manager.tasks["job_x"] = asyncio.create_task(asyncio.sleep(30))
    assert await manager.cancel("job_x") is True
    assert token.cancelled
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await manager.cancel("job_x") is False and await manager.cancel("nope") is False