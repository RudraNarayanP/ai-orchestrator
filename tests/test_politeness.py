"""Rate-limit politeness (backlog item 11)."""

from __future__ import annotations

import time

import pytest

from backend.api import app as api_app
from backend.models import ProviderStatus
from backend.orchestrator.politeness import PolitenessGate
from backend.orchestrator.runner import ResearchRunner
from backend.settings import ProviderConfig, ResearchConfig
from browser.adapters.base import ChatAdapter
from tests.conftest import FakeAdapter, base_settings


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(round(seconds, 3))
        self.now += seconds


def gate(**cfg):
    clock = Clock()
    base = dict(min_provider_spacing_s=2.0, rate_limit_backoff_s=30.0, rate_limit_backoff_max_s=200.0, rate_limit_max_wait_s=45.0)
    base.update(cfg)
    return PolitenessGate(ResearchConfig(**base), clock=clock, sleep=clock.sleep), clock


async def test_first_question_to_a_site_is_not_delayed():
    g, clock = gate()
    assert await g.before("chatgpt") is None and clock.slept == []


async def test_questions_to_one_site_are_spaced():
    g, clock = gate()
    g.after("chatgpt", "completed")
    clock.now += 0.5
    assert await g.before("chatgpt") is None
    assert clock.slept == [1.5], "2s spacing minus the 0.5s already elapsed"
    g.after("chatgpt", "completed")
    clock.now += 5
    await g.before("chatgpt")
    assert clock.slept == [1.5], "no wait once the spacing has passed"


async def test_other_sites_are_not_slowed_by_this_one():
    g, clock = gate()
    g.after("chatgpt", "rate_limited")
    assert await g.before("gemini") is None and clock.slept == []


async def test_backoff_doubles_per_consecutive_limit_and_is_capped():
    g, clock = gate(rate_limit_max_wait_s=1000)
    seen = []
    for _ in range(5):
        g.after("chatgpt", "rate_limited")
        seen.append(round(g.backoff_remaining("chatgpt")))
    assert seen == [30, 60, 120, 200, 200]


async def test_a_short_backoff_is_waited_out_and_a_long_one_is_declined():
    g, clock = gate()
    g.after("chatgpt", "rate_limited")  # 30s <= 45s -> wait
    assert await g.before("chatgpt") is None
    assert clock.slept and clock.slept[-1] == pytest.approx(30, abs=0.01)
    g.after("chatgpt", "rate_limited")  # 60s > 45s -> decline, do not sleep
    slept = list(clock.slept)
    reason = await g.before("chatgpt")
    assert reason and "backing off" in reason and "2 in a row" in reason
    assert clock.slept == slept


async def test_a_completed_answer_resets_the_backoff():
    g, clock = gate()
    g.after("chatgpt", "rate_limited")
    g.after("chatgpt", "completed")
    assert g.backoff_remaining("chatgpt") == 0
    g.after("chatgpt", "rate_limited")
    assert round(g.backoff_remaining("chatgpt")) == 30, "the count restarts after a success"


async def test_other_failures_neither_start_nor_clear_a_backoff():
    g, clock = gate()
    g.after("chatgpt", "rate_limited")
    g.after("chatgpt", "timeout")
    assert g.backoff_remaining("chatgpt") > 0
    g2, _ = gate()
    g2.after("gemini", "broken")
    assert g2.backoff_remaining("gemini") == 0


# ------------------------------------------------------------------- the runner


class Limited(FakeAdapter):
    async def ask(self, job_id, prompt, round_no=1, emit=None):
        self.calls.append({"prompt": prompt, "round": round_no})
        from backend.models import ProviderResponse

        return ProviderResponse(job_id=job_id, round=round_no, provider=self.provider, prompt=prompt, status=ProviderStatus.RATE_LIMITED, error="readiness=rate_limited")


def runner_for(adapter_cls, **research):
    settings = base_settings(research={"mode": "STANDARD", "max_rounds": 3, "max_workers": 4, "min_independent_sources": 2, **research})
    adapter = adapter_cls("chatgpt", {"answer": "KEY CLAIMS\n1. Acme shipped in 2026."}, settings)
    return ResearchRunner(settings, {"chatgpt": adapter}, engine=None), adapter


async def test_a_rate_limited_site_is_not_asked_again_straight_away():
    runner, adapter = runner_for(Limited, rate_limit_backoff_s=120, rate_limit_max_wait_s=5)
    first = await runner._ask("chatgpt", "q1", 1)
    second = await runner._ask("chatgpt", "q2", 2)
    assert first.status == ProviderStatus.RATE_LIMITED and len(adapter.calls) == 1
    assert second.status == ProviderStatus.RATE_LIMITED and "backing off" in second.error
    assert len(adapter.calls) == 1, "the second question must not reach the site"
    assert runner.health["chatgpt"] == "rate_limited"


async def test_a_short_backoff_is_waited_then_the_site_is_asked_again():
    class FlakyOnce(FakeAdapter):
        n = 0

        async def ask(self, job_id, prompt, round_no=1, emit=None):
            FlakyOnce.n += 1
            if FlakyOnce.n == 1:
                return await Limited.ask(self, job_id, prompt, round_no, emit)
            return await super().ask(job_id, prompt, round_no, emit)

    runner, adapter = runner_for(FlakyOnce, rate_limit_backoff_s=0.3, rate_limit_max_wait_s=5)
    await runner._ask("chatgpt", "q1", 1)
    started = time.monotonic()
    second = await runner._ask("chatgpt", "q2", 2)
    assert time.monotonic() - started >= 0.25
    assert second.status == ProviderStatus.COMPLETED and runner.politeness.backoff_remaining("chatgpt") == 0


async def test_questions_to_one_site_are_spaced_in_the_runner():
    runner, adapter = runner_for(FakeAdapter, min_provider_spacing_s=0.3)
    await runner._ask("chatgpt", "q1", 1)
    started = time.monotonic()
    await runner._ask("chatgpt", "q2", 2)
    assert time.monotonic() - started >= 0.25


async def test_a_cancelled_job_does_not_sit_out_a_backoff():
    import asyncio

    from backend.cancel import CancelToken, JobCancelled

    settings = base_settings(research={"mode": "STANDARD", "max_rounds": 3, "max_workers": 4, "min_independent_sources": 2, "min_provider_spacing_s": 30})
    token = CancelToken()
    runner = ResearchRunner(settings, {"chatgpt": FakeAdapter("chatgpt", {"answer": "x"}, settings)}, engine=None, cancel=token)
    await runner._ask("chatgpt", "q1", 1)
    asyncio.get_running_loop().call_later(0.2, token.cancel)
    started = time.monotonic()
    with pytest.raises(JobCancelled):
        await runner._ask("chatgpt", "q2", 2)
    assert time.monotonic() - started < 3


def test_the_manager_shares_one_gate_across_jobs(tmp_path):
    settings = base_settings(storage={"db_path": str(tmp_path / "p.db")})
    manager = api_app.JobManager(settings, api_app.Store(settings), api_app.EventBroker())
    assert isinstance(manager.politeness, PolitenessGate)


# --------------------------------------------------------------- the adapter retry


class Stub:
    async def close_tab(self, provider, key=None):
        return True


async def test_the_adapter_never_retries_straight_into_a_rate_limit():
    """Regression: with max_retries > 0 a rate_limited readiness was retried immediately."""
    settings = base_settings()
    cfg = ProviderConfig(enabled=True, label="X", url="https://x.test/", max_retries=3)
    adapter = ChatAdapter(Stub(), settings, "chatgpt", cfg)
    attempts = []

    async def attempt(**kwargs):
        attempts.append(1)
        kwargs["response"].note(ProviderStatus.RATE_LIMITED, error="readiness=rate_limited")
        return False

    adapter._attempt = attempt
    response = await adapter.ask("j", "q", 1)
    assert response.status == ProviderStatus.RATE_LIMITED and len(attempts) == 1


async def test_other_failures_still_get_their_retries():
    settings = base_settings()
    cfg = ProviderConfig(enabled=True, label="X", url="https://x.test/", max_retries=2)
    adapter = ChatAdapter(Stub(), settings, "chatgpt", cfg)
    attempts = []

    async def attempt(**kwargs):
        attempts.append(1)
        kwargs["response"].note(ProviderStatus.BROKEN, error="no new message")
        return False

    adapter._attempt = attempt
    await adapter.ask("j", "q", 1)
    assert len(attempts) == 3

@pytest.mark.parametrize("readiness", ["login_wall", "blocked"])
async def test_a_login_wall_or_block_is_not_retried(readiness):
    """Live finding: Copilot/DeepSeek/Meta AI/Pi login walls were each re-opened and re-asked before giving up."""
    settings = base_settings()
    cfg = ProviderConfig(enabled=True, label="X", url="https://x.test/", max_retries=3)
    adapter = ChatAdapter(Stub(), settings, "chatgpt", cfg)
    attempts = []

    async def attempt(**kwargs):
        attempts.append(1)
        kwargs["response"].note(ProviderStatus.LOGGED_OUT, error=f"readiness={readiness}")
        return False

    adapter._attempt = attempt
    response = await adapter.ask("j", "q", 1)
    assert len(attempts) == 1 and response.error == f"readiness={readiness}"
