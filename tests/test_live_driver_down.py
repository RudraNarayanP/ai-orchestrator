"""When the live browser itself is unusable (extension not connected in Edge), the job says so and stops asking."""

from __future__ import annotations

from backend.models import Job, JobStatus, ResearchMode
from backend.orchestrator.runner import ResearchRunner
from tests.conftest import FakeAdapter

EDGE = "browser.chrome_use_browser is 'edge' but the chrome-use extension is not connected in Microsoft Edge."


async def test_an_unusable_live_browser_fails_the_job_with_the_reason_and_asks_nobody_else(settings):
    adapters = {name: FakeAdapter(name, {"status": "failed", "error": f"LiveChromeUnavailable: {EDGE}"}, settings)
                for name in settings.providers}
    runner = ResearchRunner(settings, adapters, engine=None)
    job = Job(question="Under the Data Protection Act 2018, what is the minimum age of digital consent in the UK?",
              mode=ResearchMode.STANDARD, max_rounds=2)
    job = await runner.run(job)
    assert job.status == JobStatus.FAILED
    assert "live browser is not ready" in (job.error or "") and "Microsoft Edge" in job.error
    assert sum(len(a.calls) for a in adapters.values()) == 1, "after the first provider says the browser is unusable nobody else is opened"
    assert job.final is None or "Couldn't verify" not in (job.final.answer or "")
    assert runner.health == {}, "a down browser says nothing about any provider's health"


async def test_an_ordinary_provider_failure_does_not_stop_the_job(settings, net):
    scripts = {name: {"answer": "x"} for name in settings.providers}
    scripts["chatgpt"] = {"status": "failed", "error": "readiness=blocked"}
    adapters = {name: FakeAdapter(name, s, settings) for name, s in scripts.items()}
    runner = ResearchRunner(settings, adapters, engine=None)
    job = await runner.run(Job(question="When did Acme release the Bolt?", mode=ResearchMode.STANDARD, max_rounds=1))
    assert runner._driver_down is None
    assert sum(len(a.calls) for a in adapters.values()) > 1
