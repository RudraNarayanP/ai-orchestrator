"""A research job that crashes must leave a traceback behind (live: ua-closed-session died with a bare
"IndexError: list index out of range" and nothing to follow)."""

from __future__ import annotations

from backend.models import Job, ResearchMode
from backend.orchestrator.runner import ResearchRunner
from backend.research import claims as claim_ops
from tests.conftest import adapters_from


class RecordingBus:
    def __init__(self):
        self.events = []

    async def emit(self, kind, message, **payload):
        self.events.append((kind, message, payload))


async def test_crash_records_a_traceback(settings, net, monkeypatch):
    async def boom(*a, **k):
        items = []
        return items[3]

    monkeypatch.setattr(claim_ops, "extract_claims", boom)
    scripts = {"chatgpt": {"answer": "Acme released the Bolt in March 2026.", "citations": []}}
    bus = RecordingBus()
    runner = ResearchRunner(settings, adapters_from(scripts, settings), engine=None, bus=bus)
    job = await runner.run(Job(question="When did Acme release the Bolt?", mode=ResearchMode.STANDARD, max_rounds=1))
    assert job.status.value == "failed"
    assert "IndexError" in job.error
    errors = [p for kind, _, p in bus.events if kind == "error"]
    assert errors and "Traceback" in errors[0].get("traceback", "") and "boom" in errors[0]["traceback"]