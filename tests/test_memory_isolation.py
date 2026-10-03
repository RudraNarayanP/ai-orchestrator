"""Memory is context, never evidence.

A sentinel "remembered" fact must reach every provider that starts a conversation (primary, escalation,
targeted follow-ups), the same block for each, and must never appear in claims, evidence, sources, the
verdicts or the final answer. Switched off, it appears nowhere. Evals and tests never touch the real file.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.evidence.sources import SourceTier
from backend.memory.service import MemoryService
from backend.memory.store import MemoryStore
from backend.models import Job
from backend.orchestrator.runner import ResearchRunner
from tests.conftest import adapters_from, base_settings

SENTINEL = "ZXQ-SENTINEL-4417"
BLOCK = f"Relevant context about the user:\n- Is fond of {SENTINEL}\nUse when relevant. Don't mention it was supplied."

SCRIPTS = {
    "chatgpt": {"answer": "I couldn't verify that reliably; no solid data is available.", "citations": []},
    "gemini": {"answer": "Acme released the Bolt in March 2024.", "citations": [{"url": "https://reuters.com/old", "title": "Acme Bolt March 2024 release"}]},
    "copilot": {"answer": "Acme released the Bolt in March 2025.", "citations": [{"url": "https://acme.com/press", "title": "Acme Bolt press release 2025"}]},
    "search": {"answer": "results", "citations": [{"url": "https://acme.com/press", "title": "Acme Bolt press release 2025"}]},
}


class FakeMemory:
    def __init__(self, block=BLOCK):
        self.block = block
        self.learned: list[str] = []

    def context_for(self, question, *, project=None, conversation=None):
        hit = SimpleNamespace(memory=SimpleNamespace(memory_id="mem_x"), public=lambda: {"memory_id": "mem_x", "content": f"Is fond of {SENTINEL}", "why": ["test"]})
        return self.block, SimpleNamespace(kind="personal", hits=[hit] if self.block else [])

    def learn(self, message, *, project=None, conversation=None):
        self.learned.append(message)
        return SimpleNamespace(added=[], forgotten=0, skipped="")


async def run(memory):
    from tests.conftest import adapters_from  # noqa: F811

    settings = base_settings()
    adapters = adapters_from(SCRIPTS, settings)
    runner = ResearchRunner(settings, adapters, engine=None, verifier=None, memory=memory)
    job = await runner.run(Job(question="When did Acme release the Bolt?", max_rounds=3))
    return job, adapters


@pytest.fixture
def net_ok(net):
    net.confirm("https://reuters.com/old", tier=SourceTier.JOURNALISM)
    net.confirm("https://acme.com/press", tier=SourceTier.PRIMARY_OFFICIAL)
    return net


def _fresh_prompts(adapters):
    return [(n, c) for n, a in adapters.items() if n != "search" for c in a.calls if not c["continue_thread"]]


async def test_every_provider_that_starts_a_conversation_gets_the_same_block(net_ok):
    job, adapters = await run(FakeMemory())
    fresh = _fresh_prompts(adapters)
    assert {n for n, _ in fresh} >= {"chatgpt", "gemini", "copilot"}, "the scenario must reach escalation"
    for name, call in fresh:
        assert BLOCK in call["prompt"], f"{name} round {call['round']} is missing the memory block"
    assert len({n for n, c in fresh if c["round"] >= 2}) >= 1, "targeted follow-ups must carry it too"


async def test_memory_never_leaks_into_evidence_claims_or_answer(net_ok):
    job, _ = await run(FakeMemory())
    assert job.final is not None
    for label, part in {
        "claims": [c.model_dump(mode="json") for c in job.claims],
        "evidence": [e.model_dump(mode="json") for e in job.evidence],
        "final": job.final.model_dump(mode="json"),
        "disagreements": [d.model_dump(mode="json") for d in job.disagreements],
        "follow_ups": [f.model_dump(mode="json") for f in job.follow_ups],
    }.items():
        assert SENTINEL not in json.dumps(part, default=str), f"memory leaked into {label}"


async def test_memory_context_is_not_serialised_and_used_list_is_exposed(net_ok):
    job, _ = await run(FakeMemory())
    assert "memory_context" not in job.model_dump()
    assert job.memory_used and job.memory_used[0]["why"] == ["test"]


async def test_the_users_question_not_the_answer_is_what_gets_learned(net_ok):
    mem = FakeMemory()
    job, _ = await run(mem)
    assert mem.learned == ["When did Acme release the Bolt?"]


async def test_no_memory_service_means_no_block(net_ok):
    _job, adapters = await run(None)
    assert not any(SENTINEL in c["prompt"] for a in adapters.values() for c in a.calls)


async def test_empty_block_injects_nothing(net_ok):
    _job, adapters = await run(FakeMemory(block=""))
    assert not any("Relevant context about the user" in c["prompt"] for a in adapters.values() for c in a.calls)


async def test_broken_memory_never_breaks_research(net_ok):
    class Broken(FakeMemory):
        def context_for(self, *a, **k):
            raise RuntimeError("disk on fire")

        def learn(self, *a, **k):
            raise RuntimeError("disk on fire")

    job, _ = await run(Broken())
    assert job.final is not None


def test_test_and_eval_configs_never_open_the_real_memory_file():
    assert base_settings().memory.enabled is False
    import scripts.eval_hard as eval_hard

    src = open(eval_hard.__file__, encoding="utf-8").read()
    assert '["memory"]["enabled"] = False' in src or "['memory']['enabled'] = False" in src or '"enabled": False' in src


def test_real_service_block_and_inject_off(tmp_path):
    svc = MemoryService(MemoryStore(tmp_path / "m.db"))
    svc.learn("My cat is called Zorbo")
    block, ret = svc.context_for("what is my cat called?")
    assert "Zorbo" in block
    svc.store.set_setting("inject", "off")
    block, ret = svc.context_for("what is my cat called?")
    assert block == ""