"""Escalation architecture tests (the sequential-first correction).

Each test asserts on *how many* researchers were used, not just on the answer --
because the whole point is that a system which asks everyone everything passes
while still being wasteful and, on a shared-source question, wrong.
"""

from __future__ import annotations

import pytest

from backend.models import Job, ProviderStatus, ResearchMode, SourceCheckStatus
from backend.orchestrator.runner import ResearchRunner
from backend.evidence.sources import SourceTier
from tests.conftest import adapters_from


async def run_job(settings, scripts, question, *, mode=ResearchMode.STANDARD, max_rounds=3, verifier=None):
    adapters = adapters_from(scripts, settings)
    runner = ResearchRunner(settings, adapters, engine=None, verifier=verifier)
    job = Job(question=question, mode=mode, max_rounds=max_rounds)
    finished = await runner.run(job)
    return finished, adapters, runner


def asked(adapters, round_no=None):
    return {
        name
        for name, adapter in adapters.items()
        if any(c["round"] == round_no or round_no is None for c in adapter.calls)
    }


def chat_researchers(adapters, round_no=None):
    """Providers that were asked to *research*, not the evidence transport."""
    return {n for n in asked(adapters, round_no) if n != "search"}


def call_count(adapters):
    return sum(len(adapter.calls) for adapter in adapters.values())


# --------------------------------------------------------------- Test A

async def test_trivial_question_never_touches_a_browser(settings):
    scripts = {"chatgpt": {"answer": "should never be called"}}
    job, adapters, _ = await run_job(settings, scripts, "What is 2 + 2?")

    assert job.final is not None
    assert job.final.answer.strip() == "4"
    assert call_count(adapters) == 0, "arithmetic must be computed, not asked"
    assert job.browser_sessions_used == 0
    assert job.verifier_calls == 0
    assert job.stop_reason and "level 0" in job.stop_reason


async def test_arithmetic_variants_stay_at_level_zero(settings):
    for question, expected in [("calculate 17 * 23", "391"), ("100 minus 42", "58"), ("what is 10 / 4", "2.5")]:
        job, adapters, _ = await run_job(settings, {"chatgpt": {"answer": "no"}}, question)
        assert job.final.answer.strip() == expected, question
        assert call_count(adapters) == 0


# --------------------------------------------------------------- Test B

async def test_primary_with_two_confirmed_independent_sources_stops_early(settings, net):
    net.confirm("https://reuters.com/a", tier=SourceTier.JOURNALISM)
    net.confirm("https://openai.com/news/x", tier=SourceTier.PRIMARY_OFFICIAL)
    scripts = {
        "chatgpt": {
            "answer": (
                "DIRECT ANSWER\nAcme released the Bolt in March 2026.\n\n"
                "KEY CLAIMS\n1. Acme released the Bolt in March 2026."
            ),
            "citations": [
                {"url": "https://reuters.com/a", "title": "Acme releases Bolt in March 2026"},
                {"url": "https://openai.com/news/x", "title": "Acme Bolt launch, March 2026"},
            ],
        },
        "gemini": {"answer": "must not be asked"},
        "copilot": {"answer": "must not be asked"},
        "search": {"answer": "must not be asked"},
    }
    job, adapters, _ = await run_job(settings, scripts, "When did Acme release the Bolt?")

    assert chat_researchers(adapters) == {"chatgpt"}, "no swarm when the primary closed it"
    assert job.verifier_calls == 0, "the adversarial pass is an escalation layer, not a toll booth"
    assert job.final is not None and "March 2026" in job.final.answer
    assert len(job.escalation_log) == 1


async def test_confident_but_unsourced_answer_escalates(settings, net):
    """Confidence in tone is not evidence. This is the core of the design."""
    scripts = {
        "chatgpt": {"answer": "Acme definitely released the Bolt in March 2026, I am certain.", "citations": []},
        "gemini": {"answer": "The Acme Bolt launched March 2026.", "citations": [{"url": "https://bbc.com/x", "title": "Acme Bolt launch March 2026"}]},
        "copilot": {"answer": "Acme Bolt release was March 2026.", "citations": [{"url": "https://theverge.com/y", "title": "Acme Bolt released March 2026"}]},
        "search": {"answer": "results", "citations": [{"url": "https://apnews.com/z", "title": "Acme Bolt March 2026 release"}]},
    }
    net.confirm("https://bbc.com/x")
    net.confirm("https://theverge.com/y")
    net.confirm("https://apnews.com/z")
    job, adapters, _ = await run_job(settings, scripts, "When did Acme release the Bolt?")

    assert len(chat_researchers(adapters)) > 1, "an unsourced confident answer must not end the job"
    assert any(step.level.value >= 2 for step in job.escalation_log)


# --------------------------------------------------------------- Test C

async def test_primary_failure_phrase_triggers_parallel_research_with_context(settings, net):
    scripts = {
        "chatgpt": {
            "answer": "I couldn't verify this reliably. I don't have enough information about the Acme Bolt release date.",
            "citations": [],
        },
        "gemini": {"answer": "Acme Bolt released March 2026.", "citations": [{"url": "https://reuters.com/b", "title": "Acme Bolt March 2026"}]},
        "copilot": {"answer": "The Bolt launch was March 2026 per Acme.", "citations": [{"url": "https://acme.com/news", "title": "Acme announces Bolt"}]},
        "qwen": {"answer": "Acme Bolt: March 2026.", "citations": [{"url": "https://acme.com/news", "title": "Acme announces Bolt"}]},
        "search": {"answer": "results", "citations": [{"url": "https://apnews.com/q", "title": "Acme Bolt release date March 2026"}]},
    }
    net.confirm("https://reuters.com/b")
    net.confirm("https://acme.com/news", tier=SourceTier.PRIMARY_OFFICIAL)
    net.confirm("https://apnews.com/q")
    job, adapters, _ = await run_job(settings, scripts, "When did Acme release the Bolt?")

    escalation = [step for step in job.escalation_log if step.level.value >= 2]
    assert escalation, "no swarm despite an explicit inability to verify"
    assert "partial_success" not in escalation[0].triggered_by
    secondary_prompts = [c["prompt"] for name, a in adapters.items() if name != "chatgpt" for c in a.calls]
    assert any("What it could NOT establish" in p for p in secondary_prompts), "escalation must carry failure context"
    assert any("Do not simply repeat" in p for p in secondary_prompts)
    assert job.final is not None and "March 2026" in job.final.answer


# --------------------------------------------------------------- Test D

async def test_contradiction_reaches_verifier_and_generates_targeted_followup(settings, net):
    """A contradiction can only exist after more than one researcher answers, so
    the primary must first fail to earn the swarm."""
    net.confirm("https://reuters.com/old", tier=SourceTier.JOURNALISM)
    net.confirm("https://acme.com/press", tier=SourceTier.PRIMARY_OFFICIAL)
    scripts = {
        "chatgpt": {"answer": "I couldn't verify this reliably; no solid data is available.", "citations": []},
        "gemini": {
            "answer": "Acme released the Bolt in March 2024.",
            "citations": [{"url": "https://reuters.com/old", "title": "Acme Bolt March 2024 release"}],
        },
        "copilot": {
            "answer": "Acme released the Bolt in March 2025.",
            "citations": [{"url": "https://acme.com/press", "title": "Acme Bolt press release 2025"}],
        },
        "search": {"answer": "results", "citations": [{"url": "https://acme.com/press", "title": "Acme Bolt press release 2025"}]},
    }
    job, adapters, runner = await run_job(settings, scripts, "When did Acme release the Bolt?")

    assert chat_researchers(adapters) >= {"gemini", "copilot"}, "a lone hedge must earn the swarm"
    assert job.disagreements, "a 2024 vs 2025 conflict on the same subject must be detected"
    assert job.verifier_calls >= 1, "conflicting evidence is precisely when the verifier is required"
    material = [d for d in job.disagreements if d.severity == "material"]
    assert material
    assert job.follow_ups, "the verifier should ask a specific question, not re-ask the original"
    assert any("Resolve this" in f.question or "primary or official source" in f.question for f in job.follow_ups)
    asked_rounds = {c["round"] for a in adapters.values() for c in a.calls}
    assert max(asked_rounds) >= 2, "targeted round 2 should have run"
    for call in [c for a in adapters.values() for c in a.calls if c["round"] >= 2 and a.provider != "search"]:
        assert "Please answer the question again" not in call["prompt"]
        assert "Do not rely on what another AI model said" in call["prompt"]


# --------------------------------------------------------------- Test E

async def test_high_stakes_requires_primary_source_before_stopping(settings, net):
    net.confirm("https://example.com/news1", tier=SourceTier.JOURNALISM)
    net.confirm("https://example.com/news2", tier=SourceTier.JOURNALISM)
    scripts = {
        "chatgpt": {
            "answer": "Yes, you need a visa. Applicants for Singapore must apply for an entry visa before travel.",
            "citations": [
                {"url": "https://example.com/news1", "title": "Singapore visa rules for applicants"},
                {"url": "https://example.com/news2", "title": "Entry visa requirements Singapore applicants"},
            ],
        },
        "gemini": {
            "answer": "A visa is required for most nationalities travelling to Singapore.",
            "citations": [{"url": "https://ica.gov.sg/visa", "title": "Official Singapore visa requirements"}],
        },
        "copilot": {"answer": "Visa required before travel to Singapore.", "citations": [{"url": "https://example.com/news1", "title": "Singapore visa rules"}]},
        "search": {"answer": "results", "citations": [{"url": "https://example.com/news2", "title": "Singapore entry visa requirements"}]},
    }
    net.confirm("https://ica.gov.sg/visa", tier=SourceTier.GOVERNMENT)
    job, adapters, _ = await run_job(settings, scripts, "Do I need a visa to apply for admission in Singapore?", max_rounds=2)

    assert job.analysis and job.analysis.high_stakes
    assert job.level.value >= 3 or len(chat_researchers(adapters)) > 1, "consequential questions must not stop at one confident answer"
    assert job.verifier_calls >= 1
    assert job.final is not None
    if not any(e.tier.value in {"primary_official", "government", "original_research"} for e in job.evidence if e.check_status.value == "confirmed"):
        assert job.final.confidence.value != "high"


# --------------------------------------------------------------- Test F

async def test_partial_success_targets_only_the_open_claim(settings, net):
    for url in ["https://a.test/1", "https://b.test/2", "https://c.test/3", "https://d.test/4"]:
        net.confirm(url)
    net.fail("https://weak.test/price", status=SourceCheckStatus.MISMATCH)

    primary_answer = "\n".join(
        [
            "KEY CLAIMS",
            "1. Acme reported revenue of $4.2 billion in 2025.",
            "2. The company employs 12000 people.",
            "3. Acme is based in San Francisco.",
            "4. Acme acquired Nimbus in 2024.",
            "5. The Bolt costs $499.",
        ]
    )
    scripts = {
        "chatgpt": {
            "answer": primary_answer,
            "citations": [
                {"url": "https://a.test/1", "title": "Acme revenue 4.2 billion 2025"},
                {"url": "https://b.test/2", "title": "Acme employs 12000 people"},
                {"url": "https://c.test/3", "title": "Acme based in San Francisco"},
                {"url": "https://d.test/4", "title": "Acme acquired Nimbus 2024"},
                {"url": "https://weak.test/price", "title": "Acme Bolt costs $499"},
            ],
        },
        "gemini": {"answer": "The Bolt costs $549, not $499.", "citations": [{"url": "https://store.test/bolt", "title": "Buy the Acme Bolt for $549"}]},
        "search": {"answer": "results", "citations": [{"url": "https://store.test/bolt", "title": "Acme Bolt price $549"}]},
        "copilot": {"answer": "must not be needed"},
    }
    net.confirm("https://store.test/bolt", tier=SourceTier.PRIMARY_OFFICIAL)
    job, adapters, _ = await run_job(settings, scripts, "What can you tell me about Acme's Bolt and the company around it?")

    steps = [s.triggered_by for s in job.escalation_log]
    assert any("partial_success" in t for t in steps), f"expected targeted expansion, got {job.escalation_log}"
    round2 = [c for name, a in adapters.items() if name not in ("chatgpt", "search") for c in a.calls if c["round"] >= 2]
    assert round2, "the open claim should have been researched"
    assert any("Establish whether this is true" in c["prompt"] for c in round2), [c["prompt"][:80] for c in round2]
    assert not any("What it could NOT establish" in c["prompt"] for c in round2), "partial success must not trigger a full swarm"
    assert any("$499" in c["prompt"] or "Bolt costs" in c["prompt"] for c in round2)
    established_already = ["revenue", "12000", "San Francisco"]
    assert not all(any(term in c["prompt"] for term in established_already) for c in round2)
