"""Browser-AI-first architecture: the consumer chat sites are the researchers.

Offline, with scripted fake providers. Each test asserts on what was asked of whom
and in which conversation, because the point of the design is the order of
escalation: primary AI -> same-conversation follow-up -> parallel independents ->
curator.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.evidence.sources import SourceTier
from backend.models import Job, ProviderStatus, ResearchMode
from backend.orchestrator.runner import ResearchRunner
from tests.conftest import make_response


class Scripted:
    """A provider that answers from a per-call queue and logs every prompt it gets."""

    def __init__(self, provider: str, replies: list[dict[str, Any]], log: list[dict[str, Any]]) -> None:
        self.provider = provider
        self.replies = replies
        self.log = log
        self.n = 0
        self.threads: dict[str, int] = {}

    async def ask(self, job_id, prompt, round_no=1, emit=None, continue_thread=False):
        self.n += 1
        self.log.append({"provider": self.provider, "job_id": job_id, "prompt": prompt, "round": round_no, "continue": continue_thread, "n": self.n})
        script = self.replies[min(self.n - 1, len(self.replies) - 1)]
        response = make_response(self.provider, script, job_id, round_no)
        response.prompt = prompt
        return response

    def calls(self):
        return [c for c in self.log if c["provider"] == self.provider]


async def run(settings, replies, question, *, verifier=None, max_rounds=3):
    log: list[dict[str, Any]] = []
    adapters = {name: Scripted(name, r if isinstance(r, list) else [r], log) for name, r in replies.items()}
    runner = ResearchRunner(settings, adapters, engine=None, verifier=verifier)
    job = Job(question=question, mode=ResearchMode.STANDARD, max_rounds=max_rounds)
    done = await runner.run(job)
    return done, log, adapters


SUPPORTED = {
    "answer": "DIRECT ANSWER\nAcme released the Bolt in March 2026.\n\nKEY CLAIMS\n1. Acme released the Bolt in March 2026.",
    "citations": [
        {"url": "https://reuters.com/a", "title": "Acme releases Bolt in March 2026"},
        {"url": "https://acme.com/news", "title": "Acme Bolt launch, March 2026"},
    ],
}
Q = "When did Acme release the Bolt?"


def confirm_sources(net):
    net.confirm("https://reuters.com/a", tier=SourceTier.JOURNALISM)
    net.confirm("https://acme.com/news", tier=SourceTier.PRIMARY_OFFICIAL)


# ---- test 1 is tests/test_escalation.py::test_trivial_question_never_touches_a_browser (2+2: no browser, no verifier)


# ---- test 2
async def test_supported_primary_answer_is_curated_and_done_without_escalation(settings, net):
    confirm_sources(net)
    job, log, _ = await run(settings, {"chatgpt": SUPPORTED, "gemini": {"answer": "never"}, "copilot": {"answer": "never"}}, Q)
    assert [c["provider"] for c in log] == ["chatgpt"], "a supported first answer needs no second opinion"
    assert "March 2026" in job.final.answer
    assert job.corrections == [] and len(job.escalation_log) == 1


# ---- test 3
async def test_hedged_primary_gets_a_same_conversation_follow_up_before_anyone_else(settings, net):
    confirm_sources(net)
    hedge = {"answer": "I couldn't verify this reliably. I'm not certain of the release date.", "citations": []}
    job, log, adapters = await run(settings, {"chatgpt": [hedge, SUPPORTED], "gemini": {"answer": "never"}, "copilot": {"answer": "never"}}, Q)
    order = [c["provider"] for c in log]
    assert order == ["chatgpt", "chatgpt"], "the follow-up goes to the SAME AI first, and settles it here"
    first, second = log
    assert first["continue"] is False and second["continue"] is True, "second prompt continues the first conversation"
    assert first["job_id"] == second["job_id"] == job.id, "one research id for both turns"
    assert "re-investigate" in second["prompt"] and "web search" in second["prompt"]
    assert "primary or official source" in second["prompt"] and "CANNOT ESTABLISH" in second["prompt"]
    turns = [(r.provider, r.turn, r.continued, r.thread_id) for r in job.responses]
    assert turns == [("chatgpt", 1, False, f"{job.id}:chatgpt"), ("chatgpt", 2, True, f"{job.id}:chatgpt")]
    assert "March 2026" in job.final.answer


# ---- test 4
async def test_parallel_escalation_only_after_the_follow_up_failed_and_to_two_or_three_others(settings, net):
    confirm_sources(net)
    hedge = {"answer": "I couldn't verify this reliably. I don't have enough information.", "citations": []}
    other = {"answer": "Acme released the Bolt in March 2026.", "citations": SUPPORTED["citations"]}
    replies = {"chatgpt": [hedge, hedge], "gemini": other, "qwen": other, "deepseek": other, "copilot": other, "le_chat": other}
    job, log, _ = await run(settings, replies, Q)
    order = [c["provider"] for c in log]
    assert order[:2] == ["chatgpt", "chatgpt"], "same-conversation follow-up comes first"
    parallel = [c for c in log[2:] if c["provider"] != "chatgpt" and not c["continue"] and "Original question" in c["prompt"]]
    assert 2 <= len(parallel) <= 3, [c["provider"] for c in parallel]
    for c in parallel:
        assert "using your own web search" in c["prompt"] or "own web search" in c["prompt"]
        assert "Do not simply repeat or agree" in c["prompt"]
        assert "What it could NOT establish" in c["prompt"]
    assert len({c["provider"] for c in parallel}) == len(parallel), "each independent AI gets its own conversation"


# ---- test 5
async def test_self_correction_is_kept_as_evidence_and_the_final_position_is_the_correction(settings, net):
    net.confirm("https://acme.com/news", tier=SourceTier.PRIMARY_OFFICIAL)
    net.confirm("https://reuters.com/a", tier=SourceTier.JOURNALISM)
    wrong = {"answer": "DIRECT ANSWER\nAcme released the Bolt in March 2024.\n\nKEY CLAIMS\n1. Acme released the Bolt in March 2024.", "citations": []}
    fixed = {
        "answer": (
            "DIRECT ANSWER\nAcme released the Bolt in March 2026.\n\nKEY CLAIMS\n1. Acme released the Bolt in March 2026.\n\n"
            "I was wrong earlier: the 2024 date was the prototype, because the press page dates the launch to 2026.\n"
            "VERDICT ON MY EARLIER ANSWER: CORRECTED"
        ),
        "citations": SUPPORTED["citations"],
    }
    job, log, _ = await run(settings, {"chatgpt": [wrong, fixed], "gemini": {"answer": "never"}}, Q)
    assert len(job.corrections) == 1
    c = job.corrections[0]
    assert c.verdict == "corrected" and c.provider == "chatgpt"
    assert "2024" in c.initial_claim and "2026" in c.follow_up_result and "2026" in c.final_position
    assert "prototype" in c.correction_reason or "because" in c.correction_reason
    assert job.responses[0].superseded is True, "the old answer stays in the transcript but no longer counts"
    assert "2026" in job.final.answer and "2024" not in job.final.answer
    assert not job.disagreements, "a self-correction is not a conflict between researchers"
    assert [x["provider"] for x in log] == ["chatgpt", "chatgpt"], "a correction that is sourced ends the escalation"