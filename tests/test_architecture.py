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
    hedge = {"answer": "I couldn't verify that reliably. I'm not certain of the release date.", "citations": []}
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
    hedge = {"answer": "I couldn't verify that reliably. I don't have enough information.", "citations": []}
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

# ---- test 6 (also RESEARCH_NEEDED parsing and the curator loop)
def curator_server(fake_openai, seen):
    import json
    import re

    def reply(body):
        user = body["messages"][-1]["content"]
        seen.append(user)
        first = len(seen) == 1
        rows = re.findall(r'"claim_id": "([^"]+)",\s*"claim": "([^"]+)"', user)
        verdicts = [
            {
                "claim_id": cid, "claim": text,
                "verdict": "supported" if "2025" in text else "insufficient_evidence",
                "confidence": "moderate" if "2025" in text else "low",
                "reasoning": "the opened page says so", "strong_evidence": ["https://acme.com/press"] if "2025" in text else [], "problems": [],
            }
            for cid, text in rows
        ]
        return json.dumps(
            {
                "verdicts": verdicts,
                "answer": "Acme released the Bolt in March 2025.",
                "why": "",
                "confidence": "low" if first else "moderate",
                "needs_more_research": first,
                "research_needed": (
                    [{"claim": "Acme released the Bolt in March 2025", "reason": "the AIs split between 2024 and 2025",
                      "preferred_researcher": "deepseek", "instruction": "Open the Acme press release and report the date printed on it."}]
                    if first else []
                ),
                "unresolved": ["release date"] if first else [],
            }
        )

    return fake_openai.script(reply)


def make_verifier(server):
    from backend.verification.llm import Endpoint
    from backend.verification.verifier import Verifier

    return Verifier(Endpoint(provider="openai_compatible", model="fake-model", base_url=server.base_url, timeout_s=10), min_independent_sources=2)


async def test_x_y_x_contradiction_is_judged_on_evidence_and_the_curator_gets_targeted_follow_ups(net, fake_openai):
    from tests.conftest import base_settings

    names = ["chatgpt", "gemini", "qwen", "deepseek"]
    settings = base_settings(providers={n: {"enabled": True, "label": n.title(), "url": f"https://{n}.test/"} for n in names})
    net.fail("https://blog.example/rumour")
    net.fail("https://forum.example/claims")
    net.confirm("https://acme.com/press", tier=SourceTier.PRIMARY_OFFICIAL)
    x = lambda url: {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2024.", "citations": [{"url": url, "title": "Acme Bolt March 2024 release"}]}
    y = {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2025.", "citations": [{"url": "https://acme.com/press", "title": "Acme Bolt press release March 2025"}]}
    replies = {
        "chatgpt": [
            {"answer": "I'm not certain, but I think Acme released the Bolt in March 2024.", "citations": []},
            {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2025.\nVERDICT ON MY EARLIER ANSWER: CORRECTED", "citations": []},
        ],
        "gemini": x("https://blog.example/rumour"),
        "qwen": x("https://forum.example/claims"),
        "deepseek": [y, y],
    }
    seen: list[str] = []
    verifier = make_verifier(curator_server(fake_openai, seen))
    job, log, adapters = await run(settings, replies, Q, verifier=verifier)

    assert job.verifier_calls >= 1 and len(seen) >= 2
    first = seen[0]
    for needle in ("provider_chatgpt_round1", "provider_gemini_round", "provider_deepseek_round", "re-investigate", "SELF-CORRECTIONS", "PROMPT WE SENT"):
        assert needle in first, f"the curator must see the full transcripts: missing {needle!r}"
    assert "Provider agreement" not in first or "not evidence" in first
    deepseek = [c for c in log if c["provider"] == "deepseek"]
    assert len(deepseek) >= 2, "the curator's RESEARCH_NEEDED went to the preferred researcher"
    assert deepseek[-1]["continue"] is True and "Open the Acme press release" in deepseek[-1]["prompt"], "...inside its existing conversation"
    assert job.follow_ups and job.follow_ups[0].target_providers == ["deepseek"]
    assert "2025" in job.final.answer and "2024" not in job.final.answer, "the evidenced position wins, not a head count"


async def test_research_needed_loop_stops_at_the_limit(net, fake_openai):
    from tests.conftest import base_settings

    names = ["chatgpt", "gemini", "deepseek"]
    settings = base_settings(providers={n: {"enabled": True, "label": n.title(), "url": f"https://{n}.test/"} for n in names})
    import json
    seen: list[str] = []

    def always_more(body):
        seen.append(body["messages"][-1]["content"])
        return json.dumps(
            {"verdicts": [], "answer": "I couldn't verify that reliably.", "confidence": "low", "needs_more_research": True,
             "research_needed": [{"claim": "the date", "reason": "still open", "preferred_researcher": "gemini", "instruction": "Find the date again."}],
             "unresolved": ["the date"]}
        )

    fake_openai.script(always_more)
    hedge = {"answer": "I couldn't verify that reliably. I don't have enough information.", "citations": []}
    job, log, _ = await run(settings, {n: hedge for n in names}, Q, verifier=make_verifier(fake_openai), max_rounds=2)
    assert job.status.value == "completed"
    assert len(seen) <= 4, "a curator that always wants more must still stop at the round limit"
    assert len(log) <= 14


def test_research_needed_parser_accepts_the_structured_and_the_legacy_forms():
    from backend.verification.verifier import parse_research_needed

    got = parse_research_needed(
        {
            "research_needed": [{"claim": "Bolt launched 2025", "reason": "conflict", "preferred_researcher": "Le Chat", "instruction": "Open the press page."}],
            "follow_ups": [{"question": "Is it 2025?", "target_providers": ["gemini"]}, {"question": ""}, "junk"],
        }
    )
    assert got[0]["target_providers"] == ["le_chat"] and "Open the press page." in got[0]["question"] and "Bolt launched 2025" in got[0]["question"]
    assert got[1]["question"] == "Is it 2025?" and got[1]["target_providers"] == ["gemini"]
    assert len(got) == 2
    text = 'blah\nRESEARCH_NEEDED: {"claim": "c", "reason": "r", "preferred_researcher": "qwen", "instruction": "do it"}\n'
    assert parse_research_needed(text)[0]["target_providers"] == ["qwen"]
    assert parse_research_needed("nothing here") == [] and parse_research_needed(None) == []


# ---- test 7
async def test_question_b_never_sees_anything_from_question_a(settings, net):
    net.confirm("https://zephyr.example/launch", tier=SourceTier.PRIMARY_OFFICIAL)
    net.confirm("https://reuters.com/zephyr", tier=SourceTier.JOURNALISM)
    net.confirm("https://quasar.example/price", tier=SourceTier.PRIMARY_OFFICIAL)
    net.confirm("https://bbc.com/quasar", tier=SourceTier.JOURNALISM)
    a_text = "DIRECT ANSWER\nThe Zephyr drone launched in March 2026.\n\nKEY CLAIMS\n1. The Zephyr drone launched in March 2026."
    zephyr = {"answer": a_text, "citations": [{"url": "https://zephyr.example/launch", "title": "Zephyr drone launched March 2026"}, {"url": "https://reuters.com/zephyr", "title": "Zephyr drone March 2026 launch"}]}
    quasar = {"answer": "DIRECT ANSWER\nThe Quasar telescope costs $1,200.\n\nKEY CLAIMS\n1. The Quasar telescope costs $1,200.",
              "citations": [{"url": "https://quasar.example/price", "title": "Quasar telescope costs $1,200"}, {"url": "https://bbc.com/quasar", "title": "Quasar telescope $1,200 price"}]}
    log: list[dict[str, Any]] = []
    # chatgpt's second ever reply is the STALE chat: it re-serves question A's answer to question B
    adapters = {"chatgpt": Scripted("chatgpt", [zephyr, zephyr], log), "gemini": Scripted("gemini", [quasar], log)}
    runner = ResearchRunner(settings, adapters, engine=None)
    job_a = await runner.run(Job(question="When did the Zephyr drone launch?", mode=ResearchMode.STANDARD, max_rounds=3))
    job_b = await runner.run(Job(question="How much does the Quasar telescope cost?", mode=ResearchMode.STANDARD, max_rounds=3))

    assert job_a.id != job_b.id
    calls_b = [c for c in log if c["job_id"] == job_b.id]
    assert calls_b and all(c["job_id"] != job_a.id for c in calls_b)
    assert calls_b[0]["continue"] is False, "B starts a new conversation, never continuing A's"
    assert not any("Zephyr" in c["prompt"] for c in calls_b), "nothing of A in anything sent for B"
    assert all(r.job_id == job_b.id and r.thread_id.startswith(job_b.id) for r in job_b.responses)
    blob = lambda objs: " ".join(o.model_dump_json() for o in objs)
    assert "Zephyr" not in blob(job_b.claims) and "Zephyr" not in blob(job_b.evidence) and "Zephyr" not in blob(job_b.responses), "claims/evidence/raw responses are per research"
    assert "zephyr" not in (job_b.final.answer + job_b.final.why + " ".join(s.url for s in job_b.final.sources)).lower()
    assert "$1,200" in job_b.final.answer
    assert {c.id for c in job_a.claims}.isdisjoint({c.id for c in job_b.claims})
    stale = [r for r in job_b.responses if r.provider == "chatgpt"][0]
    assert stale.status.value == "failed" and stale.error.startswith("off_topic"), "a stale reply is discarded, not analysed"


# ---- test 8
async def test_citation_stays_linked_claim_evidence_source_to_the_final_answer_and_opened_differs_from_mentioned(settings, net, monkeypatch):
    from backend.evidence import search_http

    async def no_discovery(*a, **k):
        raise AssertionError("OmniBrain must not run its own web discovery by default")

    monkeypatch.setattr(search_http, "search", no_discovery)
    settings.search.own_discovery = False
    leg = "https://www.legislation.gov.uk/ukpga/2018/12/section/9"
    blog = "https://someblog.example/dpa-age"
    net.confirm(leg, tier=SourceTier.PRIMARY_OFFICIAL)
    net.confirm(blog, tier=SourceTier.JOURNALISM)
    answer = (
        "DIRECT ANSWER\nThe age is 13.\n\nKEY CLAIMS\n1. The minimum age to consent to information society services in the UK is 13.\n\n"
        f"SOURCE LINKS\n{leg} OPENED\n{blog} MENTIONED ONLY\n"
    )
    cites = [{"url": leg, "title": "DPA 2018 section 9 minimum age to consent to information society services 13"},
             {"url": blog, "title": "UK minimum age to consent to information society services is 13"}]
    job, log, _ = await run(settings, {"chatgpt": {"answer": answer, "citations": cites}, "gemini": {"answer": "never"}}, "What is the minimum age to consent to information society services in the UK?")

    by_url = {e.url: e for e in job.evidence}
    assert by_url[leg].ai_opened is True and by_url[blog].ai_opened is False, "opened vs mentioned survives into the evidence"
    assert by_url[leg].cited_by == ["chatgpt"]
    claim_ids = {c.id for c in job.claims}
    assert by_url[leg].claim_id in claim_ids
    sources = {s.url: s for s in job.final.sources}
    assert leg in sources, "the primary page the AI opened must reach the final answer (the 'opened but not cited' defect)"
    assert sources[leg].claim_ids and set(sources[leg].claim_ids) <= claim_ids
    assert sources[leg].ai_opened is True and sources[leg].audited is True and sources[leg].cited_by == ["chatgpt"]
    if blog in sources:
        assert sources[blog].ai_opened is False
    assert [c["provider"] for c in log] == ["chatgpt"]


def test_mentioned_vs_opened_labels_are_read_from_the_answer_and_never_invented():
    from backend.models import Citation, ProviderResponse
    from backend.research.citations import mark_opened

    r = ProviderResponse(
        job_id="j", provider="gemini", prompt="",
        answer_text="SOURCE LINKS\nhttps://a.example/x - OPENED\nhttps://b.example/y (MENTIONED ONLY)\nhttps://c.example/z\n",
        citations=[Citation(url="https://a.example/x"), Citation(url="https://b.example/y/"), Citation(url="https://c.example/z"), Citation(url="https://d.example/")],
    )
    mark_opened(r)
    assert [c.ai_opened for c in r.citations] == [True, False, None, None], "unlabelled means unknown, not opened"


# ---- prompts and discovery defaults
def test_every_research_prompt_tells_the_ai_to_use_its_own_web_search():
    from backend.research.prompts import escalation_prompt, research_prompt, thread_follow_up_prompt

    first = " ".join(research_prompt("When did Acme release the Bolt?", provider="gemini").split())
    for needle in ("do the research yourself", "own web search", "primary sources", "OPEN the pages", "actually opened", "NO WEB ACCESS"):
        assert needle in first, needle
    follow = " ".join(thread_follow_up_prompt("Acme released the Bolt in 2025", "sources disagree").split())
    for needle in ("re-investigate", "own web search", "primary or official source", "open it", "I CANNOT ESTABLISH THIS", "VERDICT ON MY EARLIER ANSWER"):
        assert needle in follow, needle
    esc = escalation_prompt("Q?", provider="qwen", primary="chatgpt", established=[], unresolved=["Acme released the Bolt in 2025"], contradictions=[], sources=[])
    esc = " ".join(esc.split())
    for needle in ("own web search", "Do not simply repeat or agree", "Acme released the Bolt in 2025", "fresh conversation"):
        assert needle in esc, needle


def test_omnibrain_does_not_run_its_own_web_discovery_by_default():
    from backend.settings import Settings

    assert Settings.model_validate({"providers": {}}).search.own_discovery is False


def test_hedges_in_the_spec_are_all_detected():
    from backend.research.router import is_failure_phrase

    for text in (
        "I can't verify that.", "I couldn't find a source.", "I'm not certain about this.", "This may be outdated.",
        "I don't have enough information.", "I couldn't confirm the figure.", "The sources are unclear.",
    ):
        assert is_failure_phrase(text), text
    assert not is_failure_phrase("The age is 13, per section 9 of the Act.")

def test_threads_corrections_and_opened_flags_are_persisted_per_research(tmp_path):
    from backend.models import Evidence, ProviderResponse, SelfCorrection
    from backend.storage.db import Store
    from tests.conftest import base_settings

    settings = base_settings(storage={"db_path": str(tmp_path / "t.db"), "artifacts_dir": str(tmp_path / "a")})
    store = Store(settings)
    job = Job(question="Q?")
    job.responses.append(ProviderResponse(job_id=job.id, provider="chatgpt", prompt="p", thread_id=f"{job.id}:chatgpt", turn=2, continued=True, superseded=True))
    job.corrections.append(SelfCorrection(job_id=job.id, provider="chatgpt", initial_claim="X", follow_up_result="Y", verdict="corrected", final_position="Y"))
    job.evidence.append(Evidence(job_id=job.id, url="https://a.example/x", ai_opened=False, cited_by=["chatgpt"]))
    store.save_job(job)
    extra = store.job_snapshot(job.id)["extra"]
    assert extra["threads"][0]["turn"] == 2 and extra["threads"][0]["continued"] is True and extra["threads"][0]["superseded"] is True
    assert extra["corrections"][0]["initial_claim"] == "X" and extra["corrections"][0]["final_position"] == "Y"
    assert extra["evidence_flags"][0]["ai_opened"] is False and extra["evidence_flags"][0]["cited_by"] == ["chatgpt"]

async def test_two_requests_to_one_ai_in_the_same_round_are_two_turns(settings, net):
    import asyncio

    log: list[dict[str, Any]] = []
    adapter = Scripted("gemini", [{"answer": "one"}, {"answer": "two"}, {"answer": "three"}], log)
    runner = ResearchRunner(settings, {"gemini": adapter}, engine=None, verifier=None)
    first = await runner._ask("gemini", "Q about Acme", 1, job_id="rid-1", role="primary")
    a, b = await asyncio.gather(
        runner._ask("gemini", "follow-up one", 2, job_id="rid-1", continue_thread=True),
        runner._ask("gemini", "follow-up two", 2, job_id="rid-1", continue_thread=True),
    )
    assert first.turn == 1 and not first.continued
    assert sorted([a.turn, b.turn]) == [2, 3], "concurrent requests to one chat must not share a turn number"
    assert a.continued and b.continued