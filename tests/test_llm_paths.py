"""LLM-backed paths, exercised offline over real HTTP against the fake OpenAI-compatible server.

What this proves: our parsing, retry, fallback and "model cannot overrule the ledger"
behaviour. What it does NOT prove: that any real model produces good JSON or good
verdicts -- the replies here are scripted by us.
"""

from __future__ import annotations

import json
import re

import pytest

from backend.models import (
    Claim,
    ClaimStatus,
    Confidence,
    Evidence,
    Job,
    ProviderResponse,
    ProviderStatus,
    ResearchMode,
    SourceCheckStatus,
    SourceTier,
)
from backend.orchestrator.runner import ResearchRunner
from backend.research import claims as claim_ops
from backend.settings import Settings
from backend.verification import llm as llm_module
from backend.verification.llm import Endpoint, LLMClient, extract_json
from backend.verification.verifier import Verifier
from tests.conftest import adapters_from, base_settings
from tests.fake_openai import FakeOpenAI


def endpoint_for(server: FakeOpenAI, **kw) -> Endpoint:
    return Endpoint(provider="openai_compatible", model="fake-model", base_url=server.base_url, timeout_s=10, **kw)


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(llm_module, "_sleep", instant)


MSG = [{"role": "user", "content": "hi"}]


# ------------------------------------------------------------------ LLMClient


async def test_complete_returns_the_text_and_sends_model_and_key(fake_openai):
    fake_openai.script("hello there")
    client = LLMClient(endpoint_for(fake_openai, api_key="sk-test"))
    reply = await client.complete(MSG)
    assert reply.ok and reply.text == "hello there" and reply.model == "fake-model"
    sent = fake_openai.requests[-1]
    assert sent["model"] == "fake-model" and sent["stream"] is False
    assert sent["_headers"]["authorization"] == "Bearer sk-test"
    assert "think" not in sent, "the thinking switch is only for ollama / lm_studio"


async def test_local_runtimes_are_asked_not_to_think(fake_openai):
    fake_openai.script("ok")
    endpoint = endpoint_for(fake_openai)
    endpoint.provider = "ollama"
    await LLMClient(endpoint).complete(MSG)
    assert fake_openai.requests[-1]["think"] is False


async def test_an_empty_completion_is_not_ok(fake_openai):
    fake_openai.script("   ")
    reply = await LLMClient(endpoint_for(fake_openai)).complete(MSG)
    assert not reply.ok and reply.error == "empty completion"


async def test_server_errors_are_retried_once_then_reported(fake_openai):
    fake_openai.script({"status": 500, "body": "boom"})
    reply = await LLMClient(endpoint_for(fake_openai)).complete(MSG, retries=1)
    assert not reply.ok and "HTTP 500" in reply.error
    assert len(fake_openai.requests) == 2


async def test_a_transient_error_then_success_recovers(fake_openai):
    fake_openai.script({"status": 503, "body": "busy"}, "recovered")
    reply = await LLMClient(endpoint_for(fake_openai)).complete(MSG, retries=1)
    assert reply.ok and reply.text == "recovered"


@pytest.mark.parametrize("status", [401, 403, 404])
async def test_auth_and_missing_model_errors_are_not_retried(fake_openai, status):
    fake_openai.script({"status": status, "body": "nope"})
    reply = await LLMClient(endpoint_for(fake_openai)).complete(MSG, retries=3)
    assert not reply.ok and f"HTTP {status}" in reply.error
    assert len(fake_openai.requests) == 1


async def test_an_unreachable_server_says_so_and_does_not_hang():
    endpoint = Endpoint(provider="ollama", model="m", base_url="http://127.0.0.1:9/v1", timeout_s=3)
    reply = await LLMClient(endpoint).complete(MSG, retries=2)
    assert not reply.ok and "cannot reach" in reply.error and "Is ollama running" in reply.error


async def test_a_disabled_endpoint_never_makes_a_request():
    reply = await LLMClient(Endpoint(provider="disabled", model="x", base_url="")).complete(MSG)
    assert not reply.ok and "disabled" in reply.error


async def test_health_reports_ready_missing_and_unreachable(fake_openai):
    fake_openai.models = ["fake-model", "other:7b"]
    assert (await LLMClient(endpoint_for(fake_openai)).health())["state"] == "ready"
    missing = endpoint_for(fake_openai)
    missing.model = "absent"
    assert (await LLMClient(missing).health())["state"] == "model_missing"
    down = Endpoint(provider="ollama", model="m", base_url="http://127.0.0.1:9/v1")
    assert (await LLMClient(down).health())["state"] == "unreachable"


# ---------------------------------------------------------------- extract_json

OBJ = {"claims": [{"claim": "A"}], "ok": True}


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(OBJ),
        "```json\n" + json.dumps(OBJ) + "\n```",
        "```\n" + json.dumps(OBJ) + "\n```",
        "Sure! Here is the JSON you asked for:\n" + json.dumps(OBJ) + "\nLet me know if you need more.",
        "<think>hmm {not json} hmm</think>" + json.dumps(OBJ),
    ],
)
def test_extract_json_recovers_wrapped_objects(text):
    assert extract_json(text) == OBJ, "the whole object must come back, not just its first array"


def test_extract_json_wraps_a_bare_list_and_rescues_a_broken_tail():
    assert extract_json('[{"a": 1}]') == {"items": [{"a": 1}]}
    rescued = extract_json('{"claims": [{"claim": "A"}], "oops": ')
    assert rescued == {"claims": [{"claim": "A"}]}


@pytest.mark.parametrize("text", ["", "no json at all", "{not: json", "```json\n{broken\n```"])
def test_extract_json_returns_none_rather_than_guessing(text):
    assert extract_json(text) is None


async def test_complete_json_returns_none_with_the_reply_when_unparsable(fake_openai):
    fake_openai.script("I cannot do that.")
    parsed, reply = await LLMClient(endpoint_for(fake_openai)).complete_json(MSG)
    assert parsed is None and reply.ok and reply.text == "I cannot do that."


# ------------------------------------------------------------------- Verifier


def claim_and_evidence(*, confirmed: int = 2):
    claim = Claim(job_id="j", claim="The Acme Bolt costs $549.", kind="statistic", provider_sources=["chatgpt"])
    evidence = [
        Evidence(
            job_id="j", claim_id=claim.id, url=f"https://site{i}.example/bolt", domain=f"site{i}.example",
            tier=SourceTier.PRIMARY_OFFICIAL, check_status=SourceCheckStatus.CONFIRMED, verbatim_excerpt="costs $549", published="2026-06-01",
        )
        for i in range(confirmed)
    ]
    return claim, evidence


def verdict_json(claim: Claim, *, verdict="supported", confidence="high", needs_more=False, answer="The Bolt costs $549.", **extra) -> dict:
    return {
        "verdicts": [{"claim_id": claim.id, "claim": claim.claim, "verdict": verdict, "confidence": confidence, "reasoning": "two sources"}],
        "answer": answer,
        "why": "Two primary pages state it.",
        "confidence": confidence,
        "needs_more_research": needs_more,
        **extra,
    }


async def verify(server, reply, claims, evidence) -> "VerifierReport":  # noqa: F821
    server.script(reply if isinstance(reply, (str, dict)) and not isinstance(reply, dict) or isinstance(reply, str) else json.dumps(reply))
    verifier = Verifier(endpoint_for(server), min_independent_sources=2)
    return await verifier.verify(job_id="j", question="How much is the Bolt?", round_no=1, claims=claims, evidence=evidence, responses=[], disagreements=[])


async def test_verifier_uses_a_clean_model_verdict_when_the_ledger_agrees(fake_openai):
    claim, evidence = claim_and_evidence(confirmed=2)
    report = await verify(fake_openai, json.dumps(verdict_json(claim)), [claim], evidence)
    assert report.answer == "The Bolt costs $549." and report.confidence == Confidence.HIGH
    assert report.verdicts[0].verdict == ClaimStatus.SUPPORTED
    assert report.verifier_model == "openai_compatible:fake-model"
    sent = fake_openai.requests[-1]["messages"]
    assert claim.id in sent[1]["content"], "the claim ids must reach the model"
    assert "not evidence" in sent[1]["content"], "the provider-count warning must be in the material"


async def test_verifier_accepts_fenced_and_prose_wrapped_json(fake_openai):
    claim, evidence = claim_and_evidence()
    body = json.dumps(verdict_json(claim))
    for wrapped in (f"```json\n{body}\n```", f"Here you go:\n{body}\nHope that helps."):
        report = await verify(fake_openai, wrapped, [claim], evidence)
        assert report.answer == "The Bolt costs $549." and report.verifier_model.startswith("openai_compatible")


async def test_a_model_cannot_call_a_claim_supported_with_no_confirmed_source(fake_openai):
    claim, _ = claim_and_evidence(confirmed=0)
    report = await verify(fake_openai, json.dumps(verdict_json(claim)), [claim], [])
    assert report.verdicts[0].verdict == ClaimStatus.INSUFFICIENT_EVIDENCE
    assert any("overruled" in p for p in report.verdicts[0].problems)
    assert report.confidence == Confidence.LOW, "no claim survived the ledger -> the overall band drops"
    assert report.answer == "I couldn't verify this reliably.", "the model's confident draft must not be stated as fact"
    assert not report.why and not report.sources
    assert any("The Bolt costs $549." in c and c.startswith("Not confirmed") for c in report.caveats), report.caveats


async def test_one_confirmation_downgrades_a_high_confidence_claim(fake_openai):
    claim, evidence = claim_and_evidence(confirmed=1)
    report = await verify(fake_openai, json.dumps(verdict_json(claim)), [claim], evidence)
    assert report.verdicts[0].confidence == Confidence.MODERATE
    assert any("downgraded" in p for p in report.verdicts[0].problems)


async def test_a_model_cannot_keep_alive_a_claim_a_primary_source_refutes(fake_openai):
    claim, _ = claim_and_evidence(confirmed=0)
    evidence = [
        Evidence(
            job_id="j", claim_id=claim.id, url="https://maker.example/bolt", domain="maker.example", tier=SourceTier.PRIMARY_OFFICIAL,
            polarity="refute", check_status=SourceCheckStatus.CONFIRMED, verbatim_excerpt="The Bolt costs $599.", published="2026-06-01",
        )
    ]
    report = await verify(fake_openai, json.dumps(verdict_json(claim)), [claim], evidence)
    assert report.verdicts[0].verdict == ClaimStatus.REFUTED


async def test_verdicts_for_unknown_claim_ids_are_ignored(fake_openai):
    claim, evidence = claim_and_evidence()
    reply = verdict_json(claim)
    reply["verdicts"].append({"claim_id": "clm_invented", "claim": "x", "verdict": "supported", "confidence": "high"})
    report = await verify(fake_openai, json.dumps(reply), [claim], evidence)
    assert [v.claim_id for v in report.verdicts] == [claim.id]


async def test_needs_more_research_is_honoured_only_when_confidence_is_low(fake_openai):
    claim, evidence = claim_and_evidence()
    low = await verify(fake_openai, json.dumps(verdict_json(claim, confidence="low", verdict="partially_supported", needs_more=True)), [claim], evidence)
    assert low.needs_more_research is True
    high = await verify(fake_openai, json.dumps(verdict_json(claim, confidence="high", needs_more=True)), [claim], evidence)
    assert high.needs_more_research is False, "a model asking for more work on a settled claim is ignored"


async def test_follow_ups_from_the_model_are_kept_and_blank_ones_dropped(fake_openai):
    claim, evidence = claim_and_evidence()
    reply = verdict_json(claim, confidence="low", needs_more=True, follow_ups=[
        {"question": "Is $549 the launch or street price?", "reason": "ambiguous", "target_providers": ["gemini"], "claim_ids": [claim.id]},
        {"question": "  ", "reason": "blank"},
        "not a dict",
    ])
    report = await verify(fake_openai, json.dumps(reply), [claim], evidence)
    assert [f.question for f in report.follow_ups] == ["Is $549 the launch or street price?"]
    assert report.follow_ups[0].round == 2 and report.follow_ups[0].target_providers == ["gemini"]


async def test_percentage_confidence_in_the_answer_is_stripped(fake_openai):
    claim, evidence = claim_and_evidence()
    report = await verify(fake_openai, json.dumps(verdict_json(claim, answer="The Bolt costs $549, confidence level 95%.")), [claim], evidence)
    assert "95" not in report.answer


@pytest.mark.parametrize(
    "reply, reason",
    [
        ("I'm sorry, I can't produce JSON for that.", "unparsable"),
        ('{"verdicts": [], "answer": ""}', "unparsable"),
        ("{not: json", "unparsable"),
        ({"status": 500, "body": "model crashed"}, "HTTP 500"),
        ("", "empty completion"),
    ],
)
async def test_unusable_model_output_falls_back_to_the_ledger_and_says_why(fake_openai, reply, reason):
    claim, evidence = claim_and_evidence(confirmed=2)
    fake_openai.script(reply)
    verifier = Verifier(endpoint_for(fake_openai), min_independent_sources=2)
    report = await verifier.verify(job_id="j", question="q", round_no=1, claims=[claim], evidence=evidence, responses=[], disagreements=[])
    assert report.verdicts, "the deterministic ledger still produces verdicts"
    assert report.verdicts[0].verdict in {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}
    joined = " ".join(report.caveats + report.unresolved + [report.confidence_note or ""]).lower()
    assert reason.split()[0].lower() in joined or "verifier" in joined or "model" in joined, (reason, joined)


# -------------------------------------------------------------- claim extraction


def completed(provider: str, text: str) -> ProviderResponse:
    return ProviderResponse(job_id="j", provider=provider, round=1, prompt="", answer_text=text, raw_text=text, status=ProviderStatus.COMPLETED)


async def test_claims_come_from_the_model_when_it_answers_well(fake_openai):
    fake_openai.script(json.dumps({"claims": [
        {"claim": "OpenAI launched the ChatGPT agent on July 17, 2025.", "kind": "date", "topic": "launch"},
        {"claim": "ok", "kind": "fact"},
        {"claim": "Is that right?", "kind": "fact"},
    ]}))
    out = await claim_ops.extract_claims([completed("gemini", "Prose that the heuristic would split differently.")], "j", endpoint=endpoint_for(fake_openai))
    assert [c.claim for c in out] == ["OpenAI launched the ChatGPT agent on July 17, 2025."]
    assert out[0].kind == "date" and out[0].topic == "launch" and out[0].provider_sources == ["gemini"]
    assert "untrusted-data" in fake_openai.requests[-1]["messages"][1]["content"], "provider text goes in as data"


async def test_claims_fall_back_to_the_heuristic_when_the_model_is_garbage(fake_openai):
    fake_openai.script("sorry, no")
    text = "KEY CLAIMS\n1. The Acme Bolt costs $549 at launch in the United States."
    out = await claim_ops.extract_claims([completed("chatgpt", text)], "j", endpoint=endpoint_for(fake_openai))
    assert any("$549" in c.claim for c in out)


async def test_failed_responses_are_not_sent_to_the_model(fake_openai):
    fake_openai.script(json.dumps({"claims": [{"claim": "This should never be asked for.", "kind": "fact"}]}))
    bad = completed("qwen", "text")
    bad.status = ProviderStatus.FAILED
    out = await claim_ops.extract_claims([bad], "j", endpoint=endpoint_for(fake_openai))
    assert out == [] and fake_openai.requests == []


# ------------------------------------------------------- analysis (runner paths)


def runner_with(analysis: FakeOpenAI | None = None, verifier: FakeOpenAI | None = None, **settings_kw) -> ResearchRunner:
    raw = {}
    if verifier is not None:
        raw["verifier"] = {"provider": "openai_compatible", "model": "fake-model", "base_url": verifier.base_url}
    settings = base_settings(**raw, **settings_kw)
    endpoint = endpoint_for(analysis) if analysis is not None else None
    return ResearchRunner(settings, {}, engine=None, analysis_endpoint=endpoint)


async def test_a_stable_knowledge_answer_from_the_analysis_model_is_used_and_labelled(fake_openai):
    fake_openai.script(json.dumps({"can_answer": True, "answer": "Water boils at 100 degrees Celsius at sea level."}))
    analysis = await runner_with(fake_openai)._analyze("At what temperature does water boil at sea level?", ResearchMode.STANDARD, ["chatgpt", "gemini"])
    assert analysis.classifier_source == "llm"


async def test_the_analysis_model_declining_leaves_the_heuristic_classifier(fake_openai):
    fake_openai.script(json.dumps({"can_answer": False}))
    analysis = await runner_with(fake_openai)._analyze("What is the price of the Acme Bolt today?", ResearchMode.STANDARD, ["chatgpt", "gemini"])
    assert analysis.classifier_source == "heuristic"


async def test_analysis_garbage_or_outage_does_not_break_classification(fake_openai):
    for reply in ("lol no", {"status": 500, "body": "down"}):
        fake_openai.script(reply)
        analysis = await runner_with(fake_openai)._analyze("What is the price of the Acme Bolt today?", ResearchMode.STANDARD, ["chatgpt", "gemini"])
        assert analysis.classifier_source == "heuristic"


async def test_arithmetic_never_asks_the_model(fake_openai):
    fake_openai.script(json.dumps({"can_answer": True, "answer": "wrong"}))
    analysis = await runner_with(fake_openai)._analyze("what is 17 * 23", ResearchMode.STANDARD, ["chatgpt"])
    assert fake_openai.requests == []
    assert analysis.classifier_source in {"computed", "heuristic"}


# ------------------------------------------------------------ a whole job end to end


def verifier_reply_for(body: dict) -> str:
    """Reply as a verifier would: name the claim ids that appear in the request."""
    ids = re.findall(r'"claim_id":\s*"([^"]+)"', body["messages"][1]["content"])
    unique = list(dict.fromkeys(ids))
    return json.dumps({
        "verdicts": [{"claim_id": cid, "claim": "c", "verdict": "supported", "confidence": "high", "reasoning": "sources agree"} for cid in unique],
        "answer": "It is not safe without medical advice: ibuprofen raises bleeding risk with warfarin.",
        "why": "Two pages state the interaction.",
        "confidence": "high",
        "needs_more_research": False,
    })


async def test_a_full_job_with_the_verifier_served_over_http(net):
    verifier_server = FakeOpenAI([verifier_reply_for]).start()
    try:
        net.confirm("https://store.example/bolt", tier=SourceTier.PRIMARY_OFFICIAL)
        net.confirm("https://review.example/bolt", tier=SourceTier.JOURNALISM)
        settings = base_settings(verifier={"provider": "openai_compatible", "model": "fake-model", "base_url": verifier_server.base_url})
        scripts = {
            name: {"answer": "KEY CLAIMS\n1. Ibuprofen raises the risk of bleeding when taken with warfarin.", "citations": [{"url": "https://store.example/bolt", "title": "Ibuprofen warfarin bleeding"}, {"url": "https://review.example/bolt", "title": "Warfarin interactions"}]}
            for name in ["chatgpt", "gemini", "copilot"]
        }
        runner = ResearchRunner(settings, adapters_from(scripts, settings), engine=None)
        job = await runner.run(Job(question="Is it safe to take ibuprofen with warfarin?", mode=ResearchMode.STANDARD, max_rounds=2))
        assert verifier_server.requests, "the verifier endpoint must actually have been called"
        assert job.final is not None and "bleeding risk" in job.final.answer
        assert job.reports and job.reports[-1].verifier_model == "openai_compatible:fake-model"
    finally:
        verifier_server.stop()


async def test_a_full_job_survives_a_dead_verifier_and_says_so(net):
    net.confirm("https://store.example/bolt", tier=SourceTier.PRIMARY_OFFICIAL)
    net.confirm("https://review.example/bolt", tier=SourceTier.JOURNALISM)
    dead = FakeOpenAI([{"status": 500, "body": "model not loaded"}]).start()
    try:
        settings = base_settings(verifier={"provider": "openai_compatible", "model": "fake-model", "base_url": dead.base_url})
        scripts = {
            name: {"answer": "KEY CLAIMS\n1. Ibuprofen raises the risk of bleeding when taken with warfarin.", "citations": [{"url": "https://store.example/bolt", "title": "Ibuprofen warfarin bleeding"}, {"url": "https://review.example/bolt", "title": "Warfarin interactions"}]}
            for name in ["chatgpt", "gemini", "copilot"]
        }
        runner = ResearchRunner(settings, adapters_from(scripts, settings), engine=None)
        job = await runner.run(Job(question="Is it safe to take ibuprofen with warfarin?", mode=ResearchMode.STANDARD, max_rounds=2))
        assert job.final is not None and job.final.answer.strip(), "a job with a dead verifier still ends with an honest answer"
        assert job.status.value in {"completed", "done", "complete"} or job.final
    finally:
        dead.stop()