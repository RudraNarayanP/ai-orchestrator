"""Evidence over votes (spec sections 10, 12, 16, 28).

The headline case these tests exist for: most of the models are confidently,
unanimously wrong, each repeating the same unsourced figure, while a single
answer is backed by a primary document. A council that votes gets the majority
answer. OmniBrain has to get the evidenced one.
"""

from __future__ import annotations

from backend.evidence.sources import TIER_WEIGHT, freshness, tier_for
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
from backend.research import router
from backend.verification.llm import Endpoint
from backend.verification.verifier import Verifier, build_final_answer
from tests.conftest import adapters_from


def response(provider: str, answer: str, urls: list[str], *, job_id: str = "job1") -> ProviderResponse:
    from backend.models import Citation

    r = ProviderResponse(
        job_id=job_id,
        provider=provider,
        round=1,
        prompt="",
        answer_text=answer,
        raw_text=answer,
        citations=[Citation(url=u, title=u.split("/")[-1]) for u in urls],
        status=ProviderStatus.COMPLETED,
    )
    r.pages_visited = urls
    return r


async def test_majority_wrong_minority_evidenced_loses_the_vote(settings, net):
    """Six providers say the Bolt costs $499 and cite nothing that holds up.
    One says $549 and cites the store. The store wins."""
    for url in ["https://blog.example/rumour", "https://forum.example/claims"]:
        net.fail(url, status=SourceCheckStatus.MISMATCH)
    net.confirm("https://store.example/bolt", tier=SourceTier.PRIMARY_OFFICIAL)

    wrong = "KEY CLAIMS\n1. The Acme Bolt costs $499."
    scripts = {name: {"answer": wrong, "citations": [{"url": "https://blog.example/rumour", "title": "Acme Bolt costs $499"}]} for name in ["chatgpt", "gemini", "copilot", "qwen", "le_chat"]}
    scripts["meta_ai"] = {"answer": wrong, "citations": [{"url": "https://forum.example/claims", "title": "Acme Bolt $499 report"}]}
    scripts["deepseek"] = {"answer": "KEY CLAIMS\n1. The Acme Bolt costs $549.", "citations": [{"url": "https://store.example/bolt", "title": "Buy the Acme Bolt for $549"}]}
    scripts["search"] = {"answer": "results", "citations": [{"url": "https://store.example/bolt", "title": "Acme Bolt $549 price"}]}

    job, adapters, runner = await _run(settings, scripts, "How much does the Acme Bolt cost?")

    priced = {c.claim: c for c in job.claims}
    wrong_claim = next((c for c in job.claims if "$499" in c.claim), None)
    right_claim = next((c for c in job.claims if "$549" in c.claim), None)
    assert wrong_claim is not None and right_claim is not None, f"claims were {list(priced)}"
    assert len(wrong_claim.provider_sources) > len(right_claim.provider_sources), "the wrong figure should be the popular one"

    verdicts = {v.claim_id: v for v in (job.reports[-1].verdicts if job.reports else [])}
    assert verdicts, "the ledger should have produced verdicts"
    assert verdicts[right_claim.id].verdict in {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}
    assert verdicts[wrong_claim.id].verdict in {ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.CONTESTED, ClaimStatus.REFUTED}
    assert "$549" in job.final.answer, f"answer followed the vote instead of the evidence: {job.final.answer}"
    assert "$499" not in job.final.answer


async def test_unanimous_unsourced_consensus_earns_nothing(settings, net):
    scripts = {
        name: {"answer": "The Acme Bolt costs $499."}
        for name in ["chatgpt", "gemini", "copilot", "qwen", "le_chat", "meta_ai"]
    }
    scripts["search"] = {"answer": "results", "citations": []}
    job, _, _ = await _run(settings, scripts, "How much does the Acme Bolt cost?")
    assert job.final.answer.strip() in {"I don't know.", "I couldn't verify this reliably."} or "Insufficient" in job.final.confidence_label
    assert job.final.confidence in {Confidence.NONE, Confidence.LOW}


async def test_provider_agreement_never_raises_a_verdict_on_its_own():
    endpoint = Endpoint(provider="disabled", model="none", base_url="")
    verifier = Verifier(endpoint, min_independent_sources=2)
    job_id = "j"
    text = "The Acme Bolt costs $549."
    alone = Claim(job_id=job_id, claim=text, kind="statistic", provider_sources=["chatgpt"])
    popular = Claim(job_id=job_id, claim=text, kind="statistic", provider_sources=["chatgpt", "gemini", "copilot", "qwen", "le_chat"])
    v_alone = verifier._verdict_for(alone, [])
    v_popular = verifier._verdict_for(popular, [])
    assert v_alone.verdict == v_popular.verdict == ClaimStatus.INSUFFICIENT_EVIDENCE
    assert v_alone.confidence == v_popular.confidence
    assert "not counted" in v_alone.reasoning.lower()


async def test_conflicting_primary_sources_are_reported_as_a_conflict(settings, net):
    net.confirm("https://gov.example/a", tier=SourceTier.GOVERNMENT, published="2026-01-02")
    net.confirm("https://company.example/b", tier=SourceTier.PRIMARY_OFFICIAL, published="2026-01-03")
    scripts = {
        "chatgpt": {
            "answer": "KEY CLAIMS\n1. The filing deadline was 15 March 2026.",
            "citations": [{"url": "https://gov.example/a", "title": "filing deadline 15 March 2026"}],
        },
        "gemini": {
            "answer": "KEY CLAIMS\n1. The filing deadline was 30 March 2026.",
            "citations": [{"url": "https://company.example/b", "title": "filing deadline 30 March 2026"}],
        },
        "search": {"answer": "r", "citations": [{"url": "https://gov.example/a", "title": "filing deadline 15 March 2026"}]},
    }
    job, _, _ = await _run(settings, scripts, "When is the filing deadline?", max_rounds=2)
    assert job.disagreements, "two official sources disagreeing must surface as a conflict"
    assert job.final is not None
    assert job.final.confidence in {Confidence.LOW, Confidence.MODERATE}, "conflict must not present as high confidence"
    assert job.follow_ups, "a conflict between primaries should trigger targeted research, not a coin flip"


# ------------------------------------------------------------ unit guarantees


def test_tier_weighting_prefers_primary_over_social():
    assert TIER_WEIGHT[SourceTier.PRIMARY_OFFICIAL] > TIER_WEIGHT[SourceTier.JOURNALISM] > TIER_WEIGHT[SourceTier.SOCIAL] > TIER_WEIGHT[SourceTier.AI_UNSOURCED]
    assert tier_for("https://www.sec.gov/Archives/abcd.htm").value == "government"
    assert tier_for("https://nature.com/articles/s41586-026").value == "original_research"
    assert tier_for("https://x.com/anyone/status/1").value == "social_media"
    assert tier_for("https://example.com/blog").value == "unknown"


def test_freshness_bands():
    assert freshness("2026-09-01T00:00:00Z", now=_now())["verdict"] == "fresh"
    assert freshness("2015-01-01", now=_now())["verdict"] == "outdated"
    assert freshness(None)["known"] is False


def _now():
    from datetime import datetime, timezone

    return datetime(2026, 10, 3, tzinfo=timezone.utc)


def test_contradiction_finder_catches_figure_and_date_and_polarity():
    def claim(text, provider):
        return Claim(job_id="j", claim=text, kind="fact", provider_sources=[provider])

    a = [claim("The settlement was $1.2 billion.", "chatgpt"), claim("The settlement was $800 million.", "gemini")]
    assert claim_ops.find_contradictions(a), "different figures on the same subject must conflict"

    b = [claim("Acme launched in March 2024.", "chatgpt"), claim("Acme launched in March 2025.", "gemini")]
    found = claim_ops.find_contradictions(b)
    assert found and found[0]["kind"] == "date"

    c = [claim("The Bolt supports Wi-Fi 7 out of the box.", "chatgpt"), claim("The Bolt does not support Wi-Fi 7 out of the box.", "gemini")]
    assert claim_ops.find_contradictions(c), "a yes/no split is a contradiction"

    agree = [claim("The Bolt supports Wi-Fi 7.", "chatgpt"), claim("The Bolt supports Wi-Fi 7.", "gemini")]
    assert not claim_ops.find_contradictions(agree), "agreement is not a conflict"


def test_arithmetic_never_reaches_a_browser():
    for question, expected in [("2 + 2", "4"), ("what is 12% of 250", None), ("17*23", "391"), ("100 / 8", "12.5")]:
        computed, _ = router.try_arithmetic(question)
        if expected is None:
            continue
        assert computed == expected, (question, computed)


def test_pasted_urls_are_not_claims():
    """Sites paste their reference list inline; that is a citation, not an assertion."""
    body = "\n".join(
        [
            "KEY CLAIMS",
            "1. OpenAI launched the ChatGPT agent on July 17, 2025.",
            "2. [https://help.openai.com/en/articles/11794368](https://help.openai.com/en/articles/11794368)",
            "3. https://indianexpress.com/article/technology/tech-news-technology/openai-rolls-out",
        ]
    )
    pairs = claim_ops.heuristic_claims(
        ProviderResponse(job_id="j", provider="gemini", prompt="", answer_text=body, status=ProviderStatus.COMPLETED)
    )
    texts = [t for t, _ in pairs]
    assert any("July 17, 2025" in t for t in texts), texts
    assert not any("http" in t for t in texts), f"a bare link became a claim: {texts}"
    assert claim_ops.link_like("[https://a.example/x](https://a.example/x)")
    assert not claim_ops.link_like("OpenAI launched the agent on 17 July 2025 in the US and Canada.")


def test_malformed_arithmetic_is_refused_not_evaluated():
    for evil in ["__import__('os').system('calc')", "1; import socket", "open('C:/Windows/win.ini').read()"]:
        computed, _ = router.try_arithmetic(evil)
        assert computed is None, evil


async def _run(settings, scripts, question, *, max_rounds=3):
    adapters = adapters_from(scripts, settings)
    runner = ResearchRunner(settings, adapters, engine=None)
    job = Job(question=question, mode=ResearchMode.STANDARD, max_rounds=max_rounds)
    out = await runner.run(job)
    return out, adapters, runner
