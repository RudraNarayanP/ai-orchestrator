"""A reused chat window that answers an earlier question must not feed the ledger (live eval defect)."""

from __future__ import annotations

from backend.evidence.sources import SourceTier
from backend.models import ProviderStatus
from backend.research.router import off_topic, question_from_prompt
from tests.test_resilience import run_job

UA_Q = "Under the Family Code of Ukraine, what is the minimum marriageable age and can a court allow it earlier?"
STALE = (
    "DIRECT ANSWER\nThe Employment Rights Act 2025 received Royal Assent on 18 December 2025 and enacts "
    "structural amendments to the Employment Rights Act 1996, staged across 2026 and 2027 rather than 2028."
)
GOOD = "The minimum marriageable age in Ukraine is 18; under the Family Code a court can allow marriage from 16 if it is in the person's interests."


def test_off_topic_detects_a_reply_about_a_different_subject():
    assert off_topic(UA_Q, STALE)
    assert not off_topic(UA_Q, GOOD)


def test_off_topic_is_not_triggered_by_short_or_lowercase_answers():
    assert not off_topic(UA_Q, "I don't know.")
    assert not off_topic("how tall is mount everest in metres right now", "It is about 8,849 metres above sea level according to the 2020 survey by Nepal and China.")


def test_question_is_recovered_from_round_one_and_targeted_prompts():
    assert question_from_prompt(UA_Q + "\n\nSearch the web before answering.") == UA_Q
    assert question_from_prompt("Targeted verification task.\n\nOriginal question: " + UA_Q + "\n\nCheck this point") == UA_Q


async def test_a_stale_reply_is_dropped_before_it_reaches_the_ledger(settings, net):
    net.confirm("https://zakon.rada.gov.ua/laws/show/2947-14", tier=SourceTier.GOVERNMENT)
    scripts = {
        "chatgpt": {"answer": GOOD, "citations": [{"url": "https://zakon.rada.gov.ua/laws/show/2947-14", "title": "Family Code of Ukraine"}]},
        "gemini": {"answer": STALE, "citations": [{"url": "https://legislation.gov.uk/ukpga/2025/36", "title": "Employment Rights Act 2025"}]},
    }
    job, _adapters, _ = await run_job(settings, scripts, UA_Q)
    gem = [r for r in job.responses if r.provider == "gemini"]
    assert gem and all(r.status == ProviderStatus.FAILED and (r.error or "").startswith("off_topic") for r in gem)
    assert all(not r.answer_text for r in gem)
    assert not any("Employment Rights" in c.claim for c in job.claims)