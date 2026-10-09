"""Regression for the live uk-dpa-age run: false "figure/date" conflicts on claims that do not answer the asked
question (enactment date vs 'section 9'; '16 years ... read as 13 years' vs '13 years') kept research going."""

from __future__ import annotations

from backend.evidence.sources import SourceTier, tier_for
from backend.models import Claim
from backend.research import citations as ops
from backend.research.claims import comparable_text, find_contradictions
from tests.test_early_stop import LAW_Q, URL9, providers
from tests.test_architecture import run

# the exact claim texts from the live run (data/eval/eval_20261004_003949.json)
REAL = [
    "Section 9 of the Data Protection Act 2018 sets the UK Article 8 age threshold at 13 years for children's consent in relation to information society services.",
    "The Data Protection Act 2018, section 9, modifies Article 8 of the UK GDPR so that the EU GDPR's 16-year threshold is read as 13 years in the UK.",
    "Section 9 of the Data Protection Act 2018 states that in Article 8(1) of the GDPR, references to '16 years' are to be read as references to '13 years'.",
    "The Data Protection Act 2018 was originally enacted on 25 May 2018.",
    "The primary legislation is section 9 of the Data Protection Act 2018.",
    "As of 4 October 2026, the minimum age is 13 years old for a child to give their own consent to the processing of personal data.",
    "Under the Data Protection Act 2018 in the UK, the minimum age at which a child can consent to information society services is 13 years old.",
]


def claims(texts):
    return [Claim(id=f"c{i}", job_id="j", claim=t, kind="fact", provider_sources=["chatgpt"]) for i, t in enumerate(texts)]


def test_the_live_false_conflicts_are_not_conflicts():
    found = find_contradictions(claims(REAL[:5]))
    assert found == [], [(f["kind"], f["detail"], f["left"].claim[:40], f["right"].claim[:40]) for f in found]


def test_identifiers_and_names_are_not_figures():
    assert "9" not in comparable_text("The primary legislation is section 9 of the Data Protection Act 2018.")
    assert "2018" not in comparable_text("the Data Protection Act 2018")
    assert "25 May 2018" in comparable_text("enacted on 25 May 2018")


def test_a_replaced_figure_is_not_a_competing_answer_but_a_real_disagreement_still_is():
    assert "16" not in comparable_text("references to 16 years are to be read as 13 years")
    real = find_contradictions(claims([
        "The minimum age at which a child can consent to information society services is 13 years.",
        "The minimum age at which a child can consent to information society services is 16 years.",
    ]))
    assert real and real[0]["kind"] == "figure"


def test_regulators_are_primary():
    for url in ("https://ico.org.uk/for-organisations/x", "https://www.ofcom.org.uk/y", "https://www.parliament.uk/z"):
        assert tier_for(url) == SourceTier.GOVERNMENT
    assert tier_for("https://example.org.uk/") != SourceTier.GOVERNMENT


def test_ai_listed_urls_under_an_opened_heading_count_as_opened_by_ai():
    labels = ops.labels_from_text("Pages I opened:\nhttps://www.legislation.gov.uk/ukpga/2018/12/section/9\n\nMentioned only:\nhttps://c.example/z")
    assert labels["https://www.legislation.gov.uk/ukpga/2018/12/section/9"] is True
    assert labels["https://c.example/z"] is False


SIDE = (
    "DIRECT ANSWER\nIt is 13 under section 9 of the Data Protection Act 2018.\n\n"
    "KEY CLAIMS\n1. The minimum age of consent to information society services in the UK is 13 under section 9 of the Data Protection Act 2018.\n"
    "2. The Data Protection Act 2018 was originally enacted on 25 May 2018.\n"
    "3. The primary legislation is section 9 of the Data Protection Act 2018.\n\n"
    f"Pages I opened:\nSection 9 Data Protection Act 2018 minimum age 13 consent information society services: {URL9}\n"
)


async def test_a_side_conflict_does_not_block_the_early_stop(net):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    settings = providers("chatgpt", "gemini", "qwen")
    job, log, _ = await run(settings, {"chatgpt": {"answer": SIDE, "citations": []}, "gemini": {"answer": "never"}, "qwen": {"answer": "never"}}, LAW_Q)
    assert [c["provider"] for c in log] == ["chatgpt"], "one round, no follow-up, no parallel AIs"
    assert job.assessments[0].sufficient and job.assessments[0].strong_primary
    assert any(c.ai_opened is True for r in job.responses for c in r.citations), "listed under an opened heading = opened by the AI"


# live uk-dpa-age run, level 3: every provider said 13, yet gemini-vs-gemini and chatgpt-vs-chatgpt splits were
# flagged material and the answer became "Couldn't verify that one."
def _claim(cid, text, *providers):
    return Claim(id=cid, job_id="j", claim=text, kind="fact", provider_sources=list(providers))


SELF_SPLIT = [
    _claim("g1", "A child aged 13 can consent to information society services in the UK.", "gemini"),
    _claim("g2", "A child aged 13 cannot consent to information society services in the UK without a parent.", "gemini"),
    _claim("o1", "Acme Corp released the Bolt router in March 2024.", "chatgpt"),
    _claim("o2", "Acme Corp released the Bolt router in March 2025.", "chatgpt"),
]


def test_claims_from_one_provider_are_never_sources_disagreeing():
    found = find_contradictions(SELF_SPLIT)
    assert found, "the pairs still look like conflicts on their own"
    assert all(f["same_provider"] and not f["material"] for f in found)


def test_a_cross_provider_split_is_still_material():
    found = find_contradictions([
        _claim("g", "Acme Corp released the Bolt router in March 2024.", "gemini"),
        _claim("o", "Acme Corp released the Bolt router in March 2025.", "chatgpt"),
    ])
    assert found and found[0]["material"] and not found[0]["same_provider"]
    shared = find_contradictions([
        _claim("g", "Acme Corp released the Bolt router in March 2024.", "gemini"),
        _claim("o", "Acme Corp released the Bolt router in March 2025.", "gemini", "chatgpt"),
    ])
    assert shared and shared[0]["material"], "a second provider on one side makes it a real disagreement"


async def test_runner_records_no_disagreement_for_one_provider_against_itself():
    from types import SimpleNamespace

    from backend.models import Job
    from backend.orchestrator.runner import ResearchRunner

    emitted = []

    async def emit(*args, **kwargs):
        emitted.append(args)

    fake = SimpleNamespace(cancel=SimpleNamespace(raise_if_cancelled=lambda: None), _emit=emit)
    job = Job(question="Q?")
    result = await ResearchRunner._disagreements(fake, job, SELF_SPLIT, 1)
    assert result == [] and job.disagreements == [] and emitted == []
    cross = SELF_SPLIT + [_claim("x", "Acme Corp released the Bolt router in March 2023.", "copilot")]
    result = await ResearchRunner._disagreements(fake, Job(question="Q?"), cross, 1)
    assert result and all(d.severity == "material" for d in result)
    assert all(len(set(",".join(d.positions).split(","))) > 1 for d in result)
