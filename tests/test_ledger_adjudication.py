"""The claim ledger, the final confidence and the stop note are one judgement, not three.

Live SAR run 2026-10-09: the curator supported six claims with opened ICO and GOV.UK
pages and the answer said High confidence, while all 24 persisted claim rows still read
`unverified` and the stop note said "5 material claim(s) still unconfirmed". The verdicts
were never written back onto the claims, so the audit trail contradicted the answer.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.models import Claim, ClaimStatus, Confidence, Job, ResearchMode, SourceCheckStatus
from backend.orchestrator.runner import ResearchRunner
from backend.research import claims as claim_ops
from backend.storage.db import Store
from backend.evidence.sources import SourceTier
from backend.verification.verifier import Verifier
from backend.verification.llm import Endpoint
from tests.conftest import adapters_from, base_settings

Q = "When did Acme release the Bolt?"
PRESS = "https://acme.example/press"
REGISTRY = "https://gov.example/device-register"


def curator(*, decide, answer: str, confidence: str, why: str = "both opened pages state it", needs_more: bool = False):
    """A curator that returns one verdict per claim it was shown, decided by `decide(text)`."""
    seen: list[str] = []

    def reply(body: dict[str, Any]) -> str:
        user = body["messages"][-1]["content"]
        seen.append(user)
        rows = re.findall(r'"claim_id": "([^"]+)",\s*"claim": "([^"]+)"', user)
        # decide() returning None models a curator that simply did not judge that wording.
        verdicts = [{"claim_id": cid, "claim": text, **decide(text)} for cid, text in rows if decide(text) is not None]
        return json.dumps(
            {
                "verdicts": verdicts,
                "answer": answer,
                "why": why,
                "confidence": confidence,
                "needs_more_research": needs_more,
                "research_needed": [],
                "unresolved": [],
            }
        )

    reply.seen = seen
    return reply


def make_curator(server) -> Verifier:
    return Verifier(Endpoint(provider="openai_compatible", model="fake-model", base_url=server.base_url, timeout_s=10), min_independent_sources=2)


async def ask(settings, scripts, question=Q, *, verifier=None, max_rounds=2):
    adapters = adapters_from(scripts, settings)
    runner = ResearchRunner(settings, adapters, engine=None, verifier=verifier)
    job = Job(question=question, mode=ResearchMode.STANDARD, max_rounds=max_rounds)
    return await runner.run(job)


def on_point_supported(text: str) -> dict[str, Any]:
    if "March 2026" in text:
        return {
            "verdict": "supported", "confidence": "high",
            "reasoning": "the press release and the register both give March 2026",
            "strong_evidence": [PRESS, REGISTRY], "problems": [],
        }
    return {"verdict": "insufficient_evidence", "confidence": "low", "reasoning": "no opened page gives it", "strong_evidence": [], "problems": []}


# --------------------------------------------------------------- verdict -> ledger


async def test_curator_verdicts_land_on_the_claims_they_judged(net, fake_openai):
    """A supported verdict and an insufficient one sit on different claim rows -- the
    ledger does not inherit the headline confidence."""
    net.confirm(PRESS, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="Acme released the Bolt in March 2026.")
    net.confirm(REGISTRY, tier=SourceTier.GOVERNMENT, excerpt="Acme released the Bolt in March 2026.")
    split = {
        "chatgpt": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
        "gemini": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": REGISTRY, "title": "Acme released the Bolt in March 2026 (register)"}]},
        "qwen": {"answer": "KEY CLAIMS\n1. The Bolt weighs 1.2 kg.", "citations": []},
        "search": {"answer": "results", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
    }
    fn = curator(decide=on_point_supported, answer="Acme released the Bolt in March 2026.", confidence="high")
    job = await ask(base_settings(), split, verifier=make_curator(fake_openai.script(fn)))

    assert job.verifier_calls >= 1 and fn.seen, "the curator was actually consulted"
    for verdict in job.reports[-1].verdicts:
        row = next((c for c in job.claims if c.id == verdict.claim_id), None)
        assert row is not None, f"a verdict for a claim that is not in the ledger: {verdict.claim!r}"
        assert row.status == verdict.verdict and row.confidence == verdict.confidence, (
            f"{row.claim!r}: ledger says {row.status}/{row.confidence}, the curator said {verdict.verdict}/{verdict.confidence}"
        )
    settled = [c for c in job.claims if "2026" in c.claim]
    rejected = [c for c in job.claims if "weighs" in c.claim]
    assert settled and rejected
    for claim in settled:
        assert claim.status == ClaimStatus.SUPPORTED, f"{claim.claim!r} still {claim.status}"
        assert claim.confidence is not None and claim.rationale
        assert PRESS in claim.supporting, "the evidence the verdict cited is on the claim"
    for claim in rejected:
        assert claim.status is not ClaimStatus.SUPPORTED, "the unproven side claim must not inherit the answer's support"
    assert job.final.confidence == Confidence.HIGH


async def test_nothing_confirmed_keeps_the_ledger_and_the_answer_agree(net, fake_openai):
    """The other direction: no page confirms anything, so no claim may read supported and
    the answer must not assert a date."""
    net.fail(PRESS, status=SourceCheckStatus.MISMATCH)
    net.fail(REGISTRY, status=SourceCheckStatus.MISMATCH)
    scripts = {
        "chatgpt": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": PRESS, "title": "Bolt"}]},
        "gemini": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": REGISTRY, "title": "Register"}]},
        "search": {"answer": "results", "citations": []},
    }
    decide = lambda text: {"verdict": "supported", "confidence": "high", "reasoning": "the pages say so", "strong_evidence": [PRESS], "problems": []}
    fn = curator(decide=decide, answer="Acme released the Bolt in March 2026.", confidence="high")
    job = await ask(base_settings(), scripts, verifier=make_curator(fake_openai.script(fn)))

    assert job.claims and all(c.status != ClaimStatus.SUPPORTED for c in job.claims), [
        (c.claim, c.status) for c in job.claims if c.status == ClaimStatus.SUPPORTED
    ]
    assert job.final.confidence in {Confidence.LOW, Confidence.NONE}, job.final.confidence
    assert "March 2026" not in job.final.answer, "nothing confirmed it, so the answer must not assert it"
    assert job.final.answer.lower().startswith(
        ("i don't know", "i couldn't", "i could not", "i cannot", "i can't", "couldn't verify", "i can't verify")
    ), job.final.answer


async def test_the_deterministic_ledger_adjudicates_without_a_curator(net):
    """No verifier endpoint: the ledger itself decides, and its decision is what the
    claim rows must carry -- not `unverified` forever."""
    net.confirm(PRESS, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="Acme released the Bolt in March 2026.")
    net.confirm(REGISTRY, tier=SourceTier.GOVERNMENT, excerpt="Acme released the Bolt in March 2026.")
    scripts = {
        "chatgpt": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
        "gemini": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": REGISTRY, "title": "Acme released the Bolt in March 2026 (register)"}]},
        "search": {"answer": "results", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
    }
    job = await ask(base_settings(), scripts)
    assert job.reports, "the deterministic report still ran"
    supported = [c for c in job.claims if c.status == ClaimStatus.SUPPORTED]
    assert supported, [(c.claim, c.status) for c in job.claims]
    assert job.final.confidence in {Confidence.HIGH, Confidence.MODERATE}
    assert all(c.confidence is not None for c in supported)


async def test_wordings_the_curator_never_judged_are_not_reported_as_unconfirmed(net, fake_openai):
    """The SAR shape: a long transcript, one claim the curator actually judged. The rest
    must not be promoted, and a stop note must not call an unjudged wording unconfirmed."""
    net.confirm(PRESS, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="Acme released the Bolt in March 2026.")
    net.confirm(REGISTRY, tier=SourceTier.GOVERNMENT, excerpt="Acme released the Bolt in March 2026.")
    side = "The press release page was last updated on 3 April 2026."
    scripts = {
        "chatgpt": {
            "answer": f"KEY CLAIMS\n1. Acme released the Bolt in March 2026.\n2. {side}",
            "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}],
        },
        "gemini": {
            "answer": "KEY CLAIMS\n1. The Bolt launch date was March 2026.",
            "citations": [{"url": REGISTRY, "title": "Acme released the Bolt in March 2026 (register)"}],
        },
        "qwen": {"answer": "KEY CLAIMS\n1. The Bolt weighs 1.2 kg.", "citations": []},
        "search": {"answer": "results", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
    }
    # the curator returns a verdict for the one wording it judged and nothing else
    def only_primary(text: str):
        return on_point_supported(text) if text == "Acme released the Bolt in March 2026." else None

    fn = curator(decide=only_primary, answer="Acme released the Bolt in March 2026.", confidence="high")
    job = await ask(base_settings(), scripts, verifier=make_curator(fake_openai.script(fn)))

    judged = {v.claim_id for v in job.reports[-1].verdicts}
    assert len(judged) == 1, "one wording judged, the rest left alone"
    settled = [c for c in job.claims if c.claim == "Acme released the Bolt in March 2026."]
    assert settled and settled[0].status == ClaimStatus.SUPPORTED
    unjudged = [c for c in job.claims if c.id not in judged]
    assert unjudged
    for claim in unjudged:
        assert claim.status == ClaimStatus.UNVERIFIED, f"an unjudged wording was promoted: {claim.claim!r}"
        assert claim.rationale and "not among them" in claim.rationale, claim.claim
    note = job.stop_reason or ""
    assert "still unconfirmed" not in note, note
    assert "did not judge" in note, note
    assert job.final.confidence == Confidence.HIGH, note


async def test_an_unproven_side_claim_is_not_phrased_as_doubt_about_the_answer(net, fake_openai):
    """Judged, not proven, and beside the point: the ledger must say so and the note must
    not dress it up as uncertainty about the release date being asked."""
    net.confirm(PRESS, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="Acme released the Bolt in March 2026.")
    net.confirm(REGISTRY, tier=SourceTier.GOVERNMENT, excerpt="Acme released the Bolt in March 2026.")
    scripts = {
        "chatgpt": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
        "gemini": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": REGISTRY, "title": "Acme released the Bolt in March 2026 (register)"}]},
        "qwen": {"answer": "KEY CLAIMS\n1. The Bolt weighs 1.2 kg.", "citations": []},
        "search": {"answer": "results", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
    }
    fn = curator(decide=on_point_supported, answer="Acme released the Bolt in March 2026.", confidence="high")
    job = await ask(base_settings(), scripts, verifier=make_curator(fake_openai.script(fn)))

    weight = [c for c in job.claims if "weighs" in c.claim]
    assert weight and weight[0].status == ClaimStatus.INSUFFICIENT_EVIDENCE, [(c.claim, c.status) for c in job.claims]
    note = job.stop_reason or ""
    assert "side claim" in note, note
    assert "still unconfirmed" not in note, note


async def test_an_unproven_claim_that_answers_the_question_is_called_unconfirmed(net, fake_openai):
    """The guard has both directions: when the claim the curator could not prove IS the
    thing that was asked, the note must keep saying so."""
    net.confirm(PRESS, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="Acme released the Bolt in March 2026.")
    net.confirm(REGISTRY, tier=SourceTier.GOVERNMENT, excerpt="Acme released the Bolt in March 2026.")
    scripts = {
        "chatgpt": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
        "gemini": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": REGISTRY, "title": "Acme released the Bolt in March 2026 (register)"}]},
        "qwen": {"answer": "KEY CLAIMS\n1. Acme released the Bolt on 17 March 2026.", "citations": []},
        "search": {"answer": "results", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
    }
    fn = curator(decide=on_point_supported, answer="Acme released the Bolt in March 2026.", confidence="high")
    job = await ask(base_settings(), scripts, verifier=make_curator(fake_openai.script(fn)))

    day = [c for c in job.claims if "17 March" in c.claim]
    assert day and day[0].status == ClaimStatus.INSUFFICIENT_EVIDENCE, [(c.claim, c.status) for c in job.claims]
    note = job.stop_reason or ""
    assert "still unconfirmed" in note, note
    assert "17 March 2026" in note, f"the note must name what is open, not just count it: {note}"
    assert job.final.confidence is not Confidence.HIGH, (
        f"an unproven claim that answers the question cannot sit under High: {job.final.confidence}"
    )
    assert any("17 March" in c for c in job.final.caveats), job.final.caveats


# ----------------------------------------------------------------- the helpers


def test_annotate_unadjudicated_says_it_was_not_judged_without_changing_status():
    claims = [
        Claim(id="clm_a", job_id="j", claim="Acme released the Bolt in March 2026.", status=ClaimStatus.SUPPORTED),
        Claim(id="clm_b", job_id="j", claim="The Bolt launch date was March 2026."),
    ]
    assert claim_ops.annotate_unadjudicated(claims, {"clm_a"}) == 1
    assert claims[1].status == ClaimStatus.UNVERIFIED, "annotating is not adjudicating"
    assert "not among them" in claims[1].rationale
    assert claim_ops.annotate_unadjudicated(claims, {"clm_a", "clm_b"}) == 0, "a judged claim keeps its own reason"


async def test_the_persisted_ledger_matches_the_answer_it_shipped(net, fake_openai, tmp_path):
    """The audit trail is the promise: history and export must show the verdicts, not a
    wall of `unverified` next to a confident answer."""
    net.confirm(PRESS, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="Acme released the Bolt in March 2026.")
    net.confirm(REGISTRY, tier=SourceTier.GOVERNMENT, excerpt="Acme released the Bolt in March 2026.")
    scripts = {
        "chatgpt": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
        "gemini": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": REGISTRY, "title": "Acme released the Bolt in March 2026 (register)"}]},
        "qwen": {"answer": "KEY CLAIMS\n1. The Bolt weighs 1.2 kg.", "citations": []},
        "search": {"answer": "results", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
    }
    fn = curator(decide=on_point_supported, answer="Acme released the Bolt in March 2026.", confidence="high")
    settings = base_settings(storage={"db_path": str(tmp_path / "ledger.db")})
    job = await ask(settings, scripts, verifier=make_curator(fake_openai.script(fn)))
    Store(settings).save_job(job)

    store = Store(base_settings(storage={"db_path": str(tmp_path / "ledger.db")}))
    rows = store.job_snapshot(job.id)["claims"]
    by_status = {}
    for row in rows:
        by_status.setdefault(row["status"], []).append(row)
    assert "supported" in by_status, [r["status"] for r in rows]
    assert not all(r["status"] == "unverified" for r in rows)
    settled = [r for r in rows if r["status"] == "supported"]
    assert all(r["confidence"] for r in settled), "a settled claim carries how settled it is"
    assert all(r["rationale"] for r in settled), "and why"
    assert all(r["supporting"] for r in settled), "and the pages it was settled on"
    assert job.final.confidence.value == store.job_snapshot(job.id)["confidence"]


# ---------------------------------------------------- a conflict cannot read as settled


async def test_a_material_conflict_caps_the_curated_answer(net, fake_openai):
    """The curator may bless both sides; if the ledger says they materially conflict, the
    answer cannot present as High -- note, ledger and confidence have to agree."""
    net.confirm(PRESS, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="Acme released the Bolt in March 2026.")
    net.confirm(REGISTRY, tier=SourceTier.GOVERNMENT, excerpt="Acme released the Bolt in March 2027.")
    scripts = {
        "chatgpt": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2026.", "citations": [{"url": PRESS, "title": "Acme released the Bolt in March 2026"}]},
        "gemini": {"answer": "KEY CLAIMS\n1. Acme released the Bolt in March 2027.", "citations": [{"url": REGISTRY, "title": "Acme released the Bolt in March 2027"}]},
        "search": {"answer": "results", "citations": []},
    }

    def bless_everything(text: str) -> dict[str, Any]:
        return {"verdict": "supported", "confidence": "high", "reasoning": "a page states it", "strong_evidence": [PRESS, REGISTRY], "problems": []}

    fn = curator(decide=bless_everything, answer="Acme released the Bolt in March 2026.", confidence="high")
    job = await ask(base_settings(), scripts, verifier=make_curator(fake_openai.script(fn)))

    assert [d for d in job.disagreements if d.severity == "material"], [(d.topic, d.severity) for d in job.disagreements]
    assert job.final.confidence is not Confidence.HIGH, f"conflict reported as settled: {job.final.confidence}"
    contested = [c for c in job.claims if c.status == ClaimStatus.CONTESTED]
    assert contested, [(c.claim, c.status) for c in job.claims]
    note = job.stop_reason or ""
    assert "conflict" in note.lower(), note


def test_apply_verdicts_only_touches_the_claims_it_was_given():
    claims = [
        Claim(id="clm_a", job_id="j", claim="Acme released the Bolt in March 2026."),
        Claim(id="clm_b", job_id="j", claim="The Bolt weighs 1.2 kg."),
    ]
    verdicts = [
        type("V", (), {
            "claim_id": "clm_a", "verdict": ClaimStatus.SUPPORTED, "confidence": Confidence.HIGH,
            "reasoning": "two pages say so", "strong_evidence": [PRESS],
        })()
    ]
    assert claim_ops.apply_verdicts(claims, verdicts) == 1
    assert claims[0].status == ClaimStatus.SUPPORTED and claims[0].confidence == Confidence.HIGH
    assert claims[0].supporting == [PRESS]
    assert claims[1].status == ClaimStatus.UNVERIFIED and claims[1].confidence is None


def test_apply_verdicts_puts_refuting_evidence_on_the_contradicting_side():
    from backend.models import ClaimVerdict

    claim = Claim(id="clm_a", job_id="j", claim="The Bolt costs $499.")
    verdict = ClaimVerdict(
        claim_id="clm_a", claim=claim.claim, verdict=ClaimStatus.REFUTED, confidence=Confidence.HIGH,
        reasoning="the store lists $549", strong_evidence=[PRESS],
    )
    claim_ops.apply_verdicts([claim], [verdict])
    assert claim.status == ClaimStatus.REFUTED
    assert claim.contradicting == [PRESS] and claim.supporting == []


