"""CI-locked replay of answer assembly (no browser, no SQLite).

The fixture is a sanitized export of job_261010234454_b92a0462 -- the DPA/ISS age smoke run whose
stored headline was wrong while the curator synthesis and caveats were largely right.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend.models import ProviderResponse, ProviderStatus, VerifierReport
from backend.research.legislation_frame import temporal_authority_checks
from backend.research.style import is_claim_check
from backend.verification.verifier import build_final_answer

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "assembly" / "dpa_iss_age_replay.json"

MIXED_INTENT = (
    "Is it true that UK law uses 16 as the minimum age for ISS consent, "
    "and what provision establishes the age?"
)


@pytest.fixture(scope="module")
def replay_payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def replay_final(replay_payload: dict):
    report = VerifierReport.model_validate(replay_payload["report"])
    responses = [
        ProviderResponse(
            job_id=report.job_id,
            provider=row["provider"],
            round=row["round"],
            role="primary",
            status=ProviderStatus(row["status"]),
            prompt="",
            answer_text="",
        )
        for row in replay_payload["responses"]
    ]
    return build_final_answer(
        report,
        responses,
        replay_payload["rounds_run"],
        replay_payload["question"],
    )


def test_fixture_replay_identifies_age_13_not_16(replay_final):
    low = replay_final.answer.lower()
    assert re.search(r"\b13\b", replay_final.answer)
    assert "not 16" in low or "does not apply the gdpr default" in low or "not the gdpr default" in low


def test_fixture_replay_names_current_uk_gdpr_article_8(replay_final):
    checks = temporal_authority_checks(replay_final.answer, replay_final.caveats)
    assert checks["current_art8"], replay_final.answer


def test_fixture_replay_treats_section_9_as_historical_not_operative(replay_final):
    checks = temporal_authority_checks(replay_final.answer, replay_final.caveats)
    assert checks["s9_historical"], replay_final.answer
    assert checks["s9_not_operative_now"]


def test_fixture_replay_no_refusal_headline_on_supported_informational_answer(replay_final):
    low = replay_final.answer.lower()
    assert replay_final.truth_state == "TRUE"
    assert "nah, that doesn't work like that" not in low
    assert "couldn't find that documented" not in low
    assert "couldn't verify that one" not in low


def test_mixed_intent_question_stays_a_claim_check():
    assert is_claim_check(MIXED_INTENT)


def test_legislation_frame_does_not_fire_without_omitted_section_caveat():
    from backend.research.legislation_frame import apply_uk_iss_age_temporal_frame

    q = "Under the Data Protection Act 2018, what is the minimum age for ISS consent in the UK?"
    base = "The minimum age is 13; the UK does not use 16 as the default."
    assert apply_uk_iss_age_temporal_frame(base, [], q) == base
