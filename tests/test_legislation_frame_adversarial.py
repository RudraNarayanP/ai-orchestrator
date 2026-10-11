"""Adversarial assembly cases for UK ISS age authority — wording variants, not one golden phrase."""

from __future__ import annotations

from backend.research.legislation_frame import apply_uk_iss_age_temporal_frame, temporal_authority_checks

Q = (
    "Under the Data Protection Act 2018, what is the minimum age at which a child can consent to "
    "information society services in the UK?"
)
AGE_OK = (
    "Under the UK Data Protection Act 2018, a child can consent at age 13; "
    "the UK does not apply the GDPR default age of 16."
)
OMITTED_IN_CAVEAT = (
    "Although section 9 was omitted from the Data Protection Act 2018 after the Brexit transition period, "
    "the age threshold of 13 years is now directly embedded in the retained UK GDPR."
)
OMITTED_IN_WHY = (
    "The legislation.gov.uk text shows section 9 of the Data Protection Act 2018 was omitted with effect "
    "from 31 December 2020; Article 8(1) UK GDPR now states 13 years directly."
)
OPERATIVE_S9 = (
    "The operative provision is section 9 of the Data Protection Act 2018, which substitutes 13 for 16."
)


def test_omission_only_in_why_still_frames_current_and_historical_authority():
    out = apply_uk_iss_age_temporal_frame(AGE_OK, [], Q, why=OMITTED_IN_WHY)
    checks = temporal_authority_checks(out, [])
    assert checks["current_art8"] and checks["s9_historical"]
    assert checks["s9_not_operative_now"]
    assert not checks["authority_unresolved"]


def test_no_omission_signal_does_not_invent_article_8():
    out = apply_uk_iss_age_temporal_frame(AGE_OK, [], Q, why="")
    checks = temporal_authority_checks(out, [])
    assert out == AGE_OK
    assert not checks["current_art8"]
    assert not checks["s9_historical"]


def test_operative_section_9_without_omission_stays_unresolved_on_authority():
    out = apply_uk_iss_age_temporal_frame(OPERATIVE_S9, [], Q, why="")
    checks = temporal_authority_checks(out, [])
    assert checks["age_13_not_16"]
    assert not checks["current_art8"]
    assert checks["authority_unresolved"]
    assert checks["s9_not_operative_now"]
    assert "operative provision is section 9" not in out.lower()
