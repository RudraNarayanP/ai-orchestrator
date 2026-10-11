"""Temporal UK legislation framing (assembly layer only)."""

from __future__ import annotations

from backend.research.legislation_frame import apply_uk_iss_age_temporal_frame, temporal_authority_checks

Q = "Under the Data Protection Act 2018, what is the minimum age at which a child can consent to information society services in the UK?"

OMITTED_CAVEAT = (
    "Although section 9 was omitted from the Data Protection Act 2018 after the Brexit transition period, "
    "the age threshold of 13 years is now directly embedded in the retained UK GDPR."
)


def test_frames_current_art8_and_historical_section_9():
    base = (
        "Under the UK Data Protection Act 2018, a child can consent at age 13; "
        "the UK does not apply the GDPR default age of 16."
    )
    out = apply_uk_iss_age_temporal_frame(base, [OMITTED_CAVEAT], Q, why="")
    checks = temporal_authority_checks(out, [OMITTED_CAVEAT])
    assert checks["current_art8"]
    assert checks["s9_historical"]
    assert checks["s9_not_operative_now"]
    assert "originally" in out.lower() and "omitted" in out.lower()


def test_replaces_operative_section_9_wording():
    base = "The operative provision is section 9 of the Data Protection Act 2018, which substitutes 13 for 16."
    out = apply_uk_iss_age_temporal_frame(base, [OMITTED_CAVEAT], Q, why="")
    assert "operative provision is section 9" not in out.lower()
    assert "article 8(1)" in out.lower()
