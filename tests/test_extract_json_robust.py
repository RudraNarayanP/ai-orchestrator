"""extract_json must never raise on odd model output (live: IndexError killed a research job in claim extraction)."""

from __future__ import annotations

import pytest

from backend.verification.llm import _repair_truncated, extract_json


def test_comma_after_the_first_object_closed_does_not_raise():
    _repair_truncated('{"a": 1}, {"claims": [{"claim": "x"},')  # used to raise IndexError (empty stack)


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}, {"claims": [{"claim": "x"},',
        '{"a": 1},',
        'Sure: {\'claims\': []}, then {"claims": [{"claim": "y"}, {"claim": ',
        '}, {',
        '],[',
        '{"claims": [{"claim": "ok"}, {"claim": "cut off',
        '',
    ],
)
def test_extract_json_never_raises(text):
    extract_json(text)  # may return None or a dict, never an exception


def test_a_truncated_reply_is_still_repaired():
    out = extract_json('{"claims": [{"claim": "a"}, {"claim": "b"}, {"claim": "cut')
    assert out and out["claims"][0]["claim"] == "a"