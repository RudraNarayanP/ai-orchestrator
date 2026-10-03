"""Answer clean-up rules learned from live logged-out runs."""

from __future__ import annotations

import pytest

from browser.adapters.base import BROKEN_RE, ChatAdapter
from backend.settings import ProviderConfig
from tests.conftest import base_settings


def adapter():
    return ChatAdapter(object(), base_settings(), "chatgpt", ProviderConfig(enabled=True, label="X", url="https://x.test/"))


def test_chatgpt_logged_out_attribution_heading_does_not_leak_into_the_answer():
    """Live: the captured ChatGPT answer began with the lone line '#### :' (a heading whose text was visually hidden)."""
    raw = "#### :\nThe **Eiffel Tower** was completed on **March 31, 1889**."
    assert adapter()._clean(raw) == "The **Eiffel Tower** was completed on **March 31, 1889**."
    assert adapter()._clean("ChatGPT said:\nParis.") == "Paris."


def test_headings_glued_to_the_previous_line_are_split():
    """Live Gemini capture: '...is 300 meters (984 feet).EVIDENCE' and '...[link]SOURCE DATES' were one line,
    so 'EVIDENCE' became part of a claim and the section parser lost its sections."""
    raw = "KEY CLAIMS\n\n- The tower is 300 meters (984 feet).EVIDENCE\n\n- It was built for the fair.SOURCE LINKS\n\n- https://a.example/x"
    cleaned = adapter()._clean(raw)
    assert "feet).EVIDENCE" not in cleaned
    assert "(984 feet).\nEVIDENCE" in cleaned and "fair.\nSOURCE LINKS" in cleaned
    assert adapter()._clean("There is some evidence of this.") == "There is some evidence of this."
    assert adapter()._clean("The EVIDENCE is clear.") == "The EVIDENCE is clear.", "only a heading at the end of a line is split"


def test_real_headings_survive():
    assert adapter()._clean("## Details\n- one") == "## Details\n- one"


@pytest.mark.parametrize("text", [
    "Confirm your age to continue What year were you born? 2000 Continue",
    "Please verify your age",
    "Enter your date of birth",
    "Hey, I'm Pi. But first, what should I call you? Preferred name",
])
def test_age_gates_read_as_blocked(text):
    assert BROKEN_RE.search(text)