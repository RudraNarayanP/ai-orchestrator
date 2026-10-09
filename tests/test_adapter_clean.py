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


def test_inline_citation_chip_labels_are_removed_but_prose_is_not():
    """Live Gemini/ChatGPT product answers: '... comfort. Headphones Addict', '... hinge. SoundGuys', '... Sony UK+1'."""
    from backend.models import Citation
    from browser.adapters.base import strip_chip_labels

    cites = [Citation(url="https://headphonesaddict.com/sony-review/", title="x"), Citation(url="https://sundr.ca/failure-timeline/x", title="y")]
    raw = (
        "- The headphones deliver class-leading noise cancelling. Headphones Addict\n"
        "- Users report that ear cushions flake over time. Sundr\n"
        "- Sony warns that overstretching causes loose hinges. Sony UK+1\n"
        "- It launched in 2022. Sony says it is the best yet.\n"
        "- Prices start at $399. Amazon"
    )
    got = strip_chip_labels(raw, cites).split("\n")
    assert got[0].endswith("noise cancelling.") and got[1].endswith("over time.") and got[2].endswith("loose hinges.")
    assert got[3] == "- It launched in 2022. Sony says it is the best yet."
    assert got[4].endswith("$399."), "a bullet's trailing chip is removed even with no citations to match"
    live = strip_chip_labels("- The Eiffel Tower's construction was completed on March 31, 1889. La tour Eiffel", None)
    assert live.endswith("1889."), live
    assert strip_chip_labels("- It opened in 1889. It is still the tallest structure in the city.", None).endswith("city.")
    assert strip_chip_labels("Prices start at $399. Amazon", None).endswith("Amazon"), "outside a list an unmatched word stays"
    assert strip_chip_labels("- Sony specifies about 3.5 hours for a full charge. RTINGS.com", None).endswith("charge.")


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


def test_pi_read_aloud_and_more_options_controls_are_not_part_of_the_answer():
    """Live (pi.ai 2026-10-08, parallel run): the answer ended with Pi's "Read aloud" / "More options" controls."""
    raw = "The Eiffel Tower was completed in 1889.\nSource: Britannica.\n\nRead aloud\n\nMore options"
    assert adapter()._clean(raw) == "The Eiffel Tower was completed in 1889.\nSource: Britannica."
    assert "read aloud" in adapter()._clean("You can read aloud the passage below.").lower(), "only a whole control line is dropped"


def test_the_page_library_version_check_matches_the_library():
    """base._install used to look for version 3 while the library said 4, so it re-injected on every call."""
    import inspect

    from browser.adapters import base
    from browser.adapters.dom_library import DOM_LIBRARY_JS, DOM_LIBRARY_VERSION

    assert f"window.__omnibrain.version === {DOM_LIBRARY_VERSION}) return 'already'" in DOM_LIBRARY_JS
    assert f"    version: {DOM_LIBRARY_VERSION},\n" in DOM_LIBRARY_JS
    assert "DOM_LIBRARY_VERSION" in inspect.getsource(base.ChatAdapter._install)
