"""The stored excerpt for a claim with a figure shows the figure (live: Oxford DPhil 100,000 words)."""

from backend.evidence.sources import FetchedPage, check_support


def test_excerpt_is_centred_on_the_claims_figure_not_the_first_common_word():
    filler = "Examination regulations for the doctoral degree. " * 30
    text = "Oxford DPhil regulations, schedule 1000 applies. " + filler + "The thesis shall not exceed 100,000 words, excluding the bibliography. " + filler
    page = FetchedPage(url="https://examregs.example.ac.uk/r", ok=True, status=200, text=text)
    out = check_support("The maximum length of an Oxford DPhil thesis is 100,000 words.", page)
    assert out["status"].value == "confirmed"
    assert "100,000 words" in out["excerpt"]


def test_excerpt_still_works_without_a_figure():
    page = FetchedPage(url="https://x.example/r", ok=True, status=200, text="Theft Act 1968. A person is guilty of theft if he dishonestly appropriates property belonging to another.")
    out = check_support("A person is guilty of theft if he dishonestly appropriates property belonging to another.", page)
    assert out["excerpt"] and "dishonestly" in out["excerpt"]