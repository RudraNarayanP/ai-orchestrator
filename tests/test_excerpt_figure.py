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

def test_the_excerpt_is_the_passage_where_the_figure_is_about_the_subject_not_the_menu():
    """Replayed from job_261015234541_22b1544d (live IPC theft): the stored verbatim excerpt for the rows that
    "confirmed" Section 378 was "Skip to main content IPC2BNS Converter Shop Services ...", because the first
    place the number is written is a Popular-IPC-Sections menu. The figure was on the page; what it was said
    *about* was not in the record, and the eval read the answer's section number as one no opened page carried."""
    chrome = (
        "Skip to main content IPC2BNS Converter Shop Services Free Resources Donate Blog Legal Help "
        "Popular IPC Sections Section 378 Section 379 Section 420 Section 304 "
    )
    body = (
        "Section 378 of the Indian Penal Code, 1860 defines theft: whoever intending to take dishonestly any "
        "movable property out of the possession of any person without that person's consent, moves that property."
    )
    page = FetchedPage(url="https://www.ipc2bns.in/section/ipc-379", ok=True, status=200, text=chrome + "x" * 400 + body)
    out = check_support("Section 378 and Section 379 of the Indian Penal Code defined and punished theft.", page)
    assert "Indian Penal Code" in out["excerpt"], out["excerpt"][:200]
    assert "Skip to main content" not in out["excerpt"], out["excerpt"][:200]


def test_every_figure_the_claim_stands_on_is_in_the_recorded_passage():
    """A section-mapping page puts "IPC 378" and "IPC 379" on different lines. One window cannot hold both, and an
    excerpt that carries only one leaves the other figure looking invented (the live IPC theft run, twice)."""
    text = (
        "Home / Sections / IPC 379 to BNS 303. IPC Section 379 (Theft) under the old Indian Penal Code 1860 is now "
        "covered under Section 303 of the Bharatiya Nyaya Sanhita, 2023. " + "mapping table of offences. " * 12
        + "IPC Section 378 defines theft: whoever intending to take dishonestly any movable property out of the "
        "possession of any person. Indian Penal Code 1860, section 378 punishment three years."
    )
    page = FetchedPage(url="https://ipc2bns.example/section/ipc-379", ok=True, status=200, text=text)
    out = check_support("Section 378 and Section 379 of the Indian Penal Code, 1860 defined and punished theft.", page)
    for figure in ("378", "379", "1860"):
        assert figure in (out["excerpt"] or ""), (figure, out["excerpt"][:200])
