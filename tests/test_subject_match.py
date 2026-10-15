"""A page about the right idea but the wrong subject is not evidence (live eval defect).

"The University of Oxford defines plagiarism as ..." was confirmed by another university's
library page with near-identical wording, and the answer then called it Oxford's own guidance.
"""

from __future__ import annotations

from backend.evidence.sources import FetchedPage, check_support, subject_names
from backend.models import SourceCheckStatus

CLAIM = "The University of Oxford defines plagiarism as presenting someone else's work or ideas as your own without full acknowledgement."
NTU = (
    "Plagiarism is presenting someone else's work or ideas as your own, without full acknowledgement. "
    "Nottingham Trent University library guidance on plagiarism and Turnitin for students."
)
OXFORD = (
    "University of Oxford: Plagiarism is presenting someone else's work or ideas as your own, "
    "with or without their consent, by incorporating it into your work without full acknowledgement."
)


def page(text, url):
    return FetchedPage(url=url, final_url=url, status=200, title="t", text=text, ok=True)


def test_subject_names_pick_proper_names_not_generic_words():
    assert subject_names(CLAIM) == ["oxfor"]
    assert subject_names("Andrew Wiles proved Fermat's Last Theorem; the Poincar\u00e9 conjecture is separate.") == ["wiles", "ferma", "theor", "poinc"]
    assert subject_names("the minimum age is 13") == []


def test_a_page_about_another_institution_does_not_confirm_the_claim():
    got = check_support(CLAIM, page(NTU, "https://www.ntu.ac.uk/m/library/plagiarism-and-turnitin"))
    assert got["status"] != SourceCheckStatus.CONFIRMED
    assert any(str(m).startswith("subject:") for m in got["missing"])


def test_the_named_institution_page_still_confirms():
    assert check_support(CLAIM, page(OXFORD, "https://www.ox.ac.uk/students/academic/guidance/skills/plagiarism"))["status"] == SourceCheckStatus.CONFIRMED


def test_accents_do_not_break_the_name_match():
    claim = "Perelman proved the Poincar\u00e9 conjecture between 2002 and 2003."
    ok = check_support(claim, page("Grigori Perelman proved the Poincare conjecture; preprints appeared in 2002 and 2003.", "https://e.example/p"))
    assert ok["status"] == SourceCheckStatus.CONFIRMED


def test_claim_ids_are_internal_vocabulary():
    from backend.research.style import INTERNAL_VOCAB_RE, plain_caveats

    assert INTERNAL_VOCAB_RE.search("see clm_0ee3a7f218db40a0 for detail")
    assert plain_caveats(["clm_0ee3a7f218db40a0", "Check the date yourself."]) == ["Check the date yourself."]

# ---- 2026-10-15 live IPC run: a section menu is not a page that says the thing

IPC_CLAIM = "Section 378 and Section 379 of the legacy Indian Penal Code formerly defined and punished theft."
_FILLER = "Cookie consent language, advertising slots, newsletter signup, careers page, privacy policy links, "
MENU_ONLY = (
    "Skip to main content IPC2BNS Converter Shop Services Free Resources Donate Blog Legal Help Consult a Lawyer "
    "Popular sections Section 378 Section 379 Section 420 theft definition punishment imprisonment three years fine "
    "map of the code to the new law sections list " + _FILLER * 20
    + " about the Bharatiya Nyaya Sanhita and the Indian Penal Code, formerly the code of criminal law in India, "
    "definition of theft and its punishment, a mapping tool for advocates and litigants."
)
SUBSTANCE = (
    MENU_ONLY
    + " Section 378 Indian Penal Code, 1860 defined theft: whoever intending to take dishonestly any movable property "
    "out of the possession of any person, that person's moveable property without his consent. Section 379 of the "
    "Indian Penal Code punished theft with imprisonment of either description for a term which may extend to three years."
)


def test_a_figure_that_only_appears_in_a_list_of_sections_does_not_confirm():
    """The page we opened is a converter site whose first words are navigation, and the only place its numbers sit is a
    menu of popular sections. That is not the sentence the claim stands on, so it must not be filed as confirmed."""
    got = check_support(IPC_CLAIM, page(MENU_ONLY, "https://www.ipc2bns.in/section/ipc-379"))
    assert got["status"] != SourceCheckStatus.CONFIRMED, got
    assert any(str(m).startswith("figure only in text") for m in got["missing"]), got["missing"]


def test_the_passage_that_names_the_figure_and_the_subject_is_the_excerpt_kept():
    got = check_support(IPC_CLAIM, page(SUBSTANCE, "https://www.ipc2bns.in/section/ipc-379"))
    assert got["status"] == SourceCheckStatus.CONFIRMED, got
    assert "Indian Penal Code" in (got["excerpt"] or ""), got["excerpt"]
    assert "378" in (got["excerpt"] or "")
