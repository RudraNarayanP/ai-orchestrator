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