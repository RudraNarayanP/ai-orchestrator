"""Statute-text questions are researched, not answered from memory (live eval: Theft Act s.1 answered in 8 s with nothing opened)."""

from __future__ import annotations

import pytest

from backend.models import EscalationLevel
from backend.research import router


@pytest.mark.parametrize(
    "q",
    [
        "How does section 1 of the Theft Act 1968 define theft?",
        "What does Article 10 of the Constitution of Ukraine say?",
        "What is the minimum marriageable age under the Family Code of Ukraine?",
    ],
)
def test_law_text_questions_are_never_answered_from_model_memory(q):
    analysis = router.classify(q, llm_stable_answer="Confident answer from memory.")
    assert not analysis.can_answer_directly
    assert analysis.needs_web_research
    assert analysis.starting_level != EscalationLevel.DIRECT


def test_ordinary_stable_knowledge_still_answers_directly():
    analysis = router.classify("Give me the definition of recursion", llm_stable_answer="A function calling itself.")
    assert analysis.can_answer_directly