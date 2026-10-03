"""Memory retrieval: general questions get nothing, personal ones get a few, irrelevant memory stays out."""

from __future__ import annotations

import pytest

from backend.memory import MemoryService, MemoryStore
from backend.memory.schema import MemoryType, Source

FACTS = [
    "Lives in Kyiv",
    "Works at Acme Robotics as a backend developer",
    "Prefers concise answers without preamble",
    "Is allergic to peanuts",
    "Owns a cat called Miso",
    "Plays the cello on weekends",
    "Is learning Rust",
    "Has a brother called Taras",
]


@pytest.fixture
def svc():
    s = MemoryStore(":memory:")
    for f in FACTS:
        s.add(f, slot="residence" if f.startswith("Lives") else None)
    s.add("Project Atlas database is PostgreSQL 16", memory_type=MemoryType.PROJECT, project="atlas")
    yield MemoryService(s)
    s.close()


@pytest.mark.parametrize("q", ["What is the capital of France?", "how tall is Everest", "explain quantum tunnelling", "what is 2 + 2"])
def test_general_questions_inject_nothing(svc, q):
    block, ret = svc.context_for(q)
    assert ret.kind == "general" and ret.budget == 0 and not ret.hits and block == ""


def test_personal_question_gets_the_relevant_memory_only(svc):
    block, ret = svc.context_for("what's my cat called?")
    assert [h.memory.content for h in ret.hits] == ["Owns a cat called Miso"]
    assert "Kyiv" not in block and "cello" not in block


def test_every_hit_explains_why(svc):
    _b, ret = svc.context_for("where do I live?")
    assert ret.hits and all(h.why for h in ret.hits)


def test_budget_grows_with_complexity(svc):
    _b, personal = svc.context_for("where do I live?")
    _b, project = svc.context_for("Which database does Atlas use?", project="atlas")
    assert personal.budget < project.budget
    assert len(personal.hits) <= personal.budget
    from backend.memory.retrieve import BUDGETS
    assert BUDGETS["general"] == 0 < BUDGETS["personal"] < BUDGETS["project"] < BUDGETS["complex"]


def test_budget_is_a_hard_cap(svc):
    for i in range(12):
        svc.store.add(f"Colleague number {i} named Person{i} works on the platform team")
    _b, ret = svc.context_for("which colleagues work on the platform team?")
    assert len(ret.hits) <= ret.budget


def test_block_has_header_footer_and_no_metadata(svc):
    block, ret = svc.context_for("where do I live?")
    assert block.startswith("Relevant context about the user:")
    assert "current instructions override" in block
    assert "mem_" not in block and "confidence" not in block


def test_inject_switch_turns_retrieval_off(svc):
    svc.store.set_setting("inject", "off")
    block, ret = svc.context_for("where do I live?")
    assert block == "" and not ret.hits


def test_explicit_beats_inferred_in_the_same_slot_at_retrieval(svc):
    svc.store.add("Lives in Odesa", slot="residence", source=Source.MODEL_INFERRED)  # rejected: explicit Kyiv exists
    _b, ret = svc.context_for("where do I live?")
    assert [h.memory.content for h in ret.hits] == ["Lives in Kyiv"]


def test_project_memory_not_offered_to_non_project_questions(svc):
    _b, ret = svc.context_for("which database do I use, PostgreSQL?")
    assert not any(h.memory.project == "atlas" for h in ret.hits)


def test_stage_timings_are_reported(svc):
    _b, ret = svc.context_for("where do I live?")
    assert "vector_ms" in ret.stages and ret.elapsed_ms >= 0