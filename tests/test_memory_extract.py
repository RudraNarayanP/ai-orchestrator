"""Memory extraction: only what the user said, never secrets, honouring opt-outs."""

from __future__ import annotations

import pytest

from backend.memory import MemoryService, MemoryStore
from backend.memory.extract import extract
from backend.memory.schema import MemoryType, Source


def kept(msg):
    return extract(msg).candidates


def test_dont_remember_saves_nothing():
    e = extract("Don't remember this: my sister is called Anya")
    assert not e.candidates and e.skipped == "no_remember"


@pytest.mark.parametrize("msg", [
    "my password is hunter2",
    "My api key is sk-abcdef1234567890abcdef",
    "here is my token ghp_abcdefghijklmnopqrstuvwxyz0123456789",
])
def test_secrets_are_never_stored(msg):
    e = extract(msg)
    assert not e.candidates and e.skipped == "secret"


def test_secret_next_to_a_fact_keeps_nothing():
    assert not kept("my password is hunter2 and I live in Kyiv")


@pytest.mark.parametrize("msg", ["Do I live in Kyiv?", "What is the capital of France?", "Where do I work?"])
def test_questions_are_not_facts(msg):
    assert not kept(msg)


def test_hypotheticals_are_not_facts():
    assert not kept("If I lived in Paris I would eat croissants")


def test_chitchat_is_dropped():
    assert not kept("ok thanks")


def test_plain_statement_is_explicit_and_neutral():
    (c,) = kept("I live in Kyiv.")
    assert c.source == Source.USER_EXPLICIT and c.content == "Lives in Kyiv" and c.slot == "residence"


def test_soft_preferences_are_inferred_not_explicit():
    (c,) = kept("I like dark roast coffee")
    assert c.source == Source.MODEL_INFERRED and c.memory_type == MemoryType.PREFERENCE


def test_remember_that_is_explicit():
    (c,) = kept("Remember that I am allergic to peanuts")
    assert c.source == Source.USER_EXPLICIT


def test_correction_is_flagged_and_time_tail_trimmed():
    (c,) = kept("Actually I live in Lviv now")
    assert c.correction and c.content == "Lives in Lviv"


def test_forget_is_parsed():
    e = extract("Forget that I live in Kyiv")
    assert e.forget and not e.candidates


def test_two_facts_in_one_message_become_two_memories():
    cs = kept("I am a backend developer and I prefer concise answers")
    assert {c.memory_type for c in cs} == {MemoryType.FACT, MemoryType.PREFERENCE}


def test_two_different_goals_do_not_replace_each_other():
    svc = MemoryService(MemoryStore(":memory:"))
    svc.learn("My goal is to finish the Rust book")
    svc.learn("I want to learn Go this year")
    goals = [m for m in svc.store.list(status="ACTIVE") if m.memory_type == MemoryType.GOAL]
    assert len(goals) == 2


def test_learn_applies_forget_and_respects_capture_switch():
    svc = MemoryService(MemoryStore(":memory:"))
    svc.learn("I live in Kyiv")
    assert svc.store.count(status="ACTIVE") == 1
    svc.learn("Forget that I live in Kyiv")
    assert svc.store.count() == 0
    svc.store.set_setting("capture", "off")
    out = svc.learn("I live in Odesa")
    assert svc.store.count() == 0 and "off" in out.skipped