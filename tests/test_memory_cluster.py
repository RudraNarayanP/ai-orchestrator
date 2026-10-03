"""The family-finance scenario: facts + the user's interpretation, a vague follow-up, a connected cluster, unrelated memories left out.

The fixture is shared with scripts/memory_eval.py (`cluster`), which also runs it with the neural embedder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.memory import MemoryService, MemoryStore
from backend.memory.schema import MemoryType
from backend.memory.topics import is_interpretation, topics_for

FIX = json.loads((Path(__file__).resolve().parent.parent / "data" / "eval" / "fixtures" / "memory_family_finance.json").read_text(encoding="utf-8"))


@pytest.fixture
def world():
    svc = MemoryService(MemoryStore(":memory:"))
    key = {}
    for it in FIX["learn"]:
        res = svc.learn(it["say"])
        assert res.added and res.added[0].memory is not None, it
        key[it["key"]] = res.added[0].memory
    for it in FIX["remember"]:
        key[it["key"]] = svc.remember(it["text"], slot=it.get("slot")).memory
    inv = {m.memory_id: k for k, m in key.items()}
    return svc, key, inv


def ask(world, q):
    svc, _key, inv = world
    block, ret = svc.context_for(q)
    return block, ret, [inv[h.memory.memory_id] for h in ret.hits]


# ---------------------------------------------------------------- (1) facts vs the user's interpretation
def test_the_interpretation_is_stored_as_a_view_never_as_a_fact(world):
    _svc, key, _inv = world
    assert key["view"].memory_type == MemoryType.INTERPRETATION
    assert key["view"].content.startswith("User's view:")
    for k in ("coaching", "land"):
        assert key[k].memory_type == MemoryType.FACT
    facts = [m for m in world[0].store.list(status="ACTIVE") if m.memory_type == MemoryType.FACT]
    assert not any("valuing" in m.content for m in facts)


@pytest.mark.parametrize("typed", [
    "I feel like my parents don't care about my future",
    "I may read their refusal to fund as not valuing my education",
    "I'm worried that they will never support me",
    "It seems like my dad doesn't value education",
])
def test_feelings_typed_as_a_fact_are_reclassified_by_the_store_itself(typed):
    s = MemoryStore(":memory:")
    r = s.add(typed, memory_type="fact")
    assert r.memory.memory_type == MemoryType.INTERPRETATION and r.notes
    r2 = s.add(typed + " again", memory_type="preference", slot="pref:x")
    assert r2.memory.memory_type == MemoryType.INTERPRETATION and r2.memory.slot is None


@pytest.mark.parametrize("plain", ["I think the meeting is at 5", "My parents spent about INR 4 lakh on Allen coaching", "Lives in Kyiv"])
def test_ordinary_statements_are_not_mistaken_for_interpretations(plain):
    assert not is_interpretation(plain)


def test_injected_interpretation_is_labelled_as_the_users_view(world):
    block, ret, keys = ask(world, "why can't my parents fund this?")
    assert "view" in keys
    line = next(ln for ln in block.splitlines() if "valuing" in ln)
    assert "the user's own view, not an established fact" in line
    assert "never state them as facts" in block
    bare = [ln for ln in block.splitlines() if ln.startswith("- ") and "valuing" in ln and "view" not in ln]
    assert not bare


def test_view_note_is_absent_when_no_view_is_injected(world):
    block, _ret, keys = ask(world, "what is my gaming rank?")
    assert keys == ["gaming"] and "never state them as facts" not in block


def test_an_interpretation_never_supersedes_or_merges_with_a_fact(world):
    svc, key, _ = world
    before = svc.store.count(status="ACTIVE")
    svc.learn("I feel my parents refuse to fund my education because they don't value it")
    assert svc.store.get(key["coaching"].memory_id).status.value == "ACTIVE"
    assert svc.store.get(key["land"].memory_id).status.value == "ACTIVE"
    assert svc.store.count(status="ACTIVE") == before + 1
    assert svc.store.check_invariants() == []


# ---------------------------------------------------------------- (2)+(3) cluster retrieval, vague follow-ups
def test_follow_up_retrieves_the_connected_cluster_and_nothing_unrelated(world):
    q = FIX["queries"][0]
    _b, ret, keys = ask(world, q["q"])
    assert set(q["must_include"]) <= set(keys), keys
    assert q["min"] <= len(keys) <= q["max"], keys
    assert not set(keys) & set(FIX["unrelated"]), keys
    assert ret.kind == "cluster" and all(h.why for h in ret.hits)


def test_vague_reference_to_a_past_decision_finds_it_and_its_context(world):
    q = FIX["queries"][1]
    _b, _ret, keys = ask(world, q["q"])
    assert set(q["must_include"]) <= set(keys), keys
    assert q["min"] <= len(keys) <= q["max"]
    assert not set(keys) & set(FIX["unrelated"])


def test_connected_hits_say_how_they_are_connected(world):
    _b, ret, _k = ask(world, "why can't my parents fund this?")
    assert any(any("connected to your question" in w for w in h.why) for h in ret.hits)


def test_unrelated_questions_do_not_pull_the_family_finance_cluster(world):
    for q in ("what is my gaming rank?", "tell me about my cat", "where do I live?"):
        _b, _ret, keys = ask(world, q)
        assert not set(keys) & set(FIX["cluster"]), (q, keys)
    _b, ret, keys = ask(world, "what is the tuition fee at Oxford University?")
    assert keys == [] and ret.kind == "general"


def test_a_non_sensitive_match_cannot_drag_in_sensitive_neighbours(world):
    _b, _ret, keys = ask(world, "I built a gaming PC, what is my gaming rank on Valorant?")
    assert not set(keys) & set(FIX["cluster"]) and not {"girlfriend", "brother"} & set(keys)


def test_other_family_members_are_not_part_of_the_money_cluster(world):
    _b, _ret, keys = ask(world, "why can't my parents fund this?")
    assert "brother" not in keys and "girlfriend" not in keys


# ---------------------------------------------------------------- (4) sensitive memories
def test_family_money_and_feelings_are_flagged_sensitive(world):
    _svc, key, _ = world
    for k in ("coaching", "land", "rejected", "view", "girlfriend", "brother"):
        assert key[k].sensitivity == "sensitive", k
    for k in ("gaming", "gym", "pc", "coffee", "cat", "city", "decision"):
        assert key[k].sensitivity == "normal", k
    assert world[0].store.stats()["sensitive"] == 6


def test_switching_sensitive_injection_off_withholds_all_of_them(world):
    svc, _key, _ = world
    svc.store.set_setting("sensitive", "0")
    for q in ("why can't my parents fund this?", "remember why I rejected that university?"):
        _b, _ret, keys = ask(world, q)
        assert not any(world[1][k].sensitivity == "sensitive" for k in keys), keys
    svc.store.set_setting("sensitive", "1")
    assert ask(world, "why can't my parents fund this?")[2]


def test_a_sensitive_memory_is_not_offered_to_a_question_about_something_else(world):
    for q in ("recommend a good podcast about ancient history", "how do I repot an orchid", "who won the 1998 world cup"):
        _b, _ret, keys = ask(world, q)
        assert not any(world[1][k].sensitivity == "sensitive" for k in keys), (q, keys)


def test_sensitive_memories_can_be_inspected_and_listed(world):
    svc, _k, _ = world
    listed = svc.store.list(sensitivity="sensitive")
    assert len(listed) == 6 and all(m.sensitivity == "sensitive" for m in listed)


def test_deleting_a_sensitive_memory_removes_it_everywhere(world):
    svc, key, _ = world
    mid = key["view"].memory_id
    assert svc.store.delete(mid)
    assert svc.store.get(mid) is None
    assert not svc.store.fts_search(["valuing"])
    assert svc.store.db.execute("select count(*) from memory_topics where memory_id=?", (mid,)).fetchone()[0] == 0
    assert mid not in {x["memory_id"] for x in svc.store.export()}
    _b, _ret, keys = ask(world, "why can't my parents fund this?")
    assert "view" not in keys and "coaching" in keys


def test_events_never_hold_memory_text(world):
    svc, _k, _ = world
    blob = json.dumps(svc.store.events(limit=500))
    assert "Allen" not in blob and "valuing" not in blob and "Leeds" not in blob


def test_topics_and_sensitivity_survive_reopen_and_backfill(tmp_path):
    p = tmp_path / "m.db"
    s = MemoryStore(p)
    s.add("Parents spent about INR 4 lakh on Allen coaching")
    s.db.execute("delete from memory_topics")
    s.db.execute("update memories set topics='[]', sensitivity='normal'")
    s.db.commit()
    s.close()
    s2 = MemoryStore(p)
    (m,) = s2.list()
    assert set(m.topics) >= {"finance", "family"} and m.sensitivity == "sensitive"
    assert [x.memory_id for x in s2.topic_memories(["finance"])] == [m.memory_id]


def test_topic_tagging_basics():
    assert topics_for("parents bought land worth INR 50 lakh") == ["family", "finance"]
    assert "decision" in topics_for("I rejected the offer")
    assert topics_for("Likes strong black coffee") == []