"""Memory store: lifecycle, supersede, isolation, deletion, invariants."""

from __future__ import annotations

import time

import pytest

from backend.memory import MemoryService, MemoryStore
from backend.memory.schema import MemoryType, Source, Status


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_correction_supersedes_and_never_two_active_contradictions(store):
    a = store.add("Lives in Kyiv", slot="residence")
    b = store.add("Lives in Lviv", slot="residence")
    assert b.action == "superseded"
    assert store.get(a.memory.memory_id).status == Status.SUPERSEDED
    active = [m for m in store.list(status="ACTIVE") if m.slot == "residence"]
    assert [m.content for m in active] == ["Lives in Lviv"]
    assert store.check_invariants() == []


def test_inferred_never_overrides_explicit(store):
    store.add("Lives in Kyiv", slot="residence", source=Source.USER_EXPLICIT)
    r = store.add("Lives in Odesa", slot="residence", source=Source.MODEL_INFERRED)
    assert r.action == "rejected"
    assert [m.content for m in store.list(status="ACTIVE")] == ["Lives in Kyiv"]


def test_explicit_overrides_inferred(store):
    store.add("Lives in Odesa", slot="residence", source=Source.MODEL_INFERRED)
    r = store.add("Lives in Kyiv", slot="residence", source=Source.USER_EXPLICIT)
    assert r.action == "superseded"
    assert [m.content for m in store.list(status="ACTIVE")] == ["Lives in Kyiv"]


def test_exact_duplicate_merges_instead_of_adding(store):
    store.add("Prefers dark mode in every editor")
    r = store.add("Prefers dark mode in every editor")
    assert r.action == "merged"
    assert store.count() == 1


def test_delete_is_physical_and_leaves_no_trace_in_search(store):
    r = store.add("Secret hobby is falconry")
    mid = r.memory.memory_id
    assert store.delete(mid)
    assert store.get(mid) is None
    assert store.fts_search(["falconry"]) == []
    assert all(mid != x for x, _ in store.vector_search(store.embedder.embed(["falconry"])[0], 5))
    assert store.check_invariants() == []


def test_delete_all_empties_everything(store):
    for i in range(5):
        store.add(f"Fact number {i} about topic{i}")
    assert store.delete_all() == 5
    assert store.count() == 0
    assert store.fts_search(["topic1"]) == []


def test_deleting_a_correction_restores_nothing_active_but_keeps_old_as_archived(store):
    a = store.add("Lives in Kyiv", slot="residence")
    b = store.add("Lives in Lviv", slot="residence")
    store.delete(b.memory.memory_id)
    assert store.get(a.memory.memory_id).status in (Status.ARCHIVED, Status.SUPERSEDED)
    assert store.check_invariants() == []


def test_update_edits_content_and_reindexes(store):
    r = store.add("Favourite language is Rust")
    store.update(r.memory.memory_id, content="Favourite language is Zig")
    assert store.fts_search(["zig"]) and not store.fts_search(["rust"])


def test_events_are_recorded(store):
    r = store.add("Uses Linux at home")
    store.delete(r.memory.memory_id)
    actions = [e["action"] for e in store.events()]
    assert "created" in actions and "deleted" in actions


def test_persists_across_reopen(tmp_path):
    p = tmp_path / "m.db"
    s = MemoryStore(p)
    s.add("Owns a cat called Miso", entities=["Miso"])
    s.close()
    s2 = MemoryStore(p)
    assert [m.content for m in s2.list()] == ["Owns a cat called Miso"]
    assert s2.vector_search(s2.embedder.embed(["cat Miso"])[0], 3)
    s2.close()


# --------------------------------------------------------------- scoping

def test_project_isolation(store):
    svc = MemoryService(store)
    store.add("Project Atlas database is PostgreSQL 16", memory_type=MemoryType.PROJECT, project="atlas")
    store.add("Project Borealis database is MongoDB", memory_type=MemoryType.PROJECT, project="borealis")
    _blk, ret = svc.context_for("which database does the project use? database version?", project="atlas")
    texts = " ".join(h.memory.content for h in ret.hits)
    assert "PostgreSQL" in texts and "MongoDB" not in texts


def test_conversation_memory_is_separate_and_temporary(store):
    svc = MemoryService(store)
    svc.note_turn("conv-1", "We are comparing two laptop models called Zephyr and Orion today")
    conv = [m for m in store.list() if m.memory_type == MemoryType.CONVERSATION]
    assert conv and conv[0].scope == "conversation"
    # other conversations do not see it
    _b, ret = svc.context_for("compare Zephyr and Orion laptops", conversation="conv-2")
    assert not any(h.memory.memory_type == MemoryType.CONVERSATION for h in ret.hits)
    # and it expires
    assert svc.prune_conversation_memory(older_than_s=0.0) >= 1
    assert not [m for m in store.list() if m.memory_type == MemoryType.CONVERSATION]


def test_goal_completion_deactivates_goal(store):
    svc = MemoryService(store)
    svc.learn("My goal is to finish the Rust book")
    goals = [m for m in store.list(status="ACTIVE") if m.memory_type == MemoryType.GOAL]
    assert goals and goals[0].goal_active
    svc.learn("I finished the Rust book")
    _b, ret = svc.context_for("what are my current goals about the Rust book?")
    assert not any(h.memory.memory_type == MemoryType.GOAL and h.memory.goal_active for h in ret.hits)


def test_history_only_on_request(store):
    svc = MemoryService(store)
    store.add("Lives in Kyiv", slot="residence")
    store.add("Lives in Lviv", slot="residence")
    _b, now = svc.context_for("where do I live?")
    assert all(h.memory.status == Status.ACTIVE for h in now.hits)
    assert any("Lviv" in h.memory.content for h in now.hits)
    _b, hist = svc.context_for("where did I live before?")
    assert any("Kyiv" in h.memory.content for h in hist.hits)


def test_expired_memory_is_not_retrieved(store):
    svc = MemoryService(store)
    r = store.add("Has a dentist appointment on the 3rd", memory_type=MemoryType.EPISODIC)
    store.update(r.memory.memory_id, expires_at=time.time() - 10)
    _b, ret = svc.context_for("when is my dentist appointment?")
    assert not ret.hits


def test_consolidate_merges_near_duplicates(store):
    store.bulk_insert([])
    store.add("Likes strong black coffee in the morning", source=Source.MODEL_INFERRED)
    store.add("Enjoys strong black coffee every morning", source=Source.MODEL_INFERRED)
    assert store.consolidate() == 0  # not duplicates at the strict threshold
    merged = store.consolidate(threshold=0.6)
    assert merged >= 1
    assert len(store.list(status="ACTIVE")) == 1
    assert store.check_invariants() == []

# ---------------------------------------------------------------- scale regressions (found by the 10k benchmark)

def test_keyword_index_survives_reopen_and_rebuilds_a_legacy_file(tmp_path):
    p = tmp_path / "legacy.db"
    s = MemoryStore(p)
    s.add("Owns a kayak stored in the garage")
    s.db.execute("delete from memory_fts_map")  # a file written before the rowid map existed
    s.db.execute("delete from memory_fts")
    s.db.commit()
    s.close()
    s2 = MemoryStore(p)
    assert s2.fts_search(["kayak"]), "the keyword index is rebuilt on open"
    r = s2.add("Owns a canoe")
    s2.update(r.memory.memory_id, content="Owns a sailboat")
    assert s2.fts_search(["sailboat"]) and not s2.fts_search(["canoe"])
    s2.close()


def test_entity_and_project_lookup_is_by_name_not_by_scanning(store):
    svc = MemoryService(store)
    store.add("Works at Acme Robotics", entities=["Acme Robotics"])
    kind, proj, ents, _h = svc.retriever.understand("what does Acme Robotics build?")
    assert ents == ["acme robotics"]
    assert svc.retriever.understand("tell me about atlas")[1] is None
    store.add("Project Atlas uses Postgres", memory_type=MemoryType.PROJECT, project="Atlas")
    assert svc.retriever.understand("tell me about atlas")[1] == "atlas"  # project cache is invalidated by the write
    store.delete_all()
    assert svc.retriever.understand("tell me about atlas")[1] is None