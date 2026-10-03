"""Memory HTTP API: list, search (with WHY), add, edit, delete, forget-all confirm, settings, off-switch."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api import app as api_app
from tests.conftest import base_settings


@pytest.fixture
def mclient(tmp_path, monkeypatch):
    async def idle(self, job, emit):
        return None

    monkeypatch.setattr(api_app.JobManager, "_run", idle)
    settings = base_settings(storage={"db_path": str(tmp_path / "api.db")}, memory={"enabled": True, "path": str(tmp_path / "mem.db"), "embedder": "hash"})
    with TestClient(api_app._make_app(settings)) as c:
        yield c


def add(c, text, **kw):
    r = c.post("/api/memory", json={"content": text, **kw})
    assert r.status_code == 200, r.text
    return r.json()["memory"]


def test_add_list_and_get_with_events(mclient):
    m = add(mclient, "Lives in Kyiv")
    assert m["source"] == "user_explicit"  # typed into the panel = explicit
    items = mclient.get("/api/memory").json()["items"]
    assert [i["content"] for i in items] == ["Lives in Kyiv"]
    one = mclient.get(f"/api/memory/{m['memory_id']}").json()
    assert one["events"] and one["events"][0]["action"] == "created"


def test_search_explains_why(mclient):
    add(mclient, "Owns a cat called Miso")
    add(mclient, "Plays the cello on weekends")
    r = mclient.post("/api/memory/search", json={"query": "what is my cat called?"}).json()
    assert [h["content"] for h in r["hits"]] == ["Owns a cat called Miso"]
    assert r["hits"][0]["why"] and "Relevant context" in r["context"]
    g = mclient.post("/api/memory/search", json={"query": "what is the capital of France?"}).json()
    assert g["hits"] == [] and g["kind"] == "general"


def test_edit_changes_content_and_search_follows(mclient):
    m = add(mclient, "Favourite language is Rust")
    r = mclient.patch(f"/api/memory/{m['memory_id']}", json={"content": "Favourite language is Zig"})
    assert r.status_code == 200 and r.json()["content"] == "Favourite language is Zig"
    hits = mclient.post("/api/memory/search", json={"query": "what is my favourite language?"}).json()["hits"]
    assert hits and "Zig" in hits[0]["content"]


def test_delete_then_404(mclient):
    m = add(mclient, "Secret hobby is falconry")
    assert mclient.delete(f"/api/memory/{m['memory_id']}").status_code == 200
    assert mclient.get(f"/api/memory/{m['memory_id']}").status_code == 404
    assert mclient.delete(f"/api/memory/{m['memory_id']}").status_code == 404
    assert mclient.post("/api/memory/search", json={"query": "falconry hobby"}).json()["hits"] == []


def test_forget_all_needs_explicit_confirmation(mclient):
    add(mclient, "Lives in Kyiv")
    assert mclient.post("/api/memory/forget-all", json={}).status_code == 400
    assert mclient.post("/api/memory/forget-all", json={"confirm": "yes"}).status_code == 400
    assert mclient.get("/api/memory/stats").json()["total"] == 1
    assert mclient.post("/api/memory/forget-all", json={"confirm": True}).json()["deleted"] == 1
    assert mclient.get("/api/memory").json()["items"] == []


def test_inject_and_capture_switches(mclient):
    add(mclient, "Owns a cat called Miso")
    s = mclient.post("/api/memory/settings", json={"inject": False}).json()
    assert s["inject"] is False and s["capture"] is True
    assert mclient.post("/api/memory/search", json={"query": "what is my cat called?"}).json()["hits"] == []
    assert mclient.post("/api/memory/settings", json={"inject": "no"}).status_code == 400
    assert mclient.post("/api/memory/settings", json={"inject": True}).json()["inject"] is True


def test_validation(mclient):
    assert mclient.post("/api/memory", json={"content": "  "}).status_code == 400
    assert mclient.post("/api/memory", json={"content": "x", "memory_type": "bogus"}).status_code == 400
    assert mclient.post("/api/memory/search", json={}).status_code == 400
    assert mclient.patch("/api/memory/mem_nope", json={"content": "x"}).status_code == 404


def test_export_contains_no_embeddings(mclient):
    add(mclient, "Lives in Kyiv")
    data = mclient.get("/api/memory/export").json()
    assert data and "embedding" not in data[0]


def test_memory_switched_off_is_503(tmp_path):
    settings = base_settings(storage={"db_path": str(tmp_path / "a.db")})
    with TestClient(api_app._make_app(settings)) as c:
        assert c.get("/api/memory").status_code == 503


def test_consolidate_endpoint(mclient):
    add(mclient, "Likes strong black coffee in the morning")
    assert mclient.post("/api/memory/consolidate").json() == {"merged": 0}

def test_interpretation_and_sensitivity_through_the_api(mclient):
    r = mclient.post("/api/memory", json={"content": "I feel like my parents don't care about my future", "memory_type": "fact"}).json()
    assert r["memory"]["memory_type"] == "interpretation" and r["memory"]["sensitivity"] == "sensitive" and r["notes"]
    mclient.post("/api/memory", json={"content": "Parents bought land worth roughly INR 50 lakh"})
    mclient.post("/api/memory", json={"content": "Owns a cat called Miso"})
    sens = mclient.get("/api/memory?sensitivity=sensitive").json()["items"]
    assert len(sens) == 2 and mclient.get("/api/memory/stats").json()["sensitive"] == 2
    s = mclient.post("/api/memory/settings", json={"sensitive": False}).json()
    assert s["sensitive"] is False and s["inject"] is True
    hits = mclient.post("/api/memory/search", json={"query": "why can't my parents fund this land purchase?"}).json()["hits"]
    assert not any(h["sensitivity"] == "sensitive" for h in hits)
    mclient.post("/api/memory/settings", json={"sensitive": True})
    mid = next(i["memory_id"] for i in sens if "land" in i["content"])
    assert mclient.patch(f"/api/memory/{mid}", json={"sensitivity": "normal"}).json()["sensitivity"] == "normal"