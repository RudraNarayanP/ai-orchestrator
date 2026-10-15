"""Official text lookup for a named law (2026-10-15)."""

import asyncio
from types import SimpleNamespace

from backend.evidence.official import is_official_host, official_link, official_queries
from backend.evidence.sources import tier_for
from backend.models import SourceTier


def test_a_named_law_gets_site_restricted_queries_for_its_jurisdiction():
    q = official_queries("Can Parliament use Article 368 to amend the basic structure of the Indian Constitution?")
    assert q and "site:indiacode.nic.in" in q[0] and "Article 368" in q[0]
    eu = official_queries("Can a company in the EU transfer data under the EU-US Data Privacy Framework?")
    assert any("site:eur-lex.europa.eu" in x for x in eu)


def test_a_case_goes_to_case_law_publishers():
    q = official_queries("Under Erie Railroad Co. v. Tompkins, must a federal court apply federal general common law?")
    assert q and "site:supreme.justia.com" in q[0] and q[0].startswith("Erie")


def test_no_law_named_means_no_lookup():
    assert official_queries("What is the capital of France?") == []


def test_official_publishers_are_primary_and_others_are_not():
    assert tier_for("https://eur-lex.europa.eu/eli/dec_impl/2023/1795/oj") == SourceTier.PRIMARY_OFFICIAL
    assert tier_for("https://www.indiacode.nic.in/handle/123") == SourceTier.PRIMARY_OFFICIAL
    assert not is_official_host("www.cookieyes.com")
    assert not official_link({"href": "https://www.cookieyes.com/x"}, "Is GDPR Article 8 in force in the EU?")


def test_the_pool_reads_official_hits_in_parallel_and_files_them(monkeypatch):
    import backend.evidence.pool as P
    import backend.evidence.search_http as H

    async def fake_search(q, **_):
        return [{"href": "https://www.indiacode.nic.in/bns-303", "title": "BNS s.303"}, {"href": "https://blog.example/x", "title": "x"}], {}

    seen = []

    async def fake_gather(job_id, links, **_):
        seen.extend(l["href"] for l in links)
        return []

    monkeypatch.setattr(H, "search", fake_search)
    monkeypatch.setattr(P, "gather_from_links", fake_gather)
    settings = SimpleNamespace(search=SimpleNamespace(per_query_timeout_s=5, engines=["duckduckgo"], fetch_body_chars=1000))
    trace = []

    async def emit(*_):
        return None

    asyncio.run(P._official_evidence("j", "Which section of the Indian Penal Code, 1860 punishes theft?", [], [], settings, 1, trace, emit))
    assert seen == ["https://www.indiacode.nic.in/bns-303"] and trace[0]["official"] == 1
