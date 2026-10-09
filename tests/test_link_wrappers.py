"""Unwrapping the provider's own link wrapper -- one hop, allowlisted, no following.

Live (Eiffel run, 2026-10-09): Meta AI's citations came out as `https://l.meta.ai/?u=<the
actual source>`, so we fetched the wrapper, logged it `unreachable`, and lost five sources we
had been handed. Reading the URL the provider put on screen is not following a redirect
anywhere: only the named wrapper host is unwrapped, only to an http(s) target, never to itself.
"""

from __future__ import annotations

from backend.evidence.pool import unwrap_redirect

TARGET = "https://www.toureiffel.paris/en/the-monument/history"
WRAPPED = f"https://l.meta.ai/?u={TARGET.replace('/', '%2F')}"


def test_the_provider_wrapper_resolves_to_the_source_it_points_at():
    target, via = unwrap_redirect(WRAPPED)
    assert target == TARGET
    assert via == "l.meta.ai"


def test_only_the_known_wrapper_is_unwrapped():
    for url in (
        TARGET,
        "https://www.google.com/url?q=https%3A%2F%2Fexample.org%2Fpage",
        "https://example.org/page?u=https%3A%2F%2Fevil.example",
    ):
        assert unwrap_redirect(url) == (url, None), url


def test_a_wrapper_that_does_not_yield_a_plain_http_target_is_left_alone():
    for url in (
        "https://l.meta.ai/",
        "https://l.meta.ai/?u=javascript%3Aalert%281%29",
        "https://l.meta.ai/?u=%2Frelative%2Fpath",
        "https://l.meta.ai/?u=https%3A%2F%2Fl.meta.ai%2F%3Fu%3Dx",
        "not a url at all",
        "",
    ):
        target, via = unwrap_redirect(url)
        assert via is None, url
        assert target == url, url


async def test_a_wrapped_citation_is_fetched_and_attributed_to_its_real_source(net):
    """The evidence row carries the real domain, and the provider still gets credit for
    citing it -- the wrapper must not break provenance."""
    from backend.models import SourceCheckStatus
    from tests.test_ledger_adjudication import ask
    from tests.conftest import base_settings

    net.confirm(TARGET, published="2026-06-01")
    scripts = {
        "meta_ai": {
            "answer": "KEY CLAIMS\n1. The Eiffel Tower was completed on 31 March 1889.",
            "citations": [{"url": WRAPPED, "title": "Eiffel Tower history completed 31 March 1889"}],
        },
        "search": {"answer": "results", "citations": []},
    }
    job = await ask(base_settings(), scripts, question="When was the Eiffel Tower completed?", max_rounds=1)

    rows = [e for e in job.evidence if e.url == TARGET]
    assert rows, [(e.url, e.check_status) for e in job.evidence]
    row = rows[0]
    assert row.claim_id, "the source is filed under the claim it states"
    assert row.check_status == SourceCheckStatus.CONFIRMED
    assert "meta_ai" in row.cited_by, "unwrapping must not lose who cited it"
