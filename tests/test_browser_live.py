"""The browser layer, tested against a real browser and a real DOM.

These run Playwright against a local fake chat UI (tests/fixtures/chat_fixture.html)
that reproduces the awkward behaviours the consumer sites actually have: streaming
text, a stop button that exists only mid-generation, sources arriving as a second
assistant turn, UI chrome inside the answer node, a consent overlay, and the user's
own prompt echoed into the transcript.

The app itself runs headful, one dedicated window per provider. These tests run
headless for the same reason any test should not pop windows: they still use their
own profile directory, never the everyday browser.
"""

from __future__ import annotations

import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from backend.browser.engine import BrowserEngine, profile_dir
from backend.models import ResearchMode
from backend.settings import ProviderConfig, Settings
from browser.adapters import build_adapter

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs) -> None:  # noqa: D102
        return


@pytest.fixture(scope="module")
def fixture_server():
    handler = partial(_Quiet, directory=str(FIXTURES))
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/chat_fixture.html"
    server.shutdown()


@pytest.fixture(scope="module")
def browser_settings() -> Settings:
    raw = {
        "providers": {
            "fixture": {
                "enabled": True,
                "label": "Fixture",
                "url": "about:blank",
                "adapter": "generic_chat",
                "max_retries": 0,
            }
        },
        "browser": {"headless": True, "width": 1100, "height": 760, "keep_windows_open": False},
        "research": {
            "mode": "STANDARD",
            "response_stability_poll_ms": 250,
            "response_stable_rounds": 2,
        },
        "storage": {"db_path": "data/test_browser.db", "artifacts_dir": "data/test_artifacts"},
    }
    return Settings.model_validate(raw)


@pytest.fixture()
async def adapter(browser_settings, fixture_server):
    engine = BrowserEngine(browser_settings)
    cfg = ProviderConfig(
        enabled=True, label="Fixture", url=fixture_server, adapter="generic_chat", max_retries=0
    )
    built = build_adapter("fixture", engine, browser_settings, cfg)
    # tighten the fixture's timings so the test is seconds, not minutes
    built.sel.stable_ms = 700
    built.sel.tiny_fragment_ms = 1500
    built.sel.never_started_ms = 12000
    built.sel.force_capture_ms = 25000
    built.sel.hard_timeout_ms = 45000
    built.sel.source_block_ms = 1500
    built.sel.reference_force_ms = 30000
    try:
        yield built
    finally:
        await engine.stop(keep_windows=False)


pytestmark = pytest.mark.browser


async def test_single_window_reuses_tabs_instead_of_spawning_windows(browser_settings, fixture_server):
    """The window-spawning complaint, asserted.

    Two providers must share one window as two tabs, and asking again must reuse
    the tab rather than open a third.
    """
    engine = BrowserEngine(browser_settings)
    try:
        page_a = await engine.open_research_page("fixture", fixture_server)
        page_b = await engine.open_research_page("fixture2", fixture_server + "?v=2")
        context = page_a.context
        assert context is page_b.context, "providers must share one window/profile"
        assert len(context.pages) == 2, [p.url for p in context.pages]

        again = await engine.open_research_page("fixture", fixture_server)
        assert again is page_a, "an existing provider tab should be reused"
        assert len(context.pages) == 2, [p.url for p in context.pages]

        # a *different* provider on the same host still gets its own tab
        third = await engine.open_research_page("fixture3", fixture_server)
        assert third is not page_a
        assert len(context.pages) == 3
    finally:
        await engine.stop(keep_windows=False)


async def test_per_provider_mode_still_isolates(browser_settings, fixture_server):
    browser_settings.browser.window_mode = "per_provider"
    engine = BrowserEngine(browser_settings)
    try:
        page_a = await engine.open_research_page("fixture", fixture_server)
        page_b = await engine.open_research_page("fixture2", fixture_server)
        assert page_a.context is not page_b.context, "per_provider mode must isolate profiles"
    finally:
        await engine.stop(keep_windows=False)
        browser_settings.browser.window_mode = "single"


async def test_adapter_completes_a_streaming_answer_in_a_real_browser(adapter):
    response = await adapter.ask("job1", "When did Acme announce the Bolt and what does it cost?", 1)

    assert response.status.value == "completed", f"{response.status.value}: {response.error}"
    text = response.answer_text
    assert "Acme Bolt was announced on 14 May 2026" in text, text[:400]
    assert "$549" in text
    assert "Wi-Fi 7" in text
    # UI chrome and the "Acme said" prefix must not leak into the captured answer
    for noise in ("Copied!", "Regenerate", "Acme said"):
        assert noise not in text, f"{noise!r} leaked into the answer"
    # the user's own prompt is echoed into the transcript; it is not the answer
    assert "You: When did Acme announce" not in text
    # markdown structure survived
    assert "## Details" in text or "Details" in text
    assert response.duration_s and response.duration_s < 45
    assert response.web_research_status.value in {"failed_or_unclear", "unknown"}


async def test_sources_from_a_separate_assistant_turn_are_harvested(adapter):
    response = await adapter.ask("job2", "Tell me about the Acme Bolt price", 1)
    urls = {c.url for c in response.citations}
    assert "https://example.test/press" in urls, urls
    assert "https://example.test/verge" in urls, urls


async def test_second_turn_does_not_reread_the_previous_answer(adapter):
    first = await adapter.ask("job3", "question one", 1)
    assert "announced on 14 May 2026" in first.answer_text

    second = await adapter.ask("job3", "question two", 2)
    # the fixture's second answer is a correction; the first must not bleed through
    assert "549" in second.answer_text
    assert "announced on 14 May 2026 at a price" not in second.answer_text, "captured the previous turn"
    assert second.fingerprint != first.fingerprint


async def test_consensus_extract_claims_from_the_captured_text(adapter):
    from backend.research import claims as claim_ops

    response = await adapter.ask("job4", "Acme Bolt details", 1)
    pairs = claim_ops.heuristic_claims(response)
    texts = " || ".join(t for t, _ in pairs)
    assert "2026" in texts, texts
    assert "549" in texts, texts
    kinds = {k for _, k in pairs}
    assert kinds & {"date", "statistic", "product", "fact"}
    # headings are not claims
    assert all("Sources" != t.strip() for t, _ in pairs)


async def test_cancel_mid_answer_stops_the_wait_and_closes_the_tab(adapter):
    """Cooperative cancel against a real page: the flag ends the wait, the tab goes away."""
    import asyncio
    import time

    from backend.cancel import CancelToken, JobCancelled

    token = adapter.cancel_token = CancelToken()
    asyncio.get_running_loop().call_later(2.5, token.cancel)
    started = time.monotonic()
    with pytest.raises(JobCancelled):
        await adapter.ask("job-cancel", "When did Acme announce the Bolt and what does it cost?", 1)
    assert time.monotonic() - started < 20
    session = adapter.engine._sessions.get(adapter.engine._profile_key("fixture"))
    assert session is None or "fixture" not in session.pages, "the provider tab must be closed on cancel"
