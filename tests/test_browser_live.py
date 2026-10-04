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

    second = await adapter.ask("job3", "question two", 2, continue_thread=True)  # same research, same conversation
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

async def test_logged_out_chatgpt_transcript_dom_is_captured(browser_settings, fixture_server):
    """Regression from the first live logged-out run: chatgpt.com served a transcript of
    ol[data-conversation-transcript] > li[data-message-role=assistant] (no data-message-author-role),
    so the answer was 'captured but empty after cleaning' and ChatGPT counted as failed.
    The fixture reproduces that DOM, including the 'ChatGPT said:' heading, which must not leak."""
    url = fixture_server.rsplit("/", 1)[0] + "/chatgpt_logged_out.html"
    settings = Settings.model_validate({
        **browser_settings.model_dump(mode="json"),
        "providers": {"chatgpt": {"enabled": True, "label": "ChatGPT", "url": url, "max_retries": 0}},
    })
    engine = BrowserEngine(settings)
    try:
        adapter = build_adapter("chatgpt", engine, settings, settings.providers["chatgpt"])
        adapter.sel.stable_ms = 700
        adapter.sel.tiny_fragment_ms = 1500
        adapter.sel.never_started_ms = 12000
        adapter.sel.force_capture_ms = 25000
        adapter.sel.hard_timeout_ms = 45000
        response = await adapter.ask("jobx", "When was the Eiffel Tower completed and how tall is it?", 1)
        assert response.status.value == "completed", f"{response.status.value}: {response.error}"
        assert "March 31, 1889" in response.answer_text and "330 metres" in response.answer_text, response.answer_text
        assert "ChatGPT said" not in response.answer_text and "You said" not in response.answer_text
        assert "When was the Eiffel Tower" not in response.answer_text, "the user's own prompt must not be read as the answer"
    finally:
        await engine.stop(keep_windows=False)

async def test_signed_in_chatgpt_without_author_role_attribute_is_captured(browser_settings, fixture_server):
    """Live via chrome-use on the user's signed-in Chrome (2026-10-04): 'answer captured but empty after cleaning' because
    the answer sits in div[data-markdown-text-style=assistant-message] and there is no data-message-author-role."""
    url = fixture_server.rsplit("/", 1)[0] + "/chatgpt_signed_in_2026.html"
    settings = Settings.model_validate({
        **browser_settings.model_dump(mode="json"),
        "providers": {"chatgpt": {"enabled": True, "label": "ChatGPT", "url": url, "max_retries": 0}},
    })
    engine = BrowserEngine(settings)
    try:
        adapter = build_adapter("chatgpt", engine, settings, settings.providers["chatgpt"])
        adapter.sel.stable_ms = 700
        adapter.sel.tiny_fragment_ms = 1500
        adapter.sel.never_started_ms = 12000
        adapter.sel.force_capture_ms = 25000
        adapter.sel.hard_timeout_ms = 45000
        response = await adapter.ask("jobs", "When was the Eiffel Tower completed and how tall is it?", 1)
        assert response.status.value == "completed", f"{response.status.value}: {response.error}"
        assert "March 31, 1889" in response.answer_text and "330 metres" in response.answer_text, response.answer_text
        assert "ChatGPT said" not in response.answer_text and "You said" not in response.answer_text
    finally:
        await engine.stop(keep_windows=False)
async def test_a_site_that_ignores_the_scripted_send_click_is_sent_with_enter(browser_settings, fixture_server):
    """Live (chat.deepseek.com in the user's Chrome): clickSend reported ok but the untrusted click did nothing and the
    prompt sat in the composer until timeout. If the prompt is still in the composer after the click, press Enter."""
    url = fixture_server.rsplit("/", 1)[0] + "/chat_ignores_scripted_click.html"
    settings = Settings.model_validate({
        **browser_settings.model_dump(mode="json"),
        "providers": {"chatgpt": {"enabled": True, "label": "ChatGPT", "url": url, "max_retries": 0}},
    })
    engine = BrowserEngine(settings)
    try:
        adapter = build_adapter("chatgpt", engine, settings, settings.providers["chatgpt"])
        adapter.sel.stable_ms = 700
        adapter.sel.tiny_fragment_ms = 1500
        adapter.sel.never_started_ms = 12000
        adapter.sel.force_capture_ms = 25000
        adapter.sel.hard_timeout_ms = 45000
        response = await adapter.ask("jobe", "When was the Eiffel Tower completed and how tall is it?", 1)
        assert response.status.value == "completed", f"{response.status.value}: {response.error}"
        assert "March 31, 1889" in response.answer_text, response.answer_text
    finally:
        await engine.stop(keep_windows=False)
async def test_safe_dismiss_closes_banners_but_never_accepts_or_agrees(browser_settings, fixture_server):
    """On the person's own Chrome (live driver) 'Accept all cookies' / 'I agree' must never be clicked; 'Got it' / 'Reject all' may."""
    url = fixture_server.rsplit("/", 1)[0] + "/banners.html"
    settings = Settings.model_validate({
        **browser_settings.model_dump(mode="json"),
        "providers": {"chatgpt": {"enabled": True, "label": "ChatGPT", "url": url, "max_retries": 0}},
    })
    engine = BrowserEngine(settings)
    try:
        adapter = build_adapter("chatgpt", engine, settings, settings.providers["chatgpt"])
        page = await engine.open_research_page("chatgpt", url)
        await adapter._install(page)
        await adapter._call(page, "dismiss", {**adapter._sel_dict, "safe_dismiss": True})
        clicked = await page.evaluate("() => window.clicked")
        assert "acc" not in clicked and "agree" not in clicked, clicked
        assert "got" in clicked, clicked
        await page.evaluate("() => { window.clicked.length = 0; }")
        await adapter._call(page, "dismiss", adapter._sel_dict)  # the dedicated-profile behaviour is unchanged
        assert "acc" in await page.evaluate("() => window.clicked")
    finally:
        await engine.stop(keep_windows=False)
def test_clean_source_url_strips_only_the_chatgpt_tracking_parameter():
    from browser.adapters.chatgpt import clean_source_url

    assert clean_source_url("https://www.legislation.gov.uk/ukpga/2018/12/introduction/2026-09-30?utm_source=chatgpt.com") == "https://www.legislation.gov.uk/ukpga/2018/12/introduction/2026-09-30"
    assert clean_source_url("https://x.example/a?id=7&utm_source=chatgpt.com") == "https://x.example/a?id=7"
    assert clean_source_url("https://x.example/a?utm_source=newsletter") == "https://x.example/a?utm_source=newsletter"


async def test_chatgpt_source_chips_are_opened_and_their_urls_captured(browser_settings, fixture_server):
    """Live (2026-10-04): the answer named 'legislation.gov.uk' in a chip button and the citations list was empty, because the
    URL only exists in the dialog the chip opens. The URL is read from that dialog, without the tracking parameter."""
    url = fixture_server.rsplit("/", 1)[0] + "/chatgpt_source_chip.html"
    settings = Settings.model_validate({
        **browser_settings.model_dump(mode="json"),
        "providers": {"chatgpt": {"enabled": True, "label": "ChatGPT", "url": url, "max_retries": 0}},
    })
    engine = BrowserEngine(settings)
    try:
        adapter = build_adapter("chatgpt", engine, settings, settings.providers["chatgpt"])
        adapter.sel.stable_ms = 700
        adapter.sel.tiny_fragment_ms = 1500
        adapter.sel.never_started_ms = 12000
        adapter.sel.force_capture_ms = 25000
        adapter.sel.hard_timeout_ms = 45000
        response = await adapter.ask("jobc", "Which year was the Data Protection Act passed?", 1)
        assert response.status.value == "completed", f"{response.status.value}: {response.error}"
        urls = [c.url for c in response.citations]
        assert urls == ["https://www.legislation.gov.uk/ukpga/2018/12/introduction/2026-09-30"], urls
        assert response.citations[0].title == "Data Protection Act 2018"
    finally:
        await engine.stop(keep_windows=False)

async def test_an_age_gate_is_reported_blocked_and_never_answered(browser_settings, fixture_server):
    """Live finding: chat.qwen.ai put 'Confirm your age ... What year were you born? [Continue]' over its composer.
    The old flow typed into the composer anyway and ended 'answer captured but empty after cleaning'.
    We must not fill in or click through someone's age declaration: report it blocked, type nothing, retry nothing."""
    url = fixture_server.rsplit("/", 1)[0] + "/age_gate.html"
    settings = Settings.model_validate({
        **browser_settings.model_dump(mode="json"),
        "providers": {"qwen": {"enabled": True, "label": "Qwen", "url": url, "max_retries": 2}},
    })
    engine = BrowserEngine(settings)
    try:
        adapter = build_adapter("qwen", engine, settings, settings.providers["qwen"])
        events = []

        async def emit(kind, message, provider=None, round_no=None):
            events.append(message)

        response = await adapter.ask("jobg", "What is the capital of France?", 1, emit=emit)
        assert response.status.value == "failed" and "readiness=blocked" in response.error, (response.status, response.error)
        page = await engine.open_research_page("qwen", url)
        assert await page.evaluate("window.__ageClicked === true") is False, "the age gate's Continue button must never be pressed"
        assert await page.input_value("#q") == "", "nothing may be typed while the gate is up"
        assert not any("retrying" in e for e in events), events
    finally:
        await engine.stop(keep_windows=False)


async def test_a_terms_of_service_gate_is_reported_blocked_and_never_accepted(browser_settings, fixture_server):
    """Live finding (Le Chat, 2026-10-04): a modal 'You must accept our Terms of Service and Privacy Policy' sat over the
    composer. The old flow reported 'answer captured but empty after cleaning'. Accepting terms is the person's own
    legal act: neither 'Accept and continue' nor 'I agree' may be pressed, nothing is typed, and the reason is named."""
    url = fixture_server.rsplit("/", 1)[0] + "/terms_gate.html"
    settings = Settings.model_validate({
        **browser_settings.model_dump(mode="json"),
        "providers": {"le_chat": {"enabled": True, "label": "Le Chat", "url": url, "max_retries": 2}},
    })
    engine = BrowserEngine(settings)
    try:
        adapter = build_adapter("le_chat", engine, settings, settings.providers["le_chat"])
        events = []

        async def emit(kind, message, provider=None, round_no=None):
            events.append(message)

        response = await adapter.ask("jobt", "What is the capital of France?", 1, emit=emit)
        assert response.status.value == "failed" and "readiness=blocked" in response.error, (response.status, response.error)
        assert "terms-of-service" in response.error, response.error
        page = await engine.open_research_page("le_chat", url)
        assert await page.evaluate("window.__tosClicked === true") is False, "the terms button must never be pressed"
        assert (await page.inner_text("#q")).strip() == "", "nothing may be typed while the gate is up"
        assert not any("retrying" in e for e in events), events
    finally:
        await engine.stop(keep_windows=False)

async def test_a_sign_in_page_reached_on_send_is_reported_logged_out_not_empty(browser_settings, fixture_server):
    """Live finding (Le Chat, 2026-10-04, after the terms were accepted): sending the first message redirected to
    v2.auth.mistral.ai/login. The adapter said 'answer captured but empty after cleaning'. It is a login wall."""
    url = fixture_server.rsplit("/", 1)[0] + "/signin_on_submit.html"
    settings = Settings.model_validate({
        **browser_settings.model_dump(mode="json"),
        "providers": {"le_chat": {"enabled": True, "label": "Le Chat", "url": url, "max_retries": 1}},
    })
    engine = BrowserEngine(settings)
    try:
        adapter = build_adapter("le_chat", engine, settings, settings.providers["le_chat"])
        response = await adapter.ask("jobl", "What is the capital of France?", 1)
        assert response.status.value == "logged_out", (response.status, response.error)
        assert "empty after cleaning" not in (response.error or ""), response.error
        assert "login" in (response.error + response.detail).lower() or "sign" in (response.error + response.detail).lower(), (response.error, response.detail)
    finally:
        await engine.stop(keep_windows=False)


async def test_an_answer_inside_nested_wrappers_is_captured_once(browser_settings, fixture_server):
    """Regression from the first live Gemini run: model-response > message-content > .markdown matched three
    selectors, so the answer text appeared two or three times in the stored response."""
    url = fixture_server.rsplit("/", 1)[0] + "/gemini_nested.html"
    settings = Settings.model_validate({
        **browser_settings.model_dump(mode="json"),
        "providers": {"gemini": {"enabled": True, "label": "Gemini", "url": url, "max_retries": 0}},
    })
    engine = BrowserEngine(settings)
    try:
        adapter = build_adapter("gemini", engine, settings, settings.providers["gemini"])
        adapter.sel.stable_ms = 700
        adapter.sel.tiny_fragment_ms = 1500
        adapter.sel.never_started_ms = 12000
        adapter.sel.force_capture_ms = 25000
        adapter.sel.hard_timeout_ms = 45000
        response = await adapter.ask("jobn", "When was the Eiffel Tower completed?", 1)
        assert response.status.value == "completed", f"{response.status.value}: {response.error}"
        assert response.answer_text.count("March 31, 1889") == 1, response.answer_text
        assert "330 metres" in response.answer_text
    finally:
        await engine.stop(keep_windows=False)

async def test_ai_mode_that_never_answers_gives_up_early_and_is_not_retried(browser_settings, fixture_server):
    """Live logged-out run: AI Mode produced nothing and the adapter waited 404 s (two 200 s attempts)."""
    import time

    url = fixture_server.rsplit("/", 1)[0] + "/ai_mode_silent.html"
    settings = Settings.model_validate({
        **browser_settings.model_dump(mode="json"),
        "providers": {"google_ai": {"enabled": True, "label": "Google AI", "url": url, "max_retries": 1}},
    })
    engine = BrowserEngine(settings)
    events = []

    async def emit(kind, msg, provider=None, round_no=None):
        events.append(str(msg))

    try:
        adapter = build_adapter("google_ai", engine, settings, settings.providers["google_ai"])
        adapter.sel.never_started_ms = 3000
        adapter.sel.hard_timeout_ms = 60000
        started = time.time()
        response = await adapter.ask("jobq", "Is anything there?", 1, emit=emit)
        assert response.status.value == "broken", (response.status, response.error, events)
        assert "no-response-element" in (response.error or "")
        assert "AI Mode response is ready" in (response.detail or ""), "the failure must say what the page was showing"
        assert time.time() - started < 30, "should give up long before the hard timeout"
        assert not any("retrying" in e for e in events), events
    finally:
        await engine.stop(keep_windows=False)
