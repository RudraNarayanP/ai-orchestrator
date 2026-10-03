"""The web UI in a real (headless) browser: History, export, keyboard use, narrow viewport.

Runs the real FastAPI app on a local port with a throw-away database; no research
job is executed (the runner is stubbed), so no chat site is ever contacted.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn
from playwright.async_api import async_playwright

from backend.api import app as api_app
from tests.conftest import base_settings
from tests.test_history_export import finished_job

pytestmark = pytest.mark.browser

LONG = "https://example.com/" + "a" * 160


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def server(tmp_path, monkeypatch):
    async def idle(self, job, emit):
        return None

    monkeypatch.setattr(api_app.JobManager, "_run", idle)
    app = api_app._make_app(base_settings(storage={"db_path": str(tmp_path / "ui.db")}))
    store = app.state.store
    older = finished_job("What does warfarin interact with?")
    newer = finished_job("Is it safe to take ibuprofen with warfarin?")
    newer.created_at = older.created_at + 60
    newer.final.answer = "Not without medical advice. " + LONG
    for job in (older, newer):
        store.save_job(job)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    assert srv.started, "test server did not start"
    yield {"url": f"http://127.0.0.1:{port}", "newer": newer, "older": older}
    srv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
async def page(server):
    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(headless=True, channel="chrome")
        except Exception:  # noqa: BLE001 -- bundled chromium when Chrome is not installed
            browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1100, "height": 760}, accept_downloads=True)
        page = await context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        await page.goto(server["url"])
        await page.wait_for_selector("#question")
        page.js_errors = errors
        yield page
        await browser.close()
        assert not errors, f"uncaught page errors: {errors}"


async def test_history_lists_past_questions_newest_first_and_opens_one(page, server):
    await page.click("#openHistory")
    await page.wait_for_selector(".history-item")
    texts = await page.locator(".history-item .h-q").all_inner_texts()
    assert texts == ["Is it safe to take ibuprofen with warfarin?", "What does warfarin interact with?"]
    assert await page.locator("#openHistory").get_attribute("aria-expanded") == "true"

    await page.locator(".history-item").nth(1).click()
    await page.wait_for_selector(".msg.ai .answer-text")
    assert "ibuprofen raises bleeding risk" in await page.locator(".msg.ai .answer-text").inner_text()
    assert await page.locator("#history").is_hidden(), "opening a job closes the panel"
    assert await page.locator(".msg.user .q").inner_text() == "What does warfarin interact with?"
    assert await page.locator("summary", has_text="Evidence we opened").count() == 1


async def test_export_links_download_the_markdown_and_json(page, server):
    await page.click("#openHistory")
    await page.locator(".history-item").first.click()
    await page.wait_for_selector(".export-link")
    labels = await page.locator(".export-link").all_inner_texts()
    assert labels == ["Markdown", "JSON"]
    async with page.expect_download() as info:
        await page.locator(".export-link", has_text="Markdown").click()
    download = (await info.value)
    assert download.suggested_filename == f"omnibrain-{server['newer'].id}.md"


async def test_empty_history_says_so(tmp_path, monkeypatch):
    # a separate, empty database
    async def idle(self, job, emit):
        return None

    monkeypatch.setattr(api_app.JobManager, "_run", idle)
    app = api_app._make_app(base_settings(storage={"db_path": str(tmp_path / "empty.db")}))
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(f"http://127.0.0.1:{port}")
            await page.click("#openHistory")
            await page.wait_for_selector("#historyList .hint")
            assert "Nothing asked yet" in await page.locator("#historyList").inner_text()
            await browser.close()
    finally:
        srv.should_exit = True
        thread.join(timeout=10)


async def test_escape_closes_panels_and_slash_focuses_the_question(page):
    await page.click("#openHistory")
    assert await page.locator("#history").is_visible()
    await page.keyboard.press("Escape")
    assert await page.locator("#history").is_hidden()
    assert await page.evaluate("document.activeElement.id") == "question"

    await page.click("#openSettings")
    assert await page.locator("#settings").is_visible() and await page.locator("#history").is_hidden()
    await page.keyboard.press("Escape")
    assert await page.locator("#settings").is_hidden()

    await page.evaluate("document.activeElement.blur()")
    await page.keyboard.press("/")
    assert await page.evaluate("document.activeElement.id") == "question"
    assert await page.input_value("#question") == "", "the slash must not be typed into the box"


async def test_arrow_keys_move_through_history(page):
    await page.click("#openHistory")
    await page.wait_for_selector(".history-item")
    await page.locator(".history-item").first.focus()
    await page.keyboard.press("ArrowDown")
    assert await page.evaluate("document.activeElement.dataset.jobId") is not None
    assert await page.locator(".history-item").nth(1).evaluate("el => el === document.activeElement")
    await page.keyboard.press("ArrowUp")
    assert await page.locator(".history-item").first.evaluate("el => el === document.activeElement")


async def test_enter_sends_and_shift_enter_makes_a_new_line(page):
    await page.fill("#question", "")
    await page.focus("#question")
    await page.keyboard.type("first line")
    await page.keyboard.press("Shift+Enter")
    await page.keyboard.type("second line")
    assert await page.input_value("#question") == "first line\nsecond line"
    assert await page.locator(".msg.user").count() == 0, "Shift+Enter must not send"

    async with page.expect_request(lambda r: r.url.endswith("/api/jobs") and r.method == "POST") as info:
        await page.keyboard.press("Enter")
    request = await info.value
    assert request.post_data_json["question"] == "first line\nsecond line"
    assert await page.input_value("#question") == ""


async def test_a_rejected_question_shows_the_reason_instead_of_hanging(page):
    await page.fill("#question", "x" * 4001)
    await page.press("#question", "Enter")
    await page.wait_for_selector(".err:not(:empty)")
    assert "too long" in await page.locator(".err").first.inner_text()
    assert await page.locator("#live").is_hidden()


@pytest.mark.parametrize("width", [320, 375])
async def test_narrow_viewport_has_no_horizontal_scroll(page, width):
    await page.set_viewport_size({"width": width, "height": 700})
    overflow = "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    assert await page.evaluate(overflow) <= 0, "empty state overflows"
    await page.click("#openHistory")
    await page.wait_for_selector(".history-item")
    assert await page.evaluate(overflow) <= 0, "history panel overflows"
    await page.locator(".history-item").first.click()  # its answer contains a 180-character unbroken URL
    await page.wait_for_selector(".msg.ai .answer-text")
    assert await page.evaluate(overflow) <= 0, "a long unbroken URL in an answer overflows"
    for selector in ["#question", "#askBtn", "#openHistory", "#openSettings", "#newThread"]:
        box = await page.locator(selector).bounding_box()
        assert box and box["x"] >= 0 and box["x"] + box["width"] <= width + 1, f"{selector} is cut off at {width}px: {box}"