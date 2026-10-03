"""The Memory panel in a real browser: add, see why, edit/delete, toggles, forget-all. No research job runs."""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn
from playwright.async_api import async_playwright

from backend.api import app as api_app
from tests.conftest import base_settings

pytestmark = pytest.mark.browser


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def server(tmp_path, monkeypatch):
    async def idle(self, job, emit):
        return None

    monkeypatch.setattr(api_app.JobManager, "_run", idle)
    app = api_app._make_app(base_settings(storage={"db_path": str(tmp_path / "ui.db")}, memory={"enabled": True, "path": str(tmp_path / "mem.db"), "embedder": "hash"}))
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    assert srv.started
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
async def page(server):
    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(headless=True, channel="chrome")
        except Exception:  # noqa: BLE001
            browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1100, "height": 800})
        page = await context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        await page.goto(server)
        await page.wait_for_selector("#question")
        yield page
        await browser.close()
        assert not errors, errors


async def test_memory_panel_add_search_why_delete_and_forget_all(page):
    await page.click("#openMemory")
    await page.wait_for_function("document.querySelector('#memStats').textContent.length > 0")
    assert await page.locator("#memory").is_visible() and await page.locator("#history").is_hidden()
    assert await page.locator("#memInject").is_checked() and await page.locator("#memCapture").is_checked()

    await page.fill("#memNew", "Owns a cat called Miso")
    await page.click("#memAdd")
    await page.wait_for_selector("#memList .mem")
    card = page.locator("#memList .mem").first
    assert "Owns a cat called Miso" in await card.inner_text()
    chips = await card.locator(".chip").all_inner_texts()
    assert "you said it" in chips and any(c.startswith("confidence") for c in chips) and "active" in chips

    await page.fill("#memQuery", "what is my cat called?")
    await page.click("#memSearch")
    await page.wait_for_selector("#memSearchOut .mem-why")
    assert "why:" in await page.locator("#memSearchOut .mem-why").first.inner_text()
    await page.fill("#memQuery", "what is the capital of France?")
    await page.click("#memSearch")
    await page.wait_for_function("document.querySelectorAll('#memSearchOut .mem').length === 0")
    assert "general" in await page.locator("#memSearchOut").inner_text()

    await page.click("#memInject")  # switch off
    await page.reload()
    await page.click("#openMemory")
    await page.wait_for_selector("#memList .mem")
    await page.wait_for_function("document.querySelector('#memStats').textContent.length > 0")
    assert not await page.locator("#memInject").is_checked()

    await page.locator("#memList .mem button", has_text="delete").first.click()
    await page.wait_for_selector("#memList .hint")
    assert "Nothing stored" in await page.locator("#memList").inner_text()

    await page.fill("#memNew", "Lives in Kyiv")
    await page.click("#memAdd")
    await page.wait_for_selector("#memList .mem")
    page.once("dialog", lambda d: d.dismiss())
    await page.click("#memForgetAll")
    await page.wait_for_timeout(300)
    assert await page.locator("#memList .mem").count() == 1, "dismissing the confirm keeps everything"
    page.once("dialog", lambda d: d.accept())
    await page.click("#memForgetAll")
    await page.wait_for_selector("#memList .hint")


async def test_escape_closes_memory_panel_and_project_box_exists(page):
    await page.click("#openMemory")
    await page.keyboard.press("Escape")
    assert await page.locator("#memory").is_hidden()
    assert await page.locator("#project").is_visible()