"""The Threads screen in a real browser against a real HTTP server (fake provider adapters): create, chat, rotation, provider switch,
recall, failure handling, persistence across a reload, delete -- with no console errors."""

from __future__ import annotations

import pytest
from playwright.async_api import async_playwright

from tests.test_thread_http import FakeAdapter, start_server

pytestmark = pytest.mark.browser

LONG = "Let's keep going about the Munich trip details: hotels near the Messe, trains from the airport, and the conference schedule. "


@pytest.fixture
def adapters():
    return {"chatgpt": FakeAdapter("chatgpt"), "gemini": FakeAdapter("gemini"), "copilot": FakeAdapter("copilot", fail=True)}


@pytest.fixture
def server(tmp_path, monkeypatch, adapters):
    app, srv, th, base = start_server(tmp_path, monkeypatch, adapters)
    yield base
    srv.should_exit = True
    th.join(timeout=10)


@pytest.fixture
async def page(server):
    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(headless=True, channel="chrome")
        except Exception:  # noqa: BLE001
            browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1200, "height": 900})
        page = await context.new_page()
        problems: list[str] = []
        page.on("pageerror", lambda exc: problems.append("pageerror: " + str(exc)))
        page.on("console", lambda msg: problems.append("console: " + msg.text) if msg.type == "error" and "502" not in msg.text else None)
        page.on("response", lambda r: problems.append(f"http {r.status}: {r.url}") if r.status >= 400 and r.status != 502 else None)
        page.on("requestfailed", lambda req: problems.append("requestfailed: " + req.url))
        page.on("dialog", lambda d: d.accept())
        await page.goto(server)
        await page.wait_for_selector("#question")
        yield page
        await browser.close()
        assert not problems, problems


async def send(page, text, provider=None):
    if provider:
        await page.select_option("#thProvider", provider)
    before = await page.locator(".th-ai").count()
    await page.fill("#thText", text)
    await page.press("#thText", "Enter")
    await page.wait_for_function("n => document.querySelectorAll('.th-ai').length > n", arg=before)


async def test_thread_screen_end_to_end(page, adapters):
    await page.click("#openThreads")
    await page.wait_for_selector("#thList")
    assert await page.locator("#threads").is_visible() and await page.locator("#memory").is_hidden()
    await page.wait_for_function("document.querySelector('#thList').textContent.includes('No threads yet')")

    await page.fill("#thTitle", "Munich trip")
    await page.fill("#thProject", "Munich")
    await page.click("#thCreate")
    await page.wait_for_selector("#thDetail:not([hidden])")
    assert "Munich trip" in await page.locator("#thHeading").inner_text() and "project: Munich" in await page.locator("#thHeading").inner_text()
    options = await page.locator("#thProvider option").all_inner_texts()
    assert len(options) >= 3 and "Search" not in options

    # a normal chat: one provider chat, the reply is attributed to its provider, the turn shows what it was given
    await send(page, "My codeword for this thread is TANGERINE-42.")
    assert await page.locator(".th-user").count() == 1 and "chatgpt" in (await page.locator(".th-ai .th-who").first.inner_text()).lower()
    assert "chatgpt reply 1" in await page.locator(".th-ai .th-text").first.inner_text()
    chips = await page.locator("#thSegments .chip").all_inner_texts()
    assert len(chips) == 1 and chips[0].startswith("ChatGPT A") and "in use" in chips[0]
    await page.locator(".th-ctx summary").first.click()
    assert "no personal memory used" in await page.locator(".th-ctx summary").first.inner_text()

    # fill the small chat until it rotates: the screen says so, in plain words
    for i in range(10):
        await send(page, f"{LONG}(message {i})")
        if await page.locator(".rotation").count():
            break
    rot = await page.locator(".rotation").first.inner_text()
    assert "ChatGPT B" in rot and "nearly full" in rot and "tokens of context" in rot
    chips = await page.locator("#thSegments .chip").all_inner_texts()
    assert len(chips) >= 2 and "closed" in chips[0] and "in use" in chips[-1] and await page.locator("#thSegments .seg-open").count() == 1
    assert adapters["chatgpt"].calls[-1]["prompt"].startswith("OMNIBRAIN CONTINUATION CONTEXT")

    # switch provider mid-thread: the new provider gets the thread; the screen shows a new chat for Gemini
    await send(page, "Which codeword did I give you?", provider="gemini")
    assert "TANGERINE-42" in adapters["gemini"].calls[0]["prompt"]
    texts = await page.locator(".rotation").all_inner_texts()
    assert any("Gemini A" in t and "changed provider" in t for t in texts)
    assert (await page.locator("#thSegments .chip").all_inner_texts())[-1].startswith("Gemini A")
    assert "gemini" in (await page.locator(".th-ai .th-who").last.inner_text()).lower()

    # find something said earlier
    await page.fill("#thRecall", "TANGERINE codeword")
    await page.click("#thRecallBtn")
    await page.wait_for_selector("#thRecallOut .mem")
    assert "TANGERINE-42" in await page.locator("#thRecallOut").inner_text()

    # a provider that fails: the message stays in the thread and the screen says so
    total = await page.locator(".th-user").count()
    await page.select_option("#thProvider", "copilot")
    await page.fill("#thText", "This one will fail")
    await page.press("#thText", "Enter")
    await page.wait_for_function("document.querySelector('#thStatus').classList.contains('err')")
    assert "saved in the thread" in await page.locator("#thStatus").inner_text()
    assert await page.locator(".th-user").count() == total + 1 and "This one will fail" in await page.locator("#thMessages").inner_text()

    # reload: the thread is still there
    count = await page.locator(".th-msg").count()
    await page.reload()
    await page.wait_for_selector("#question")
    await page.click("#openThreads")
    await page.wait_for_selector(".th-card")
    assert "Munich trip" in await page.locator(".th-card").first.inner_text() and f"{count} messages" in await page.locator(".th-card").first.inner_text()
    await page.locator(".th-open").first.click()
    await page.wait_for_selector(".th-msg")
    assert await page.locator(".th-msg").count() == count and await page.locator(".rotation").count() >= 2

    # Escape closes the panel; deleting asks first and empties the list
    await page.keyboard.press("Escape")
    assert await page.locator("#threads").is_hidden()
    await page.click("#openThreads")
    await page.locator(".th-open").first.click()
    await page.click("#thDelete")
    await page.wait_for_function("document.querySelector('#thList').textContent.includes('No threads yet')")


async def test_provider_text_is_never_html(page, adapters):
    adapters["chatgpt"].fail = False
    original = adapters["chatgpt"].ask

    async def hostile(key, prompt, rnd, emit=None, **kw):
        r = await original(key, prompt, rnd, emit, **kw)
        r.answer_text = '<img src=x onerror="window.__pwned=1"><b>bold</b>'
        return r

    adapters["chatgpt"].ask = hostile
    await page.click("#openThreads")
    await page.click("#thCreate")
    await page.wait_for_selector("#thDetail:not([hidden])")
    await send(page, "hi")
    assert "<img" in await page.locator(".th-ai .th-text").first.inner_text()
    assert await page.evaluate("window.__pwned === undefined") and await page.locator(".th-ai img, .th-ai b").count() == 0