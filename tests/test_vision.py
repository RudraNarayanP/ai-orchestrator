"""Vision fallback: canvas-drawn UI is answered, human checks are not touched.

The fallback is driven by a scripted vision client (no model is available here), so
these tests prove the *mechanics and the safety rules* -- what gets clicked, typed,
refused and reported -- not the accuracy of any real vision model. That part is
UNVERIFIED and listed as such in AGENTS.md.
"""

from __future__ import annotations

import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from backend.browser.engine import BrowserEngine
from backend.browser.vision import OpenAIVisionClient, VisionFallback, click_allowed
from backend.models import ProviderResponse, ProviderStatus
from backend.settings import ProviderConfig, Settings, VisionConfig
from browser.adapters import build_adapter

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PROMPT = "When did Acme launch the Bolt and what does it cost?"


# --------------------------------------------------------------------- offline


@pytest.mark.parametrize(
    "info,label,allowed",
    [
        ({"tag": "canvas", "type": "", "text": "canvas"}, "message box", True),
        ({"tag": "textarea", "type": "", "text": "textarea composer-input"}, "", True),
        ({"tag": "canvas", "type": "", "text": "canvas"}, "I'm not a robot", False),
        ({"tag": "button", "type": "", "text": "button Sign in"}, "", False),
        ({"tag": "button", "type": "", "text": "button Continue with Google"}, "", False),
        ({"tag": "button", "type": "", "text": "button Upgrade to Plus"}, "", False),
        ({"tag": "button", "type": "", "text": "button Pay now"}, "", False),
        ({"tag": "input", "type": "password", "text": "input"}, "", False),
        ({"tag": "iframe", "type": "", "text": "iframe"}, "", False),
        ({"tag": None, "type": "", "text": ""}, "", False),
    ],
)
def test_click_allowed_never_operates_login_payment_or_human_checks(info, label, allowed):
    ok, why = click_allowed(info, label)
    assert ok is allowed, why


async def test_vision_client_sends_the_screenshot_and_parses_fenced_json(fake_openai):
    fake_openai.script('Sure!\n```json\n{"scene": "chat", "composer": {"x": 10, "y": 20}}\n```')
    cfg = VisionConfig(provider="openai_compatible", model="vis", base_url=fake_openai.base_url, api_key="k")
    client = OpenAIVisionClient(cfg)
    got = await client.ask(PNG_MAGIC + b"fake", "locate", "where is the box?")
    assert got == {"scene": "chat", "composer": {"x": 10, "y": 20}}
    sent = fake_openai.requests[-1]
    content = sent["messages"][0]["content"]
    assert content[0]["type"] == "text" and "where is the box?" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent["_headers"]["authorization"] == "Bearer k"


async def test_vision_client_reports_garbage_and_server_errors_as_none(fake_openai):
    cfg = VisionConfig(provider="openai_compatible", model="vis", base_url=fake_openai.base_url)
    client = OpenAIVisionClient(cfg)
    fake_openai.script("I cannot read that image, sorry.")
    assert await client.ask(PNG_MAGIC, "locate", "x") is None
    assert "not JSON" in (client.last_error or "")
    fake_openai.script({"status": 500, "body": "boom"})
    assert await client.ask(PNG_MAGIC, "locate", "x") is None


def _bare_adapter(vision_provider: str = "disabled"):
    raw = {
        "providers": {"fixture": {"enabled": True, "label": "Fixture", "url": "about:blank", "adapter": "generic_chat"}},
        "vision": {"provider": vision_provider, "model": "m", "base_url": "http://127.0.0.1:1/v1"},
    }
    settings = Settings.model_validate(raw)
    engine = BrowserEngine(settings)
    return build_adapter("fixture", engine, settings, settings.providers["fixture"])


def _resp(status: ProviderStatus) -> ProviderResponse:
    r = ProviderResponse(job_id="j", provider="fixture", prompt="p")
    r.status = status
    return r


def test_vision_is_off_by_default_and_only_for_broken_before_send():
    off = _bare_adapter("disabled")
    assert not off._vision_eligible(_resp(ProviderStatus.BROKEN))

    on = _bare_adapter("openai_compatible")
    assert on._vision_eligible(_resp(ProviderStatus.BROKEN))
    # login wall, rate limit, plain failure, timeout: a different problem, not selector drift
    for status in (ProviderStatus.LOGGED_OUT, ProviderStatus.RATE_LIMITED, ProviderStatus.FAILED, ProviderStatus.TIMEOUT):
        assert not on._vision_eligible(_resp(status)), status
    # BROKEN after the prompt went out must not re-submit the question
    on._prompt_sent = True
    assert not on._vision_eligible(_resp(ProviderStatus.BROKEN))


def test_unconfigured_fallback_is_unavailable():
    settings = Settings.model_validate({"providers": {}})
    assert VisionFallback(settings).available is False


# --------------------------------------------------------------------- browser


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs) -> None:  # noqa: D102
        return


@pytest.fixture(scope="module")
def site():
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(_Quiet, directory=str(FIXTURES)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="module")
def vision_settings() -> Settings:
    return Settings.model_validate(
        {
            "providers": {
                "fixture": {"enabled": True, "label": "Fixture", "url": "about:blank", "adapter": "generic_chat", "max_retries": 0}
            },
            "browser": {"headless": True, "width": 1100, "height": 760, "keep_windows_open": False},
            "vision": {
                "provider": "openai_compatible",
                "model": "scripted",
                "base_url": "http://127.0.0.1:1/v1",
                "wait_s": 40,
                "stable_s": 1.5,
                "max_calls": 6,
            },
            "storage": {"db_path": "data/test_vision.db", "artifacts_dir": "data/test_artifacts"},
        }
    )


class ScriptedVision:
    """Stands in for the vision model. It 'sees' the canvas through the fixture's
    own state, which makes it a perfect reader -- the point is to test what the
    fallback *does* with what it is told, not whether a model can read pixels."""

    def __init__(self, engine: BrowserEngine, provider: str = "fixture", *, scene: str = "chat", composer_label: str | None = None, send_label: str | None = None, use_send: bool = True) -> None:
        self.engine = engine
        self.provider = provider
        self.scene = scene
        self.composer_label = composer_label
        self.send_label = send_label
        self.use_send = use_send
        self.calls: list[str] = []
        self.shot_sizes: list[int] = []

    def page(self):
        return self.engine._sessions["shared"].pages[self.provider]

    async def ask(self, png: bytes, task: str, instruction: str) -> dict[str, Any] | None:
        assert png.startswith(PNG_MAGIC), "the model must be handed a real PNG screenshot"
        self.calls.append(task)
        self.shot_sizes.append(len(png))
        state = await self.page().evaluate("window.__canvas")
        if task == "locate":
            composer = dict(state["layout"]["composer"])
            send = dict(state["layout"]["send"]) if self.use_send else None
            if self.composer_label:
                composer["label"] = self.composer_label
            if send and self.send_label:
                send["label"] = self.send_label
            return {"scene": self.scene, "composer": composer, "send": send}
        return {
            "scene": "chat",
            "answer": state["shown"] if state["submitted"] else "",
            "complete": bool(state["finished"]),
        }


@pytest.fixture()
async def rig(vision_settings, site):
    engine = BrowserEngine(vision_settings)
    cfg = ProviderConfig(enabled=True, label="Fixture", url=site + "/canvas_chat.html", adapter="generic_chat", max_retries=0)
    adapter = build_adapter("fixture", engine, vision_settings, cfg)
    adapter.sel.never_started_ms = 5000
    try:
        yield engine, adapter, site
    finally:
        await engine.stop(keep_windows=False)


async def _ask(adapter, engine, client, url: str, prompt: str = PROMPT):
    adapter.cfg.url = url
    adapter.cfg.new_chat_url = None
    adapter.vision_client = client
    return await adapter.ask("job-v", prompt, 1)


@pytest.mark.browser
async def test_canvas_ui_with_no_accessible_input_is_answered_by_vision_fallback(rig):
    engine, adapter, site = rig
    client = ScriptedVision(engine)
    response = await _ask(adapter, engine, client, site + "/canvas_chat.html")

    assert response.status == ProviderStatus.COMPLETED, (response.status, response.error, response.detail)
    assert "14 May 2026" in response.answer_text and "$549" in response.answer_text
    assert response.detail == "vision fallback", "provenance must be recorded"
    assert response.citations == [], "nothing was read from the DOM, so nothing can be cited"
    assert response.error is None
    assert client.calls[0] == "locate" and "transcribe" in client.calls
    state = await client.page().evaluate("window.__canvas")
    # the prompt really reached the canvas app, typed as one line and submitted by the arrow
    assert state["submitted"] == PROMPT
    clicks = [e for e in state["events"] if e["t"] == "click"]
    assert [(c["x"], c["y"]) for c in clicks] == [(340, 510), (840, 510)]


@pytest.mark.browser
async def test_exhausted_attempts_keep_the_terminal_status_not_launching(rig):
    """Regression: the last failed attempt used to be reset to LAUNCHING."""
    engine, adapter, site = rig
    adapter.settings.vision.provider = "disabled"
    try:
        adapter.cfg.url = site + "/canvas_chat.html"
        adapter.cfg.new_chat_url = None
        response = await adapter.ask("job-r", PROMPT, 1)
    finally:
        adapter.settings.vision.provider = "openai_compatible"
    assert response.status == ProviderStatus.BROKEN, (response.status, response.error)
    assert response.status.terminal
    assert response.detail != "vision fallback"


@pytest.mark.browser
async def test_canvas_captcha_is_reported_blocked_and_never_touched(rig):
    engine, adapter, site = rig
    client = ScriptedVision(engine, scene="captcha")
    response = await _ask(adapter, engine, client, site + "/canvas_chat.html?captcha=1")

    assert response.status == ProviderStatus.FAILED
    assert "blocked" in (response.error or "")
    assert (response.detail or "").startswith("vision fallback")
    assert response.answer_text == ""
    state = await client.page().evaluate("window.__canvas")
    assert state["events"] == [], f"the captcha page must receive no clicks or keys: {state['events']}"
    assert "transcribe" not in client.calls


@pytest.mark.browser
async def test_dom_human_check_stops_the_fallback_even_if_the_model_says_chat(rig):
    engine, adapter, site = rig
    client = ScriptedVision(engine, scene="chat")  # the model is wrong; the DOM is not
    response = await _ask(adapter, engine, client, site + "/canvas_chat.html?iframe=1")

    assert response.status == ProviderStatus.FAILED
    assert "blocked" in (response.error or "")
    assert client.calls == [], "no screenshot should even be sent when the DOM shows a human check"
    state = await client.page().evaluate("window.__canvas")
    assert state["events"] == []


@pytest.mark.browser
async def test_model_pointing_at_a_restricted_control_is_refused_before_anything_is_typed(rig):
    engine, adapter, site = rig
    client = ScriptedVision(engine, send_label="Sign in")
    response = await _ask(adapter, engine, client, site + "/canvas_chat.html")

    assert response.status == ProviderStatus.BROKEN  # unchanged: the fallback did not rescue it
    assert "refused to click send button" in (response.detail or "")
    state = await client.page().evaluate("window.__canvas")
    assert state["events"] == [] and state["typed"] == "", "nothing may be clicked or typed once a target is refused"


@pytest.mark.browser
async def test_real_login_button_under_the_pointer_is_never_pressed(vision_settings, site):
    engine = BrowserEngine(vision_settings)
    try:
        page = await engine.open_research_page("fixture", "about:blank")
        await page.set_content(
            "<body style='margin:0'><button id=b style='position:absolute;left:100px;top:100px;width:200px;height:60px' "
            "onclick='window.__pressed=true'>Sign in</button></body>"
        )

        class PointsAtLogin:
            async def ask(self, png, task, instruction):
                return {"scene": "chat", "composer": {"x": 150, "y": 120, "label": "message box"}, "send": None}

        fb = VisionFallback(vision_settings, client=PointsAtLogin())
        outcome = await fb.run(page, "hello there")
        assert outcome.kind == "failed" and "refused" in outcome.reason
        assert await page.evaluate("!!window.__pressed") is False
    finally:
        await engine.stop(keep_windows=False)


@pytest.mark.browser
async def test_working_dom_path_never_calls_the_vision_model(rig):
    engine, adapter, site = rig
    client = ScriptedVision(engine)
    response = await _ask(adapter, engine, client, site + "/chat_fixture.html", prompt="Acme Bolt?")
    assert response.status == ProviderStatus.COMPLETED
    assert response.detail != "vision fallback"
    assert client.calls == []