"""A site that refuses us must be reported as refusing us, with what the page said.

Live evidence (2026-10-10, isolated headless profile, `data/probe/chatgpt_1791630885.json`):
chatgpt.com returned a Cloudflare browser check -- ``title: "Just a moment..."``, an empty
``<body>``, no input and no button. The readiness probe only ever looked at page *text*, so it
classified that as `readiness=unknown` (our word for selector drift), then retried the same
interstitial and spent a vision fallback on a page with nothing to read. Cloudflare names
itself in the title, so the title is the evidence.

Nothing here evades a check: the outcome is a truer record and fewer wasted requests.
"""

from __future__ import annotations

import asyncio

from backend.models import ProviderResponse, ProviderStatus
from backend.settings import ProviderConfig
from browser.adapters.base import ChatAdapter
from tests.conftest import base_settings

CF_STATE = {"state": "unknown", "url": "https://chatgpt.test/", "inputHere": False, "credentialForm": False,
            "loginWall": False, "rateLimited": False, "blocked": False, "gate": "", "bodyLength": 0,
            "bodyHead": "", "title": "Just a moment...", "visibility": "visible"}


def chat_adapter(**setting_overrides) -> ChatAdapter:
    return ChatAdapter(object(), base_settings(**setting_overrides), "chatgpt",
                       ProviderConfig(enabled=True, label="X", url="https://chatgpt.test/"))


def probe_returns(state: dict) -> None:
    """Stand in for the injected DOM library: one fixed answer to the `ready` probe."""
    async def _call(page, fn, *args):
        assert fn == "ready", fn
        return dict(state)
    return _call


async def test_a_cloudflare_interstitial_is_blocked_not_selector_drift():
    adapter = chat_adapter()
    adapter._call = probe_returns(CF_STATE)
    state = await adapter.readiness(object())
    assert state["state"] == "blocked", "an empty challenge page is a refusal, not a drifting selector"
    assert "cloudflare" in state["gate"]


async def test_a_page_we_can_type_into_is_never_downgraded_by_the_same_words():
    """The phrase alone must not cost us a working provider -- only the absence of a composer does."""
    adapter = chat_adapter()
    adapter._call = probe_returns({**CF_STATE, "inputHere": True,
                                   "bodyHead": "Just a moment while we restore your chats",
                                   "title": "ChatGPT"})
    assert (await adapter.readiness(object()))["state"] == "ready"


async def test_an_age_gate_still_reads_as_blocked():
    """The pre-existing title path (BROKEN_RE) must survive the new rule -- the gate wording comes
    from the page probe, which never saw a gate here."""
    adapter = chat_adapter()
    adapter._call = probe_returns({**CF_STATE, "title": "Confirm your age", "bodyHead": ""})
    assert (await adapter.readiness(object()))["state"] == "blocked"


async def test_the_blocked_page_is_recorded_with_the_only_text_it_rendered():
    """`detail` is where a human goes to attribute the blocker; an empty body used to leave it blank."""
    adapter = chat_adapter()
    page = type("Page", (), {"url": "https://chatgpt.test/"})()

    async def _page(fresh=True):
        return page

    async def _settle(p):
        return None

    async def _open_conversation(p):
        return p

    async def _prepare(p, emit, round_no):
        return {"ok": False, **CF_STATE, "state": "blocked", "gate": "cloudflare browser check"}

    adapter._page, adapter._settle = _page, _settle
    adapter._open_conversation, adapter.prepare = _open_conversation, _prepare

    response = ProviderResponse(job_id="j", round=1, provider="chatgpt", prompt="q",
                                started_at=0.0, status=ProviderStatus.LAUNCHING)
    stopped = await adapter._attempt(page_setup=False, response=response, prompt="q", round_no=1,
                                     emit=lambda *a, **k: asyncio.sleep(0))
    assert stopped is False
    assert response.status == ProviderStatus.FAILED
    assert response.error == "readiness=blocked (cloudflare browser check)"
    assert response.detail == "Just a moment..."


def test_a_blocked_provider_does_not_burn_a_vision_fallback():
    """Vision is for selector drift. A refusal has nothing on screen to read."""
    adapter = chat_adapter(vision={"provider": "openai", "model": "vision-1", "base_url": "http://x.test/v1"})
    blocked = ProviderResponse(job_id="j", provider="chatgpt", prompt="", status=ProviderStatus.FAILED,
                               error="readiness=blocked (cloudflare browser check)")
    drifted = ProviderResponse(job_id="j", provider="chatgpt", prompt="", status=ProviderStatus.BROKEN,
                               error="readiness=unknown")
    assert adapter._vision_eligible(blocked) is False
    assert adapter._vision_eligible(drifted) is True, "the fallback itself is unchanged"


async def test_a_blocked_provider_is_not_asked_again_in_the_same_job():
    """The retry list already exempts readiness=blocked; a misclassification put ChatGPT back in it."""
    adapter = chat_adapter()
    page = type("Page", (), {"url": "https://chatgpt.test/"})()

    async def _page(fresh=True):
        return page

    async def _settle(p):
        return None

    async def _open_conversation(p):
        return p

    async def _prepare(p, emit, round_no):
        return {"ok": False, **CF_STATE, "state": "blocked", "gate": "cloudflare browser check"}

    adapter._page, adapter._settle = _page, _settle
    adapter._open_conversation, adapter.prepare = _open_conversation, _prepare

    calls = []

    async def _emit(kind, message, provider, round_no):
        calls.append(message)

    response = await adapter.ask("j", "When was the bridge finished?", 1, emit=_emit)
    assert response.status == ProviderStatus.FAILED
    assert not any("retrying" in c for c in calls), calls
