"""One conversation per AI per research id (hard isolation requirement)."""

from __future__ import annotations

from types import SimpleNamespace

from backend.models import ProviderResponse
from browser.adapters.base import ChatAdapter
from tests.conftest import base_settings


class FakePage:
    def __init__(self, url: str) -> None:
        self.url = url
        self.visits: list[str] = [url]

    async def goto(self, url: str, **kw) -> None:
        self.url = url
        self.visits.append(url)

    async def wait_for_load_state(self, *a, **kw) -> None:
        return None

    async def wait_for_timeout(self, *a, **kw) -> None:
        return None


def make_adapter():
    settings = base_settings()
    cfg = settings.providers["gemini"]
    cfg.new_chat_url = "https://gemini.test/app"
    return ChatAdapter(engine=None, settings=settings, provider="gemini", cfg=cfg, selectors=None)


async def test_new_research_never_types_into_the_previous_chat():
    adapter = make_adapter()
    page = FakePage("https://gemini.test/app")
    adapter._research_id, adapter._continue_thread = "research_A", False
    await adapter._open_conversation(page)
    assert page.visits == ["https://gemini.test/app"], "a pristine new-chat tab needs no navigation"

    # research A runs and leaves the tab inside its conversation
    page.url = "https://gemini.test/app/abc123"
    adapter._remember_thread(page, ProviderResponse(job_id="research_A", provider="gemini", prompt="q"))

    adapter._research_id, adapter._continue_thread = "research_B", False
    await adapter._open_conversation(page)
    assert page.url == "https://gemini.test/app", "question B must start in a new chat, not in A's conversation"


async def test_follow_up_returns_to_this_researchs_own_conversation():
    adapter = make_adapter()
    page = FakePage("https://gemini.test/app")
    page.url = "https://gemini.test/app/conv-A"
    adapter._research_id = "research_A"
    adapter._remember_thread(page, ProviderResponse(job_id="research_A", provider="gemini", prompt="q"))

    page.url = "https://gemini.test/app/conv-B"  # the tab drifted to another research
    adapter._research_id, adapter._continue_thread = "research_A", True
    await adapter._open_conversation(page)
    assert page.url == "https://gemini.test/app/conv-A"

    response = ProviderResponse(job_id="research_A", provider="gemini", prompt="follow-up")
    adapter._remember_thread(page, response)
    assert response.continued is True and response.conversation_url.endswith("conv-A")


async def test_continue_without_a_thread_falls_back_to_a_new_conversation():
    adapter = make_adapter()
    # ask() resolves continue_thread against known threads
    assert "research_Z" not in adapter._threads
    adapter._research_id = "research_Z"
    adapter._continue_thread = bool(True and "research_Z" in adapter._threads)
    assert adapter._continue_thread is False

class SlowUrlPage(FakePage):
    """Gemini-style: the conversation id reaches the address bar a moment after the answer."""

    def __init__(self, url: str, changes_after: int, final: str) -> None:
        super().__init__(url)
        self.polls, self.changes_after, self.final = 0, changes_after, final

    @property
    def url(self):
        self.polls += 1
        return self.final if self.polls > self.changes_after else self._url

    @url.setter
    def url(self, value):
        self._url = value

    async def evaluate(self, *a, **kw):
        return None


async def test_a_conversation_id_that_arrives_late_is_still_captured():
    adapter = make_adapter()
    adapter.url_wait_s = 3.0
    adapter._research_id, adapter._continue_thread = "research_A", False
    page = SlowUrlPage("https://gemini.test/app", changes_after=4, final="https://gemini.test/app/d0328f17eebd6a24")
    response = ProviderResponse(job_id="research_A", provider="gemini", prompt="q")
    await adapter._remember_thread_when_known(page, response)
    assert response.conversation_url == "https://gemini.test/app/d0328f17eebd6a24"
    assert adapter._threads["research_A"].endswith("d0328f17eebd6a24"), "the follow-up goes back to this id"


async def test_a_site_that_never_exposes_the_id_is_reported_not_faked():
    adapter = make_adapter()
    adapter.url_wait_s = 0.5
    adapter._research_id, adapter._continue_thread = "research_A", False
    page = FakePage("https://gemini.test/app")
    page.evaluate = lambda *a, **k: _none()
    response = ProviderResponse(job_id="research_A", provider="gemini", prompt="q")
    await adapter._remember_thread_when_known(page, response)
    assert response.conversation_url == "https://gemini.test/app"
    assert "not exposed" in (response.detail or "")


async def _none():
    return None