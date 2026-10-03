"""Pi (pi.ai) adapter.

Pi's composer is a contenteditable rather than a textarea, and its answers are
short, warm and almost never sourced. That makes it useful for framing and
useless as evidence, so its responses carry the lowest independent weight and
are expected to be marked as not having researched.
"""

from __future__ import annotations

from backend.models import ProviderResponse, WebResearchStatus
from browser.adapters.base import ChatAdapter


class PiAdapter(ChatAdapter):
    name = "pi"

    def _assess_web_research(self, response: ProviderResponse) -> None:
        super()._assess_web_research(response)
        if not response.citations:
            response.web_research_status = WebResearchStatus.FAILED_OR_UNCLEAR
            response.detail = (
                (response.detail or "")
                + " Pi answered without retrievable sources; treated as opinion, not evidence."
            ).strip()
