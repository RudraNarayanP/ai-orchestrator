"""Meta AI adapter.

Meta AI has no guest path: a cold profile is a login wall, full stop. The job
here is to say so precisely -- with the window left open so the user can log in
once -- instead of retrying into a loop.
"""

from __future__ import annotations

from typing import Any

from backend.models import ProviderResponse
from browser.adapters.base import ChatAdapter


class MetaAIAdapter(ChatAdapter):
    name = "meta_ai"

    async def ask(self, job_id: str, prompt: str, round_no: int = 1, emit=None, continue_thread: bool = False) -> ProviderResponse:
        response = await super().ask(job_id, prompt, round_no, emit, continue_thread=continue_thread)
        if response.status.value == "logged_out":
            response.detail = (
                "Meta AI requires a login in its own window. Open the OmniBrain Meta AI "
                "window, sign in once, and re-run -- the profile will remember you."
            )
        return response
