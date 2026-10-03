"""Adapter registry: provider key -> adapter class.

Adding a provider is either one line here plus a selectors entry, or nothing at
all if the site behaves like a normal chat UI (config-only, via ``generic_chat``).
"""

from __future__ import annotations

from typing import Type

from backend.settings import ProviderConfig, Settings
from backend.browser.engine import BrowserEngine
from browser.adapters.base import ChatAdapter
from browser.adapters.chatgpt import ChatGPTAdapter
from browser.adapters.copilot import CopilotAdapter
from browser.adapters.gemini import GeminiAdapter
from browser.adapters.google_ai import GoogleAIAdapter
from browser.adapters.le_chat import LeChatAdapter
from browser.adapters.meta_ai import MetaAIAdapter
from browser.adapters.pi import PiAdapter
from browser.adapters.search import SearchAdapter
from browser.adapters.selectors import selectors_for

ADAPTERS: dict[str, Type[ChatAdapter]] = {
    ChatGPTAdapter.name: ChatGPTAdapter,
    GeminiAdapter.name: GeminiAdapter,
    GoogleAIAdapter.name: GoogleAIAdapter,
    CopilotAdapter.name: CopilotAdapter,
    MetaAIAdapter.name: MetaAIAdapter,
    LeChatAdapter.name: LeChatAdapter,
    PiAdapter.name: PiAdapter,
    SearchAdapter.name: SearchAdapter,
    "generic_chat": ChatAdapter,
}


def build_adapter(
    provider: str,
    engine: BrowserEngine,
    settings: Settings,
    cfg: ProviderConfig,
) -> ChatAdapter:
    key = cfg.adapter or provider
    cls = ADAPTERS.get(key) or ADAPTERS.get("generic_chat") or ChatAdapter
    kwargs = {}
    if cls is SearchAdapter:
        kwargs["engine"] = (cfg.extra or {}).get("engine", settings.search.engines[0])
    return cls(engine, settings, provider, cfg, selectors=selectors_for(provider))


def enabled_providers(settings: Settings) -> list[str]:
    return [name for name, cfg in settings.providers.items() if cfg.enabled]


__all__ = [
    "ADAPTERS",
    "ChatAdapter",
    "build_adapter",
]
