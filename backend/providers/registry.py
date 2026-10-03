"""Provider registry: settings -> live adapter instances.

The runner only ever talks to the ``ask(job_id, prompt, round_no, emit)``
protocol, so adapters are independently replaceable and a config-only provider is
indistinguishable from a hand-written one.
"""

from __future__ import annotations

from typing import Any

from backend.settings import Settings
from browser.adapters import build_adapter


class ProviderCatalog:
    def __init__(self, settings: Settings, engine: Any) -> None:
        self.settings = settings
        self.engine = engine
        self._adapters: dict[str, Any] = {}

    def enabled(self) -> list[str]:
        return [name for name, cfg in self.settings.providers.items() if cfg.enabled]

    def adapter(self, provider: str) -> Any | None:
        cfg = self.settings.providers.get(provider)
        if cfg is None or not cfg.enabled:
            return None
        if provider not in self._adapters:
            self._adapters[provider] = build_adapter(provider, self.engine, self.settings, cfg)
        return self._adapters[provider]

    def all(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.enabled():
            adapter = self.adapter(name)
            if adapter is not None:
                out[name] = adapter
        return out

    def label(self, provider: str) -> str:
        cfg = self.settings.providers.get(provider)
        return cfg.label if cfg else provider


def endpoint_for(settings: Settings, name: str):
    from backend.verification.llm import Endpoint

    section = getattr(settings, name, None)
    if section is None:
        return None
    endpoint = Endpoint.from_config(section)
    return endpoint if endpoint.enabled else None
