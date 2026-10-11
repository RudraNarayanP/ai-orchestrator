"""Configuration checks that turn a bad settings file into a one-line, fixable message.

``load_settings`` raises :class:`backend.settings.ConfigError` for a file that cannot be read at all (YAML syntax,
a wrong type). ``check_settings`` is for a file that loads but would misbehave: a mistyped key that pydantic would
silently ignore, a provider list that is empty, a verifier with no key, a data folder that cannot be written. Each
finding says which key, what is wrong and what to do. Errors stop ``serve``; warnings are printed and the run goes on.
Nothing here reads a secret: a key is only ever tested for being empty or not.
"""

from __future__ import annotations

import difflib
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.settings import Settings


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warning"
    key: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.key}: {self.message}"


def unknown_keys(raw: dict[str, Any], model: type = Settings, prefix: str = "") -> list[Finding]:
    """Keys in the file that no setting reads (usually a typo). Free-form dicts are not walked."""
    out: list[Finding] = []
    fields = model.model_fields
    for key, value in (raw or {}).items():
        path = f"{prefix}{key}"
        if key not in fields:
            near = difflib.get_close_matches(str(key), list(fields), n=1, cutoff=0.75)
            hint = f" Did you mean '{prefix}{near[0]}'?" if near else ""
            out.append(Finding("warning", path, f"unknown setting, nothing reads it.{hint}"))
            continue
        sub = _model_of(fields[key].annotation)
        if sub is not None and isinstance(value, dict):
            out += unknown_keys(value, sub, path + ".")
        elif key == "providers" and isinstance(value, dict):
            from backend.settings import ProviderConfig

            grouped: dict[str, list[str]] = {}
            for name, block in value.items():
                if isinstance(block, dict):
                    for f in unknown_keys(block, ProviderConfig, ""):
                        grouped.setdefault(f.key, []).append(str(name))
            for sub, names in grouped.items():
                where = f"providers.{names[0]}.{sub}" if len(names) == 1 else f"providers.*.{sub}"
                near = difflib.get_close_matches(sub, list(ProviderConfig.model_fields), n=1, cutoff=0.75)
                hint = f" Did you mean '{near[0]}'?" if near else " Remove it."
                who = f" (in {', '.join(names)})" if len(names) > 1 else ""
                out.append(Finding("warning", where, f"unknown setting, nothing reads it.{hint}{who}"))
    return out


def _model_of(annotation: Any) -> type | None:
    try:
        from pydantic import BaseModel

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
    except Exception:  # noqa: BLE001
        pass
    return None


def _has_key(block: Any) -> bool:
    return bool(getattr(block, "api_key", "")) or bool(os.environ.get("OPENROUTER_API_KEY"))


def port_in_use(host: str, port: int) -> bool:
    probe = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
    try:
        with socket.create_connection((probe, port), timeout=0.5):
            return True
    except OSError:
        return False


def _writable_dir(path: Path) -> str | None:
    """None when ``path`` (a folder, or the folder of a file) can be created and written, else the reason."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".omnibrain_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return None
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"


def check_settings(settings: Settings, raw: dict[str, Any] | None = None) -> list[Finding]:
    out: list[Finding] = []
    if raw:
        out += unknown_keys(raw)
    enabled = {n: p for n, p in settings.providers.items() if p.enabled}
    if not enabled:
        out.append(Finding("error", "providers", "no provider is enabled, so there is nobody to ask. Set 'enabled: true' on at least one provider in config/settings.yaml."))
    for name, p in settings.providers.items():
        if not str(p.url).lower().startswith(("http://", "https://")):
            out.append(Finding("error", f"providers.{name}.url", f"'{p.url}' is not an http(s) address."))
    if settings.research.max_rounds < 1:
        out.append(Finding("error", "research.max_rounds", "must be at least 1."))
    if settings.research.time_budget_s < 0:
        out.append(Finding("error", "research.time_budget_s", "must be 0 (no budget) or a number of seconds."))
    if settings.verifier.time_cap_s < 0:
        out.append(Finding("error", "verifier.time_cap_s", "must be 0 (no cap) or a number of seconds."))
    if settings.research.max_workers < 1:
        out.append(Finding("error", "research.max_workers", "must be at least 1."))
    if settings.research.swarm_providers < 1:
        out.append(Finding("error", "research.swarm_providers", "must be at least 1."))
    if not 0.1 <= settings.threads.rotate_at <= 1.0:
        out.append(Finding("error", "threads.rotate_at", f"is {settings.threads.rotate_at}; use a share between 0.1 and 1.0."))
    for bad in ("nav_timeout_ms", "chrome_use_timeout_s"):
        if getattr(settings.browser, bad) <= 0:
            out.append(Finding("error", f"browser.{bad}", "must be greater than 0."))
    for name, modes in settings.research.provider_modes.items():
        if name not in settings.providers:
            out.append(Finding("warning", f"research.provider_modes.{name}", f"'{name}' is not a configured provider, so these modes are never used. Known: {', '.join(settings.providers)}."))
        elif not settings.providers[name].enabled:
            out.append(Finding("warning", f"research.provider_modes.{name}", f"'{name}' is switched off, so these modes are never used."))
    b = settings.browser
    if b.driver == "chrome_use" and b.chrome_use_background and b.chrome_use_front_providers:
        unknown = [n for n in b.chrome_use_front_providers if n not in settings.providers]
        if unknown:
            out.append(Finding("warning", "browser.chrome_use_front_providers", f"unknown provider(s): {', '.join(unknown)}."))
    if b.driver == "chrome_use" and b.chrome_use_path and not Path(b.chrome_use_path).exists():
        out.append(Finding("error", "browser.chrome_use_path", f"'{b.chrome_use_path}' does not exist. Remove the line to look 'chrome-use' up on PATH, or run scripts\\install_chrome_use.ps1."))
    for role in ("verifier", "analysis", "vision"):
        block = getattr(settings, role)
        provider = getattr(block, "provider", "disabled")
        if provider == "disabled":
            continue
        url = str(getattr(block, "base_url", "") or "")
        if not url.lower().startswith(("http://", "https://")):
            out.append(Finding("error", f"{role}.base_url", f"'{url}' is not an http(s) address; set it, or set {role}.provider: disabled."))
        elif provider == "openrouter" and not _has_key(block):
            out.append(Finding("warning", f"{role}.api_key", "OpenRouter needs a key: put it in config/settings.yaml (git-ignored) or set OPENROUTER_API_KEY. Without it the evidence ledger still decides, without a model to adjudicate."))
        if not getattr(block, "model", ""):
            out.append(Finding("error", f"{role}.model", "is empty."))
    for label, folder in (("storage.db_path", Path(settings.storage.db_path).parent), ("storage.artifacts_dir", Path(settings.storage.artifacts_dir)), ("storage.log_path", Path(settings.storage.log_path).parent)):
        why = _writable_dir(folder)
        if why:
            out.append(Finding("error", label, f"{folder} cannot be written ({why})."))
    return out


def environment_findings() -> list[Finding]:
    """Things a fresh clone can get wrong that are not in the settings file."""
    out: list[Finding] = []
    if sys.version_info < (3, 11):
        out.append(Finding("error", "python", f"{sys.version.split()[0]} is too old; OmniBrain needs Python 3.11 or newer."))
    for module, pip in (("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("pydantic", "pydantic"), ("httpx", "httpx"), ("yaml", "pyyaml"), ("aiosqlite", "aiosqlite")):
        try:
            __import__(module)
        except ImportError:
            out.append(Finding("error", f"package {pip}", f"is not installed. Run: python -m pip install -r requirements.txt"))
    for module, pip, why in (("numpy", "numpy", "memory search falls back to keyword-only"), ("pypdf", "pypdf", "PDF sources cannot be read")):
        try:
            __import__(module)
        except ImportError:
            out.append(Finding("warning", f"package {pip}", f"is not installed ({why}). Run: python -m pip install -r requirements.txt"))
    return out


def tracked_secret_findings(root: Path) -> list[Finding]:
    """Warn when the real settings file is not git-ignored (it may hold the OpenRouter key)."""
    ignore = root / ".gitignore"
    try:
        text = ignore.read_text(encoding="utf-8")
    except OSError:
        return []
    if "config/settings.yaml" not in text:
        return [Finding("warning", ".gitignore", "config/settings.yaml is not listed, so an API key in it could be committed. Add the line 'config/settings.yaml'.")]
    return []