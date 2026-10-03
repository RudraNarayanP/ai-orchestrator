"""Configuration: config/settings.yaml merged over config/settings.example.yaml,
with environment overrides. Secrets stay on disk in the backend's config and are
never sent to the frontend (section 25)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
BROWSER_DIR = ROOT / "browser"
PROFILES_DIR = BROWSER_DIR / "profiles"
ADAPTERS_DIR = BROWSER_DIR / "adapters"


class ProviderConfig(BaseModel):
    enabled: bool = True
    url: str
    label: str
    profile: str | None = None
    """Name of the dedicated Chrome profile for this provider. Defaults to the
    provider key, so each site gets its own isolated window + login state."""

    tab_role: Literal["own_window", "shared_tab"] = "own_window"
    adapter: str | None = None
    """Adapter class name in browser/adapters/. Defaults to the provider key,
    then falls back to the generic chat adapter -- so a new site that behaves
    like a normal chat UI needs config only, no new code."""
    timeout_s: int = 150
    max_retries: int = 1
    new_chat_url: str | None = None
    weight: float = 1.0
    requires_login: bool = True
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class BrowserConfig(BaseModel):
    channel: str = "chrome"
    """Drives the installed Chrome/Edge binary rather than a bundled Chromium.
    Real Chrome behaves like the user's browser, which matters for sites that
    distrust automation-shaped browsers."""

    executable: str | None = None
    headless: bool = False
    width: int = 1440
    height: int = 900
    window_offset_x: int = 60
    window_offset_y: int = 40
    """The OmniBrain window is placed here. It is always a window of our own."""

    window_mode: Literal["single", "per_provider"] = "single"
    """single: ONE window, one tab per provider, tabs reused between runs.
       per_provider: an isolated profile (and window) per site, which is stricter
       about cross-site login state but multiplies windows.
       Both modes stay inside browser/profiles and never touch your everyday
       browser -- there is no 'attach to my running Chrome' default."""

    reuse_tabs: bool = True
    """Adopt an already-open OmniBrain tab for a provider instead of opening
    another one. This is what stops 'random new chrome windows'."""

    cdp_url: str | None = None
    """Optional: drive an existing Chrome via DevTools, e.g.
    http://127.0.0.1:9222. Chrome 136+ ignores the debug port on the *default*
    profile, so that Chrome must be launched by you with a non-default
    --user-data-dir. Nothing here copies or reads your cookies."""

    user_agent_seed: str | None = None
    nav_timeout_ms: int = 45000
    settle_ms: int = 900
    locale: str = "en-US"
    max_concurrent_profiles: int = 4
    """In single mode this caps concurrent tabs instead."""

    focus_nudge_after_s: float = 25.0
    """Only bring a background tab forward after this long with no growth."""

    args: list[str] = Field(default_factory=list)
    keep_windows_open: bool = True


class VerifierConfig(BaseModel):
    provider: Literal["ollama", "lm_studio", "openrouter", "openai_compatible", "disabled"] = "ollama"
    model: str = "qwen3:14b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_s: int = 240
    deep_verification: bool = True
    inspect_cited_sources: bool = True
    max_sources_per_claim: int = 3


class AnalysisConfig(BaseModel):
    """Question analysis, claim extraction and synthesis can run on a cheaper
    endpoint than the adversarial verifier."""

    provider: str = "ollama"
    model: str = "qwen3:14b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_s: int = 120


class ResearchConfig(BaseModel):
    mode: Literal["QUICK", "STANDARD", "DEEP_RESEARCH"] = "STANDARD"
    max_rounds: int = 3
    max_workers: int = 4
    per_provider_concurrency: int = 1
    swarm_providers: int = 5
    """How many independent researchers a level-2 escalation uses. Small because
    each one is a browser session, but large enough that an evidenced minority
    actually gets a turn: with four slots and six families, the one provider
    holding the correct figure can simply never be picked."""

    follow_up_providers: int = 3
    min_independent_sources: int = 2
    claim_batch_size: int = 12
    response_stability_poll_ms: int = 700
    response_stable_rounds: int = 3
    hard_response_timeout_s: int = 180


class SearchConfig(BaseModel):
    """Real web research for evidence gathering, run in the dedicated browser
    windows -- not an API."""

    engines: list[str] = Field(default_factory=lambda: ["google", "duckduckgo"])
    max_results: int = 8
    fetch_body_chars: int = 12000
    http_fallback: bool = True
    per_query_timeout_s: int = 40


class StorageConfig(BaseModel):
    db_path: str = str(DATA_DIR / "omnibrain.db")
    artifacts_dir: str = str(DATA_DIR / "artifacts")
    keep_raw_responses: bool = True
    max_events_per_job: int = 4000


class Settings(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    tone: str = (
        "casual, short, light humour -- never forced. Use past context when it "
        "is genuinely relevant. Never guess, never invent a statistic, never "
        "fake precision. Be definitive when the evidence supports it, and say "
        "plainly when it does not."
    )

    def public_dict(self) -> dict[str, Any]:
        """Config handed to the frontend, with every secret stripped."""
        data = self.model_dump(mode="json")
        for section in ("verifier", "analysis"):
            block = data.get(section, {})
            has_key = bool(getattr(self, section).api_key)
            if "api_key" in block:
                block["api_key"] = "***configured***" if has_key else ""
        return data


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return loaded if isinstance(loaded, dict) else {}


def load_settings(path: Path | None = None) -> Settings:
    cfg_path = path or CONFIG_DIR / "settings.yaml"
    if not cfg_path.exists():
        cfg_path = CONFIG_DIR / "settings.example.yaml"
    raw = _read_yaml(cfg_path)
    raw = _apply_env(raw)
    return Settings.model_validate(raw)


def _apply_env(raw: dict[str, Any]) -> dict[str, Any]:
    """MODEL_PROVIDER / MODEL_NAME / BASE_URL / API_KEY / TEMPERATURE /
    MAX_TOKENS from the environment win over the file (section 12)."""
    ver = raw.setdefault("verifier", {})
    env_map = {
        "provider": "OMNIBRAIN_VERIFIER_PROVIDER",
        "model": "OMNIBRAIN_VERIFIER_MODEL",
        "base_url": "OMNIBRAIN_VERIFIER_BASE_URL",
        "api_key": "OMNIBRAIN_VERIFIER_API_KEY",
        "temperature": "OMNIBRAIN_VERIFIER_TEMPERATURE",
        "max_tokens": "OMNIBRAIN_VERIFIER_MAX_TOKENS",
    }
    for key, env in env_map.items():
        val = os.environ.get(env)
        if val is not None:
            ver[key] = float(val) if key == "temperature" else (
                int(val) if key == "max_tokens" else val
            )
    if os.environ.get("OMNIBRAIN_MODE"):
        raw.setdefault("research", {})["mode"] = os.environ["OMNIBRAIN_MODE"]
    if os.environ.get("OMNIBRAIN_MAX_ROUNDS"):
        raw.setdefault("research", {})["max_rounds"] = int(os.environ["OMNIBRAIN_MAX_ROUNDS"])
    return raw


def save_settings(settings: Settings, path: Path | None = None) -> Path:
    out = path or CONFIG_DIR / "settings.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(settings.model_dump(mode="json"), fh, sort_keys=False, allow_unicode=True)
    return out


_SETTINGS: Settings | None = None


def settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def reload_settings() -> Settings:
    global _SETTINGS
    _SETTINGS = load_settings()
    return _SETTINGS
