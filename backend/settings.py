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

    adapter: str | None = None
    """Adapter class name in browser/adapters/. Defaults to the provider key,
    then falls back to the generic chat adapter -- so a new site that behaves
    like a normal chat UI needs config only, no new code."""
    max_retries: int = 1
    new_chat_url: str | None = None
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

    driver: Literal["playwright", "chrome_use"] = "playwright"
    """playwright (default): OmniBrain's own dedicated Chrome profile, as described above.
       chrome_use: OPT-IN. Drive new tabs in your everyday, already-signed-in Chrome through
       the third-party ``chrome-use`` CLI + its Chrome extension (see backend/browser/live_chrome.py).
       Hard-limited in code to the AI provider domains; no cookies/profile are copied or read;
       captcha / Cloudflare / age / sign-in pages are reported as needing you, never touched."""

    chrome_use_path: str | None = None
    """Full path to chrome-use(.exe) if it is not on PATH (scripts/install_chrome_use.ps1 installs it
    under %LOCALAPPDATA%; pass that path here if PATH was not refreshed)."""

    chrome_use_session: str = "omnibrain"
    """chrome-use session name: OmniBrain's tabs live in their own colored tab group."""

    chrome_use_browser: str | None = None
    """Pin a Chrome profile (id or Google-account e-mail) when several profiles run the extension."""

    chrome_use_timeout_s: float = 60.0

    chrome_use_background: bool = True
    """True (default): tabs are opened and driven in the background and your Chrome window is never raised or
    focused (except the providers you list in ``chrome_use_front_providers``). A provider that cannot work without
    focus is reported as ``needs_focus`` instead of stealing it. False: OmniBrain's tab is shown in front so you can
    watch it work."""

    chrome_use_front_providers: list[str] = Field(default_factory=list)
    """Only with ``chrome_use_background: true``: providers whose tab may still be raised before typing/sending."""

    chrome_use_front_on_failure: bool = False
    """If a provider cannot be driven in the background (page never renders its composer, text does not land),
    raise just that provider's tab once and retry. Off by default: nothing is brought to the front unasked."""

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
    """For OpenRouter this may stay empty if OPENROUTER_API_KEY is set in the environment."""

    fallback_models: list[str] = Field(default_factory=list)
    """Tried in order when the model is rate limited (429), gone (404) or returns nothing."""

    headers: dict[str, str] = Field(default_factory=dict)
    """Extra request headers, e.g. {"HTTP-Referer": "http://localhost", "X-Title": "OmniBrain"} for OpenRouter."""

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
    fallback_models: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_s: int = 120


class VisionConfig(BaseModel):
    """Level-2 fallback: read the page as pixels when the DOM path is BROKEN.

    Off unless a vision-capable OpenAI-compatible endpoint is configured. It only
    ever runs after selector drift; it never clicks a captcha, a login control or
    a payment control (see backend/browser/vision.py).
    """

    provider: str = "disabled"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    fallback_models: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 1500
    timeout_s: int = 90
    max_calls: int = 6
    """Hard cap on vision requests per fallback attempt (1 locate + transcriptions)."""

    wait_s: int = 90
    """How long to wait for the on-screen answer to stop changing."""

    stable_s: float = 3.0


class ResearchConfig(BaseModel):
    mode: Literal["QUICK", "STANDARD", "DEEP_RESEARCH"] = "STANDARD"
    max_rounds: int = 3
    max_workers: int = 4
    per_provider_concurrency: int = 1
    min_provider_spacing_s: float = 2.0
    """Minimum pause between two questions to the same site."""
    rate_limit_backoff_s: float = 30.0
    """After a rate_limited result: wait this long before asking that site again, doubling per consecutive limit."""
    rate_limit_backoff_max_s: float = 600.0
    rate_limit_max_wait_s: float = 45.0
    """A backoff longer than this is not waited out: the site is skipped for this question instead."""
    swarm_providers: int = 3
    """How many independent researchers a level-2 escalation uses. Small because
    each one is a browser session, but large enough that an evidenced minority
    actually gets a turn: with four slots and six families, the one provider
    holding the correct figure can simply never be picked."""

    follow_up_providers: int = 3
    min_independent_sources: int = 2
    claim_batch_size: int = 12
    response_stability_poll_ms: int = 700
    response_stable_rounds: int = 3


class SearchConfig(BaseModel):
    """Real web research for evidence gathering, run in the dedicated browser
    windows -- not an API."""

    engines: list[str] = Field(default_factory=lambda: ["google", "duckduckgo"])
    max_results: int = 8
    fetch_body_chars: int = 12000
    review_queries: int = 3
    """Product/service questions: review-site searches per job (Reddit, G2, app stores...). 0 disables."""
    refutation_queries: int = 3
    """Material claims per round that get a counter-query (claim + correction / rebuttal / contradicts / actually). 0 disables."""

    per_query_timeout_s: int = 40

    own_discovery: bool = False
    """False (default): the AIs do the web research and OmniBrain only opens the URLs they cited,
    to audit them. True: OmniBrain also runs its own searches (DuckDuckGo/Bing, review sites) and
    files what it finds as evidence -- off by default because that is OmniBrain doing the research."""


class StorageConfig(BaseModel):
    db_path: str = str(DATA_DIR / "omnibrain.db")
    artifacts_dir: str = str(DATA_DIR / "artifacts")
    keep_raw_responses: bool = True
    max_events_per_job: int = 4000
    log_path: str = str(DATA_DIR / "omnibrain.log")
    """Rotating log file (2 MB x 3). `run.py serve|ask --log-file PATH` overrides it."""


class MemoryConfig(BaseModel):
    """Long-term memory (backend/memory). Local only: the file below never leaves this machine."""

    enabled: bool = True
    path: str = str(DATA_DIR / "memory.db")
    embedder: str = "auto"
    """auto = the local ONNX model when installed, else the built-in hashing embedder; hash | fastembed to force one."""
    model: str = "BAAI/bge-small-en-v1.5"


class ThreadConfig(BaseModel):
    """The unlimited OmniBrain thread (backend/thread). Local only."""

    enabled: bool = True
    path: str = str(DATA_DIR / "threads.db")
    rotate_at: float = 0.8
    """Rotate a provider chat when it reaches this share of its (approximate) context limit."""
    packet_budget_tokens: int = 6000
    recent_messages: int = 10
    limits: dict[str, int] = Field(default_factory=dict)
    """Per-provider context limit in approximate tokens, e.g. {chatgpt: 32000}. Missing providers use backend/thread/service.py DEFAULT_LIMITS."""


class Settings(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    threads: ThreadConfig = Field(default_factory=ThreadConfig)
    tone: str = (
        "casual, short, light humour -- never forced. Use past context when it "
        "is genuinely relevant. Never guess, never invent a statistic, never "
        "fake precision. Be definitive when the evidence supports it, and say "
        "plainly when it does not."
    )

    def public_dict(self) -> dict[str, Any]:
        """Config handed to the frontend, with every secret stripped."""
        data = self.model_dump(mode="json")
        for section in ("verifier", "analysis", "vision"):
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
    cfg_path = path or (Path(os.environ["OMNIBRAIN_CONFIG"]) if os.environ.get("OMNIBRAIN_CONFIG") else CONFIG_DIR / "settings.yaml")
    if not cfg_path.exists():
        cfg_path = CONFIG_DIR / "settings.example.yaml"
    raw = _read_yaml(cfg_path)
    raw = _apply_env(raw)
    return Settings.model_validate(raw)


def _apply_env(raw: dict[str, Any]) -> dict[str, Any]:
    """MODEL_PROVIDER / MODEL_NAME / BASE_URL / API_KEY / TEMPERATURE /
    MAX_TOKENS from the environment win over the file (section 12)."""
    for section, prefix in (("verifier", "OMNIBRAIN_VERIFIER"), ("analysis", "OMNIBRAIN_ANALYSIS"), ("vision", "OMNIBRAIN_VISION")):
        block = raw.setdefault(section, {})
        env_map = {
            "provider": f"{prefix}_PROVIDER",
            "model": f"{prefix}_MODEL",
            "base_url": f"{prefix}_BASE_URL",
            "api_key": f"{prefix}_API_KEY",
            "temperature": f"{prefix}_TEMPERATURE",
            "max_tokens": f"{prefix}_MAX_TOKENS",
        }
        for key, env in env_map.items():
            val = os.environ.get(env)
            if val is not None:
                block[key] = float(val) if key == "temperature" else (
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
