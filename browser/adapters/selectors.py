"""Per-provider DOM knowledge: ordered fallback arrays, nothing hardcoded
inline in logic.

PROVENANCE RULE -- important for trusting this file. Each field carries
``verified``:

* ``probe``   = observed live on this machine by ``scripts/probe.py``
  against the real site, with a timestamp in ``verified_at``.
* ``prior``   = taken from an open-source council project's selector map
  (MIT-licensed: jumas45/no-api-llm-council, AmT42/agent-council-browser).
  Plausible, unconfirmed against today's build of the site.
* ``guess``   = a generic pattern only. The adapter's content-based search is
  expected to carry the load; do not trust these ids.

Anything not marked ``probe`` must be treated as provisional. Run
``python run.py probe <provider> --login`` after signing in, then
``python run.py promote-selectors <provider>``, to promote entries to ``probe``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FieldSet:
    """One logical thing (the input box, the send button...) with an ordered
    list of ways to find it, best-evidence first."""

    css: list[str] = field(default_factory=list)
    aria: list[str] = field(default_factory=list)
    testids: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    text_regex: str | None = None
    verified: str = "guess"
    verified_at: str | None = None


@dataclass
class SelectorSet:
    provider: str
    input: FieldSet = field(default_factory=FieldSet)
    send: FieldSet = field(default_factory=FieldSet)
    stop: FieldSet = field(default_factory=FieldSet)
    quick_answer: FieldSet = field(default_factory=FieldSet)
    """Some providers surface an "Answer now" / "fast answer" control while a
    reasoning model is still thinking. Clicked only in QUICK mode (ADR-0004 in
    jumas45/no-api-llm-council), never for verification rounds."""
    response_root: list[str] = field(default_factory=list)
    assistant_message: list[str] = field(default_factory=list)
    streaming: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    login_wall: list[str] = field(default_factory=list)
    login_markers: list[str] = field(default_factory=list)
    rate_limit_markers: list[str] = field(default_factory=list)
    busy_markers: list[str] = field(default_factory=list)
    dismiss: FieldSet = field(default_factory=FieldSet)
    new_chat: list[str] = field(default_factory=list)
    # Some UIs put the answer behind a mode switch (thinking / fast / deep).
    preflight_clicks: list[str] = field(default_factory=list)
    notes: str = ""

    # Tuning: the hidden-window penalty doubles these, per prior art finding #3.
    stable_ms: int = 2500
    tiny_fragment_ms: int = 4500
    never_started_ms: int = 12000
    source_block_ms: int = 9000
    force_capture_ms: int = 50000
    reference_force_ms: int = 60000
    hard_timeout_ms: int = 180000
    scan_recent_blocks: int = 4
    # jumas45/no-api-llm-council (MIT): a stuck "generating" flag must not hold
    # capture hostage, and a recent change means "still working", not "done".
    stuck_ms: int = 9000
    active_ms: int = 4000
    overrun_factor: int = 2


SELECTORS: dict[str, SelectorSet] = {
    "chatgpt": SelectorSet(
        provider="chatgpt",
        input=FieldSet(
            css=[
                "#prompt-textarea",
                'textarea#mobile-composer-prompt',
                'textarea[aria-label="Chat with ChatGPT"]',
                'div[contenteditable="true"]#prompt-textarea',
                'div[contenteditable="true"][data-testid^="prompt"]',
                "textarea[data-id]",
            ],
            placeholders=["Ask ChatGPT", "Message ChatGPT", "Ask anything"],
            aria=["Chat with ChatGPT", "Send message"],
            testids=["prompt-textarea"],
            verified="probe",
            verified_at="2026-10-03 logged-out landing composer; ladder merged with jumas45 (MIT)",
        ),
        send=FieldSet(
            aria=["Send message", "Send prompt", "Send"],
            testids=["send-button"],
            css=[
                'button[data-testid="send-button"]',
                "#composer-submit-button",
                'button[aria-label="Send prompt"]',
                'button[aria-label*="Send"]',
            ],
            text_regex=r"^(send|prompt)$",
            verified="probe",
            verified_at="2026-10-03 button[aria-label=Send message]; ladder from jumas45/no-api-llm-council (MIT)",
        ),
        stop=FieldSet(
            aria=["Stop generating", "Stop response", "Stop streaming", "Stop regenerating", "Arrêter", "Annuler"],
            css=[
                'button[data-testid="stop-button"]',
                'button[aria-label="Stop streaming"]',
                "button.generate-button",
                'button[aria-label*="Stop"]',
            ],
            text_regex=r"(stop|arr|êter|annuler)",
            verified="prior",
        ),
        quick_answer=FieldSet(
            aria=["Quick answer", "Answer now", "Fast answer"],
            text_regex=r"(answer now|quick answer|fast answer|get a quick answer)",
            verified="prior",
        ),
        response_root=['[role="log"]', "main"],
        assistant_message=[
            '[data-message-author-role="assistant"]',
            "div[data-message-id] .markdown",
            "article",
        ],
        streaming=['[class*="result-streaming"]', "[data-testid=conversation-turn-content] .opacity-50"],
        sources=[
            'a[data-search-result-url]',
            "a[data-message-id] cite a",
            "button[aria-label^='Source']",
            ".sources-list-condensed",
        ],
        login_wall=[
            'a[href*="auth/sign_in"]',
            "div#log-out-overlay",
            'button[aria-label*="Log in"]',
        ],
        login_markers=['[data-testid="user-avatar"]', "#user-menu", 'button[aria-label*="profile"]'],
        dismiss=FieldSet(
            text_regex=r"(maybe later|not now|got it|no thanks|accept all|i agree|continue without an account)",
            aria=["Close", "Dismiss"],
            verified="prior",
        ),
        new_chat=['[data-testid="sidebar-new-chat-button"]', 'a[href="https://chatgpt.com/")'],
        notes=(
            "The 'Sources' block renders as its own assistant turn, so capture "
            "must scan the last N assistant roots from the baseline count rather "
            "than only the newest one, and give reference-like blocks a longer "
            "stable window. Exclude <aside> and [role=complementary]. "
            "Thresholds taken from AmT42/agent-council-browser's answer-detection "
            "doc (MIT): 3s stable normally, 4.5s for a tiny fragment, 12s if no "
            "generating UI was ever seen, 9s for reference-like blocks, force "
            "capture at 45s (60s reference-like), 180s hard timeout, doubled when "
            "the window is hidden."
        ),
        stable_ms=3000,
        tiny_fragment_ms=4500,
        never_started_ms=12000,
        source_block_ms=9000,
        force_capture_ms=45000,
        reference_force_ms=60000,
        hard_timeout_ms=180000,
        stuck_ms=9000,
        active_ms=4000,
    ),
    "gemini": SelectorSet(
        provider="gemini",
        input=FieldSet(
            css=[
                "rich-textarea .ql-editor",
                'div.ql-editor[contenteditable="true"]',
                'div[contenteditable="true"][role="textbox"]',
                'textarea[aria-label*="Promp"]',
            ],
            aria=["Enter a prompt for Gemini", "Ask Gemini", "Enter a prompt", "Prompt"],
            placeholders=["Ask Gemini"],
            verified="probe",
            verified_at="2026-10-03 div.ql-editor[role=textbox][aria-label=Enter a prompt for Gemini]",
        ),
        send=FieldSet(
            aria=["Send button", "Send message", "Send"],
            css=["button.send-button", 'button[aria-label*="Send"]', 'mat-icon[fonticon="send"]'],
            text_regex=r"^(send|submit)$",
            verified="prior",
        ),
        stop=FieldSet(
            aria=["Stop response"],
            css=["button.send-button.stop", "button.stop-icon", 'button[aria-label*="Stop"]', ".stop-icon"],
            verified="prior",
        ),
        response_root=["model-response", '[data-test-id*="message"]', "main"],
        assistant_message=[
            "message-content .markdown",
            "message-content",
            ".model-response-text",
            "model-response",
            '[data-test-id="response-text"]',
        ],
        streaming=[".response-streaming", ".cdk-text-selection-stylesheet", ".message-content.italic"],
        sources=[
            'a[aria-label*="source"]',
            ".source-anchor",
            'mat-icon[data-mat-icon-name="link"]',
            ".gsid-source-item",
        ],
        login_wall=['a[aria-label*="Google apps"]', 'form[action*="ServiceLogin"]'],
        login_markers=['a[href*="myaccount.google.com"]', '.avatar-menu-button', 'img[alt*="profile picture"]'],
        dismiss=FieldSet(
            text_regex=r"(got it|i agree|accept all|no thanks|later|not now|continue)",
            verified="prior",
        ),
        new_chat=['button[aria-label*="New chat"]', '[aria-label="New chat"]'],
        preflight_clicks=[
            'button[aria-label*="Thinking"]',
            'mat-button-toggle:has-text("Pro")',
        ],
        notes="Quill (.ql-editor) contenteditable; Enter submits, Shift+Enter is a newline. Thresholds per AmT42 (MIT): 1.2s stable, 15s force capture, 120s hard timeout.",
        stable_ms=1200,
        force_capture_ms=15000,
        hard_timeout_ms=120000,
    ),
    "copilot": SelectorSet(
        provider="copilot",
        input=FieldSet(
            css=["textarea#searchbox", 'div[contenteditable="true"][role="textbox"]', "#searchbox"],
            placeholders=["Ask me anything", "Message Bing"],
            aria=["Type a message to Copilot", "Search the web or chat"],
            verified="prior",
        ),
        send=FieldSet(
            css=['button[aria-label="Send"]', "button.send-button"],
            aria=["Send"],
            text_regex=r"^(send|submit)$",
            verified="guess",
        ),
        stop=FieldSet(aria=["Stop generating", "Stop"], text_regex=r"^stop$", verified="guess"),
        response_root=['#bnp-container', '[data-id^="c_"]', "main"],
        assistant_message=['#bnp-container .c03-only-mobile', '[data-message-id]', ".aac-send-turn"],
        streaming=['.c03-animContent', '[aria-busy="true"]'],
        sources=['a.cite-sup', 'a[sdk-append="href"]', '.citation'],
        login_wall=['a[id*="login-cta"]', 'button[id*="login"]'],
        login_markers=["#user-id-button", '.profile-avatar', 'a[id*="account"]'],
        dismiss=FieldSet(text_regex=r"(accept all|reject all|i agree|got it|no thanks|close)", verified="prior"),
        new_chat=['a[aria-label*="New topic"]', 'button[aria-label*="New topic"]'],
        notes="Copilot frequently shows a cookie/consent wall on a fresh profile.",
    ),
    "google_ai": SelectorSet(
        provider="google_ai",
        input=FieldSet(
            css=['textarea[name="q"]', 'textarea[aria-label*="Search"]', "#APjFqb"],
            placeholders=["Search Google", "Ask"],
            aria=["Search", "Ask Google"],
            verified="prior",
        ),
        send=FieldSet(
            css=['button[aria-label="Search"]', 'button[jsname="sbDdd"]', 'span.R48SH'],
            aria=["Search"],
            verified="prior",
        ),
        stop=FieldSet(verified="guess"),
        response_root=["#main", "[data-hveid]", ".A8SBwf"],
        assistant_message=["#search", ".s", "[data-rtid]"],
        streaming=['[data-eh-index]', ".u8JNpb"],
        sources=["#search a", 'a[LST]', 'div[data-visibility] a'],
        login_wall=["#consent-bump", 'form[action*="consent"]'],
        login_markers=[],
        dismiss=FieldSet(
            css=["button.L2mgTb", "form[role='dialog'] button"],
            text_regex=r"(accept all|i agree|reject all|got it)",
            verified="prior",
        ),
        notes=(
            "AI Mode (udm=50) is queried by submitting the search box, not by "
            "chatting; the AI answer streams into the results column and takes "
            "materially longer than a chat answer, so the hard timeout is the "
            "controlling limit here."
        ),
        hard_timeout_ms=200000,
        stable_ms=4000,
        force_capture_ms=70000,
    ),
    "meta_ai": SelectorSet(
        provider="meta_ai",
        input=FieldSet(
            css=['div[contenteditable="true"][role="textbox"]', "textarea[aria-label*='Ask Meta']", "#msg_input"],
            placeholders=["Ask Meta AI anything", "Message"],
            aria=["Ask Meta AI anything", "Message"],
            verified="guess",
        ),
        send=FieldSet(aria=["Send", "Send message"], text_regex=r"^(send|go)$", verified="guess"),
        stop=FieldSet(aria=["Stop"], verified="guess"),
        response_root=["[role='log']", "main"],
        assistant_message=['[data-message-author-role="assistant"]', ".x1y2bd81", "[role='listitem']"],
        streaming=["[aria-busy='true']"],
        sources=["a[href^='http']"],
        login_wall=['a[href*="login"]', 'button[name="login"]'],
        login_markers=['[aria-label*="profile"]'],
        dismiss=FieldSet(text_regex=r"(allow all cookies|decline all cookies|got it|not now|i agree)", verified="guess"),
        notes="Meta AI pushes a hard login wall on a cold profile; expect LOGGED_OUT often.",
        hard_timeout_ms=150000,
    ),
    "le_chat": SelectorSet(
        provider="le_chat",
        input=FieldSet(
            css=["textarea#chat-composer-input", 'div[contenteditable="true"][role="textbox"]', "textarea"],
            placeholders=["Message", "Ask me anything"],
            aria=["Message Le Chat", "Chat input"],
            verified="prior",
        ),
        send=FieldSet(
            css=['button[type="submit"]', '[data-testid="chat-composer-submit"]'],
            aria=["Send", "Submit"],
            verified="prior",
        ),
        stop=FieldSet(aria=["Stop", "Stop generating"], css=["[data-testid='chat-composer-stop']"], verified="prior"),
        response_root=["[role='log']", "main"],
        assistant_message=['[data-testid*="message"][data-role="assistant"]', ".Message_assistant", "[data-role='assistant']"],
        streaming=["[data-streaming='true']", ".MarkdownRenderer_streaming"],
        sources=["a[href^='http']", '.Message_citation', '[data-testid*="citation"]'],
        login_wall=['a[href*="auth"]', 'button:has-text("Login")'],
        login_markers=['[data-testid*="user-menu"]'],
        dismiss=FieldSet(text_regex=r"(accept|got it|maybe later)", verified="guess"),
        notes="Le Chat supports web browsing and Python tools; ask for sources explicitly.",
    ),
    "pi": SelectorSet(
        provider="pi",
        input=FieldSet(
            css=['textarea[name="text"]', 'textarea[aria-label*="Message"]', "div[contenteditable='true']"],
            placeholders=["Message Pi", "Write a message"],
            aria=["Message input", "Message Pi"],
            verified="guess",
        ),
        send=FieldSet(css=['button[type="submit"]'], aria=["Send"], verified="guess"),
        stop=FieldSet(aria=["Stop"], verified="guess"),
        response_root=["[role='log']", "main"],
        assistant_message=['[data-testid*="message"]', ".markdown"],
        streaming=["[aria-busy='true']"],
        sources=["a[href^='http']"],
        login_wall=['button:has-text("Log in")', 'a[href*="login"]'],
        login_markers=[],
        dismiss=FieldSet(verified="guess"),
        notes=(
            "Pi keeps no reliable scrollable history in an automated profile and "
            "caps context aggressively -- capture must happen live, and Pi's "
            "answers count as opinion unless it actually browsed."
        ),
        hard_timeout_ms=120000,
        force_capture_ms=40000,
    ),
    "qwen": SelectorSet(
        provider="qwen",
        input=FieldSet(
            css=[
                "textarea.message-input-textarea",
                'div[contenteditable="true"][role="textbox"]',
                'textarea[aria-label*="input"]',
                "#root textarea",
            ],
            placeholders=["Ask Qwen", "Ask anything", "Message", "Send a message"],
            aria=["Send message", "Chat input", "input"],
            verified="probe",
            verified_at="2026-10-03 textarea.message-input-textarea[placeholder=Ask Qwen]",
        ),
        send=FieldSet(
            css=['button[type="submit"]', '[class*="send-btn"]', '[data-testid*="send"]'],
            aria=["Send", "send message", "Submit"],
            text_regex=r"^(send|submit)$",
            verified="guess",
        ),
        stop=FieldSet(aria=["Stop", "stop generating"], css=['[class*="stop"]'], text_regex=r"^stop$", verified="guess"),
        response_root=['[role="log"]', "main"],
        assistant_message=['[data-message-author-role="assistant"]', '[class*="message-tts"]', '[class*="markdown"]'],
        streaming=['[aria-busy="true"]', '[class*="loading"]'],
        sources=["a[href^='http']"],
        login_wall=['button:has-text("Log in")', 'a[href*="oauth"]', 'a[href*="login"]'],
        login_markers=['[class*="avatar"]', '[class*="user-info"]'],
        dismiss=FieldSet(text_regex=r"(accept|got it|i agree|maybe later|not now|close|allow)", verified="guess"),
        notes=(
            "chat.qwen.ai. Selectors unprobed -- candidate scoring carries this "
            "until scripts/probe.py runs against a signed-in session. Qwen often "
            "offers a 'thinking' toggle that changes latency a lot."
        ),
        hard_timeout_ms=180000,
    ),
    "deepseek": SelectorSet(
        provider="deepseek",
        input=FieldSet(
            css=[
                "#chat-input",
                'textarea[placeholder*="Ask"]',
                'div[contenteditable="true"][role="textbox"]',
                "textarea",
            ],
            placeholders=["Ask anything", "Hello there", "How can I help you today"],
            aria=["Chat Input", "chat input", "input"],
            testids=["chat-input"],
            verified="guess",
        ),
        send=FieldSet(
            css=['[aria-label="Send"]', '.sendBtn', 'button[type="submit"]'],
            aria=["Send", "Send button"],
            text_regex=r"^(send|submit)$",
            verified="guess",
        ),
        stop=FieldSet(aria=["Stop", "Stop generating"], css=['[class*="stop"]'], verified="guess"),
        response_root=['[role="log"]', "main", "#root"],
        assistant_message=['[class*="markdown"]', '[data-message-id]', '[class*="answer"]'],
        streaming=['[class*="loading"]', '[aria-busy="true"]'],
        sources=["a[href^='http']"],
        login_wall=['button:has-text("Sign in")', 'a[href*="login"]', 'button:has-text("Log in")'],
        login_markers=['[class*="avatar"]'],
        dismiss=FieldSet(text_regex=r"(accept|got it|i agree|maybe later|not now|close)", verified="guess"),
        notes=(
            "chat.deepseek.com. Unprobed. DeepSeek's web chat runs its reasoning "
            "mode by default, which is slow: the hard timeout is generous and the "
            "answer block may only appear after a long thinking phase."
        ),
        hard_timeout_ms=210000,
        never_started_ms=30000,
        force_capture_ms=90000,
    ),
    "search": SelectorSet(
        provider="search",
        input=FieldSet(
            css=['textarea[name="q"]', "#APjFqb", 'input[name="q"]', "#searchbox", 'input[aria-label*="Search"]'],
            aria=["Search", "Search the web"],
            placeholders=["Search"],
            verified="prior",
        ),
        send=FieldSet(css=['button[aria-label="Search"]', "#searchbox-submit", 'button[type="submit"]'], verified="prior"),
        response_root=["#main", "#results", "[role='main']"],
        assistant_message=[".g", "div.s", '[data-hveid]'],
        sources=["#rso a", "#results a", ".b_algo a"],
        login_wall=["#consent-bump"],
        dismiss=FieldSet(
            css=["button.L2mgTb", 'input[aria-label*="Accept"]', 'form[role="dialog"] button'],
            text_regex=r"(accept all|i agree|reject all)",
            verified="prior",
        ),
        notes=(
            "Not a chat provider: this adapter harvests result links plus the "
            "visible snippet, which is what independent evidence is built from. "
            "DuckDuckGo's html endpoint is the no-consent-wall fallback engine."
        ),
        hard_timeout_ms=60000,
        stable_ms=1200,
        force_capture_ms=20000,
    ),
}


# A deliberately neutral default so an unknown site is configurable, not fatal.
GENERIC = SelectorSet(
    provider="generic_chat",
    input=FieldSet(
        css=[
            'div[contenteditable="true"][role="textbox"]',
            'textarea[role="textbox"]',
            "textarea:not([type=hidden])",
            '[contenteditable="true"]',
            '[role="textbox"]',
        ],
        placeholders=["Ask", "Message", "Type", "Prompt"],
        verified="guess",
    ),
    send=FieldSet(
        aria=["Send", "Submit", "Send message"],
        css=['button[type="submit"]', '[data-testid*="send"]', '[class*="send-button"]'],
        text_regex=r"^(send|submit|ask|go)$",
        verified="guess",
    ),
    stop=FieldSet(aria=["Stop", "Stop generating"], css=['[class*="stop"]'], text_regex=r"^stop$", verified="guess"),
    response_root=['[role="log"]', "main"],
    assistant_message=['[data-message-author-role="assistant"]', "[class*='assistant']", "article"],
    sources=["a[href^='http']"],
    login_markers=['[aria-label*="account"]', "[class*='avatar']"],
    dismiss=FieldSet(text_regex=r"(accept all|i agree|got it|maybe later|not now|no thanks|close)", verified="guess"),
    notes="Config-driven provider: relies on candidate scoring rather than ids.",
)


#: written by ``run.py promote-selectors``; overlays the ladders above (see promote.py)
PROMOTED_PATH = Path(__file__).with_name("promoted_selectors.json")


def load_promotions(path: Path | None = None) -> dict[str, dict[str, Any]]:
    target = Path(path) if path else PROMOTED_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def merge_fieldset(base: FieldSet, promoted: dict[str, Any]) -> FieldSet:
    """Observed locators first, the old ladder behind them as fallbacks."""

    def front(new: list[str] | None, old: list[str]) -> list[str]:
        return list(dict.fromkeys([*(new or []), *old]))

    return FieldSet(
        css=front(promoted.get("css"), base.css),
        aria=front(promoted.get("aria"), base.aria),
        testids=front(promoted.get("testids"), base.testids),
        placeholders=front(promoted.get("placeholders"), base.placeholders),
        text_regex=promoted.get("text_regex") or base.text_regex,
        verified=promoted.get("verified", "probe"),
        verified_at=promoted.get("verified_at"),
    )


def selectors_for(provider: str, promotions: dict[str, dict[str, Any]] | None = None) -> SelectorSet:
    base = SELECTORS.get(provider, GENERIC)
    promo = (promotions if promotions is not None else load_promotions()).get(provider)
    if not promo:
        return base
    merged = copy.deepcopy(base)
    for name in ("input", "send"):
        if isinstance(promo.get(name), dict):
            setattr(merged, name, merge_fieldset(getattr(base, name), promo[name]))
    return merged
