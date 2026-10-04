"""The opt-in live-Chrome driver, tested against a fake ``chrome-use`` (no browser, no network).

What is pinned here: the provider-domain allowlist, the URL re-check before every action,
the closed set of chrome-use commands (and that a refused command never starts a process),
new-tabs-only, the mapping from the Playwright surface to CLI calls, error handling, and
that a login / captcha / age page is reported as the user's, never touched.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from backend.browser import live_chrome as lc
from backend.browser.factory import create_engine
from backend.browser.live_chrome import (
    ChromeUseRunner,
    LiveChromeEngine,
    LiveChromeError,
    LiveChromeNeedsUser,
    LiveChromeRefused,
    LiveChromeUnavailable,
    url_allowed,
)
from backend.models import ProviderStatus
from backend.settings import Settings
from tests.conftest import base_settings

FAKE = Path(__file__).with_name("fake_chrome_use.py")
USER_TAB = "https://user-own-tab.example/inbox"


class Fake:
    """Handle on one fake chrome-use: drives its state, reads back what it was asked."""

    def __init__(self, tmp: Path) -> None:
        self.dir = tmp
        self.state = {"tabs": {"t1": {"url": USER_TAB, "title": "My mail"}}, "active": "t1", "next": 1, "rules": [], "redirects": {}}
        self.write()
        (tmp / "calls.jsonl").write_text("", encoding="utf-8")
        self.runner = ChromeUseRunner([sys.executable, str(FAKE)], session="omnibrain", timeout_s=20)

    def write(self) -> None:
        (self.dir / "state.json").write_text(json.dumps(self.state), encoding="utf-8")

    def update(self, **kw) -> None:
        self.state = json.loads((self.dir / "state.json").read_text(encoding="utf-8"))
        self.state.update(kw)
        self.write()

    def read(self) -> dict:
        return json.loads((self.dir / "state.json").read_text(encoding="utf-8"))

    def rule(self, contains: str, **kw) -> None:
        rules = self.read()["rules"] + [{"contains": contains, **kw}]
        self.update(rules=rules)

    def set_tab_url(self, tab: str, url: str) -> None:
        tabs = self.read()["tabs"]
        tabs[tab]["url"] = url
        self.update(tabs=tabs)

    def calls(self) -> list[dict]:
        text = (self.dir / "calls.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def verbs(self) -> list[list[str]]:
        return [c["argv"] for c in self.calls()]

    def clear_calls(self) -> None:
        (self.dir / "calls.jsonl").write_text("", encoding="utf-8")


@pytest.fixture
def fake(tmp_path, monkeypatch) -> Fake:
    monkeypatch.setenv("FAKE_CU_DIR", str(tmp_path))
    return Fake(tmp_path)


def live_settings(artifacts: str | None = None, **browser) -> Settings:
    cfg = {"driver": "chrome_use", "settle_ms": 1, **browser}
    extra = {"storage": {"artifacts_dir": artifacts}} if artifacts else {}
    return base_settings(
        browser=cfg, providers={"chatgpt": {"enabled": True, "label": "ChatGPT", "url": "https://chatgpt.com/"}}, **extra
    )


@pytest.fixture
def engine(fake, tmp_path) -> LiveChromeEngine:
    return LiveChromeEngine(live_settings(str(tmp_path / "artifacts")), runner=fake.runner)


# ------------------------------------------------------------------ allowlist


@pytest.mark.parametrize(
    "url",
    [
        "https://chatgpt.com/",
        "https://www.chatgpt.com/c/abc",
        "https://gemini.google.com/app",
        "https://copilot.microsoft.com/",
        "https://copilot.com/",
        "https://www.copilot.com/?fromcode=abc&sessionId=1",
        "https://meta.ai/",
        "https://www.meta.ai/",
        "https://chat.mistral.ai/chat",
        "https://pi.ai/talk",
        "https://chat.deepseek.com/",
        "https://chat.qwen.ai/",
        "https://www.google.com/search?udm=50",
        "https://google.com/search?q=hello&udm=50&aep=1",
    ],
)
def test_provider_urls_are_allowed(url):
    assert url_allowed(url)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "about:blank",
        "chrome://settings",
        "file:///C:/Windows/win.ini",
        "http://chatgpt.com/",  # not https
        "https://chatgpt.com.evil.com/",
        "https://evil.com/chatgpt.com",
        "https://chatgpt.com@evil.com/",
        "https://evil.com@chatgpt.com/x",  # userinfo is never accepted
        "https://sub.chatgpt.com/",  # subdomains are not implied
        "https://auth.copilot.com/",
        "https://copilot.com.evil.com/",
        "https://evil.com/copilot.com",
        "https://login.microsoftonline.com/",
        "https://bing.com/chat",
        "http://copilot.com/",
        "https://chatgpt.com:8443/",
        "https://chat.openai.com/",
        "https://mail.google.com/",
        "https://accounts.google.com/",
        "https://google.com/search?q=cats",  # plain Search is not AI Mode
        "https://www.google.com/search?udm=14",
        "https://www.google.com/maps?udm=50",
        "https://www.google.com/",
        "https://reuters.com/world",
        USER_TAB,
    ],
)
def test_everything_else_is_refused(url):
    assert not url_allowed(url)
    with pytest.raises(LiveChromeRefused):
        lc.assert_allowed_url(url)


async def test_a_refused_url_never_starts_a_process(fake, engine):
    with pytest.raises(LiveChromeRefused):
        await fake.runner.run("open", "https://evil.example/")
    with pytest.raises(LiveChromeRefused):
        await fake.runner.run("tab", "new", "https://evil.example/")
    with pytest.raises(LiveChromeRefused):
        await engine.open_research_page("fetch", "https://www.reuters.com/a", key="fetch_1")  # evidence fetching stays off
    assert fake.calls() == [] and fake.runner.calls == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["cookies"],
        ["cookies", "set", "a", "b"],
        ["state", "save", "x.json"],
        ["state", "load", "x.json"],
        ["auth", "login", "--bwu"],
        ["auth", "save", "x"],
        ["storage", "local"],
        ["network", "route", "*"],
        ["addinitscript", "x"],
        ["humanize", "human"],
        ["tab", "adopt", "chatgpt"],
        ["tab", "duplicate"],
        ["tab", "new"],  # no blank tabs
        ["tab", "new", "about:blank"],
        ["tab", "select", "chatgpt.com"],  # tab ids only
        ["extension", "install"],
        ["connect", "9222"],
        ["snapshot", "-i"],
        ["download", "@e1", "x"],
        ["eval", "document.cookie"],  # scripts only via stdin
        ["open", "https://chatgpt.com/", "--humanize", "human"],
        ["open", "https://chatgpt.com/", "--launch"],
        ["open", "https://chatgpt.com/", "--profile", "auto"],
        ["open", "https://chatgpt.com/", "--init-script", "x.js"],
        ["press", "Enter; rm -rf /"],
        ["click", "@e1"],  # refs/selectors are not used: coordinates only
        ["click", "#send"],
        ["keyboard", "type", "--humanize"],
        ["mouse", "move", "1", "2"],
        [],
    ],
)
async def test_commands_outside_the_closed_set_are_refused_before_any_process(fake, argv):
    with pytest.raises(LiveChromeRefused):
        await fake.runner.run(*argv)
    assert fake.calls() == []


async def test_child_process_has_the_stealth_and_humanize_knobs_forced_off(fake, monkeypatch):
    for k, v in {
        "AGENT_BROWSER_STEALTH": "1",
        "AGENT_BROWSER_HUMANIZE": "human",
        "AGENT_BROWSER_HIDE_CANVAS": "1",
        "AGENT_BROWSER_BLOCK_WEBRTC": "1",
        "AGENT_BROWSER_TIMEZONE": "Asia/Tokyo",
        "AGENT_BROWSER_USER_AGENT": "spoof",
        "AGENT_BROWSER_PROFILE": "auto",
        "AGENT_BROWSER_ALLOWED_DOMAINS": "x.com",
        "AGENT_BROWSER_FORCE_LAUNCH": "1",  # would make chrome-use launch its own browser
        "AGENT_BROWSER_NO_AUTO_CONNECT": "1",
        "AGENT_BROWSER_CDP": "9222",
        "AGENT_BROWSER_ENGINE": "lightpanda",
        "CI": "true",  # chrome-use treats CI as force-launch
    }.items():
        monkeypatch.setenv(k, v)
    await fake.runner.run("status")
    env = fake.calls()[0]["env"]
    assert env == {"AGENT_BROWSER_STEALTH": "0", "AGENT_BROWSER_HUMANIZE": "off", "AGENT_BROWSER_AUTO_CONNECT": "1"}


async def test_argv_shape_is_session_then_json_then_command(fake):
    await fake.runner.run("get", "title")
    call = fake.calls()[0]
    assert call["flags"] == {"--session": "omnibrain", "--json": True}
    assert call["argv"] == ["get", "title"]
    pinned = ChromeUseRunner([sys.executable, str(FAKE)], session="s2", browser="me@example.com")
    assert pinned.build_argv(["status"])[-6:] == ["--session", "s2", "--browser", "me@example.com", "--json", "status"]


# ------------------------------------------------------------------ new tabs only


async def test_only_tabs_this_engine_created_are_ever_selected_or_closed(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    other = await engine.open_research_page("gemini", "https://gemini.google.com/app")
    await page.evaluate("() => 1")
    await other.evaluate("() => 2")
    await page.evaluate("() => 3")
    await engine.stop(keep_windows=False)
    argvs = fake.verbs()
    created = {"t2", "t3"}  # t1 is the user's own tab
    assert [a for a in argvs if a[:2] == ["tab", "new"]] == [["tab", "new", "https://chatgpt.com/"], ["tab", "new", "https://gemini.google.com/app"]]
    for a in argvs:
        if a[:2] in (["tab", "select"], ["tab", "close"]):
            assert a[2] in created, a
        assert a[0] not in {"cookies", "state", "auth", "storage", "network"}
        assert a[:2] != ["tab", "adopt"] and a[:2] != ["tab", "list"]
    assert "t1" in fake.read()["tabs"], "the user's own tab must survive"
    assert set(fake.read()["tabs"]) == {"t1"}


async def test_a_tab_is_reused_not_reopened(fake, engine):
    a = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    b = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    assert a is b
    assert sum(1 for v in fake.verbs() if v[:2] == ["tab", "new"]) == 1


async def test_stop_can_leave_the_tabs_open(fake, engine):
    await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    await engine.stop(keep_windows=True)
    assert "t2" in fake.read()["tabs"]


async def test_close_tab_closes_only_that_providers_tab(fake, engine):
    await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    await engine.open_research_page("gemini", "https://gemini.google.com/app")
    assert await engine.close_tab("chatgpt") is True
    assert await engine.close_tab("chatgpt") is False
    assert set(fake.read()["tabs"]) == {"t1", "t3"}


# ------------------------------------------------------------------ URL re-check


async def test_url_is_reread_before_every_action(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.clear_calls()
    await page.evaluate("() => document.title")
    await page.keyboard.press("Escape")
    # every action is preceded by a fresh `get url`; an input action also gets the challenge probe (an eval)
    assert fake.verbs() == [
        ["get", "url"], ["eval", "--stdin"],
        ["get", "url"], ["eval", "--stdin"], ["press", "Escape"],
    ]


async def test_a_tab_that_moved_off_the_provider_is_not_acted_on(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.set_tab_url("t2", "https://some-news-site.example/story")
    fake.clear_calls()
    with pytest.raises(LiveChromeRefused) as ei:
        await page.evaluate("() => document.body.innerText")
    assert not isinstance(ei.value, LiveChromeNeedsUser)
    for call in (page.keyboard.insert_text("hi"), page.keyboard.press("Enter"), page.mouse.wheel(0, 5), page.title()):
        with pytest.raises(LiveChromeRefused):
            await call
    assert not any(v[0] in ("eval", "keyboard", "press", "mouse", "click", "get") and v[:2] != ["get", "url"] for v in fake.verbs())
    assert page.url == "https://some-news-site.example/story"  # and the shim says where it is


async def test_in_page_navigation_is_noticed_between_actions(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    await page.evaluate("() => 1")
    fake.set_tab_url("t2", "https://chatgpt.com/c/123")
    await page.evaluate("() => 2")
    assert page.url == "https://chatgpt.com/c/123"


@pytest.mark.parametrize(
    "url,kind",
    [
        ("https://v2.auth.mistral.ai/login?x=1", "login"),
        ("https://accounts.google.com/v3/signin", "login"),
        ("https://auth.openai.com/log-in", "login"),
        ("https://consent.google.com/m?continue=x", "consent"),
        ("https://challenges.cloudflare.com/cdn-cgi/x", "captcha"),
    ],
)
async def test_sign_in_consent_and_challenge_pages_are_the_users(fake, engine, url, kind):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.set_tab_url("t2", url)
    fake.clear_calls()
    with pytest.raises(LiveChromeNeedsUser) as ei:
        await page.evaluate("() => 1")
    assert ei.value.kind == kind and "needs you" in str(ei.value)
    assert [v[0] for v in fake.verbs() if v[0] != "get"] == []


async def test_a_login_redirect_while_opening_is_reported_and_the_tab_is_left_for_the_user(fake, engine):
    fake.update(redirects={"https://chat.mistral.ai/chat": "https://v2.auth.mistral.ai/login"})
    with pytest.raises(LiveChromeNeedsUser) as ei:
        await engine.open_research_page("le_chat", "https://chat.mistral.ai/chat")
    assert ei.value.kind == "login"
    assert "t2" in fake.read()["tabs"], "OmniBrain must not close a page the user may be signing in on"


async def test_goto_is_checked_before_and_after(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.clear_calls()
    with pytest.raises(LiveChromeRefused):
        await page.goto("https://evil.example/")
    assert fake.calls() == []
    fake.update(redirects={"https://chatgpt.com/auth": "https://auth.openai.com/log-in"})
    with pytest.raises(LiveChromeNeedsUser):
        await page.goto("https://chatgpt.com/auth", wait_until="domcontentloaded")
    assert ["open", "https://chatgpt.com/auth"] in fake.verbs()


async def test_captcha_and_age_gates_stop_input_but_can_still_be_read(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.rule("captchaFrame", value={"captchaFrame": False, "cf": True, "human": False, "age": False, "title": "Just a moment..."})
    fake.rule("readyState", value="complete")
    assert await page.evaluate("() => document.readyState") == "complete"  # reading the page (to report it) still works
    fake.clear_calls()
    with pytest.raises(LiveChromeNeedsUser) as ei:
        await page.keyboard.insert_text("my question")
    assert ei.value.kind == "captcha"
    with pytest.raises(LiveChromeNeedsUser):
        await page.keyboard.press("Enter")
    with pytest.raises(LiveChromeNeedsUser):
        await page.mouse.click(10, 10)
    assert not any(v[0] in ("keyboard", "press", "click") for v in fake.verbs())
    fake.update(rules=[])
    fake.rule("captchaFrame", value={"captchaFrame": False, "cf": False, "human": False, "age": True, "title": "x"})
    with pytest.raises(LiveChromeNeedsUser) as ei2:
        await page.keyboard.type("hello")
    assert ei2.value.kind == "age"


# ------------------------------------------------------------------ command mapping


async def test_evaluate_sends_the_script_on_stdin_and_decodes_the_result(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.rule("document.title", value={"a": [1, 2]})
    out = await page.evaluate("(args) => document.title + args[0]", ["x"])
    assert out == {"a": [1, 2]}
    call = [c for c in fake.calls() if c["argv"] == ["eval", "--stdin"]][-1]
    assert "document.title + args[0]" in call["stdin"] and '["x"]' in call["stdin"]
    assert "eval(" not in call["stdin"] and "new Function" not in call["stdin"]  # CSP-proof: source is embedded, never eval'd


async def test_trailing_semicolon_iife_sources_are_accepted(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    await page.evaluate("\n(() => { return 'installed'; })();\n")
    script = [c for c in fake.calls() if c["argv"] == ["eval", "--stdin"]][-1]["stdin"]
    assert ";\n)" not in script and "'installed'" in script


async def test_page_script_errors_surface_as_errors(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.rule("boom", error="ReferenceError: nope is not defined")
    with pytest.raises(LiveChromeError, match="nope is not defined"):
        await page.evaluate("() => boom()")


async def test_handles_and_elements_are_re_resolved_by_their_script(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.rule("instanceof Element", value=True)
    handle = await page.evaluate_handle("(cfg) => window.__omnibrain.pickEl(cfg, 'input')", {"sel": "#q"})
    element = handle.as_element()
    assert element is not None
    fake.rule("getBoundingClientRect", value={"x": 60, "y": 35, "w": 100, "h": 30})  # the page script returns the box centre
    fake.clear_calls()
    await element.scroll_into_view_if_needed(timeout=4000)
    await element.click(timeout=4000)
    await element.focus()
    clicks = [v for v in fake.verbs() if v[0] == "click"]
    assert clicks == [["click", "60", "35"]]  # centre of the box, viewport coordinates
    scripts = [c["stdin"] for c in fake.calls() if c["argv"] == ["eval", "--stdin"]]
    assert any("pickEl(cfg, 'input')" in s and '"sel": "#q"' in s for s in scripts)
    # an element passed as an argument is resolved in-page, not serialised
    fake.clear_calls()
    await page.evaluate("(args) => window.__omnibrain.typeInto(args[0], args[1])", ["plain", "text"])
    await page.evaluate("(args) => window.__omnibrain.typeInto(args[0], args[1])", [element, "text"])
    plain, with_el = [c["stdin"] for c in fake.calls() if c["argv"] == ["eval", "--stdin"]][-2:]
    assert '{"__ob_el__": 0}' not in plain and '["plain", "text"]' in plain
    assert '[{"__ob_el__": 0}, "text"]' in with_el and "pickEl(cfg, 'input')" in with_el


async def test_a_non_element_handle_has_no_element(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    handle = await page.evaluate_handle("() => 5")
    assert handle.as_element() is None


async def test_locator_count_and_last_click(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.rule("querySelectorAll(\"button[aria-controls='sources']\").length", value=2)
    chips = page.locator("button[aria-controls='sources']")
    assert await chips.count() == 2
    fake.rule("getBoundingClientRect", value={"x": 5, "y": 5, "w": 10, "h": 10})
    fake.clear_calls()
    await chips.last.click(timeout=4000)
    scripts = [c["stdin"] for c in fake.calls() if c["argv"] == ["eval", "--stdin"]]
    assert any("list[list.length - 1]" in s for s in scripts)
    assert ["click", "5", "5"] in fake.verbs()


async def test_keyboard_mouse_navigation_and_pixels_map_to_cli_calls(fake, engine, tmp_path):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.clear_calls()
    prompt = "Line one\nLine \"two\" — ünïcode ✓ -- not a flag"
    await page.keyboard.insert_text(prompt)
    await page.keyboard.press("Enter")
    await page.keyboard.press("Control+a")
    await page.keyboard.type("short", delay=6)
    await page.mouse.wheel(0, 900)
    await page.mouse.wheel(0, -1)
    await page.mouse.click(12.4, 30.6)
    assert await page.title() == "New tab"
    shot = await page.screenshot(type="png", scale="css", full_page=False)
    assert shot.startswith(b"\x89PNG")
    saved = await engine.snapshot(page, "probe")
    assert saved.exists() and saved.read_bytes().startswith(b"\x89PNG")
    calls = [c for c in fake.calls() if c["argv"][0] not in ("get", "eval", "tab")]
    by_verb = [c["argv"] for c in calls]
    assert ["keyboard", "inserttext", "--stdin"] in by_verb
    assert [c for c in calls if c["argv"] == ["keyboard", "inserttext", "--stdin"]][0]["stdin"] == prompt
    assert ["press", "Enter"] in by_verb and ["press", "Control+a"] in by_verb
    assert ["keyboard", "type", "short"] in by_verb
    assert ["mouse", "wheel", "900", "0"] in by_verb and ["mouse", "wheel", "-1", "0"] in by_verb
    assert ["click", "12", "31"] in by_verb
    assert ["get", "title"] in [c["argv"] for c in fake.calls()]


async def test_typed_text_that_looks_like_a_flag_is_not_sent(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.clear_calls()
    with pytest.raises(LiveChromeRefused):
        await page.keyboard.type("--humanize human")
    assert not any(v[0] == "keyboard" for v in fake.verbs())


async def test_wait_for_load_state_polls_ready_state_and_times_out(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    await page.wait_for_load_state("domcontentloaded", timeout=2000)
    fake.update(rules=[{"contains": "document.readyState", "value": "loading"}])
    with pytest.raises(LiveChromeError, match="timeout"):
        await page.wait_for_load_state("load", timeout=400)


async def test_wait_for_timeout_keeps_page_url_fresh(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.set_tab_url("t2", "https://chatgpt.com/c/abc")
    await page.wait_for_timeout(5)
    assert page.url == "https://chatgpt.com/c/abc"


def _activations(fake) -> list[list[str]]:
    return [v for v in fake.verbs() if "--activate" in v]


def _bg(fake, tmp_path, **browser) -> LiveChromeEngine:
    return LiveChromeEngine(live_settings(str(tmp_path / "a"), chrome_use_background=True, **browser), runner=fake.runner)


async def test_default_is_foreground_the_new_omnibrain_tab_is_shown_and_only_that_tab(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    assert ["tab", "select", "t2", "--activate"] in fake.verbs()
    other = await engine.open_research_page("gemini", "https://gemini.google.com/app")
    fake.clear_calls()
    await engine.focus_tab(page)
    assert ["tab", "select", "t2", "--activate"] in fake.verbs()
    assert not [v for v in fake.verbs() if v[:2] == ["tab", "select"] and v[2] == "t1"], "the user's own tab is never selected"
    assert engine.escalate_focus("chatgpt") is False  # nothing to escalate: everything is already shown
    assert other.tab_id == "t3"


async def test_background_mode_never_raises_a_tab_by_default(fake, tmp_path):
    eng = _bg(fake, tmp_path)
    page = await eng.open_research_page("chatgpt", "https://chatgpt.com/")
    other = await eng.open_research_page("gemini", "https://gemini.google.com/app")
    await page.evaluate("() => 1")
    await other.evaluate("() => 2")
    await eng.focus_tab(page)
    await page.bring_to_front()
    assert _activations(fake) == [], "nothing may raise a tab unless the provider is opted in or escalated"


async def test_background_mode_opt_in_provider_is_raised_others_stay_back(fake, tmp_path):
    eng = _bg(fake, tmp_path, chrome_use_front_providers=["chatgpt"])
    page = await eng.open_research_page("chatgpt", "https://chatgpt.com/")
    gem = await eng.open_research_page("gemini", "https://gemini.google.com/app")
    fake.clear_calls()
    await eng.focus_tab(page)
    assert ["tab", "select", "t2", "--activate"] in fake.verbs()
    fake.clear_calls()
    await eng.focus_tab(gem)
    assert _activations(fake) == []


async def test_background_mode_escalates_only_the_failing_provider_once(fake, tmp_path):
    eng = _bg(fake, tmp_path)
    page = await eng.open_research_page("chatgpt", "https://chatgpt.com/")
    gem = await eng.open_research_page("gemini", "https://gemini.google.com/app")
    assert eng.escalate_focus("gemini") is True
    assert eng.escalate_focus("gemini") is False
    fake.clear_calls()
    await page.bring_to_front()
    assert _activations(fake) == []
    await gem.bring_to_front()
    assert ["tab", "select", "t3", "--activate"] in fake.verbs()


async def test_background_mode_escalation_can_be_off_and_force_is_for_explicit_user_actions(fake, tmp_path):
    eng = _bg(fake, tmp_path, chrome_use_front_on_failure=False)
    page = await eng.open_research_page("chatgpt", "https://chatgpt.com/")
    assert eng.escalate_focus("chatgpt") is False
    fake.clear_calls()
    await page.bring_to_front()
    assert _activations(fake) == []
    await page.bring_to_front(force=True)
    assert ["tab", "select", "t2", "--activate"] in fake.verbs()


async def test_the_adapter_escalates_only_after_the_background_attempt_failed(fake, tmp_path):
    from browser.adapters import build_adapter

    eng = _bg(fake, tmp_path)
    page = await eng.open_research_page("chatgpt", "https://chatgpt.com/")
    adapter = build_adapter("chatgpt", eng, eng.settings, eng.settings.providers["chatgpt"])
    fake.clear_calls()
    assert _activations(fake) == []
    assert await adapter._escalate_focus(page) is True
    assert ["tab", "select", "t2", "--activate"] in fake.verbs()
    assert await adapter._escalate_focus(page) is False


async def test_screenshot_to_a_path_and_content(fake, engine, tmp_path):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    out = tmp_path / "s.png"
    await page.screenshot(path=str(out))
    assert out.read_bytes().startswith(b"\x89PNG")
    fake.rule("outerHTML", value="<html></html>")
    assert await page.content() == "<html></html>"


# ------------------------------------------------------------------ error handling


async def test_missing_binary_is_a_clear_unavailable_error():
    runner = ChromeUseRunner(["definitely-not-a-real-binary-xyz"])
    with pytest.raises(LiveChromeUnavailable, match="install_chrome_use"):
        await runner.run("status")
    with pytest.raises(LiveChromeUnavailable):
        await LiveChromeEngine(live_settings(), runner=runner).start()


async def test_cli_failures_become_errors_with_a_hint(fake):
    fake.update(fail={"get": "relay not connected to the extension"})
    with pytest.raises(LiveChromeError, match="chrome-use status"):
        await fake.runner.run("get", "url")


async def test_unparseable_output_and_timeouts_are_errors(fake):
    fake.update(garbage=True)
    with pytest.raises(LiveChromeError, match="no JSON"):
        await fake.runner.run("status")
    fake.update(garbage=False, sleep=3)
    quick = ChromeUseRunner([sys.executable, str(FAKE)], timeout_s=0.5)
    with pytest.raises(LiveChromeError, match="timed out"):
        await quick.run("status")


async def test_a_failed_tab_select_is_reported_and_retried_next_time(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    other = await engine.open_research_page("gemini", "https://gemini.google.com/app")
    await page.evaluate("() => 1")
    tabs = fake.read()["tabs"]
    tabs.pop("t3")
    fake.update(tabs=tabs)
    with pytest.raises(LiveChromeError, match="Could not resolve"):
        await other.evaluate("() => 1")
    assert engine._selected is None
    await page.evaluate("() => 2")  # the other tab still works


async def test_tab_new_without_an_id_is_an_error(fake, engine, monkeypatch):
    async def bad(*a, **k):
        return {"url": "x"}

    monkeypatch.setattr(engine.runner, "run", bad)
    with pytest.raises(LiveChromeError, match="no usable tab id"):
        await engine.open_research_page("chatgpt", "https://chatgpt.com/")


async def test_start_tolerates_an_unhealthy_status(fake):
    fake.update(fail={"status": "extension not connected"})
    eng = LiveChromeEngine(live_settings(), runner=fake.runner)
    await eng.start()  # no raise: the first real command will say what is wrong
    assert any("not healthy" in line for line in eng.log)


async def test_start_says_when_the_extension_route_is_not_ready(fake):
    fake.update(extension={"relayUp": False})
    eng = LiveChromeEngine(live_settings(), runner=fake.runner)
    await eng.start()
    assert any("not connected through the chrome-use extension" in line for line in eng.log)
    fake.update(extension={"relayUp": True, "hostInstalled": False})
    await LiveChromeEngine(live_settings(), runner=fake.runner).start()


async def test_the_first_tab_is_refused_and_closed_if_chrome_is_not_attached_through_the_extension(fake, engine):
    """If the relay is down chrome-use might be driving a browser it launched itself: not the user's Chrome."""
    fake.update(extension={"relayUp": False})
    with pytest.raises(LiveChromeUnavailable, match="refusing to continue"):
        await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    assert set(fake.read()["tabs"]) == {"t1"}, "the tab it made is closed again"
    assert not any(v[0] in ("eval", "keyboard", "press", "click") for v in fake.verbs())
    # once the relay is up it works, and the status check is only repeated until it has passed once
    fake.update(extension={"relayUp": True})
    await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    fake.clear_calls()
    await engine.open_research_page("gemini", "https://gemini.google.com/app")
    assert not any(v[0] == "status" for v in fake.verbs())


async def test_a_closed_page_refuses_further_use(fake, engine):
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    await page.close()
    assert page.is_closed()
    with pytest.raises(LiveChromeError, match="closed"):
        await page.evaluate("() => 1")


# ------------------------------------------------------------------ wiring


def test_default_driver_is_still_playwright():
    from backend.browser.engine import BrowserEngine

    s = base_settings()
    assert s.browser.driver == "playwright"
    assert isinstance(create_engine(s), BrowserEngine)
    assert isinstance(create_engine(live_settings()), LiveChromeEngine)


def test_chrome_use_is_found_on_path_or_in_the_installer_folder(tmp_path, monkeypatch):
    exe = tmp_path / "Programs" / "chrome-use" / "chrome-use.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert lc.resolve_chrome_use() == str(exe)
    assert lc.resolve_chrome_use(r"D:\tools\chrome-use.exe") == r"D:\tools\chrome-use.exe"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "empty"))
    assert lc.resolve_chrome_use() == "chrome-use"


def test_example_config_documents_the_driver_and_keeps_the_default():
    from backend.settings import CONFIG_DIR
    import yaml

    raw = yaml.safe_load((CONFIG_DIR / "settings.example.yaml").read_text(encoding="utf-8"))
    assert raw["browser"].get("driver", "playwright") == "playwright"


# ------------------------------------------------------------------ adapters keep working, and stop at gates


async def test_adapter_reports_a_login_redirect_as_logged_out_without_touching_the_page(fake, engine):
    from browser.adapters import build_adapter

    s = live_settings()
    fake.update(redirects={"https://chatgpt.com/": "https://auth.openai.com/log-in"})
    adapter = build_adapter("chatgpt", engine, s, s.providers["chatgpt"])
    response = await adapter.ask("job1", "What is 2+2?")
    assert response.status == ProviderStatus.LOGGED_OUT
    assert response.error.startswith("readiness=login_wall") and "needs you" in response.error
    assert not any(v[0] in ("keyboard", "press", "click", "mouse") for v in fake.verbs())


async def test_adapter_reports_a_page_that_moves_to_a_captcha_as_blocked(fake, engine):
    from browser.adapters import build_adapter

    s = live_settings()
    adapter = build_adapter("chatgpt", engine, s, s.providers["chatgpt"])
    fake.update(redirects={"https://chatgpt.com/": "https://challenges.cloudflare.com/cdn-cgi/challenge"})
    response = await adapter.ask("job2", "hello")
    assert response.status == ProviderStatus.FAILED
    assert response.error.startswith("readiness=blocked")
    assert not any(v[0] in ("keyboard", "press", "click") for v in fake.verbs())


async def test_adapter_dom_unavailable_wrapping_still_finds_the_gate(fake, engine):
    """Mid-run redirect: the adapter's own `_call` wraps errors in DOMUnavailable; the gate must survive that."""
    from browser.adapters import build_adapter

    s = live_settings()
    adapter = build_adapter("chatgpt", engine, s, s.providers["chatgpt"])
    fake.rule("window.__omnibrain", set_url="https://accounts.google.com/signin", value=None)
    response = await adapter.ask("job3", "hello")
    assert response.status == ProviderStatus.LOGGED_OUT
    assert "needs you" in (response.error or "")
    assert not any(v[0] in ("keyboard", "press", "click") for v in fake.verbs())


async def test_a_background_daemon_holding_our_output_handles_does_not_hang_the_first_call(fake):
    """Real chrome-use leaves a daemon behind on the first command; it inherits stdout/stderr (live regression)."""
    import time

    fake.update(daemon=12)
    t0 = time.monotonic()
    data = await fake.runner.run("status")
    assert data["cliVersion"]
    assert time.monotonic() - t0 < 8


async def test_a_new_tab_that_still_reads_about_blank_is_waited_for_not_refused(fake, engine):
    """Live regression: right after `tab new <url>` real Chrome reports about:blank for a moment."""
    fake.update(blank_gets=3)
    page = await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    assert page.url == "https://chatgpt.com/"


async def test_a_new_tab_stuck_on_about_blank_is_an_error_and_is_closed(fake, engine, monkeypatch):
    monkeypatch.setattr(lc, "NEW_TAB_SETTLE_S", 0.6)
    fake.update(blank_gets=1000)
    with pytest.raises(LiveChromeError, match="about:blank"):
        await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    assert ["tab", "close", "t2"] in fake.verbs()
    assert "t2" not in fake.read()["tabs"]
    assert "t1" in fake.read()["tabs"]  # the user's own tab was never touched


async def test_a_new_tab_that_lands_off_the_allowlist_is_refused_and_our_tab_closed(fake, engine):
    fake.update(redirects={"https://chatgpt.com/": "https://evil.example/"})
    with pytest.raises(LiveChromeRefused):
        await engine.open_research_page("chatgpt", "https://chatgpt.com/")
    assert "t2" not in fake.read()["tabs"]
    assert "t1" in fake.read()["tabs"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the wrapper JS")
def test_wrapped_script_survives_dom_nodes_and_cycles():
    """Live regression (Gemini): a script returning an element / cyclic object must not throw."""
    import subprocess

    probe = (
        "class Node {}; globalThis.Node = Node; class Window {}; globalThis.Window = Window;"
        "const cyc = {a: 1}; cyc.self = cyc;"
        "const run = async (js) => JSON.parse(await eval(js));"
        "(async () => { const out = [];"
        "out.push(await run(" + json.dumps(lc.wrap_for_cli("(async () => new Node())()")) + "));"
        "out.push(await run(" + json.dumps(lc.wrap_for_cli("(async () => cyc)()")) + "));"
        "const shared = {k: 1}; globalThis.shared = shared;"
        "out.push(await run(" + json.dumps(lc.wrap_for_cli("(async () => [shared, shared, {again: shared}])()")) + "));"
        "out.push(await run(" + json.dumps(lc.wrap_for_cli("(async () => ({n: 1, f() {}, s: 'x'}))()")) + "));"
        "out.push(await run(" + json.dumps(lc.wrap_for_cli("(async () => { throw new Error('boom'); })()")) + "));"
        "console.log(JSON.stringify(out)); })();"
    )
    done = subprocess.run(["node", "-e", probe], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        {"ok": True, "v": {}},
        {"ok": True, "v": {"a": 1, "self": None}},
        {"ok": True, "v": [{"k": 1}, {"k": 1}, {"again": {"k": 1}}]},  # shared (not cyclic) references stay intact
        {"ok": True, "v": {"n": 1, "s": "x"}},
        {"ok": False, "e": "boom"},
    ]


async def test_a_dialog_over_the_page_is_the_users_to_answer_but_escape_still_works(fake, engine):
    """Live (pi.ai): 'Memory just got better ... Continue to Pi' sat over the composer; clicks landed on it."""
    page = await engine.open_research_page("pi", "https://pi.ai/")
    fake.rule("captchaFrame", value={"captchaFrame": False, "cf": False, "human": False, "age": False, "title": "Pi",
                                     "modal": "Memory just got better  Now Pi can remember relevant details across your chats"})
    with pytest.raises(LiveChromeNeedsUser) as ei:
        await page.check_gate()
    assert ei.value.kind == "consent" and "Memory just got better" in str(ei.value)
    fake.clear_calls()
    with pytest.raises(LiveChromeNeedsUser):
        await page.keyboard.insert_text("my question")
    with pytest.raises(LiveChromeNeedsUser):
        await page.mouse.click(10, 10)
    assert not [v for v in fake.verbs() if v[0] in ("click", "keyboard")], "nothing may be typed/clicked under a dialog"
    await page.keyboard.press("Escape")  # closing a dialog is always allowed
    assert ["press", "Escape"] in fake.verbs()
    # and a clean page passes
    fake.update(rules=[])
    fake.rule("captchaFrame", value={"captchaFrame": False, "cf": False, "human": False, "age": False, "title": "Pi", "modal": ""})
    await page.check_gate()


async def test_copilot_redirect_to_copilot_com_is_followed_but_a_microsoft_login_is_not(fake, engine):
    """Live: copilot.microsoft.com redirects to copilot.com (allowed); a sign-in at login.microsoftonline.com is the user's."""
    fake.update(redirects={"https://copilot.microsoft.com/": "https://copilot.com/?fromcode=x&sessionId=1"})
    page = await engine.open_research_page("copilot", "https://copilot.microsoft.com/")
    assert page.url.startswith("https://copilot.com/")
    fake.set_tab_url("t2", "https://login.microsoftonline.com/common/oauth2/authorize")
    with pytest.raises(LiveChromeNeedsUser):
        await page.evaluate("() => 1")


def test_chrome_use_calls_never_open_a_window_and_the_code_never_opens_explorer():
    """The user saw windows popping up repeatedly while the live driver ran. Each chrome-use call must be windowless and
    nothing in OmniBrain may open files/folders through the shell (os.startfile, explorer, start, Invoke-Item)."""
    import re
    import subprocess as sp

    seen = {}
    real = sp.Popen

    class Spy(real):
        def __init__(self, *a, **kw):
            seen.update(kw)
            super().__init__(*a, **kw)

    runner = ChromeUseRunner([sys.executable, "-c", "print('{\"success\": true, \"data\": {}}')"])
    sp.Popen = Spy
    try:
        runner._run_sync(runner.build_argv(["status"])[:2] + ["-c", "print(1)"], None, 20)
    finally:
        sp.Popen = real
    if sys.platform == "win32":
        assert seen.get("creationflags", 0) & sp.CREATE_NO_WINDOW
    assert seen.get("shell") in (None, False)

    root = Path(__file__).resolve().parent.parent
    bad = re.compile(r"os\.startfile|explorer(\.exe)?\b['\"\s]|Invoke-Item|shell\s*=\s*True|os\.system\(|webbrowser\.open\(\s*(?!url|f?['\"]http)")
    offenders = []
    for sub in ("backend", "browser", "scripts"):
        for f in (root / sub).rglob("*.py"):
            for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if bad.search(line) and "re.compile" not in line:
                    offenders.append(f"{f.relative_to(root)}:{n}: {line.strip()[:90]}")
    assert not offenders, offenders
