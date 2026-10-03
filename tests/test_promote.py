"""Selector promotion (backlog item 9): probe JSON -> verified="probe" selectors."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import run as run_cli
from browser.adapters import selectors
from browser.adapters.promote import PromotionError, check_probe, newest_probe, promote
from browser.adapters.selectors import SELECTORS, merge_fieldset, selectors_for
from tests.conftest import base_settings

URL = "https://chatgpt.com/"


def probe(**over):
    observed = {
        "url_seen": "https://chatgpt.com/",
        "title": "ChatGPT",
        "inputs": [
            {"tag": "input", "id": "search", "visible": True, "rect": {"w": 100, "h": 20}},
            {"tag": "div", "id": "prompt-textarea", "role": "textbox", "contenteditable": "true", "aria": "Chat with ChatGPT",
             "placeholder": "Ask anything", "visible": True, "rect": {"w": 700, "h": 60}},
            {"tag": "textarea", "id": "hidden", "visible": False, "rect": {"w": 900, "h": 90}},
        ],
        "buttons": [
            {"tag": "button", "text": "Log in", "visible": True},
            {"tag": "button", "aria": "Send message", "testid": "send-button", "visible": True},
        ],
        "banners": [],
    }
    observed.update(over.pop("observed", {}))
    data = {"provider": "chatgpt", "ts": "2026-10-03 02:08:09", "observed": observed}
    data.update(over)
    return data


def write_probe(dir_: Path, data, name="chatgpt_1790973494.json") -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def run(tmp_path, data=None, **kw):
    write_probe(tmp_path / "probe", data or probe())
    return promote("chatgpt", provider_url=URL, probe_dir=tmp_path / "probe", out_path=tmp_path / "promoted.json", **kw)


def test_composer_and_send_are_promoted_with_provenance(tmp_path):
    result = run(tmp_path)
    inp, send = result.promoted["input"], result.promoted["send"]
    assert inp["css"][0] == "#prompt-textarea" and "Chat with ChatGPT" in inp["aria"] and "Ask anything" in inp["placeholders"]
    assert inp["verified"] == "probe" and "2026-10-03 02:08:09" in inp["verified_at"] and "chatgpt_1790973494.json" in inp["verified_at"]
    assert send["aria"] == ["Send message"] and send["testids"] == ["send-button"]
    assert any("Log in" in s for s in result.skipped), "a login button must never become 'send'"
    saved = json.loads((tmp_path / "promoted.json").read_text(encoding="utf-8"))
    assert saved["chatgpt"]["input"]["verified"] == "probe" and result.written


def test_dry_run_writes_nothing(tmp_path):
    result = run(tmp_path, dry_run=True)
    assert result.promoted and not result.written and not (tmp_path / "promoted.json").exists()


def test_generated_ids_are_not_promoted_as_selectors(tmp_path):
    data = probe(observed={"inputs": [{"tag": "div", "id": "radix-12345678", "role": "textbox", "contenteditable": "true",
                                       "aria": "Message", "visible": True, "rect": {"w": 500, "h": 40}}]})
    assert run(tmp_path, data).promoted["input"]["css"] == ['div[contenteditable="true"][role="textbox"]']


def test_refusals_say_what_to_do(tmp_path):
    with pytest.raises(PromotionError, match="not signed in"):
        run(tmp_path, probe(observed={"inputs": []}))
    with pytest.raises(PromotionError, match="landed on"):
        run(tmp_path, probe(observed={"url_seen": "https://login.example.org/"}))
    with pytest.raises(PromotionError, match="robot"):
        run(tmp_path, probe(observed={"banners": [{"text": "Are you a robot?"}]}))
    with pytest.raises(PromotionError, match="recorded an error"):
        run(tmp_path, {"provider": "chatgpt", "error": "TimeoutError: x"})
    with pytest.raises(PromotionError, match="no probe"):
        promote("gemini", provider_url=URL, probe_dir=tmp_path / "probe", out_path=tmp_path / "x.json")
    assert not (tmp_path / "promoted.json").exists()


def test_no_send_button_leaves_send_unchanged_and_says_so(tmp_path):
    result = run(tmp_path, probe(observed={"buttons": [{"tag": "button", "text": "Sign in", "visible": True}]}))
    assert "send" not in result.promoted and any("no send button" in s for s in result.skipped)


def test_newest_probe_wins(tmp_path):
    d = tmp_path / "probe"
    write_probe(d, probe(), "chatgpt_100.json")
    newest = write_probe(d, probe(), "chatgpt_200.json")
    write_probe(d, probe(), "chatgpt_9.json")
    write_probe(d, probe(), "chatgpt_notes.json")
    assert newest_probe("chatgpt", d) == newest


def test_repromoting_replaces_the_provider_entry_and_keeps_others(tmp_path):
    out = tmp_path / "promoted.json"
    out.write_text(json.dumps({"gemini": {"input": {"css": ["#g"], "verified": "probe"}}, "chatgpt": {"send": {"aria": ["Old"]}}}), encoding="utf-8")
    run(tmp_path, probe(observed={"buttons": []}))
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["gemini"]["input"]["css"] == ["#g"]
    assert saved["chatgpt"]["send"]["aria"] == ["Old"], "an unobserved field keeps its earlier promotion"
    assert saved["chatgpt"]["input"]["css"][0] == "#prompt-textarea"


def test_promotions_go_in_front_of_the_ladder_and_keep_the_fallbacks(tmp_path):
    promotions = {"chatgpt": run(tmp_path).promoted}
    sel = selectors_for("chatgpt", promotions)
    base = SELECTORS["chatgpt"]
    assert sel.input.css[0] == "#prompt-textarea" and sel.input.verified == "probe" and "2026-10-03" in sel.input.verified_at
    assert set(base.input.css) <= set(sel.input.css), "old selectors stay as fallbacks"
    assert len(sel.input.css) == len(set(sel.input.css))
    assert sel.stop == base.stop and sel.hard_timeout_ms == base.hard_timeout_ms
    assert SELECTORS["chatgpt"].input.verified_at != sel.input.verified_at, "the shared default is not mutated"


def test_without_a_promotion_the_default_object_is_returned_unchanged(tmp_path):
    assert selectors_for("chatgpt", {}) is SELECTORS["chatgpt"]
    assert selectors_for("chatgpt", {"gemini": {}}) is SELECTORS["chatgpt"]
    assert selectors.load_promotions(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert selectors.load_promotions(bad) == {}


def test_cli_promote_selectors_end_to_end(tmp_path, capsys, monkeypatch):
    write_probe(tmp_path / "probe", probe())
    monkeypatch.setattr(run_cli, "load", lambda: base_settings(providers={"chatgpt": {"enabled": True, "label": "ChatGPT", "url": URL}}))
    parser = run_cli.build_parser()
    out = tmp_path / "out.json"
    args = parser.parse_args(["promote-selectors", "chatgpt", "--probe-dir", str(tmp_path / "probe"), "--out", str(out)])
    assert run_cli.cmd_promote_selectors(args) == 0
    text = capsys.readouterr().out
    assert "input:" in text and "Log in" in text and "wrote" in text and out.exists()
    # refusal exits non-zero without writing
    out.unlink()
    write_probe(tmp_path / "probe", probe(observed={"inputs": []}), "chatgpt_1999999999.json")
    assert run_cli.cmd_promote_selectors(args) == 1 and not out.exists()
    assert "not promoted" in capsys.readouterr().out
    unknown = parser.parse_args(["promote-selectors", "nope"])
    assert run_cli.cmd_promote_selectors(unknown) == 2


def test_the_committed_overlay_is_valid_json_and_only_promotes_known_fields():
    data = selectors.load_promotions()
    for provider, fields in data.items():
        assert set(fields) <= {"input", "send"}, provider
        for entry in fields.values():
            assert entry["verified"] == "probe" and entry["verified_at"]