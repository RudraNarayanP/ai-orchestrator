"""A bad settings file is a one-line fixable message, never a traceback; a fresh clone's example config is clean."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import yaml

import run as run_module
from backend import config_check
from backend.settings import ConfigError, Settings, load_settings, read_raw_settings

ROOT = Path(__file__).resolve().parent.parent


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_yaml_syntax_error_names_the_line_and_is_a_config_error(tmp_path):
    path = write(tmp_path, "browser:\n  driver: playwright\n verifier: [unclosed\n")
    with pytest.raises(ConfigError) as err:
        load_settings(path)
    assert "line" in str(err.value) and "YAML syntax error" in str(err.value) and "Traceback" not in str(err.value)


def test_a_wrong_type_names_the_key_and_the_value(tmp_path):
    path = write(tmp_path, "research:\n  max_rounds: lots\nbrowser:\n  driver: firefox\n")
    with pytest.raises(ConfigError) as err:
        load_settings(path)
    text = str(err.value)
    assert "research.max_rounds" in text and "'lots'" in text
    assert "browser.driver" in text and "'firefox'" in text


def test_a_top_level_list_is_refused_with_a_reason(tmp_path):
    with pytest.raises(ConfigError) as err:
        load_settings(write(tmp_path, "- a\n- b\n"))
    assert "mapping" in str(err.value)


def test_a_bad_environment_override_is_a_config_error(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIBRAIN_VERIFIER_MAX_TOKENS", "many")
    with pytest.raises(ConfigError) as err:
        load_settings(write(tmp_path, "{}\n"))
    assert "OMNIBRAIN_" in str(err.value)


def test_a_mistyped_key_is_reported_with_the_likely_spelling(tmp_path):
    raw = {"browser": {"chrome_use_brower": "chrome"}, "reserch": {}, "providers": {"a": {"url": "https://a", "label": "A", "enabld": True}}}
    found = {f.key: f for f in config_check.unknown_keys(raw)}
    assert "chrome_use_browser" in found["browser.chrome_use_brower"].message
    assert "reserch" in found and "research" in found["reserch"].message
    assert "enabled" in found["providers.a.enabld"].message


def test_the_same_dead_key_in_every_provider_is_one_line():
    raw = {"providers": {n: {"url": "https://x", "label": n, "weight": 1} for n in ("a", "b", "c")}}
    rows = [f for f in config_check.unknown_keys(raw) if "weight" in f.key]
    assert len(rows) == 1 and rows[0].key == "providers.*.weight" and "a, b, c" in rows[0].message


def settings_in(tmp_path, **overrides) -> Settings:
    raw = yaml.safe_load((ROOT / "config" / "settings.example.yaml").read_text(encoding="utf-8"))
    raw["storage"] = {"db_path": str(tmp_path / "d" / "o.db"), "artifacts_dir": str(tmp_path / "a"), "log_path": str(tmp_path / "l" / "o.log")}
    for key, value in overrides.items():
        raw.setdefault(key.split("__")[0], {})
        if "__" in key:
            raw[key.split("__")[0]][key.split("__")[1]] = value
        else:
            raw[key] = value
    return Settings.model_validate(raw), raw


def test_the_example_config_has_no_errors_on_a_fresh_clone(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    s, raw = settings_in(tmp_path)
    found = config_check.check_settings(s, raw)
    assert [f for f in found if f.level == "error"] == [], found
    assert [f for f in found if f.key.startswith(("browser.", "research.", "providers."))] == [], found


def test_no_enabled_provider_is_an_error(tmp_path):
    s, raw = settings_in(tmp_path)
    for p in s.providers.values():
        p.enabled = False
    found = config_check.check_settings(s)
    assert any(f.level == "error" and f.key == "providers" for f in found)


def test_openrouter_without_a_key_is_a_warning_not_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    s, _ = settings_in(tmp_path)
    found = config_check.check_settings(s)
    keys = {f.key: f.level for f in found}
    assert keys.get("verifier.api_key") == "warning" and "error" not in [f.level for f in found if f.key.endswith("api_key")]
    assert not any("sk-" in f.message for f in found), "a key value is never printed"


def test_out_of_range_and_unreachable_values_are_errors(tmp_path):
    s, _ = settings_in(tmp_path)
    s.research.max_rounds = 0
    s.threads.rotate_at = 3
    s.browser.driver = "chrome_use"
    s.browser.chrome_use_path = str(tmp_path / "nope.exe")
    s.providers["chatgpt"].url = "chatgpt.com"
    keys = {f.key for f in config_check.check_settings(s) if f.level == "error"}
    assert {"research.max_rounds", "threads.rotate_at", "browser.chrome_use_path", "providers.chatgpt.url"} <= keys


def test_modes_for_an_unknown_or_disabled_provider_are_warned(tmp_path):
    s, _ = settings_in(tmp_path)
    s.research.provider_modes = {"nobody": ["thinking"]}
    s.providers["pi"].enabled = False
    s.research.provider_modes["pi"] = ["search"]
    found = {f.key for f in config_check.check_settings(s)}
    assert {"research.provider_modes.nobody", "research.provider_modes.pi"} <= found


def test_an_unwritable_data_folder_is_an_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    s, _ = settings_in(tmp_path)
    s.storage.artifacts_dir = str(blocker / "sub")  # a folder cannot live inside a file
    assert any(f.level == "error" and f.key == "storage.artifacts_dir" for f in config_check.check_settings(s))


def test_port_in_use_is_detected():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert config_check.port_in_use("127.0.0.1", port)
    assert not config_check.port_in_use("127.0.0.1", port)


def test_serve_refuses_a_busy_port_and_an_invalid_port_with_a_reason(monkeypatch, capsys, tmp_path):
    s, _ = settings_in(tmp_path)
    monkeypatch.setattr(run_module, "load", lambda: s)
    monkeypatch.setattr(run_module, "config_findings", lambda settings, include_environment=False: [])
    args = run_module.build_parser().parse_args(["serve", "--port", "70000"])
    assert run_module.cmd_serve(args) == 2 and "not a valid port" in capsys.readouterr().out
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy = sock.getsockname()[1]
        args = run_module.build_parser().parse_args(["serve", "--port", str(busy)])
        assert run_module.cmd_serve(args) == 2
    out = capsys.readouterr().out
    assert "already in use" in out and "run.py stop" in out


def test_serve_does_not_start_on_a_config_error(monkeypatch, capsys, tmp_path):
    s, _ = settings_in(tmp_path)
    monkeypatch.setattr(run_module, "load", lambda: s)
    monkeypatch.setattr(run_module, "config_findings", lambda settings, include_environment=False: [config_check.Finding("error", "providers", "no provider is enabled")])
    args = run_module.build_parser().parse_args(["serve", "--port", "8799"])
    assert run_module.cmd_serve(args) == 2
    assert "not starting" in capsys.readouterr().out


def test_main_prints_a_config_error_instead_of_a_traceback(monkeypatch, capsys, tmp_path):
    bad = write(tmp_path, "research:\n  max_rounds: lots\n")
    monkeypatch.setenv("OMNIBRAIN_CONFIG", str(bad))
    monkeypatch.setattr(sys, "argv", ["run.py", "jobs"])
    assert run_module.main() == 2
    out = capsys.readouterr().out
    assert "config error" in out and "research.max_rounds" in out and "Traceback" not in out


def test_doctor_exits_nonzero_on_a_setup_error(monkeypatch, capsys, tmp_path):
    s, _ = settings_in(tmp_path)
    monkeypatch.setattr(run_module, "load", lambda: s)
    monkeypatch.setattr(run_module, "config_findings", lambda settings, include_environment=False: [config_check.Finding("error", "python", "too old")])
    assert run_module.cmd_doctor(None) == 1
    assert "setup error" in capsys.readouterr().out


def test_gitignore_keeps_the_real_settings_file_out_of_git():
    assert config_check.tracked_secret_findings(ROOT) == []
    assert config_check.tracked_secret_findings(Path("/definitely/not/here")) == []
    assert "config/settings.yaml" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_read_raw_settings_is_the_file_as_written(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIBRAIN_CONFIG", str(write(tmp_path, "research:\n  max_rounds: 2\n")))
    assert read_raw_settings() == {"research": {"max_rounds": 2}}

def test_doctor_does_not_tell_people_they_must_sign_up():
    cfg = Settings.model_validate({"providers": {"pi": {"url": "https://pi.ai/", "label": "Pi", "requires_login": True}}}).providers["pi"]
    for note in (run_module.provider_login_note(cfg), run_module.provider_login_note(cfg, live_browser="Google Chrome")):
        assert "guest" in note and "only after you sign in" not in note
