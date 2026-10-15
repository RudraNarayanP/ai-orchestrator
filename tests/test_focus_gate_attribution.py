"""The focus gate attributes a foreground change before it fails on it (scripts/live_focus_check.py).

The gate used to require that the foreground handle never changed for the whole run. On a desktop someone is
actually using that is unsatisfiable -- the person's own Task Manager, the shell's Task Switching and the agent's
Qoder window all change it (measured 2026-10-15: every one of the four recorded changes was one of those, and no
live verification could start). What must never happen is a window *OmniBrain could have raised* coming to the
front: a browser, a console, or File Explorer. Those still fail the gate, and a Chrome-class window whose owning
process cannot be read is treated as the violation it most likely is.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_focus_check", ROOT / "scripts" / "live_focus_check.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the focus gate is a Windows API poller")


def test_windows_only_omnibrain_could_have_raised_are_violations():
    for name, cls in [("chrome.exe", "Chrome_WidgetWin_1"), ("msedge.exe", "Chrome_WidgetWin_1"),
                      ("conhost.exe", "ConsoleWindowClass"), ("cmd.exe", "ConsoleWindowClass"),
                      ("powershell.exe", "CASCADIA_HOSTING_WINDOW_CLASS"),
                      ("windowsterminal.exe", "CASCADIA_HOSTING_WINDOW_CLASS"),
                      ("python.exe", "PythonGUIWindowClass"), ("explorer.exe", "CabinetWClass")]:
        kind, why = gate.classify_owner(name, cls, "anything")
        assert kind == "violation", (name, cls, why)


def test_the_persons_own_windows_are_recorded_but_do_not_stop_a_run():
    # the four changes the gate actually recorded on 2026-10-15, none of them OmniBrain's
    for name, cls, title in [
        ("qoder.exe", "Chrome_WidgetWin_1", "Qoder"),
        ("taskmgr.exe", "TaskManagerWindow", "Task Manager"),
        ("explorer.exe", "WorkerW", "Task Switching"),
        ("nvcontainer.exe", "CEF-OSC-WIDGET", "NVIDIA GeForce Overlay DT"),
    ]:
        kind, why = gate.classify_owner(name, cls, title)
        assert kind == "noise", (name, cls, why)


def test_an_unattributable_chrome_window_is_a_violation_not_a_pass():
    """Failing safe: if the owning process could not be read, a Chrome-class window is judged by its class."""
    kind, _ = gate.classify_owner("", "Chrome_WidgetWin_1", "Some provider page")
    assert kind == "violation"
    kind, _ = gate.classify_owner("", "WorkerW", "")
    assert kind == "noise"


def test_a_report_keeps_the_old_key_and_adds_the_verdicts():
    mon = gate.Monitor.__new__(gate.Monitor)
    mon.events = [
        {"t": 1.0, "hwnd": 11, "pid": 2, "process": "taskmgr.exe", "class": "TaskManagerWindow",
         "title": "Task Manager", "kind_verdict": "noise", "why": "taskmgr.exe / TaskManagerWindow"},
        {"t": "m", "mark": "before tab new chatgpt", "chrome_titles": [], "zorder": []},
    ]
    mon.start_fg = (10, 1, "Resident Evil 9 - Google Chrome")
    mon.end_fg = (11, 2, "Task Manager")
    mon.start_titles = mon.end_titles = []
    rep = gate.Monitor.report(mon)
    assert rep["violations"] == [] and rep["clean"] is True
    assert rep["never_changed"] is False, "the strict observation is still reported for the audit"
    assert len(rep["unrelated_changes"]) == 1
