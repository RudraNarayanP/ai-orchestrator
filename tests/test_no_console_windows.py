"""No console window may ever appear on the user's screen (live 2026-10-13: a cmd window popped up in front of the user).

Every process OmniBrain, its scripts or its tests start -- servers, git, node/chrome-use, powershell, pytest -- is created with
CREATE_NO_WINDOW on Windows. A process started from a parent without a console (pythonw, a service, an agent) otherwise gets a
NEW, visible console. Checked statically over every tracked Python file, so a new call site cannot slip in without the flag.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPAWNERS = {"Popen", "run", "call", "check_output", "check_call"}


def _tracked_py() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.split()
    return [ROOT / p for p in out]


def test_every_subprocess_call_is_created_without_a_console_window():
    missing = []
    for path in _tracked_py():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if node.func.attr in SPAWNERS and isinstance(owner, ast.Name) and owner.id in {"subprocess", "sp"}:
                if not any(k.arg == "creationflags" for k in node.keywords):
                    missing.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if node.func.attr in {"system", "startfile"} and isinstance(owner, ast.Name) and owner.id == "os":
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno} os.{node.func.attr}")
    assert not missing, "spawned without CREATE_NO_WINDOW: " + ", ".join(missing)
