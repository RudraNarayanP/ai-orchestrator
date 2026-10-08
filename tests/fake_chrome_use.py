"""A fake ``chrome-use`` for offline tests: same argv/JSON contract, no browser.

State lives in ``$FAKE_CU_DIR/state.json``; every invocation is appended to
``calls.jsonl`` (argv, stdin, and the AGENT_BROWSER_* env it was started with) so tests can
assert exactly what OmniBrain asked for. Anything outside the commands OmniBrain is allowed
to use is answered with an error -- a refused command must never even get this far.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

DIR = Path(os.environ["FAKE_CU_DIR"])
STATE = DIR / "state.json"
CALLS = DIR / "calls.jsonl"


def load() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def save(state: dict) -> None:
    STATE.write_text(json.dumps(state), encoding="utf-8")


def reply(data=None, *, ok=True, error=None, code=None) -> None:
    payload = {"id": "x", "success": ok}
    if ok:
        payload["data"] = data if data is not None else {}
    else:
        payload["error"] = error
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.exit(0 if ok else (code if code is not None else 1))


def verb_is_first_call() -> bool:
    return sum(1 for _ in CALLS.open(encoding="utf-8")) == 1


def spawn_daemon(seconds: float) -> None:
    """Like the real chrome-use: the first command leaves a background daemon that inherits our stdout/stderr."""
    import subprocess

    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        close_fds=False,
    )


def main() -> None:
    argv = sys.argv[1:]
    args: list[str] = []
    i = 0
    flags: dict[str, str | bool] = {}
    while i < len(argv):
        a = argv[i]
        if a in ("--session", "--browser"):
            flags[a] = argv[i + 1]
            i += 2
            continue
        if a == "--json":
            flags[a] = True
            i += 1
            continue
        args.append(a)
        i += 1
    stdin = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.buffer.read().decode("utf-8")
    env = {k: v for k, v in os.environ.items() if k.startswith("AGENT_BROWSER_") or k == "CI"}
    pre = load()
    if pre.get("sleep"):  # a slow command: sleep outside the state lock so other sessions are not held up
        time.sleep(float(pre["sleep"]))
    # parallel tabs run several fake processes at once: serialise state access like the real daemon would
    import atexit
    lock = DIR / "state.lock"
    for _ in range(4000):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 5:  # left behind by a killed process
                    lock.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.005)
    atexit.register(lambda: lock.unlink(missing_ok=True))
    with CALLS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"argv": args, "flags": flags, "stdin": stdin, "env": env}) + "\n")
    state = load()
    if state.get("daemon") and verb_is_first_call():
        spawn_daemon(float(state["daemon"]))
    if state.get("garbage"):
        sys.stdout.write("this is not json\n")
        sys.exit(0)
    verb = args[0] if args else ""
    flaky = state.get("fail_times") or {}
    if flaky.get(verb, {}).get("n", 0) > 0:
        flaky[verb]["n"] -= 1
        save(state)
        reply(ok=False, error=flaky[verb]["error"])
    if verb in state.get("fail", {}):
        reply(ok=False, error=state["fail"][verb])

    tabs = state["tabs"]
    sess = str(flags.get("--session") or "default")
    actives = state.setdefault("actives", {})
    active = actives.get(sess, state.get("active"))

    def set_active(tid):
        actives[sess] = tid
        state["active"] = tid

    def redirect(url: str) -> str:
        return state.get("redirects", {}).get(url, url)

    if verb == "browsers":
        reply({"browsers": state.get("browsers") or [
            {"default": True, "email": "me@example.com", "id": "chrome-profile", "wsUrl": "ws://127.0.0.1:50001/chrome"}]})
    if verb == "session" and args[1:2] == ["stop"]:
        name = args[2] if len(args) > 2 else sess
        for k in [k for k, v in tabs.items() if v.get("session") == name]:  # the daemon closes the tabs IT created
            tabs.pop(k, None)
        save(state)
        reply({"stopped": name})
    if verb == "status":
        ext = {"hostInstalled": True, "hostHealthy": True, "relayUp": True, "liveVersion": "0.5.29"}
        ext.update(state.get("extension", {}))
        reply({"cliVersion": "1.5.157", "extension": ext, "currentSession": {"name": "omnibrain", "running": True}, "sessions": []})
    if verb == "tab":
        sub = args[1] if len(args) > 1 else ""
        if sub == "new":
            state["next"] = state.get("next", 1) + 1
            tid = f"t{state['next']}"
            tabs[tid] = {"url": redirect(args[2]), "title": "New tab", "blank": state.get("blank_gets", 0), "session": sess}
            set_active(tid)
            save(state)
            reply({"tabId": tid, "targetId": f"T{tid}", "label": None, "url": tabs[tid]["url"], "total": len(tabs)})
        if sub == "select":
            if args[2] not in tabs:
                reply(ok=False, error=f"Could not resolve target tab `{args[2]}`")
            set_active(args[2])
            save(state)
            reply({"tabId": args[2]})
        if sub == "close":
            # real chrome-use: "Cannot close the last tab" of a session (state["last_tab_guard"] turns this on)
            if state.get("last_tab_guard") and sum(1 for v in tabs.values() if v.get("session") == sess) <= 1:
                reply(ok=False, error="Cannot close the last tab")
            tabs.pop(args[2], None)
            if active == args[2]:
                set_active(None)
            save(state)
            reply({"closed": args[2]})
        if sub == "list":
            reply({"tabs": [{"tabId": k, **v} for k, v in tabs.items()]})
    if verb == "get":
        if active not in tabs:
            reply(ok=False, error="no active tab")
        if args[1] == "url" and tabs[active].get("blank", 0) > 0:
            tabs[active]["blank"] -= 1  # a fresh real-Chrome tab reads about:blank until its navigation commits
            save(state)
            reply({"url": "about:blank"})
        reply({"url": tabs[active]["url"]} if args[1] == "url" else {"title": tabs[active].get("title", "")})
    if verb == "open":
        tabs[active]["url"] = redirect(args[1])
        save(state)
        reply({"url": tabs[active]["url"]})
    if verb == "eval":
        script = stdin
        for rule in state.get("rules", []):
            if rule["contains"] in script and rule.get("times", 1) != 0:
                if "times" in rule:
                    rule["times"] -= 1
                if "set_url" in rule:
                    tabs[active]["url"] = rule["set_url"]
                save(state)
                if "error" in rule:
                    reply({"result": json.dumps({"ok": False, "e": rule["error"]})})
                reply({"result": json.dumps({"ok": True, "v": rule.get("value")})})
        if "captchaFrame" in script:
            value = {"captchaFrame": False, "cf": False, "human": False, "age": False, "title": "ok"}
        elif "document.readyState" in script:
            value = "complete"
        else:
            value = None
        reply({"result": json.dumps({"ok": True, "v": value})})
    if verb == "screenshot":
        Path(args[1]).write_bytes(b"\x89PNG\r\n\x1a\nfake")
        reply({"path": args[1]})
    if verb in ("keyboard", "press", "mouse", "click"):
        reply({"done": True})
    reply(ok=False, error=f"fake chrome-use: unexpected command {args!r}")


main()
