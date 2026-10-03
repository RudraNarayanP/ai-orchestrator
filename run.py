"""OmniBrain entry point.

    python run.py doctor                 what is reachable, what is not
    python run.py serve                  start the app (http://127.0.0.1:8730)
    python run.py ask "question"         one-shot research run in the terminal
    python run.py login gemini           open that provider's window to sign in once
    python run.py probe chatgpt --login  inspect the real DOM before trusting selectors
    python run.py promote-selectors NAME turn the newest probe into verified=probe selectors
    python run.py close                  shut any OmniBrain browser windows left open

Nothing here opens your everyday Chrome. Every window OmniBrain uses is a
dedicated one with its own profile under browser/profiles/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252 and die on the glyphs we print (status dots,
# em dashes, and any non-Latin provider text).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

BANNER = r"""
  OmniBrain
  multiple AI research agents, driven through their real websites
  one dedicated Chrome window per provider -- never your own browser
"""


def ensure_config() -> Path:
    config = ROOT / "config" / "settings.yaml"
    if not config.exists():
        shutil.copyfile(ROOT / "config" / "settings.example.yaml", config)
        print(f"created {config} from the example -- edit it to change providers or the verifier")
    return config


def load():
    ensure_config()
    from backend.settings import load_settings

    return load_settings()


def cmd_doctor(_: argparse.Namespace) -> int:
    settings = load()
    from backend.verification.llm import LLMClient, Endpoint
    from backend.settings import PROFILES_DIR

    print(BANNER)
    print(f"profiles dir : {PROFILES_DIR}")
    existing = sorted(p.name for p in PROFILES_DIR.glob("omnibrain_*")) if PROFILES_DIR.exists() else []
    print(f"profiles     : {', '.join(existing) or 'none yet -- sign in with: python run.py login <provider>'}")
    print()
    print("verifier")
    endpoint = Endpoint.from_config(settings.verifier)
    if not endpoint.enabled:
        print("  disabled: the deterministic evidence ledger still works, but no model will adjudicate conflicts")
    else:
        health = asyncio.run(LLMClient(endpoint).health())
        print(f"  {settings.verifier.provider} @ {endpoint.base_url} model={endpoint.model}")
        print(f"  -> {health['state']}: {health['detail']}")
        if health.get("models"):
            print("     available: " + ", ".join(health["models"][:12]))
    print()
    print("analysis endpoint")
    analysis = Endpoint.from_config(settings.analysis)
    if not analysis.enabled:
        print("  disabled: claims fall back to the deterministic extractor (works, less subtle)")
    else:
        health = asyncio.run(LLMClient(analysis).health())
        print(f"  {analysis.provider} @ {analysis.base_url} -> {health['state']}: {health['detail']}")
    print()
    print("providers")
    for name, cfg in settings.providers.items():
        if not cfg.enabled:
            continue
        from backend.browser.engine import profile_dir

        # a provider that needs no account (Google Search) is never "not signed in"
        signed = not cfg.requires_login or (profile_dir(f"omnibrain_{name}") / "Preferences").exists()
        print(f"  {'●' if signed else '○'} {cfg.label:<18} {cfg.url}" + ("" if signed else "   (not signed in yet)"))
    print()
    print("note: automation is not hidden from these sites -- navigator.webdriver stays true.")
    print("If a site objects, OmniBrain reports the provider as blocked instead of working around it.")
    return 0


def setup_file_logging(args: argparse.Namespace, settings) -> Path:
    """Start the rotating log. ``--log-file`` wins over ``storage.log_path``."""
    from backend.logs import setup_logging

    path = Path(getattr(args, "log_file", None) or settings.storage.log_path)
    setup_logging(path, verbose=bool(getattr(args, "verbose", False)))
    return path


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from backend.logs import uvicorn_log_config

    settings = load()
    log_path = setup_file_logging(args, settings)
    print(BANNER)
    print(f"  http://{args.host}:{args.port}")
    print(f"  log file: {log_path}")
    print("  sign in once per provider from the settings panel, then ask a question.")
    uvicorn.run(
        "backend.api.app:app",
        host=args.host,
        port=args.port,
        log_level="info" if not args.verbose else "debug",
        log_config=uvicorn_log_config(log_path, verbose=args.verbose),
        reload=False,
    )
    return 0


async def _one_shot(question: str, mode: str | None, rounds: int | None, args: argparse.Namespace | None = None) -> int:
    settings = load()
    if args is not None:
        setup_file_logging(args, settings)
    from backend.logs import get_logger

    log = get_logger("cli")
    from backend.browser.engine import BrowserEngine
    from backend.models import Job, ResearchMode
    from backend.orchestrator.runner import ResearchRunner
    from backend.providers.registry import ProviderCatalog, endpoint_for

    job = Job(
        question=question,
        mode=ResearchMode((mode or settings.research.mode).upper()),
        max_rounds=int(rounds or settings.research.max_rounds),
    )
    started = time.time()

    async def emit(kind, message, provider=None, round_no=None, **payload):
        colour = {
            "escalation": "\033[36m",
            "assessment": "\033[37m",
            "verifier": "\033[35m",
            "disagreement": "\033[33m",
            "error": "\033[31m",
        }.get(kind, "")
        log.info("[%s] %s%s: %s", job.id, f"{provider} " if provider else "", kind, message[:600])
        print(f"  {colour}[{time.time() - started:5.1f}s] {kind}: {message}\033[0m")

    class ConsoleBus:
        async def emit(self, kind, message, provider=None, round_no=None, **payload):
            await emit(kind, message, provider=provider, round_no=round_no, **payload)

    engine = BrowserEngine(settings)
    await engine.start()
    try:
        catalog = ProviderCatalog(settings, engine)
        runner = ResearchRunner(
            settings,
            catalog.all(),
            engine=engine,
            bus=ConsoleBus(),
            analysis_endpoint=endpoint_for(settings, "analysis"),
        )
        finished = await runner.run(job)
        print("\n" + "=" * 72)
        answer = finished.final
        if answer is None:
            print(f"no answer produced. status={finished.status.value} error={finished.error}")
            return 1
        print(answer.answer)
        print("-" * 72)
        print(f"confidence: {answer.confidence_label}   rounds: {answer.rounds_run}   level: {finished.level.name}")
        if finished.stop_reason:
            print(f"stopped: {finished.stop_reason}")
        if answer.why:
            print(f"\nwhy: {answer.why}")
        if answer.important_disagreement:
            print(f"\ndisagreement: {answer.important_disagreement}")
        if answer.sources:
            print("\nstrongest sources:")
            for s in answer.sources[:6]:
                print(f"  - {s.title or s.url}\n    {s.url}")
        if answer.caveats:
            print("\ncaveats: " + "; ".join(answer.caveats))
        print(
            f"\nresearchers used: {', '.join(answer.providers_used) or 'none'}"
            + (f" | failed: {', '.join(answer.providers_failed)}" if answer.providers_failed else "")
        )
        print(f"browser sessions: {finished.browser_sessions_used}  verifier calls: {finished.verifier_calls}")
        print(f"job id: {finished.id}  (stored in data/omnibrain.db)")
        return 0
    finally:
        await engine.stop(keep_windows=False)


def cmd_ask(args: argparse.Namespace) -> int:
    return asyncio.run(_one_shot(args.question, args.mode, args.rounds, args))


def cmd_login(args: argparse.Namespace) -> int:
    settings = load()

    async def go() -> int:
        from backend.browser.engine import BrowserEngine

        engine = BrowserEngine(settings)
        await engine.start()
        for provider in args.providers:
            cfg = settings.providers.get(provider)
            if cfg is None:
                print(f"unknown provider {provider!r}; known: {', '.join(settings.providers)}")
                continue
            page = await engine.open_research_page(provider, cfg.url)
            await page.bring_to_front()
            print(f"\n{cfg.label}: window open at {page.url}")
        print("\nSign in inside those windows. They are separate OmniBrain profiles -- your normal Chrome is untouched.")
        input("Press Enter here once you have finished signing in (windows will then close)... ")
        await engine.stop(keep_windows=False)
        print("saved. Profiles live under browser/profiles/ and are reused from now on.")
        return 0

    return asyncio.run(go())


def cmd_close(_: argparse.Namespace) -> int:
    """Shut any Chrome process started from an OmniBrain profile.

    Matches on the profile path in the command line, so the Chrome you use every
    day is never a candidate -- its command line does not contain our profile dir.
    """
    profiles = str((ROOT / "browser" / "profiles").resolve()).lower()
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
        "Where-Object { $_.CommandLine -like '*omnibrain*' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=45,
        ).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"could not enumerate processes: {exc}")
        return 1
    killed = 0
    for pid in out.split():
        if not pid.strip().isdigit():
            continue
        try:
            detail = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                capture_output=True, text=True, timeout=20,
            ).stdout.lower()
        except Exception:  # noqa: BLE001
            detail = ""
        if profiles not in detail and "omnibrain" not in detail:
            continue
        subprocess.run(["taskkill", "/PID", pid.strip(), "/T", "/F"], capture_output=True, timeout=20)
        killed += 1
    print(f"closed {killed} OmniBrain browser process(es). Your own Chrome was not touched.")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    script = ROOT / "scripts" / "probe.py"
    forwarded = ["--url", args.url] if args.url else []
    if args.login:
        forwarded.append("--login")
    if args.wait is not None:
        forwarded += ["--wait", str(args.wait)]
    return subprocess.call([sys.executable, str(script), *args.providers, *forwarded])


def cmd_promote_selectors(args: argparse.Namespace) -> int:
    """Rewrite a provider's composer/send selectors from its newest probe (verified="probe")."""
    settings = load()
    from backend.settings import DATA_DIR
    from browser.adapters.promote import PromotionError, promote

    cfg = settings.providers.get(args.provider)
    if cfg is None:
        print(f"unknown provider {args.provider!r}; known: {', '.join(settings.providers)}")
        return 2
    try:
        result = promote(
            args.provider,
            provider_url=cfg.url,
            probe_dir=Path(args.probe_dir) if args.probe_dir else DATA_DIR / "probe",
            probe_path=Path(args.probe) if args.probe else None,
            out_path=Path(args.out) if args.out else None,
            dry_run=args.dry_run,
        )
    except PromotionError as exc:
        print(f"not promoted: {exc}")
        return 1
    print(f"{args.provider}: probe {Path(result.source).name} (observed {result.observed_at})")
    for name, entry in result.promoted.items():
        shown = {k: v for k, v in entry.items() if k not in {"verified", "verified_at"} and v}
        print(f"  {name}: {json.dumps(shown, ensure_ascii=False)}")
    for note in result.skipped:
        print(f"  - {note}")
    print(f"{'wrote' if result.written else 'dry run, nothing written:'} {result.path}")
    if result.written:
        print("  these now sit in front of the existing selectors (verified=probe); commit the file to keep the checkpoint.")
    return 0


def cmd_jobs(_: argparse.Namespace) -> int:
    settings = load()
    from backend.storage.db import Store

    store = Store(settings)
    rows = store.list_jobs(limit=20)
    if not rows:
        print("no jobs yet")
        return 0
    for row in rows:
        answer = (row["answer_text"] or "").replace("\n", " ")[:70]
        print(
            f"{row['id']}  {row['status']:<10} L{row['level'] or 0}  "
            f"{row['rounds_run'] or 0}r  {(row['browser_sessions'] or 0)}win  "
            f"{(row['question'] or '')[:44]:<44}  {answer}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnibrain", description=BANNER)
    sub = parser.add_subparsers(dest="command", required=True)
    logging_args = argparse.ArgumentParser(add_help=False)
    logging_args.add_argument("--log-file", metavar="PATH", help="write the rotating log here (default: storage.log_path, data/omnibrain.log)")

    sub.add_parser("doctor", help="check models, profiles and providers").set_defaults(func=cmd_doctor)

    serve = sub.add_parser("serve", help="start the web app", parents=[logging_args])
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8730)
    serve.add_argument("--verbose", action="store_true")
    serve.set_defaults(func=cmd_serve)

    ask = sub.add_parser("ask", help="run one research job in the terminal", parents=[logging_args])
    ask.add_argument("question")
    ask.add_argument("--mode", choices=["QUICK", "STANDARD", "DEEP_RESEARCH"])
    ask.add_argument("--rounds", type=int)
    ask.set_defaults(func=cmd_ask)

    login = sub.add_parser("login", help="open provider windows to sign in once")
    login.add_argument("providers", nargs="+")
    login.set_defaults(func=cmd_login)

    probe = sub.add_parser("probe", help="dump the real DOM of a provider page")
    probe.add_argument("providers", nargs="*")
    probe.add_argument("--url")
    probe.add_argument("--login", action="store_true")
    probe.add_argument("--wait", type=int)
    probe.set_defaults(func=cmd_probe)

    promote = sub.add_parser("promote-selectors", help="promote a provider's newest probe to verified=probe selectors")
    promote.add_argument("provider")
    promote.add_argument("--probe", help="a specific probe JSON (default: newest data/probe/<provider>_*.json)")
    promote.add_argument("--probe-dir", help=argparse.SUPPRESS)
    promote.add_argument("--out", help=argparse.SUPPRESS)
    promote.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    promote.set_defaults(func=cmd_promote_selectors)

    sub.add_parser("close", help="close leftover OmniBrain browser windows").set_defaults(func=cmd_close)
    sub.add_parser("jobs", help="list stored research jobs").set_defaults(func=cmd_jobs)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
