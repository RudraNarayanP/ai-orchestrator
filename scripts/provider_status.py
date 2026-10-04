"""Probe every enabled chat AI once and write a per-provider status table.

    python scripts/provider_status.py                 # every enabled AI, one tiny web question each
    python scripts/provider_status.py gemini google_ai

Each run re-tries every site, including ones that were walled last time: access changes. A login wall,
age gate, captcha or Cloudflare check is reported as blocked/broken -- nothing is bypassed or filled in.
Writes data/eval/provider_status_<ts>.{json,md} and records the health so routing can rank sites.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from backend.browser.engine import BrowserEngine  # noqa: E402
from backend.providers.registry import ProviderCatalog  # noqa: E402
from backend.settings import ROOT, load_settings  # noqa: E402
from backend.storage.db import Store  # noqa: E402

PROMPT = (
    "Use your own web search. Which year was the UK Data Protection Act 2018 passed, and what is the "
    "official legislation.gov.uk page for it? Reply in two lines and give the URL."
)
NOT_AI = {"search"}


async def probe(adapter, provider: str, timeout: int) -> dict:
    t0 = time.time()
    out = {"provider": provider, "status": "failed", "detail": "", "answer": "", "citations": 0, "conversation_url": None, "seconds": 0}
    try:
        r = await asyncio.wait_for(adapter.ask(f"probe_{provider}_{int(t0)}", PROMPT, 1), timeout=timeout)
        out.update(
            status=r.status.value,
            detail=(r.error or r.detail or "")[:200],
            answer=(r.answer_text or "")[:160].replace("\n", " "),
            citations=len(r.citations or []),
            conversation_url=r.conversation_url,
        )
    except asyncio.TimeoutError:
        out.update(status="timeout", detail=f"no answer within {timeout}s")
    except Exception as exc:  # noqa: BLE001
        out.update(status="failed", detail=f"{type(exc).__name__}: {exc}"[:200])
    out["seconds"] = int(time.time() - t0)
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("providers", nargs="*")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    settings = load_settings()
    names = args.providers or [n for n, c in settings.providers.items() if c.enabled and n not in NOT_AI]
    engine = BrowserEngine(settings)
    await engine.start()
    store = Store(settings)
    for n in names:  # an explicitly named provider is probed even when it is switched off in the config (read-only: one question, no sign-in)
        if n in settings.providers and not settings.providers[n].enabled:
            settings.providers[n].enabled = True
    catalog = ProviderCatalog(settings, engine)
    rows = []
    try:
        for name in names:
            print(f"probing {name} ...", flush=True)
            row = await probe(catalog.adapter(name), name, args.timeout)
            rows.append(row)
            store.record_health(name, "completed" if row["status"] == "completed" else row["status"])
            print(f"  {name}: {row['status']} in {row['seconds']}s {row['detail'][:100]}", flush=True)
    finally:
        await engine.stop()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"provider_status_{stamp}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Provider status {stamp}", "", "| provider | status | seconds | citations | note |", "|---|---|---|---|---|"]
    for r in rows:
        note = (r["detail"] or r["answer"]).replace("|", "/")[:120]
        md.append(f"| {r['provider']} | {r['status']} | {r['seconds']} | {r['citations']} | {note} |")
    (out_dir / f"provider_status_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
