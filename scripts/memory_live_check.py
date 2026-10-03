"""Live check: memory context reaches a REAL provider prompt only when relevant, and research still works.

Uses a throw-away memory file (never the user's), the real Chrome profile/providers, QUICK mode, 1 round.
  python scripts/memory_live_check.py
Writes data/eval/memory_live_check.json (prompts are truncated; no personal data - the facts are synthetic).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))

from backend.browser.engine import BrowserEngine  # noqa: E402
from backend.memory import MemoryService, MemoryStore  # noqa: E402
from backend.memory.embed import get_embedder  # noqa: E402
from backend.models import Job, ResearchMode  # noqa: E402
from backend.orchestrator.runner import ResearchRunner  # noqa: E402
from backend.providers.registry import ProviderCatalog, endpoint_for  # noqa: E402
from backend.settings import load_settings  # noqa: E402

MARK = "Relevant context about the user:"
CASES = [
    ("general", "Who won the 2022 FIFA World Cup final and what was the score?", None, False),
    ("project", "Which is better for my Atlas project: PostgreSQL or MongoDB? Give the main trade-offs.", "Atlas", True),
]


async def main() -> int:
    settings = load_settings()
    db = ROOT / "data" / "tmp_live_memory.db"
    for ext in ("", "-wal", "-shm"):
        Path(str(db) + ext).unlink(missing_ok=True)
    mem = MemoryService(MemoryStore(db, get_embedder("auto", settings.memory.model)))
    mem.store.add("Project Atlas database is PostgreSQL 16 on a single VPS", memory_type="project", project="Atlas")
    mem.store.add("Lives in Kyiv", slot="residence")
    mem.store.add("Prefers concise answers without preamble", slot="pref:answer_style")
    engine = BrowserEngine(settings)
    await engine.start()
    report = {"embedder": mem.store.embedder.name, "cases": []}
    try:
        catalog = ProviderCatalog(settings, engine)
        adapters = catalog.all()
        prompts: list[dict] = []
        for name, ad in adapters.items():
            orig = ad.ask

            async def spy(job_id, prompt, round_no=1, emit=None, continue_thread=False, _o=orig, _n=name):
                prompts.append({"provider": _n, "round": round_no, "continue_thread": bool(continue_thread), "has_memory_block": MARK in prompt, "mentions_atlas_fact": "PostgreSQL 16" in prompt,
                                "mentions_kyiv": "Kyiv" in prompt, "head": prompt[:260]})
                return await _o(job_id, prompt, round_no, emit, continue_thread)

            ad.ask = spy  # type: ignore[method-assign]
        runner = ResearchRunner(settings, adapters, engine=engine, analysis_endpoint=endpoint_for(settings, "analysis"), memory=mem)
        for label, question, project, expect_block in CASES:
            prompts.clear()
            job = Job(question=question, mode=ResearchMode.QUICK, max_rounds=1, project=project)
            finished = await runner.run(job)
            ans = finished.final
            blob = json.dumps({"claims": [c.model_dump(mode="json") for c in finished.claims], "evidence": [e.model_dump(mode="json") for e in finished.evidence],
                               "final": ans.model_dump(mode="json") if ans else None}, default=str)
            case = {
                "case": label, "question": question, "project": project, "expect_memory_block": expect_block, "memory_kind": finished.memory_kind,
                "memory_used": [m["content"] for m in finished.memory_used], "prompts_sent": len(prompts),
                "prompts_with_block": sum(p["has_memory_block"] for p in prompts), "prompt_flags": [(p["provider"], p["round"], p["continue_thread"], p["has_memory_block"]) for p in prompts], "providers": sorted({p["provider"] for p in prompts}),
                "fresh_prompts": sum(not p["continue_thread"] for p in prompts), "fresh_prompts_with_block": sum(p["has_memory_block"] for p in prompts if not p["continue_thread"]),
                "thread_followups_with_block": sum(p["has_memory_block"] for p in prompts if p["continue_thread"]),
                "block_matches_expectation": all(p["has_memory_block"] == expect_block for p in prompts if not p["continue_thread"]) if prompts else None,
                "memory_text_in_claims_evidence_or_answer": ("PostgreSQL 16 on a single VPS" in blob) or ("Kyiv" in blob and "Kyiv" not in question),
                "status": finished.status.value, "answer_chars": len(ans.answer) if ans else 0, "answer_head": (ans.answer[:200] if ans else None),
                "confidence": ans.confidence_label if ans else None, "sources": len(ans.sources) if ans else 0, "error": finished.error,
                "prompt_samples": prompts[:2],
            }
            report["cases"].append(case)
            print(json.dumps({k: v for k, v in case.items() if k != "prompt_samples"}, indent=1), flush=True)
    finally:
        await engine.stop(keep_windows=False)
        mem.store.close()
        for ext in ("", "-wal", "-shm"):
            Path(str(db) + ext).unlink(missing_ok=True)
    out = ROOT / "data" / "eval" / "memory_live_check.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))