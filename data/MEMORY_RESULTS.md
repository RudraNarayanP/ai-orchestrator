# Memory system - progress and results

Status: IN PROGRESS (updated as work lands; see the git log for exact commits).

## What exists (committed)
- `backend/memory/` - schema, store (SQLite + FTS5 + in-RAM numpy exact vectors), rule-based extractor, retriever (vector + keyword + entity candidates, lifecycle filters, multi-signal score, dynamic budget), service.
- Integration: runner injects a context block into provider prompts (primary, escalation, targeted follow-ups); learning from the USER's question only; memory is never added to claims, evidence, verifier payloads, sources or the answer.
- HTTP API `/api/memory*` (list, search-with-why, add, edit, delete, forget-all with confirm, settings, export, consolidate).
- Inspection UI: Memory panel (list with type/source/confidence/status/dates, "why would it be used" search, add/edit/delete, forget everything with confirm, inject/capture switches, export), per-answer "Memory used as context" expander, optional Project field.
- Tests: store, extract, retrieval, isolation ("memory is not evidence"), API, browser UI.
- Engine micro-benchmark: `data/eval/memory_bench_engines_*.json` (numpy exact vs sqlite-vec, 1k-1M).

## Not done yet
- Retrieval quality eval (hash vs bge-small), scale benchmark of the full pipeline, calibrated thresholds for the neural embedder.
- README / AGENTS.md memory sections.