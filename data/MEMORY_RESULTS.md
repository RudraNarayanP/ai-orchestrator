# Memory system - results

Design and the reasons for each choice: `data/MEMORY_DESIGN.md`. Raw numbers: `data/eval/memory_*.json`.

## What was built
- `backend/memory/`: SQLite source of truth (memories, FTS5 keywords, entities, topics, links), exact float32 cosine over an in-RAM numpy matrix, a rule-based extractor, a multi-signal retriever, a service that formats the context block.
- Embedder: `bge-small-en-v1.5` (fastembed, optional, `requirements-memory.txt`), with a dependency-free hash embedder as the fallback.
- Integration: the runner injects a memory block into provider prompts (fresh prompts only, not thread follow-ups). Learning is from the USER's own message only. Memory is never added to claims, evidence, verifier payloads, sources or the ledger.
- API `/api/memory*` (list with `?sensitivity=`, search with "why", add, edit, delete, forget-all with confirm, settings, export, consolidate). UI: Memory panel, per-answer "Memory used as context" expander, optional Project field.
- Fact vs the user's interpretation: a statement of opinion or feeling is stored as `interpretation`, injected as "(the user's own view, not an established fact)". Topic tags (finance, education, family, decision, health, relationship, legal, identity, ...) drive cluster expansion; sensitive topics are flagged, gated on relevance, and switchable off.

## Privacy
Local only; opt-out switches for injection and capture; secrets filtered before storing; physical delete (no tombstone text); "forget everything" with confirmation; sensitive memories need relevance or topic overlap and the user's switch; audit events carry no memory content.

## Engine choice (384-d, k=20, warm p50)
| N | numpy f32 | sqlite-vec |
|---|---|---|
| 1k | 0.03 ms | 0.67 ms |
| 10k | 0.17 ms | 12 ms |
| 100k | 7.4 ms | 237 ms |
| 1M | 68 ms | 2415 ms |

An int8 numpy variant was slower. LanceDB is the documented escape hatch above ~1M memories.

## Retrieval quality (36 paraphrase queries, 8 distractors, 16 general questions)
Hash embedder: recall 0.39-0.42, MRR 0.38-0.40, general-question and distractor suppression 1.0.
bge-small by similarity floor:
| floor | recall | MRR | precision | distractors suppressed |
|---|---|---|---|---|
| 0.45 | 0.92 | 0.88 | 0.54 | 0.25-0.50 |
| 0.50 | 0.86 | 0.82 | 0.60 | 0.88 |
| 0.55 | 0.75 | 0.72 | 0.61 | 1.0 |
Chosen floor: 0.52 (recall ~0.81, distractors suppressed 0.875).

## Scale (full pipeline, hash embedder)
| N | build | cold 1st query | warm p50 | warm p95 | DB | RAM matrix |
|---|---|---|---|---|---|---|
| 1k | 0.1 s | 13.8 ms | 4.0 ms | 6.2 ms | 2.5 MB | 1.5 MB |
| 10k | 0.7 s | 88 ms | 5.2 ms | 12.9 ms | 23.7 MB | 15 MB |
| 100k | 13.7 s | 1.6 s | 16.8 ms | 134 ms | 238 MB | 154 MB |
| 1M | 243 s | 22.5 s | 91 ms | 1.47 s | 2.37 GB | 1.5 GB |
General (non-personal) questions skip retrieval: 0.08-0.19 ms.
bge-small (floor 0.50): 1k warm p50 54 ms / p95 75 ms, recall 0.86; 10k warm p50 66 ms / p95 100 ms, recall 0.86. The ~35-39 ms query embedding dominates. Not run: bge at 100k and 1M (embedding 100k documents takes ~17 min).

## Scenarios
`scripts/memory_eval.py scenarios`: 13/13 pass with both embedders: correction/supersede, temporal, explicit vs inferred, project isolation, privacy/delete, consistency, memory-never-evidence.

## Family-finance cluster fixture (`data/eval/fixtures/memory_family_finance.json`)
"why can't my parents fund this?" returns the 5 related memories (rejected offer, decision, the user's view, land, coaching); "remember why I rejected that university?" returns 4. No unrelated memory (gaming, gym, PC, girlfriend, brother, coffee, cat, city) appears. The interpretation is labelled as the user's view. Switching sensitive injection off withholds the cluster. bge: all checks pass. Hash: 2 non-cluster recall misses (gym, PC; vocabulary gap), no leaks.

## Live check (`scripts/memory_live_check.py`, real Chrome providers, quick mode, throw-away DB)
- General question (World Cup final): 5 fresh prompts, 0 with a memory block, correct answer with sources.
- Project question (Atlas database): 5 fresh prompts, 5 with the block, 0 on thread follow-ups, `memory_used` = only the Atlas fact. The answer repeats the user's own Atlas fact ("already running on PostgreSQL 16"); the providers received it in their prompt. The check cannot say whether that string also reached the claim list.

## Bugs the tests and benchmarks found (fixed)
Every memory-using job would have errored (`_emit(kind=...)` argument clash); goal slots were constant so a new goal replaced an unrelated one; superseded rows were invisible to history lookups; quadratic FTS deletes; per-query full scans of entities/projects; timestamp equality in pruning on Windows.

## Gaps
Extraction is rule-based and topic tags are English word lists. The hash embedder's semantic recall is low (~0.4); use bge-small. No cross-encoder or EmbeddingGemma comparison. Scale numbers were measured before the topic/sensitivity changes. bge not measured at 100k/1M.