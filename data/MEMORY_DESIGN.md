# OmniBrain memory system - design note

Status: part 1 (requirements + ecosystem research, 2026-10-04). Benchmark-backed decision is in part 2 below (filled in after measuring on this PC).

## 0. Hard rules (from the user)
- Memory is **never evidence**. It is personalisation context only. It never enters claim extraction, source auditing, sufficiency, verification or citations. "Capital of France" injects nothing.
- Current user instruction > explicit memory > inferred memory. Inferred never overrides explicit.
- Local-only. No silent upload of the DB. Explicit delete, "do not remember", global off-switch for injection, inspection UI with *why retrieved*.
- Same relevant context goes to **every** provider prompt (cross-provider consistency).
- Prompt wrapper: "Relevant context about the user: ... Use when relevant. Don't mention it was supplied. Don't assume irrelevant memories apply."

## 1. Requirements -> what must be true of the store
| Need | Consequence |
|---|---|
| Runs on Windows, no server, no Docker | embedded, pip-installable, single file or folder |
| Single user, 1k-100k memories typical (1M stress) | exact or ANN search OK; no horizontal scale needed |
| Temporal + supersede (corrections) | relational columns (status, supersedes, valid times); transactional updates |
| Metadata filters (scope, project, type, status) | SQL WHERE must apply *before/with* vector search |
| Graph retrieval (entities, related memories) | a relationship table + recursive CTE is enough at this scale |
| Incremental add/edit/delete | no rebuild; delete must really delete (privacy) |
| Maintainable | fewest moving parts; already have SQLite (`backend/storage/db.py`) |

## 2. Ecosystem survey (sources: web research on 2026-10-04)
| Option | What it is | Fit for local single-user Windows memory | Verdict |
|---|---|---|---|
| **LangGraph BaseStore / LangMem** | JSON docs in `namespace`/`key`, optional embedding index, metadata filter; LangMem adds background extraction/consolidation (semantic/episodic/procedural). Prod store is Postgres; InMemoryStore is dev-only. (langchain-ai.github.io/langmem; docs.langchain.com/oss/python/concepts/memory) | Good *ideas* (namespaces, semantic/episodic/procedural split, background manager). Dragging in LangGraph + LangChain model wrappers + Postgres for one user is heavy; extraction needs a model call we already route ourselves. | **Borrow the model, not the dependency** |
| **Mem0** | extraction + multi-signal retrieval (semantic + BM25 + entity + temporal), Apache-2.0, self-host. Self-reported LoCoMo 92.5 / LongMemEval 94.4, but an independent run (Vectorize, 2026-03) scored it 49.0 on LongMemEval. (dreaming.press, digitalapplied.com) | Same multi-signal idea we plan; own vector store (Qdrant) + LLM calls on every add. Benchmarks are vendor-reported and contested. | **Borrow the retrieval signals; do not adopt** |
| **Zep / Graphiti** | bi-temporal knowledge graph; needs Neo4j/FalkorDB; Zep CE deprecated, hosted only. | Best at "what was true when", but a graph DB server is operational weight for one user. We get the needed temporal behaviour with `valid_from/valid_to` + `supersedes`. | **No (too heavy); copy bi-temporal fields** |
| **Letta (MemGPT)** | agent runtime owning core/recall/archival memory via tool calls, Postgres. | We are not letting providers self-edit memory (privacy + "memory is never evidence"). | **No** |
| **HydraDB** | Rust object-store-native graph DB; managed or self-hostable; memories + knowledge + episodic; hybrid recall. (docs.hydradb.com) | Aimed at teams / 10K-10M docs; managed API or object-store infra; "no data leaving your stack" only if self-hosted. Overkill + new dependency. | **No** (reasoning ideas only: scoped, explainable recall) |
| **pgvector** | Postgres extension. | Needs a local Postgres service on Windows; best when memories join with big relational data. | **No** |
| **Qdrant** | Rust server, HNSW, rich payload filters. Python local mode is dev-only (~20k points); Qdrant Edge is the on-device option. | Server for one user; local mode too small. | **No** |
| **Weaviate** | native hybrid vector+BM25, embedded mode exists but heavy. | Heavy, complex. | **No** |
| **Chroma** | `pip install chromadb`, `PersistentClient(path)`, single-user. | Easy, but separate store from our relational state (supersede/status joins duplicated), more deps. | Fallback |
| **LanceDB** | embedded, Lance columnar format, IVF-PQ/HNSW, FTS + RRF hybrid, dataset versioning. Pre-1.0. | Scales to millions embedded. On small data, 1.9 ms avg vs 0.5 ms for SQLite stack with identical quality (vecdb-bench). | **Escape hatch if >~300k memories** |
| **sqlite-vec + FTS5** | SQLite extension (pure C), exact brute-force KNN, same .db file; pre-1.0. Windows: pip wheel, load via `enable_load_extension` (not Microsoft-Store Python). | In vecdb-bench (small hybrid): quality identical to LanceDB (P@5 0.233, MRR 0.977 vs 0.975), 0.5 ms vs 1.9 ms. Brute force: ~1.7 s p50 at 100k x 1024-d (Go bench) - fine for background, too slow at 100k+ unless dims are small / pre-filtered. | **Primary candidate** |
| **Neo4j / GraphRAG** | graph DB server. | Needs JVM server; our graph is tiny (entity <-> memory edges). | **No**; use edge table + recursive CTE |
| **Embedding models** | `BAAI/bge-small-en-v1.5` 33M params, 384-d, int8 ONNX 32 MB, MIT; FastEmbed runs it on ONNX Runtime CPU. `google/embeddinggemma-300m` 308M, 768-d (truncatable to 128-d), multilingual (needed: user writes English + possibly Ukrainian/Hindi), <200 MB RAM quantised. | bge-small is the CPU-cheap default; EmbeddingGemma better multilingual but heavier. | Benchmark both on this PC |
| **Rerankers** | `ms-marco-MiniLM-L-6-v2` (80 MB, ~72 ms / 100 docs CPU) vs `bge-reranker-v2-m3` (568M, ~7 s / 100 docs). | Cheap MiniLM only on the final ~20 candidates, or a deterministic multi-signal rerank. | Benchmark: does a cross-encoder beat the multi-signal score? |

Caveat on vendor benchmarks: LoCoMo numbers are self-reported, harnesses differ, and the dataset has documented flaws; we do not rank on them. We benchmark on our own synthetic + real-shaped data (part 2).

## 3. Candidate architecture (to be confirmed or rejected by part 2 numbers)
SQLite (existing file, new tables) + FTS5 (keyword) + a local vector index (sqlite-vec if installable on this PC; otherwise numpy brute-force over a float32 BLOB column, which has the same exact-search behaviour) + relationship tables (`memory_entities`, `memory_links`) + a deterministic multi-signal reranker. Optional ONNX embedder (fastembed) behind an interface with a deterministic hashing-embedder fallback so tests and offline use never need a model download.

Escalation triggers (documented, measured): vector search p95 > 100 ms at the user's real size -> prefilter by scope/project + int8 / truncated dims; still > 100 ms -> LanceDB.

## 4. Decision (confirmed by the numbers in part 5)
**SQLite is the source of truth; vectors are exact float32 cosine in an in-RAM numpy matrix loaded from BLOB columns; FTS5 for keywords; plain tables for entities, topics and links; a deterministic multi-signal ranker.** No vector database server, no graph database, no LLM call on the hot path.

Why, from our own measurements (this PC, 384-d, k=20; `data/eval/memory_bench_engines_*.json`):

| N | numpy exact warm p50 | sqlite-vec exact (on disk) warm p50 | sqlite-vec insert | DB size |
|---|---|---|---|---|
| 1k | 0.03 ms | 0.67 ms | 0.02 s | 1.6 MB |
| 10k | 0.17 ms | 12 ms | 0.19 s | 16 MB |
| 100k | 7.4 ms | 237 ms | 3.9 s | 158 MB |
| 1M | 68 ms | 2415 ms | 36 s | 1572 MB |

* numpy-in-RAM is ~30x faster than sqlite-vec brute force at 100k and still 68 ms at 1M. sqlite-vec's `rowid IN` prefilter did not help; an int8 numpy variant was *slower* than float32. One user will never hold more than ~100k memories, so exact search is not a compromise: recall is 100% by construction (no ANN approximation to tune or to explain).
* The cost of RAM-resident vectors is the cold load (1.6 s at 100k, 22 s at 1M, 1.5 GB) - paid once per server start.
* **LanceDB** stays the documented escape hatch above ~1M memories (its quality equals SQLite+FTS5+sqlite-vec in vecdb-bench; it only wins on scale). **Qdrant/Weaviate/pgvector/HydraDB/Neo4j/Letta/Zep**: each needs a server or a managed service for a single local user, and none gives anything the tables below do not (see section 2).
* Mem0/Zep/LangMem are borrowed *ideas*: multi-signal retrieval (semantic + BM25 + entity + temporal), bi-temporal fields (`valid_from/valid_to/supersedes`), namespaces, a semantic/episodic split. Their published benchmark numbers are not used: they are vendor-reported and contested (Mem0: 94.4 self-reported vs 49.0 independent on LongMemEval).
* **Embedder**: `BAAI/bge-small-en-v1.5` via fastembed/ONNX (33M params, 384-d). Measured: cold load 26 s the first time (model download), 0.8 s afterwards; ~39 ms per query, ~10 ms per document in batches. Retrieval recall on our 36-query paraphrase set: **0.86 vs 0.39 for the dependency-free hash embedder** - the hash embedder is only a safe offline fallback (tests, no model download), not a recommendation. EmbeddingGemma-300m (multilingual) was **not benchmarked**; if the user writes mostly Ukrainian/Hindi it is the first thing to try. A cross-encoder rerank (MiniLM) was **not benchmarked**: the multi-signal score already reaches MRR 0.82 on the set and the cost would be ~70 ms per 100 candidates; revisit only if MRR on real use is poor.
* The embedder is a swappable interface: changing it re-embeds every memory on the next open (`embed_model` is recorded in the file).

## 5. Schema
`memories(memory_id, content, memory_type, scope, project, entities, topics, source_conversation, created_at, updated_at, last_accessed_at, access_count, importance, confidence, status, supersedes, superseded_by, embedding, metadata, source, slot, valid_from, valid_to, goal_active, sensitivity)`
* `memory_type`: preference | fact | goal | project | episodic | conversation | **interpretation**.
* `source`: `user_explicit` (the user said it, asked to remember it, corrected it, or edited it) | `model_inferred` (soft signals). **Inferred never overrides explicit**; the store refuses it (`rejected`).
* `status`: ACTIVE | SUPERSEDED (history; only seen when the question asks about the past) | ARCHIVED. Deleting is physical.
* `slot`: what a memory is *about* ("residence", "pref:units"). At most one ACTIVE memory per (slot, scope, project); a newer statement supersedes (bi-temporal `valid_to`), `check_invariants()` proves it.
* Side tables: `memory_fts` (FTS5, porter) + `memory_fts_map` (rowid map so deletes are O(log n); a delete by the unindexed column was quadratic - found by the 10k benchmark), `memory_entities`, `memory_topics` (indexed life-domain tags), `memory_links`, `memory_events` (content-free audit lines), `memory_settings`.

## 6. Retrieval pipeline
understand (general / personal / project / cluster / complex; entities and project by indexed n-gram lookup, not by scanning) -> candidates (vector top-60 + FTS5 top-60 + entity graph + project namespace) -> lifecycle and isolation filters (status, goal_active, expiry, project, conversation, sensitivity switch) -> relevance (max of vector, keyword coverage, entity) with a floor per embedder -> sensitive gate -> multi-signal score (relevance .46, importance .14, confidence .08, recency .08, source .08, project .10, graph .06) -> **cluster expansion** -> near-duplicate removal -> dynamic budget (general 0, personal 2, project 5, cluster 6, complex 8). Every hit carries `why` strings (shown in the Memory panel and by `POST /api/memory/search`). Very common query terms are skipped on stores above 20k memories (FTS5 vocab document frequency > 3%) - this took FTS from 51 ms to 17 ms at 100k and 683 ms to 227 ms at 1M.

### Facts vs the user's interpretation
A feeling, belief or reading of someone's motives ("I may read their refusal to fund as not valuing my education") is stored as `interpretation`, verbatim and quoted behind the fixed prefix `User's view:`. This is enforced **in the store**, not only by the extractor: anything that looks like one is reclassified even when it is typed in as a fact, it never gets a slot (so it can never supersede or merge with a fact), and when injected it is labelled "(the user's own view, not an established fact)" with a one-line instruction not to state it as fact. Stated facts ("Parents spent about INR 4 lakh on Allen coaching") stay `fact`.

### Context clusters
Each memory carries coarse life-domain tags (finance, family, education, decision, health, relationship, legal, identity, fitness, gaming, tech) from word lists (`backend/memory/topics.py`; heuristics, not semantics). After the normal ranking, the strongest 1-3 matches whose tags overlap the question's tags become *seeds*; memories sharing >=2 tags with the question+seeds (or one tag when the memory is a decision and the question is a vague reference - "this", "that", "remember why") join as `connected` hits. A single shared tag is not enough, which is why "brother lives in Lviv" (family only) is not dragged into a family-money question, and a gaming match can never pull the cluster. The budget rises to 6 only when a cluster is found.

### Sensitive memories
Family, money, health, relationships, legal and identity memories - and every interpretation - are flagged `sensitive`. They (a) never leave the machine except inside the context block of a prompt, (b) are only offered when strongly matched (relevance >= 0.30) **or** the question is plainly in the same domain, (c) cannot be pulled in by a non-sensitive seed, (d) can be switched off entirely (`sensitive` setting / Memory-panel checkbox), (e) are listed, filterable, editable and physically deletable; audit events never contain memory text.

## 7. Privacy and integration points
* Local file only (`data/memory.db`, gitignored). The memory file is never read by evals or tests (`memory.enabled=False` in test and eval configs).
* Learns from the **user's own message only** - never from an AI's answer. "Don't remember this", secrets (keys, passwords, tokens), questions and hypotheticals store nothing. "Forget X" deletes. Capture and injection each have a switch.
* The block is added to the prompt of every provider that starts a conversation (primary, escalation, targeted follow-ups; thread continuations already contain it). It is **never** added to claims, evidence, the verifier payload, sources or the final answer: memory is context, not evidence (`tests/test_memory_isolation.py`).

## 8. Known gaps
* Extraction is rule-based (patterns + word lists), not an LLM extractor: it misses unusual phrasings and will classify some edge sentences wrongly. The user can always add/edit/delete in the panel.
* Topic tags are English word lists; non-English statements get no tags (so no cluster/sensitivity inference) until a multilingual lexicon or embedding-based tagging is added.
* No cross-encoder reranker and no EmbeddingGemma benchmark (see section 4).
* Cold start loads all vectors (22 s at 1M); no ANN index by design.
* Results with the neural embedder at 100k/1M were not run (embedding 100k documents takes ~17 min on this CPU); those sizes were measured with the hash embedder, which exercises the same SQLite/FTS/numpy paths but not neural-embedding quality.