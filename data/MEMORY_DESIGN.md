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
