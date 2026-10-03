# OmniBrain - final report (2026-10-04)

Everything below is committed in `omnibrain/` (HEAD after this report). Test counts at the end; gaps and unverified items are listed last and are not softened.

## 1. Early-stop bug (commit 897baa5)
- Symptom: `uk-dpa-age` ran the full round budget (339 s, then 253 s) even when a primary source had already answered.
- Root cause: `find_contradictions` produced false conflicts (identifier numbers, years inside Act names, side remarks), and the key-claim gate ran after the ledger check, so unrelated conflicts blocked the stop.
- Fix: `claims.comparable_text` (identifiers, names and replaced figures are not conflicting figures); the key-claim gating runs before `ledger_ok`; regulators (ICO, Ofcom, ...) are tiered as government/primary; URLs under "Pages I opened:" count as opened by the AI and "Mentioned only:" as mentioned.
- Result: `uk-dpa-age` 339 s -> 253 s -> **49.8 s, one round**. `triv-2plus2` passes in 4 s.
- Honest caveat: in that run ChatGPT returned real URLs; in the earlier arch3 run it returned names only, so the follow-up round was correct there.
- Trap questions: the harness now judges reasoned "nothing public / no such event" answers on their support and counts "couldn't verify" as don't-know (b04ac50). `res-private-review` would now pass and `ua-closed-session` would warn. These two were re-judged by unit tests on their old texts, **not re-run live**.

## 2. Truth-state answer voice (commit 6c28ebe)
Answers are 1-3 casual sentences chosen from ledger verdicts: TRUE / PARTLY / FALSE / CONFLICT / UNVERIFIED (`style.TRUTH_*`, `render_truth`, `verifier.truth_state`, `FinalAnswer.truth_state`). Honest "I couldn't verify that reliably." lines are fixed text, never softened, no emoji or apology. `tests/test_truth.py`. Gap: there is no dedicated chat "why?" handler (only `FinalAnswer.why` and the UI expanders).

## 3. Evidence-reporting rule: "report evidence, do not invent or lecture" (commit f3ef7ac)
- In AGENTS.md non-negotiable 7 and README (Voice).
- Enforced in three places: `style.py` (`EVIDENCE_RULES`, new `BANNED_PHRASES`: "It's important to note", "worth mentioning", "That being said", "Ultimately", "keep in mind", "We cannot definitively say", "several possible explanations", "highlights the importance of", "prudent to"), the researcher/verifier/follow-up/escalation prompts (`prompts.ANTI_INJECTION`, `VERIFIER_ROLE`), and `backend/research/lint.py`, run on every final answer, its "why" and its disagreement note in `verifier.build_final_answer`.
- Lint findings: C unknown turned negative (rewritten to "I couldn't find that documented."), D invented explanation (removed), E unrequested advice/lecture (removed unless the question asks for advice or inference), F boilerplate (lead-ins stripped, filler dropped), G longer than needed (flag). Honest-uncertainty sentences are never touched; the lint never returns an empty answer. `final_check` returns the checklist (answered? facts vs unknowns separated? unknown turned negative? invented explanation? unrequested advice? useless disclaimer? shorter possible?).
- "What do we actually know?" renders Documented / Disputed / Not documented with provenance hosts and no advice (`evidence_report`).
- `tests/test_evidence_reporting.py` (49 tests) includes the tuition example (bad answer cut back to "The sources I found don't say who paid Rahul's tuition."; the good answer is untouched; a flat "parents did not pay" is rewritten when the fact is undocumented), the obvious-inference example (not challenged; a real factual correction stays), every banned phrase, advice/inference kept when asked, honest lines unchanged, the final-answer wiring, and the prompts.
- Limit: the lint is regex-based; it can miss an unusual phrasing or over-strip an edge case.

## 4. Memory system
Design: `data/MEMORY_DESIGN.md`. Results: `data/MEMORY_RESULTS.md`. Raw numbers: `data/eval/memory_*.json`.

**Architecture (and why).** SQLite is the source of truth; keywords in FTS5; entities, topics and links in plain tables; vectors are exact float32 cosine over an in-RAM numpy matrix loaded from BLOB columns; a deterministic multi-signal ranker on top. Embedder `bge-small-en-v1.5` via fastembed (optional), hash-embedder fallback with no dependencies. Chosen because it is local, one file, exact (no ANN tuning), and measured fastest at the sizes a person actually reaches; LanceDB is the documented escape hatch above about 1M memories. Rejected after review (ideas borrowed only): Qdrant, Weaviate, pgvector, Neo4j, Zep, Letta, HydraDB, Mem0. sqlite-vec was benchmarked against numpy:

| N (384-d, k=20, warm p50) | numpy float32 | sqlite-vec |
|---|---|---|
| 1k | 0.03 ms | 0.67 ms |
| 10k | 0.17 ms | 12 ms |
| 100k | 7.4 ms | 237 ms |
| 1M | 68 ms | 2415 ms |

(An int8 numpy variant was slower than float32.)

**Full pipeline scale (hash embedder; warm query includes retrieval + ranking):**

| N | build | cold first query | warm p50 | warm p95 | general question p50 | DB | RAM matrix |
|---|---|---|---|---|---|---|---|
| 1k | 0.1 s | 13.8 ms | 4.0 ms | 6.2 ms | 0.08 ms | 2.5 MB | 1.5 MB |
| 10k | 0.7 s | 88 ms | 5.2 ms | 12.9 ms | 0.08 ms | 23.7 MB | 15 MB |
| 100k | 13.7 s | 1.6 s | 16.8 ms | 134 ms | 0.19 ms | 238 MB | 154 MB |
| 1M | 243 s | 22.5 s | 91 ms | 1.47 s | 0.19 ms | 2.37 GB | 1.5 GB |

bge-small, floor 0.50: 1k warm p50 54 ms (p95 75), recall 0.86, MRR 0.82; 10k warm p50 66 ms (p95 100), recall 0.86 (query embedding, about 35-39 ms, dominates). **bge was not run at 100k or 1M** (embedding 100k documents takes about 17 minutes); hash was used for those sizes.

**Quality (36 paraphrase queries, 8 distractors, 16 general questions).** Hash embedder: recall 0.39-0.42, MRR 0.38-0.40, distractors and general questions suppressed 1.0. bge-small by floor: 0.45 -> recall 0.92 / distractors suppressed 0.25-0.50; 0.50 -> 0.86 / 0.88; 0.55 -> 0.75 / 1.0. Chosen floor 0.52 (recall about 0.81, distractors suppressed 0.875). **Recommended default is bge-small; the hash embedder is a fallback with low semantic recall.**

**Scenario suite** (`scripts/memory_eval.py scenarios`): 13/13 pass with both embedders: correction/supersede, temporal, explicit vs inferred, project isolation, privacy/delete, consistency, "memory is never evidence".

**Facts vs the user's interpretation, clusters, vague references, sensitivity** (commit 95b5e23). Opinions/feelings are stored as `interpretation` and injected as "the user's own view, not an established fact". Topic tags and cluster expansion pull related memories (decision + rejection + the user's reading + the underlying facts) for vague questions. Family-finance fixture (`data/eval/fixtures/memory_family_finance.json`): "why can't my parents fund this?" returns the 5 related memories (rejected offer, decision, the user's view, land, coaching); "remember why I rejected that university?" returns 4; none of the unrelated memories (gaming, gym, PC, girlfriend, brother, coffee, cat, city) appears; the interpretation is labelled; turning the sensitive switch off withholds the cluster. bge: all checks pass. Hash: 2 recall misses on non-cluster queries (vocabulary gap), no leaks.

**Privacy.** Local file only; separate switches for injection, capture and sensitive memories; secrets filtered before storing; learns from the USER's message only, never from provider output; physical delete (no tombstone text) and "forget everything" with confirmation; sensitive memories need relevance or topic overlap plus the switch; audit events carry no memory content; memory text is prompt context only and never enters claims, evidence, sources, the verifier or the answer (`tests/test_memory_isolation.py`).

**Inspection UI.** Memory panel (list with type, source, confidence, status, dates, sensitivity chips; "why would this be used" search; add/edit/delete; export; forget-all with confirm; inject/capture/sensitive switches), per-answer "Memory used as context" expander, optional Project field.

**Bugs the tests and benchmarks found (fixed):** every memory-using job would have errored (`_emit(kind=...)` clash in `_prepare_memory`); goal slots were constant so a new goal replaced an unrelated one; superseded rows were invisible to history lookups; quadratic FTS deletes; per-query full scans of entities/projects; timestamp equality in pruning on Windows. The full suite then caught that the memory panel's export link broke the answer export-link selector (bb65d6f).

**Live check** (`scripts/memory_live_check.py`, throw-away DB, real Chrome providers, quick mode): a general question ("Who won the 2022 World Cup final?") had 0 of 5 fresh prompts with a memory block and a correct answer; a project question (Atlas database) had 5 of 5 fresh prompts with the block, none on thread follow-ups, and `memory_used` was only the Atlas fact. The final answer repeated the user's own Atlas fact ("already running on PostgreSQL 16"); the providers had received it in their prompt, and the check cannot say whether that string also reached the claim list.

## 5. Unlimited OmniBrain thread (commits 67afb11, d244266, c2a21e9)
Design and numbers: `data/THREAD_RESULTS.md`. Code: `backend/thread/`.
- **Segments.** The thread owns the conversation; each segment is one provider chat (ChatGPT A, ChatGPT B, Gemini A ...). The user sees one thread (`GET /api/threads/{id}` shows messages only).
- **Rotation.** `ContextManager` tracks approximate tokens (chars / 4) per provider chat against a per-provider limit (`threads.limits`, defaults in code) and rotates at `threads.rotate_at` (80%). Rotating closes the segment and opens a NEW chat in the same tab slot (new adapter conversation key) with the continuation packet. A chat with fewer than two exchanges is not rotated again.
- **Compression.** L1 recent turns verbatim; L2 per-segment summary (decisions, facts, arguments, open questions, preferences, key terms, project state, entities, corrections, conclusions, rejected, direction) by the curator model when up, each item grounded in the segment's own words, with a deterministic extractor as fallback and safety net; L3 thread state (a later correction retires the fact it corrects; old segment lines fold into an archive, state stays about 21 KB); L4 only user preferences go to long-term memory through memory's own extractor. Raw messages are never deleted and are indexed (FTS5 + vectors).
- **Continuation packet** `OMNIBRAIN CONTINUATION CONTEXT`: CURRENT USER (relevant memory only), PROJECT, CONVERSATION HISTORY, RECENT DISCUSSION, RELEVANT EARLIER DISCUSSION (only when asked), ESTABLISHED FACTS, DECISIONS, OPEN QUESTIONS, CORRECTIONS, IMPORTANT USER PREFERENCES, CURRENT TASK, and the instructions to continue naturally, not restart, not ask the user to repeat, not mention the transfer. Held inside a token budget (default 6,000). It states that everything in it is conversation context, not verified evidence.
- **Provider switch.** Switching provider mid-thread closes the old segment and gives the new provider the thread state.
- **Historical recall.** "Remember that thing about Germany 8 months ago" searches that thread's raw messages and segment summaries (keywords + meaning, rank-fused, soft time boost that never filters) and injects about a dozen messages plus the relevant summary.
- **Research jobs.** `POST /api/jobs {"thread_id"}` gives the job's fresh chats the thread context (prompt only; never claims, evidence, sources or the verifier); the question and final answer join the thread. A job outside a thread, or in another thread, gets nothing.
- **API.** `POST/GET/DELETE /api/threads`, `GET /api/threads/{id}`, `/segments`, `POST /recall`, `POST /chat {text, provider}`; `threads.enabled: false` switches it off.

**Benchmark, synthetic thread, hash embedder, file DB** (25-message segments, each summarised; 10 planted facts):

| messages | segments | build | DB | segment close p50 | recall cold | recall warm p50 / p95 | hit@12 / hit@1 | plan_turn p50 | provider switch p50 | research-job packet p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 408 | 7.2 s | 27 MB | 8.9 ms | 151 ms | 5.1 / 12 ms | 10/10, 10/10 | 0.8 ms | 3.1 ms | 0.7 ms |
| 100,000 | 4,009 | 135 s | 267 MB | 28 ms | 1.0 s | 39 / 75 ms | 10/10, 10/10 | 27 ms | 57 ms | 0.4 ms |

All raw messages intact. The 100k run found and fixed three scale bugs (build 940 s -> 135 s; research-job packet 420 ms -> 0.4 ms).

**Live check** (`scripts/thread_live_check.py`, throw-away thread DB, real ChatGPT and Gemini windows, `data/eval/thread_live_check.json`): with a tiny context limit ChatGPT rotated twice (ChatGPT A -> B -> C, reason `context_limit`, packets of 371 and 412 tokens). The first question in ChatGPT C, a brand-new chat, was "Which month is my trip, and what is my codeword?" and it answered "Your trip is in May, and your codeword is TANGERINE-42." Switching to Gemini (new chat, 445-token packet) answered "You prefer window seats, and you want to avoid a layover in Frankfurt." All 12 raw messages kept; 4 chats under one thread.

## 6. Test counts
- Offline gate (`pytest -q -m "not browser"`): **610 passed** (523 before the lint and thread work; +49 evidence-reporting, +38 thread).
- Full `pytest -q` including the real-Chrome UI tests: **639 passed, 0 failed** (about 4.4 minutes). An earlier full run had one failure (the export-link collision) which was fixed and re-run.
- The 610 gate run predates the last thread-store edit (segment-read index hint); that edit was covered by `tests/test_thread.py` and by the 639-test full run, which ran after it.

## 7. Live checks done
Early-stop eval (uk-dpa-age, triv-2plus2); memory live check (general and project cases); thread live check (rotation x2 and provider switch, ChatGPT and Gemini). Key-leak check on the repo and history: 0 hits; the OpenRouter key stays only in the gitignored `config/settings.yaml`.

## 8. Gaps and unverified (stated plainly)
- The full hard-question suite was never run end to end; trap questions `res-private-review` and `ua-closed-session` were re-judged by unit tests, not re-run live.
- Whether each site obeys the OPENED/MENTIONED labelling is unverified; some providers are blocked or need login (Copilot, Le Chat, Meta AI, Pi, Qwen, DeepSeek); ChatGPT sometimes returns source names without URLs.
- Memory extraction is rule-based and the topic tags are English word lists. No cross-encoder rerank or EmbeddingGemma comparison. Scale numbers for the full pipeline were measured before the topic/sensitivity changes; bge was not run at 100k or 1M. The hash embedder's semantic recall is low (about 0.4); bge-small is the recommended default and is optional (`requirements-memory.txt`).
- The evidence lint is heuristic (regex): it can miss an odd phrasing or over-strip an edge case. There is no dedicated chat "why?" handler.
- Thread: provider context limits are approximate and the defaults are my estimates, not measured per site; token counting is chars/4. Only ChatGPT rotation and a ChatGPT -> Gemini switch were run live; other providers were not. The curator-model path of compression is unit-tested with a fake curator and was not run live against OpenRouter here. The thread benchmark used the hash embedder only; `plan_turn` p50 grows from 0.8 ms at 10k to 27 ms at 100k and the cause was not traced; vectors for a thread are loaded into RAM on first recall (1.0 s cold at 100k). A thread built with one embedder falls back to keyword-only recall if the embedder is changed. There is no thread UI beyond the API (the existing UI does not yet select a thread), and `POST /api/threads/{id}/chat` was exercised through the service-level live check, not through HTTP. L4 promotion is limited to user preferences by design.