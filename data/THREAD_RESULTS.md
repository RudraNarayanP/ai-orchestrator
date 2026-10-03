# Unlimited OmniBrain thread - design and results

## Principle
The OmniBrain thread owns the canonical conversation; provider chats are replaceable context windows underneath it. The user sees one thread (`GET /api/threads/{id}` returns messages only; segments are an implementation detail, visible at `/segments`).

## Model (`backend/thread/`)
- `store.py` - SQLite: threads, segments (one provider chat each: ChatGPT A, ChatGPT B, Gemini A ...), raw messages (append-only), FTS5 over messages and over segment summaries, exact float32 vectors per message (same embedder as memory). Every query is scoped by `thread_id`.
- `service.py` - `ContextManager` (approximate tokens = chars/4 per provider chat, per-provider limits, rotate at 80%), `ThreadService.plan_turn` (continue / rotate / provider switch / resume), `context_for_job` (packet for research jobs), `record_job`, `recall`, `chat` (provider-agnostic driver), `adapter_ask` (real browser adapters; a new segment = a new adapter conversation key = a new chat in the same tab), `endpoint_curator` (OpenRouter/local model as curator).
- `compress.py` - L2 segment summaries (deterministic extractor + grounded curator pass), L3 thread state (corrections retire the facts they correct; old segment lines fold into an archive), L4 promotion of user preferences through memory's own extractor.
- `packet.py` - the `OMNIBRAIN CONTINUATION CONTEXT` packet and the in-chat recall block, fitted to a token budget.
- `recall.py` - recall cues, query terms, "N months ago" soft time window.

## Behaviour covered by tests (`tests/test_thread.py`, 38 tests)
Rotation only at the limit, per-provider limits, no immediate re-rotation of a fresh chat; packet sections/instructions/budget (also a huge thread and a huge current message); relevant memory only in CURRENT USER; nothing lost across 40 turns and many rotations (every message stored in order, indexed, viewable as one thread); user message stored before a provider failure; provider switch (and switching back opens a new chat); historical recall (8 months back, hybrid keywords + meaning + summaries, time boost never filters, about a dozen messages, unrelated talk excluded); isolation (threads can't see each other; a job outside a thread or in another thread gets nothing; thread context never reaches claims/evidence/final; broken store never breaks research); curator used when grounded, falls back to the deterministic summary when down/unusable; corrections supersede; only durable items promoted; API; 20k-message performance in the gate.

## Benchmark (`scripts/thread_bench.py`, hash embedder, file DB, Windows laptop)
Thread built in 25-message segments, each closed, summarised and folded into the thread state; 10 planted facts at random depths.

| messages | segments | build | DB | segment close p50/p95 | recall cold | recall warm p50/p95 | hit@12 / hit@1 | plan_turn p50/p95 | rotation turn p50 | provider switch p50 | research-job packet p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 408 | 7.2 s | 27 MB | 8.9 / 26 ms | 151 ms | 5.1 / 12 ms | 10/10 / 10/10 | 0.8 / 3.6 ms | 19 ms | 3.1 ms | 0.7 ms |
| 100,000 | 4,009 | 135 s | 267 MB | 28 / 47 ms | 1.0 s | 39 / 75 ms | 10/10 / 10/10 | 27 / 43 ms | 67 ms | 57 ms | 0.4 ms |

Thread state stays about 21 KB at any length; the packet was about 1,500 tokens against a 6,000-token budget. All raw messages intact. Scale bugs the benchmark found and fixed: deleting summary index rows by column predicate (build 940 s -> 135 s), segment reads scanning the whole thread (research-job packet 420 ms -> 0.4 ms), per-turn counts and provider ordinals scanning segments/messages. `plan_turn` is still 27 ms p50 at 100k against 0.8 ms at 10k; the remaining growth was not traced.

## Live check (`scripts/thread_live_check.py`, real ChatGPT and Gemini, throw-away DB; `data/eval/thread_live_check.json`)
Tiny context limit: ChatGPT rotated twice (A -> B -> C, reason context_limit; packets 371 and 412 tokens). The first question in the brand-new ChatGPT C ("Which month is my trip, and what is my codeword?") was answered correctly (May, TANGERINE-42). Switching to Gemini (new chat, 445-token packet) correctly recalled the window-seat preference and the Frankfurt layover. 12 raw messages kept, 4 chats, one thread.

## Not measured / gaps
See the gaps list in `data/FINAL_REPORT.md`.