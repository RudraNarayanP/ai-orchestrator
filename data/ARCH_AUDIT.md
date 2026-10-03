# Architecture audit (2026-10-03, before the architecture correction)

Scope: read-only inspection of `backend/`, `browser/adapters/` and `scripts/` at commit `fae9b02`. Findings first, then what the code does against the spec.

## 1. Routing
- `runner._analyze` -> `router.classify`. Arithmetic ("2+2") never touches a model (`_obviously_arithmetic`); other stable questions can be sent to the OpenRouter analysis model (`STABLE_KNOWLEDGE_ASK`). A trivial/computed answer is returned at level 0 with no browser session and no verifier call. OK for 2+2; any other "trivial" question costs one analysis-model call (not a browser, not the verifier).
- Law/medicine/finance/visa/university categories set `analysis.high_stakes`; `STATUTE_RE` forces research. `router.select_primary` picks one provider, `select_secondaries` picks the others.

## 2. Browser sessions and provider conversations  (DEFECT)
- `BrowserEngine`: one OmniBrain window, one tab per provider, tabs reused (good, matches "one dedicated window with reused tabs").
- `engine.open_research_page` returns the cached tab and only navigates if it is blank. `ChatAdapter._page(fresh=...)` computes `new_chat_url` but nothing navigates to it. Result: **every question, every round and every provider call is typed into whatever conversation the tab was last left in**. Questions are not isolated at the conversation level; the stale-Gemini-reply leak seen in the eval is the symptom (the earlier answer was still the visible chat). `off_topic()` (commit 781cf39) papers over it after the fact.
- There is no per-AI conversation thread object, no research id carried to the adapter (`runner._ask` calls `adapter.ask("job", ...)`, a literal), and no way to ask "continue the same conversation".

## 3. Follow-ups
- Within a job there is no same-conversation follow-up. After the primary AI, the runner goes straight to (a) a targeted expansion or (b) a parallel swarm of other providers (`router.select_secondaries`), then the verifier, then targeted rounds (`_targeted_round`) that each open a new prompt (accidentally in the same tab, see 2). The spec's FIRST escalation (same AI, same conversation, re-investigate the specific claim) does not exist, and neither do self-correction records.
- Cross-question follow-ups (`memory.py`) rewrite the user's next question using the previous answer; that is a separate feature and stays.

## 4. Evidence extraction  (DEVIATION)
- `claims.extract_claims` (analysis model, heuristic fallback) turns provider answers into atomic claims.
- `evidence/pool.build_pool` opens the URLs the providers cited (good) **and** runs its own discovery: HTTP searches against DuckDuckGo/Bing for every material claim (`search_queries`), a browser "search" provider as fallback, and counter-searches (`_counter_links`). Everything found is fetched and ranked equally with AI-cited pages. So OmniBrain itself does web research and files the result as evidence next to what the AI cited. The evidence record carries `origin` (`provider`/`search`) but nothing downstream treats `search` as lower-trust, and the final answer can rest entirely on pages the AI never saw. That is the "secretly doing the research" deviation.
- `search` is also an enabled "provider" (`providers: chatgpt, gemini, search`) and can be picked as a researcher; it is a SERP scraper, not an AI.
- `Citation` has no field distinguishing "the AI mentioned this URL" from "the AI actually opened it". `ProviderResponse.web_research_status` is a coarse per-answer signal only.

## 5. OpenRouter
- Used for: claim extraction, the level-0 stable-knowledge check, follow-up rewriting, the verifier/curator (`Verifier.verify`, sees claims, fetched evidence rows, conflicts and each provider's prompt + answer fenced as untrusted data), and vision fallback. Not used as a researcher (it has no web access). Good. It already emits `follow_ups` with `target_providers`, which is a RESEARCH_NEEDED in everything but name (no `preferred_researcher`/`instruction` fields, no explicit parser, no thread awareness).

## 6. Do prompts tell the AI to use its own web search?
- Partly. `WEB_INSTRUCTION` says "search the web ... whenever the question needs current or externally verifiable information", with per-provider hints. It does not say "find the sources yourself, prefer primary sources, open and inspect them, report which ones you actually checked", and the response schema asks for "SOURCE LINKS: URLs you actually used" without separating opened from merely mentioned.

## 7. Final answer
- `style.py` produces the short answer with a closing emoji on confident answers and a plain "I don't know"/"I couldn't verify" otherwise, but the emoji can land after the first sentence of a longer answer, and "I'm not sure -- the sources disagree." is not a distinct form.
- The "opened but not cited" defect: `build_final_answer` takes sources from the verifier's `sources`; evidence that is linked to the supported claim in the ledger is not guaranteed to reach them.

## Plan (smallest changes, existing browser/provider infrastructure kept)
1. Per-research conversation threads: the adapter starts a NEW chat for each research id and each provider, remembers the thread URL, and can continue it. Runner passes the research id.
2. Prompts: explicit own-web-search instruction; opened-vs-mentioned reporting; a same-conversation re-investigation prompt; an independent-verification prompt for parallel escalation.
3. Runner: FIRST escalation = same-conversation follow-up to the same AI; record self-corrections; parallel escalation only if still unresolved; hedge detection extended.
4. Evidence: OmniBrain discovery off by default (`search.own_discovery: false`), `search` provider no longer a researcher; HTTP opening kept only to audit AI-cited URLs; citations carry mentioned/opened; discovery evidence never establishes a claim.
5. Curator: explicit RESEARCH_NEEDED parser (claim, reason, preferred researcher, instruction), thread-aware loop with a stop limit.
6. Final answer: emoji only at the end, "I'm not sure" form, supported-claim sources always carried to the answer.
7. Offline tests 1-8 plus prompt/RESEARCH_NEEDED tests; live check; README/AGENTS.md.

## Addendum: correction and status (after the change)

- Correction to the finding above: `router.select_secondaries` already excluded the `search` provider as a researcher; it only served as an evidence transport. It is now also in the eval's default skip list and own discovery is off by default.
- Changes by commit: c4875bd (one conversation per AI per research id), 659709b (primary -> same-thread follow-up -> parallel independents, own-web-search prompts, self-correction records), 84a703f (own discovery off, opened-vs-mentioned, claim-evidence-source link, RESEARCH_NEEDED loop, tests 1-8), bc2bcf5/fa7c9fa (voice), 075cb3c (persistence + eval), a05935d (turn numbering).
- Live results: `data/eval/ARCH_RESULTS.md`. Still unverified live: other sites' conversation URLs and obedience to the OPENED / MENTIONED ONLY labelling.
