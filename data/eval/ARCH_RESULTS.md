# Live results after the browser-AI-first correction (batch arch1, 2026-10-03)

Providers: chatgpt + gemini (logged out). Curator/verifier: nvidia/nemotron-3-super-120b-a12b:free via OpenRouter.
Raw: `console_arch1.txt`, `eval_20261003_223808.json`. NOTE: the JSON `threads`/`corrections`/`ai_opened`
fields are empty because the eval script read them from the wrong key (fixed in a05935d after the run);
the numbers below were read from the `job_extra` table of the same jobs.

| question | verdict | time | browser sessions | curator calls | result |
|---|---|---|---|---|---|
| triv-2plus2 | WARN (eval false positive, fixed in 08bebfc) | 4 s | 0 | 0 | "4" - classifier only |
| uk-dpa-age | PASS | 520 s | 4 | 2 | correct (13), legislation.gov.uk linked as primary; stopped at max rounds (3) |
| ua-marriage-age | WARN | 373 s | 3 | 2 | "I couldn't verify that reliably." - zakon.rada.gov.ua unreachable from this PC, the AIs could not open it either |
| uni-oxford-plagiarism | WARN | 176 s | | | "I couldn't verify that reliably." ox.ac.uk answers 403 to the audit fetch and the AIs gave nothing openable; an honest don't-know |

Observed live (this is the first real evidence for the new design):
- Each research got its OWN new chat on each site (new chatgpt.com/uc/<id>, gemini /app/<id>); the follow-up
  went back into the same URL (`continued: true`) - ChatGPT turn 1 -> follow-up turn 2 in the same thread.
- Escalation order matched the spec: primary ChatGPT -> same-thread follow-up to ChatGPT -> Gemini ->
  curator-issued targeted follow-ups to gemini/chatgpt (continuing their threads).
- One self-correction record was created in each of the two researched questions.
- Citations carry the AI's own opened/mentioned label (e.g. cookie-script.com: mentioned only; natlex.ilo.org: opened).
- No stale answer from the previous question appeared in any record.

Defects / gaps seen:
- Gemini's conversation URL stayed `https://gemini.google.com/app` in the Ukraine run (id not captured); follow-up worked only because the tab stayed in that chat.
- In the UK run Gemini's three targeted turns all got the number 2 (concurrent requests shared a turn). Fixed in a05935d with a regression test.
- ChatGPT's own citations were not captured in the DB flags (only Gemini's carry `cited_by`); the legislation.gov.uk page reached the final sources through the audit/ledger. Needs a look at ChatGPT citation extraction.
- uk-dpa-age used all 3 rounds although the answer was right at once: the curator kept asking for more research. The final answer is three sentences, longer than the spec's tiny answer, and carries a caveat "The AI reviewer wasn't available" although two curator calls ran - the final curator pass probably failed and the deterministic fallback wrote the answer.
- Runtime 6-9 minutes per hard question.


---

# Phase 2 report: early stop, provenance, explicit state, tiny answers (2026-10-04)

## Per item: what changed (commits)
1. **Early stop / over-escalation** - `6f0da74`, `ab69f34`. A primary AI that opened a primary/official page supporting the claims that answer the question
   (provenance CLAIM_SUPPORTED) ends the research: 1 round, no follow-up, no parallel AIs, at most one curator pass
   (`SufficiencyAssessment.strong_primary`). The curator may ask for more but is not followed when the evidence is strong; two targeted rounds that add no
   new confirmed claim/source also stop the loop (`Job.stop_note` says why). Side remarks (e.g. a guidance page's update date) and conflicts that do not touch the
   asked claims do not block the stop (`ab69f34`). A hedged/weak primary still gets the same-thread follow-up, then parallel independents.
2. **Provenance** - `6f0da74`. Root cause of "ChatGPT cited nothing": logged-out ChatGPT renders source chips as bare text, so its DB record had 0 citations
   although its answer listed OPENED/MENTIONED URLs. `citations.text_citations` now records the URLs the AI wrote as the AI's citations (`origin="text"`; native
   site links stay `origin="native"`; the chat's own links are excluded; nothing is searched for or invented). States MENTIONED -> OPENED -> INSPECTED -> CITED ->
   CLAIM_SUPPORTED are computed per evidence record (`provenance()`), `omnibrain_opened` is a separate flag, a page only OmniBrain opened has provenance None and
   never outranks an AI-opened one in the final sources; the curator payload shows both. Final sources carry provenance, cited_by, ai_opened, omnibrain_opened, claim_ids.
3. **Explicit state** - `56dbb8e`. `research_status`, `reviewer_status` (COMPLETED/UNAVAILABLE/INVALID_OUTPUT/NOT_RUN), `synthesis_status` (CURATED/FALLBACK/DETERMINISTIC/DIRECT),
   `fallback_reason` on every report and the final answer. The "AI reviewer wasn't available" caveat appears only for UNAVAILABLE. **Diagnosis of the uk-dpa-age fallback:**
   round 2 of that job was an upstream **HTTP 429** on the free model (UNAVAILABLE); round 3 returned a long JSON that failed to parse (**INVALID_OUTPUT**; most likely cut off by max_tokens,
   the stored raw text is capped at 4000 chars so the exact tail is not provable). The old code reported both as "reviewer wasn't available" and also printed that caveat for the
   by-design light path. Fixes: truncated-JSON repair (`_repair_truncated`), answer-first curator schema so a cut tail loses detail not the conclusion, honest status per cause.
4. **Tiny answers** - `56dbb8e`. `style.tiny()` keeps 1 sentence (2 at most, <=260 chars, abbreviation-aware), final caveats limited to 1, emoji only at the very end.
5. **Gemini conversation id** - `9e6061f`. The id reaches the address bar after the answer; the adapter now waits (up to 8 s) and falls back to an `aria-current` link; a site that never
   exposes an id is reported in `detail`, not faked. (Live: Gemini ids captured in both phase-2 runs, e.g. `/app/2ed380b4fd27139e`.)
6. **Provider pool (steering from the user)** - `cf828ce`, `8d17a3d`, `6fda7e1`. No provider is hard-coded out any more: the eval attempts all 9 AIs (only `search`, which is Google Search, not an AI, is excluded);
   a wall seen in an EARLIER run only lowers a site's rank and one escalation slot re-probes it (`prior_health`/`reprobe`); a wall in THIS job skips the site for the job; empty-answer sites are skipped for the job;
   `scripts/provider_status.py` probes every AI each run and writes `provider_status_*.md/json`; the eval prints a per-provider table. Google AI is pinned to AI Mode (`udm=50`, `new_chat_url`).
   **Bug found live and fixed:** with three providers asked at once, background tabs swallowed the prompt and captured nothing (Gemini "answer captured but empty"); the tab that types is now in front and holds the
   focus lock until its prompt is sent (regression test with a fake lock).

## Test counts
- Offline gate (`-m "not browser"`): **395 passed** (was 367). Full `pytest -q` incl. the 27 real-Chrome fixture tests: **422 passed, 0 failed** (3m50s).
- New files: `test_early_stop.py` (10), `test_states.py` (5), `test_tiny.py` (3), `test_provider_pool.py` (5), additions to `test_threads.py`, `test_architecture.py`.

## Provider status (probe 2026-10-03 23:35, `provider_status_20261003_233535.md`, logged out, no bypass)
| provider | status | note |
|---|---|---|
| chatgpt | completed (10 s) | works |
| gemini | completed (8 s) | works; conversation id captured |
| google_ai | broken | AI Mode (udm=50) loads, accepts the prompt, page says "AI Mode response is ready" but shows only animated dots, no answer text for 40+ s (screenshot `data/artifacts/google_ai_udm50_after_submit.png`). No robot check/captcha appeared this time. Skipped per job. |
| copilot | logged_out | login wall, no composer |
| meta_ai | logged_out | login wall |
| le_chat | failed | composer accepts text, answer capture empty (selector/DOM issue or login); skipped for the rest of a job |
| pi | logged_out | login wall |
| qwen | failed | readiness=blocked (age gate; deliberately not filled in) |
| deepseek | logged_out | login wall |
Eval-run table (6 questions, `eval_20261004_000036.md`): chatgpt 11 asked / 11 completed; gemini 9 / 8; copilot 8 asked, 8 logged_out; le_chat 5 asked, 5 failed; google_ai 1 broken; qwen 1 failed; meta_ai, pi, deepseek not asked (escalation picks the best available, and 2 healthy AIs answered first).

## Eval table (all providers enabled; `eval_20261004_000036.*`; before = phase-1 run `eval_20261003_223808`/batch 5)
| question | verdict | latency now | before | notes |
|---|---|---|---|---|
| triv-2plus2 | PASS (computed) | 4 s | 4 s | 0 sessions, 0 curator calls |
| uk-dpa-age | PASS | 339 s* | 520 s (batch arch1), 220-417 s (earlier) | correct 13; 1 tiny sentence + 1 caveat; legislation.gov.uk s.9 opened by the AI and cited. *This run started before `ab69f34`, so it still did follow-up + 3 rounds; a re-run with the final code is in `console_arch3.txt` |
| uni-oxford-plagiarism | PASS | 280 s | 176 s "couldn't verify" (ox.ac.uk 403 to OmniBrain's audit) | now answered from pages the AI opened; was an honest don't-know before |
| uni-ucl-appeal | WARN | 325 s | 341 s "couldn't verify" | answered (four grounds); WARN: figure "27" without an opened page behind it |
| res-ramsey-r55 | PASS | 474 s | FAIL 409 s (missing 43/46) | 43-46 with attribution |
| ua-closed-session | FAIL (harness) | 442 s | not run | answer: no plenary session on 29 Sept 2026 (unn.ua page supported "plenary suspended"), no closed-session content. The harness marks any answer to this "no public answer" question as FAIL and flags figure 29; treat as a defensible-but-unsure result, not a pass |
| res-private-review | FAIL (harness) | 73 s | not run | stated the Nature article does not provide the reviewers' reports and quoted a critic; harness marks answering an "unanswerable" as FAIL |
Totals: 3 PASS, 1 WARN, 2 FAIL (both "unanswerable" traps). No uncertainty behaviour was weakened to raise PASS counts; no don't-know was converted into an answer by changing rules.

## Latency (hard questions, real sites, free curator)
Simple factual: seconds (2+2 = 4 s, level 0). Hard research now 280-474 s; the target "several minutes" is met, but not the "couple of minutes" for a single legal fact:
the follow-up and the 80 s wasted on Le Chat's two empty attempts per use (now skipped after the first job-level failure) were the main costs; the final early-stop fix has only been tested offline and in `console_arch3.txt`.

## Remaining gaps
- Early stop was proven offline and by the re-run noted above only; other sites' obedience to OPENED/MENTIONED labels is unverified (unlabelled stays `None`, never promoted).
- Le Chat (empty capture), Google AI Mode (no answer text), Copilot/Meta AI/Pi/DeepSeek (login walls), Qwen (age gate): blocked/broken logged out; re-probed every run, not bypassed.
- Curator on free models: 429s and invalid JSON remain possible; they are now reported as such and repaired when truncated.
- Provenance "INSPECTED" means: the AI said it opened it AND OmniBrain's audit of the same page loaded; OmniBrain cannot see what the AI actually read.
- The ChatGPT/Gemini answers' source URLs are the AI's text; a site that lists sources only in an unlabelled UI chip is under-counted.
- Two "unanswerable" traps still produce confident negative answers from the AIs' text (ua-closed-session, res-private-review).
