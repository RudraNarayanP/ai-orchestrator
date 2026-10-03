# OmniBrain hard-question evaluation report

Date: 2026-10-03 (Asia/Calcutta). Code: committed through `c233429` (+ eval result commits). Offline gate at report time: **350 passed, 27 browser tests deselected** (`pytest -m "not browser"`, 36 s). The full suite including browser tests was not re-run in this session.

Stack under test: real browser providers (ChatGPT, Gemini anonymous, search), OpenRouter free models (verifier `nvidia/nemotron-3-super-120b-a12b:free`, analysis `poolside/laguna-s-2.1:free`), STANDARD mode, 2 rounds. Question set: `scripts/eval_questions.yaml` (17 questions). Raw results: `data/eval/eval_*.json|md`, `data/eval/console_*.txt`.

## 1. Question vs outcome (latest run per question)

| Question id | Kind | Latest outcome | Verdict | Time | Primary opened / cited |
|---|---|---|---|---|---|
| uk-dpa-age | answer | 13 years, high confidence (correct) | WARN: no primary cited | 220 s | 5 / 0 |
| uk-theft-s1 | answer | s.1(1) "dishonestly appropriates property belonging to another with intent to permanently deprive" | WARN: no primary cited | 417 s | 6 / 0 |
| uk-vpn-ban | false premise | "No section of the Online Safety Act 2023 bans VPNs" (correct) | WARN: no primary cited | 225 s | 6 / 0 |
| uk-future-amendment | unanswerable | Reasoned "I don't know": about 2028, hasn't happened | PASS | 565 s | 6 / 0 |
| ua-marriage-age | answer | 18 for both sexes, court may permit lower (correct) | WARN: no primary cited | 213 s | 0 / 0 |
| ua-russian-state-language | false premise | "I couldn't verify this reliably" (sources were about other countries) | WARN: IDK on answerable | 226 s | 0 / 0 |
| ua-constitution-year | answer | 28 June 1996, high confidence (correct) | WARN: no primary cited | 313 s | 0 / 0 |
| uni-oxford-plagiarism | answer | "I couldn't verify" (ox.ac.uk returned 403 to plain HTTP) | WARN: IDK on answerable | 197 s | 6 / 0 |
| uni-oxford-dphil-length | answer | 100,000 words (Social Sciences), high confidence | **PASS** | 582 s | 6 / 4 |
| uni-manchester-viva-pass-rate | unanswerable | Honest "I couldn't verify" | PASS | 465 s | 6 / 0 |
| res-wiles-poincare | false premise | "Wiles did not prove Poincare; his 1995 work proved Fermat's Last Theorem" | WARN: no primary cited | 157 s | 0 / 0 |
| res-ten-percent-brain | myth | "No, a myth" | WARN: no primary cited | 301 s | 0 / 0 |
| res-recovery-dexamethasone | answer | Dexamethasone lowered 28-day mortality in ventilated/oxygen patients | WARN: no primary cited | 157 s | 2 / 0 |
| uni-ucl-appeal | answer | batch 5, was still running when this report was written; see section 9 | - | - | - |
| res-ramsey-r55 | answer | batch 5, was still running when this report was written; see section 9 | - | - | - |
| ua-closed-session | unanswerable | **NOT RUN** | - | - | - |
| res-private-review | unanswerable | **NOT RUN** | - | - | - |

Coverage: 13 of 17 finished, 2 in flight, 2 never run. Of the 13: 3 PASS, 10 WARN, **0 FAIL on the latest code**. All 13 factual answers are correct or honestly "I don't know"; no wrong answer remained in the final runs. The WARNs are almost all "no primary source cited" (see residual risk).

## 2. Primary sources opened

- UK law: legislation.gov.uk pages were opened for DPA 2018, Theft Act 1968 and the Online Safety Act (5-6 each, 33-69 pages confirmed in total), but none was listed among the final cited sources in the last runs.
- Oxford DPhil: law.ox.ac.uk, examregs.admin.ox.ac.uk, mpls.ox.ac.uk, philosophy.site.ox.ac.uk; all 4 cited. This is the only question with primary sources in the final answer.
- Ukrainian law (zakon.rada.gov.ua): unreachable from this PC (connect timeout), 0 primary pages in every run.
- ox.ac.uk plagiarism pages: 403 to plain HTTP, so the question degrades to IDK.
- RECOVERY: 2 primary pages opened, not cited.

## 3. Unsupported claims that slipped through (all found and fixed)

1. Oxford plagiarism: "Oxford's official guidance..." stated confidently with no Oxford page opened (early run). Fixed by the subject-name check (`26aa2fa`); now an honest IDK.
2. Ukraine marriage age: "17 for women / 14", high confidence, sourced to California, Singapore and a UK SI. Caught by the unsupported-figures check; fixed by `26aa2fa`, `1e097ea`. Latest run says 18, correct.
3. Manchester viva pass rate: high-confidence irrelevant regulations facts plus "Today's date is...". Fixed by the subject and relevance checks; now an honest IDK.
4. Future amendment: answered with Employment Rights Act 2025 facts about a 2028 question. Fixed by `79267bb`.
5. Cross-question contamination: Gemini's chat returned the previous question's answer. Fixed by `781cf39` (off-topic reply is dropped and the provider counts as dead).
6. Theft Act s.1 answered from model memory in 8 s with nothing opened. Fixed by `6a8a829` (statute questions always route to research).

No unsupported figure was flagged in the final runs (`unsupported_figures` is empty for all records I read).

## 4. Voice and emoji examples

After:
- "No section of the Online Safety Act 2023 bans the use of VPNs in the UK; ... ⚖️"
- "Andrew Wiles did not prove the Poincaré conjecture; his 1995 work proved Fermat's Last Theorem. ✅ ..."
- "No, the idea that humans use only 10% of their brains is a myth; ... 🚫"
- "The current Constitution of Ukraine was adopted on 28 June 1996. ✅"
- Don't-know stays plain, no emoji: "I don't know. That's about 2028, which hasn't happened yet, so nothing published can say." and "I couldn't verify this reliably. None of the pages I opened confirms an answer, so I won't guess."

Caveats, before vs after `plain_caveats` (`0c7b88c`, `76047b3`): before "verifier model unavailable (...); verdicts come from the evidence ledger", raw `clm_...` ids, "The AI answers said I don't know..."; after "The AI reviewer wasn't available, so this rests only on the pages I could open and check." or the item is dropped, and "One AI answer claimed: "..." I couldn't confirm that from any page I opened."

Voice nits still visible: some answers repeat a sentence; an emoji sometimes follows the first sentence rather than ending the answer.

## 5. Defects fixed (each with a regression test)

| Commit | Defect | Fix |
|---|---|---|
| f1b10c8 | Evidence only filed under one claim, "No claim survived" | Evidence attributed to every claim its page text supports (`tests/test_attribution.py`) |
| 0c7b88c | Internal bookkeeping in caveats | `plain_caveats()` |
| 781cf39 | Stale provider reply for the wrong question | `off_topic()` guard (`tests/test_off_topic.py`) |
| e4d5792 | Truncated JSON from the model; supported claims missing the asked attribute | Retry with doubled max_tokens; `addresses_question()` |
| 26aa2fa | A page about another institution confirmed a claim | `subject_names()` and subject check (`tests/test_subject_match.py`) |
| 79267bb | Future-year question answered with present facts | `future_year()` |
| 60cbbc7 | Risk of the API key in the temp eval config | Key passed only via the child's env; test added |
| 76047b3 | Don't-know draft quoted back in caveats | Wording and filter |
| 6a8a829 | Law-text questions answered from memory | `STATUTE_RE` route (`tests/test_statute_route.py`) |
| 1e097ea | Verdict JSON cut off, model-free fallbacks, off-subject pages as evidence | Claim cap of 12, short reasoning, `require_names` check |
| f38b757 | Invented claim id dropped a supported verdict | `_nearest_claim_id` (Jaccard >= 0.6) |
| c233429 | Model under-rated claims the ledger confirmed, giving a false "I couldn't verify" | Ledger lifts claims and rebuilds the answer. Verified live: uk-dpa-age went from "I couldn't verify" to 13 years; uk-theft-s1 from "I couldn't verify" to the s.1(1) definition. |

## 6. Test counts

350 non-browser tests pass at HEAD, 27 browser tests deselected (about 377 collected). README test counts (~365/338) are slightly stale.

## 7. Residual risk

- Free-model variance: the same question can flip between a correct answer and an honest IDK across runs. Over-conservative IDK on answerable questions (ua-russian-state-language, uni-oxford-plagiarism) remains.
- Primary pages opened but not cited: for uk-dpa-age, uk-theft-s1 and uk-vpn-ban, legislation.gov.uk pages were opened (5-6) yet the final answer cites none, so the eval WARNs. This is the main open defect; the answers themselves were correct.
- Several correct answers (Wiles, brain myth, constitution year, marriage age) have no primary source at all.
- `check_support` cannot tell a number's direction (a page saying "16 years are to be read as 13" contains "16").
- Environment: zakon.rada.gov.ua unreachable, ox.ac.uk 403 to plain HTTP, Gemini anonymous hits a login wall after several prompts; Google AI Mode and Copilot are blocked and skipped. Ukrainian primary law cannot be checked from this PC.
- Gemini round-2 responses are repeated up to 5-6 times per round (dedup not done).
- Runtime 157-582 s per question, so the eval must run in the background in small batches.
- The 13-question sample is small. These are not statistical guarantees.
- Secret hygiene: the OpenRouter key lives only in gitignored `config/settings.yaml`; temp eval configs hold no key.

## 8. Not run

`ua-closed-session`, `res-private-review`. `uni-ucl-appeal` and `res-ramsey-r55` were in flight (batch 5); see section 9.

## 9. Batch 5 (in flight)

`data/eval/eval_20261003_122404.json` currently holds uk-dpa-age and uk-theft-s1; uni-ucl-appeal and res-ramsey-r55 are appended when finished.

## 10. Re-run

```
.venv\Scripts\python.exe scripts/eval_hard.py --only uk-dpa-age,uk-theft-s1
.venv\Scripts\python.exe scripts/eval_hard.py --category university
.venv\Scripts\python.exe scripts/eval_hard.py --limit 4
```
Run in the background with output redirected to `data/eval/console_<name>.txt`, and poll. Offline gate: `.venv\Scripts\python.exe -m pytest -q -m "not browser"`.