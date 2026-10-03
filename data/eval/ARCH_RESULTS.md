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
| uni-oxford-plagiarism | not finished when the report was written | | | | see console_arch1.txt if it completed |

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
