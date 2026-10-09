# Cloudflare, provider-inspected sources, and what we may count as evidence

Investigated 2026-10-09 after the Eiffel Tower runs kept answering "Couldn't verify that
one." while the AIs confidently cited the official site. Nothing here bypasses an access
control; the question is what we are allowed to know, and from whom.

## What actually happened live

Job `job_261009194837_4d524cb7` ("In which year was the Eiffel Tower completed, and when did
it open to the public?", STANDARD, real background Chrome, live driver):

- 18 evidence rows, **zero** of them `confirmed`. Nine sources were Cloudflare-gated
  (`toureiffel.paris`, `bie-paris.org`, `sete.toureiffel.paris`, `azertag.az`), one
  `broken_url`, several `unreachable` including `britannica.com`.
- 31 claims, all `unverified`. Answer: "Couldn't verify that one." -- correct under
  non-negotiable 5: we did not read anything that established it.
- Recorded conflicts were genuine, not artifacts: ChatGPT said the public opening coincided
  with the 1889 World's Fair, DeepSeek said it did not; a separate inauguration-date split
  (31 March vs 15 May) was graded minor. So the ledger was not silently flattening
  disagreement.

The gate is at the fetch layer, not the matcher: `fetch_page` tries HTTP, and the browser
re-read only exists for the *dedicated profile* engine. The opt-in live driver
(`browser.driver: chrome_use`) is deliberately restricted to provider sites
(`assert_allowed_url` in `browser/live_chrome.py`), so under that driver a blocked source has
no second path at all. That boundary stays: browsing arbitrary third-party sites through the
user's signed-in Chrome is not something to widen for an answer rate. The failure is now
recorded honestly (`blocked` plus "browser re-read gave no readable text") instead of the
earlier silent `unreachable`.

## Can text from a source the provider inspected be captured, with provenance?

Yes, and it already partly is. The DOM harvest (`browser/adapters/dom_library.py`,
`linksFrom`) keeps, per citation link: `href`, the label, and `snippet` = the innerText of
the surrounding source-card/`li`/`cite`, up to 400 chars. Live from that job:

- **DeepSeek renders quotations from the source.** Its captured citation text reads:
  `The BIE page for Expo 1889 Paris states: "The record-breaking structure was completed on
  31 March 1889 and the Tower opened to the public on 15 May" - 27`. We hold that string.
- **ChatGPT (logged out)** emits `url — OPENED (official Eiffel Tower website; read the
  page)`: a self-declared action, no quoted content.
- **Meta AI** rewrites links as `https://l.meta.ai/?u=<percent-encoded target>`. We fetched
  the redirector and logged `unreachable`. Unwrapping the `u=` parameter is the URL the
  provider itself embedded -- not a control bypass -- and we do not do it yet.

Provenance already distinguishes the two acts the question asks about. `Citation.provenance`
/ `Evidence.provenance` climb `MENTIONED -> OPENED -> INSPECTED -> CITED -> CLAIM_SUPPORTED`,
where `OPENED` means *the AI said it opened it* (its own label, never inferred) and `INSPECTED`
requires that **we** also opened it. The early-stop rule reads only `CLAIM_SUPPORTED`
(`runner.py:1080`), so a page we could not read can never settle a claim however loudly the AI
quotes it.

What is missing: a provider's quoted passage is currently used *only* to attribute a link to a
claim (`claim_for_link`), never as evidence content. The quote dies in `Citation.snippet`.

## Implemented 2026-10-09 (`31ac700`)

An evidence row per provider-quoted passage: `origin="provider_quote"`,
`check_status=NOT_CHECKED`, `verbatim_excerpt` = the quoted text exactly, the real source url /
title / tier, `cited_by=[provider]`, and a note naming who displayed it and in which
conversation (`quoted by deepseek in https://chat.deepseek.com/...`). It is filed under a claim
only when the displayed passage states that claim's own date and event about the same subject:
the live quote says "the record-breaking structure", never "Eiffel Tower", so it is recorded
unattached rather than borrowed.

Guaranteed by `tests/test_provider_quotes.py`: never `CONFIRMED`, never `CLAIM_SUPPORTED`, never
counted toward sufficiency or confidence (every counting rule requires `CONFIRMED`), never a
cited source in the answer (`attach_sources` requires `CONFIRMED`), and it cannot overwrite the
row for a page we tried to open -- `_merge_evidence` keeps quote rows in their own space, so a
`blocked` source stays `blocked`. The curator receives it labelled "provider-quoted, not opened
by us".

Why `NOT_CHECKED`: confirmed-counting (`pool.py`, `runner.py:1002`, `verifier.py:181`) and the
`provenance()` ladder all ignore it, so no threshold moved; the curator finally sees "DeepSeek
quotes the BIE page as saying X" as a labelled object instead of only inside a transcript, and
history shows which facts rest solely on a provider's word.

Risk to keep in view: a provider can misquote or invent a quote, so this is second-hand
documentation ("the provider says the source says"), never verification. If it ever becomes a
status that counts toward sufficiency, it weakens the one thing the system exists for.

Untested: whether opening a source card in a provider's own UI (ChatGPT's hover/click panel,
Gemini's source preview) exposes more text than the chip label. Driving those panels is normal
UI use, but it has not been probed live, so no claim is made about it.
