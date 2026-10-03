# Third-party reference and derived code

OmniBrain is not a clone of anything. These repositories were fetched, **read in
full where relevant**, and used in three distinct ways, kept separate on purpose
because two of them carry no licence at all.

Vendored under `third_party/` as shallow clones for inspection. They are
reference material, not build dependencies -- nothing in `backend/` or `browser/`
imports from them.

| Repository | Licence | Last commit | Status |
|---|---|---|---|
| `jumas45/no-api-llm-council` | **MIT** (verified in `LICENSE`) | 2026-07-22 | code ported |
| `AmT42/agent-council-browser` | **MIT** | 2026-05-03 | code + thresholds ported |
| `paritoshv/multi-llm-web-orchestrator` | **MIT** | 2026-06-16 | code ported |
| `karpathy/llm-council` | **NO LICENCE FILE** | 2025-11-22 | ideas only, no code read past the design |
| `UditSankhadasariya/LLM-council` | **NO LICENCE FILE** | 2026-04-27 | ideas only |

A repository without a licence is *all rights reserved* by default. A README line
saying "MIT" is not a grant. Nothing from the last two rows was copied; their
READMEs and publicly documented approach informed decisions that are re-implemented
here in Python under our own licence.

`Tencent/BrowserSkill` (MIT, active) was surveyed for its Chrome/CDP bridge and
network-capture design but shares no code path with OmniBrain's DOM-first layer, so
nothing was taken.

## What was ported, and from where

**`browser/adapters/selectors.py`**
- Ordered candidate ladders per field per domain (input / send / stop / response),
  so one stale selector degrades instead of breaking: `jumas45` →
  `src/content/selectors.js`.
- ChatGPT additions `#composer-submit-button`, `textarea[data-id]`,
  `button[aria-label="Stop streaming"]`; Gemini `button.send-button.stop`,
  `.stop-icon`, `message-content .markdown`, `.model-response-text`: same source.
- The `quick_answer` ("Answer now" / "fast answer") field: `jumas45` ADR-0004.
  Clicked only in QUICK mode, never during a verification round.
- Per-provider capture thresholds (ChatGPT 3s stable / 4.5s tiny fragment / 12s
  if no generating UI seen / 9s reference-like / 45s force / 60s reference force /
  180s hard; Gemini 1.2s / 15s / 120s): `AmT42` →
  `AGENTSMD/AGENTS_ANSWER_DETECTION.md`.

**`browser/adapters/dom_library.py`**
- `cleanAssistantText()` -- strips `Copied`, `Regenerate`, "`<provider> said`"
  leakage: `paritoshv` → `content-helpers.js`.
- `unwrap()` for text-bearing buttons instead of removing them. Deleting suggestion
  chips turned sentences into "I can also , , or .": `jumas45` capture code.
- Scanning the last *N* assistant roots from a pre-send baseline count, and treating
  a trailing "Sources" block as reference-like with a longer stable window; scoping
  to `[role="log"]` and excluding `<aside>` / `[role=complementary]`: `AmT42`.
- Visibility via `computedStyle` + `getBoundingClientRect` rather than `offsetParent`,
  which misreports fixed/overlay controls: `AmT42`.
- Doubled thresholds while `document.visibilityState === 'hidden'`, plus a focus
  nudge, because background windows throttle rendering and truncate answers:
  `AmT42` + `jumas45`.
- Media-only-answer placeholder so a capture returns a stated fact instead of an
  empty string: `AmT42`'s Gemini image handling.
- Sanitised, length-capped model-label sniffing, deliberately informational only:
  `AmT42`'s `PROVIDER_MODEL_HINT` rules.

**`browser/adapters/base.py`**
- Completion-decision order: quiet-and-idle wins → a stuck "generating" flag can
  only delay to `stuck_ms` → nothing outruns the force budget → hard timeout last.
  `jumas45` → `waitResult()` constants (`STABLE_MS`, `STUCK_MS`, `ACTIVE_MS`,
  `OVERRUN_FACTOR`).
- `no-response-element` as a distinct failure meaning *selector drift*, reported as
  BROKEN rather than retried into the wall: `jumas45`.
- Typing cascade verified by reading the value back after each method:
  `setNativeValue` → `execCommand('insertText')` → synthetic `ClipboardEvent`
  `paste` → raw `textContent`. `jumas45`. (OmniBrain tries genuine CDP keyboard
  input first, which is more reliable against React and less spoofy.)
- Prompt-echo stripping so a provider's echo of our own prompt is never counted as
  a second source: `AmT42` / `jumas45` sanitizers.

**`scripts/probe.py`**
- Persistent-profile selector probe to revalidate the ladders after a site redesign:
  `jumas45` → `scripts/probe-selectors.mjs`.

**Design, not code**
- Three-stage council (independent answers → anonymised peer review → chair
  synthesis) as a *pattern*: `karpathy/llm-council` (unlicensed) and
  `jumas45` (MIT). OmniBrain deliberately does **not** implement voting or
  ranking, because the brief's central rule is that evidence beats consensus.
- Fencing untrusted model output before it reaches a verifier: `jumas45` ADR-0007.
- One fresh conversation per research stage so a provider cannot simply agree with
  its own previous answer: `jumas45` ADR-0012.

## Licence position for OmniBrain itself

OmniBrain is a private, local, single-user tool. Every port above came from MIT
code, so it would remain fine even if the app were distributed. No copyleft source
was used. The two unlicensed repositories contributed no code.

MIT requires the copyright notice and permission notice to be retained in
substantial copies of the software; the derived fragments here are attributed above
and the upstream `LICENSE` files stay in `third_party/<repo>/LICENSE` alongside the
clone they came from.
