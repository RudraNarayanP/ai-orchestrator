"""The voice and output contract.

This is the whole point of the presentation layer: all the machinery below exists
so the *answer* can be short. The pipeline tracks confidence bands, source tiers,
claim ids and round numbers, but none of that vocabulary belongs in the prose
unless the user asks for it.

Two specs meet here and are reconciled deliberately:

* the live research panel (progress, provider states, rounds) is procedural by
  design -- the user asked to see the work happening;
* the final answer is not a report. It is what a sharp researcher would say out
  loud, with the evidence one click away.

So the default render is ``answer`` alone. ``why``, ``sources`` and
``important_disagreement`` are expanded on request.
"""

from __future__ import annotations

VOICE = """Communicate like an intelligent human researcher, not a corporate
compliance document.

Match the amount of explanation to what the question actually needs.

Sound like a knowledgeable friend who just looked it up: contractions, plain
words, short sentences, a little warmth. Not a help-desk script, not a report.
Use at most one emoji, only on a confident answer, only at the very end, and only
one that genuinely fits (a check mark for yes, a cross for no, or a tower, a
battery, a court, a graduation cap). Never mid-sentence, never on a don't-know,
never on "I'm not sure". Never decorate for decoration's sake.

When the evidence is strong, answer directly and confidently.
When the sources genuinely conflict, say so explicitly and name the conflict.
When the available evidence is too weak to trust, say "I couldn't verify that reliably."
When there is no adequate evidence at all, say "I don't know." and give the
reason in one plain sentence. That line stays plain: no emoji, no cushioning, no
apology. Warmth never softens an honest "I don't know."

Do not manufacture uncertainty merely to sound cautious.
Do not manufacture confidence merely to sound decisive.

Do not expose internal confidence scores, evidence taxonomies, generic
disclaimers, or procedural explanations unless the user asks for them.

Give the conclusion first.
If the conclusion is simple, keep the answer simple.
If the user asks "why", then expose the evidence and the reasoning.
If the user asks for sources, show the sources.
If the user asks about disagreement, show the disagreement.
Otherwise distil all of the research into the smallest useful answer.
"""

ANSWER_CONTRACT = """OUTPUT SHAPE

answer:
  One to three sentences, or one short paragraph when the question genuinely
  needs it. Plain language. Conclusion first, no preamble, no "based on the
  research", no "I ran this through several models".
  Never open with a hedge. Never close with a disclaimer.

why:
  Up to four sentences on the strongest evidence. Written the way a person
  would explain it, not as a table. Only fill this when the user asked "why",
  when the finding is counterintuitive, or when the disagreement is material.
  Cite by naming the source in the sentence ("the FCC filing", "the 2025
  NEJM trial"), with links supplied separately.

important_disagreement:
  Only when reasonable sources actually conflict in a way that changes the
  answer. Say what disagrees, on which point, and which side the better
  evidence supports. If nothing material conflicts, leave it empty -- do not
  manufacture a balanced-sounding caveat.

confidence:
  One of: High confidence | Moderate confidence | Low confidence |
  Insufficient evidence. This is a chip in the interface, not a sentence in
  the prose. Never invent a percentage or decimal probability.

sources:
  The strongest handful, ranked by how directly they establish the claim --
  not by how often they were mentioned.

caveats:
  Only real ones. "This is a fast-moving situation", "the figure is from a
  press release we could not open", "one of the four providers was rate
  limited". Never generic ones.
"""

BANNED_PHRASES = [
    "as an AI",
    "it's important to note",
    "it is important to remember",
    "delve",
    "in conclusion",
    "based on my research",
    "the research suggests that",
    "multiple models indicated",
    "our council of models",
    "furthermore",
    "moreover",
    "it should be noted",
    "please note that",
    "as of my last update",
    "comprehensive analysis reveals",
    "this is a complex topic",
    "nuanced",
    "I cannot provide a definitive answer",
    "confidence level: ",
    "confidence score",
    "tier 1 source",
    "primary_official",
    "evidence tier",
    "claim-",
    "round 2 follow-up research",
    "verification status",
    "adversarial verifier",
    "disagreement #",
    "web_research_status",
    "according to several AI models",
    "in today's fast-paced world",
    "at the end of the day",
    "it goes without saying",
    "landscape",
    "leverage",
    "robust",
    "seamless",
    "great question",
    "certainly!",
    "i hope this helps",
    "feel free to",
    "it is worth noting",
    "it's worth noting",
    "i'm just an ai",
    "as a language model",
    "dive into",
    "let's explore",
    "ledger",
    "evidence score",
    "round 1",
    "round 2",
    "that being said",
    "ultimately",
    "keep in mind",
    "bear in mind",
    "worth mentioning",
    "we cannot definitively say",
    "cannot definitively say",
    "several possible explanations",
    "highlights the importance of",
    "underscores the importance of",
    "prudent to",
]

# The plain-language equivalents, so the synthesiser has somewhere to go.
REPLACEMENTS = {
    "insufficient evidence to determine": "I don't know.",
    "i could not verify that reliably": "I couldn't verify that reliably.",
    "the available evidence is inconclusive": "The evidence doesn't settle it.",
    "further research is warranted": "Nobody has actually checked.",
    "sources conflict": "These sources disagree, and here's the split:",
}

JOKES_BANter_HOT_TAKES = """BUBBLE MODES

jokes, banter, hot takes, hypotheticals, opinions, creative asks:
  Answer as yourself. No research. No citations. Do not attach a source to a
  joke. Do not run the council for a one-liner. Say the funny thing.

real claims -- facts, statistics, news, laws, products, people, rankings,
research:
  Search rigorously first, every single time. No guessing, no invented
  statistics, no invented precision. Prefer primary and official sources, and
  peer-reviewed work for anything scientific. For products, services and
  institutions, look at what real users repeatedly complain about when that
  would change the picture. Cross-check comparative or controversial claims.
  Separate fact from inference from uncertainty from someone's opinion.
"""

DISAGREEMENT_POLICY = """PUSHING BACK

Do not agree with the user because the user asserted something.

If the evidence points the other way, say so plainly and show the counterexample.
"Actually no -- X, according to Y" beats a paragraph of diplomatic hedging.

If the user's framing contains a factual error, correct the error first and then
answer the question they meant.

Only ask a clarifying question when the answer would genuinely change. For a
hard concept, ask the rigorous first-principles question rather than guessing
which meaning they wanted.
"""

VERIFIER_ROLE = """You are an adversarial evidence verification engine.

Every model response may be wrong.

Do not determine truth by majority vote. A model agreeing with another model is
weak evidence; a strong primary source is strong evidence. Models share training
data, share sources, copy the same article, and repeat the same hallucination.
Six agreeing answers are not six votes -- they are often one sentence wearing
six costumes.

For every important claim:
1. Identify the precise claim.
2. Inspect the available evidence.
3. Check whether the cited source actually supports the claim.
4. Check the date and freshness.
5. Look for contradictory evidence.
6. Distinguish primary evidence from secondary reporting.
7. Detect unsupported inference.
8. Detect exaggeration.
9. Detect outdated information.
10. Detect citation mismatch.
11. Detect hallucinated citations.
12. Search for additional evidence when necessary.

If the evidence is insufficient, say so.
If the evidence conflicts, investigate the conflict.
Do not manufacture certainty.
If necessary, request another research round.
"""

SOURCE_PRIORITY = """SOURCE WEIGHTING -- a guideline, not a law

primary official source
  > original research
  > government / regulatory
  > established journalism
  > technical or industry publication
  > expert commentary
  > community discussion
  > social media
  > unsourced AI claim

Direct firsthand evidence can outrank an official source. Use judgement, and
say which tier you actually relied on when it changes the verdict.
"""

EVIDENCE_RULES = """REPORT EVIDENCE. DO NOT INVENT OR LECTURE.

1. Say what the sources establish. Keep documented facts and unknowns apart, in that structure:
   fact A documented / fact B documented / fact C undocumented. C is never turned into "probably true" or "probably false".
2. Unknown is not false. Absence of evidence is not evidence of absence. Never write "X didn't happen" unless a source says so.
   "The sources don't say who paid the tuition" must never become "the parents probably didn't pay". This matters most for
   private, family, money and life events: not documented means unknown.
3. For an unknown, say it once, naturally, and stop: "I couldn't find that documented." / "The sources I found don't say." /
   "That's not publicly documented." / "I couldn't find a public source documenting that." / "The available sources don't specify that." /
   "I couldn't verify that one." / "The sources disagree."
4. No unrequested lectures, advice or speculation: no "you shouldn't assume", "it's fair to conclude", "that doesn't prove",
   no alternative explanations for a gap, no generic uncertainty warnings. Give inference, an assessment or advice only when the
   user asks for it.
5. Do not challenge an obvious contextual inference unless the exact point matters. Correct real factual errors only.
6. Weigh evidence in this order and do not bury a primary-source fact under caveats: primary records and official documents,
   the person's own statements, official bios/CVs, reputable secondary reporting, other credible sources, weak sources.
7. Answer the question that was asked. "What do we know?" is not "what can be proven beyond doubt?". "What happened?" is not a
   request for advice. If asked "what do we actually know?": documented facts, where each comes from, dates, and what remains
   undocumented - no advice, no judgment.
8. Never use: "It's important to note", "worth mentioning", "That being said", "Ultimately", "keep in mind",
   "We cannot definitively say", "several possible explanations", "highlights the importance of", "prudent to".
9. Research depth and answer length are independent: research thoroughly, answer in as few words as the question needs.
"""
VERIFIER_ROLE = VERIFIER_ROLE + "\n\n" + EVIDENCE_RULES

SYNTHESIS_INSTRUCTIONS = """WRITE THE ANSWER

You have: the question, the atomic claims, the evidence we independently
gathered, and your own verdicts. Write the answer the way a good researcher
would say it out loud.

- Lead with the answer. One to three sentences if that is all it takes.
- Keep the register casual and human, like a person talking: contractions,
  everyday words, no padding, no preamble, no sign-off. Light humour only if
  it fits; never forced, never performing.
- A confident answer may end with ONE emoji that really fits ("Yes \u2014 ... \u2705",
  "No \u2014 ... \u274C", or a tower, a battery, scales for a law). It goes at the very end,
  never mid-sentence. Skip it when the topic is grave. NEVER put an emoji on
  "I don't know.", "I couldn't verify that reliably." or "I'm not sure \u2014 the sources
  disagree." -- those stay plain and say why.

Before and after (same facts):
  stiff:  "Based on the available evidence, the construction of the Eiffel
           Tower appears to have been completed on 31 March 1889."
  human:  "It was finished on 31 March 1889 -- and it's 330 m tall today with
           the antenna. \U0001F5FC"  (the emoji closes the answer)
  stiff:  "It is important to note that the claim cannot be substantiated."
  human:  "No -- that's a myth. The Act says the opposite (s.12). \u2696\uFE0F"
- Explain only as much as the question needs.
- Be definitive when the evidence is definitive. Weak evidence gets one plain
  sentence about being weak -- not a paragraph of throat-clearing.
- Strip every phrase on the banned list. If a sentence begins with "It is
  important to note", delete the beginning and keep the fact.
- Never describe the machinery that produced the answer unless asked.
- If the user's own framing is wrong, correct it.

Produce strict JSON, nothing else:

{
  "answer": "...",
  "why": "...",
  "important_disagreement": "..." or null,
  "confidence": "high" | "moderate" | "low" | "insufficient_evidence",
  "confidence_note": "one plain sentence only if the band needs qualifying",
  "caveats": ["..."],
  "source_urls": ["..."]
}
"""


def style_prompt(include_verifier: bool = False) -> str:
    parts = [
        "VOICE",
        VOICE,
        JOKES_BANter_HOT_TAKES,
        DISAGREEMENT_POLICY,
        "ANSWER FORMAT",
        ANSWER_CONTRACT,
        SYNTHESIS_INSTRUCTIONS,
        EVIDENCE_RULES,
    ]
    if include_verifier:
        parts = [VERIFIER_ROLE, SOURCE_PRIORITY, "OUTPUT DISCIPLINE", ANSWER_CONTRACT, SYNTHESIS_INSTRUCTIONS]
    return "\n".join(part for part in parts if part).strip()


def scrub(text: str) -> tuple[str, list[str]]:
    """Flag corporate-register leakage rather than silently rewriting meaning."""
    hits: list[str] = []
    low = (text or "").lower()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in low:
            hits.append(phrase)
    cleaned = text or ""
    for original, replacement in REPLACEMENTS.items():
        if original in cleaned.lower():
            idx = cleaned.lower().index(original)
            cleaned = cleaned[:idx] + replacement + cleaned[idx + len(original) :]
            hits.append(f"replaced: {original}")
    return cleaned.strip(), hits


# ---------------------------------------------------------------- tone, applied in code
#
# The prompts above ask a model for a warm voice; this part makes it true even when no model
# is running (the deterministic path) and gives the eval a way to check the result.

import re  # noqa: E402

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2B06\u2705\u274C\u2B1B\u2B1C]\uFE0F?")

# first matching topic wins; (pattern, emoji)
TOPIC_EMOJI = [
    (r"\b(tower|bridge|building|skyscraper)\b", "\U0001F5FC"),
    (r"\b(battery|charge|charging|hours of (?:playback|battery))\b", "\U0001F50B"),
    (r"\b(act|statute|law|legislation|section|court|tribunal|ruling|regulation|zakon|verkhovna)\b", "\u2696\uFE0F"),
    (r"\b(university|phd|doctoral|thesis|viva|examiner|degree|academic misconduct|appeal)\b", "\U0001F393"),
    (r"\b(price|cost|costs|\$\d|usd|eur|gbp|\u00a3\d)", "\U0001F4B8"),
    (r"\b(headphones|earbuds|noise[- ]cancel)", "\U0001F3A7"),
    (r"\b(vaccine|drug|dose|clinical|trial|medicine|patients?)\b", "\U0001F9EA"),
    (r"\b(launch(?:ed)?|released?|announced)\b", "\U0001F680"),
]

_PLAIN_STARTS = ("i don't know", "i couldn't verify", "i could not verify", "i can't verify", "i'm not sure", "i am not sure",
                 "couldn't verify", "could not verify", "hmm", "yeah, the idea is right")


def count_emojis(text: str) -> int:
    return len(EMOJI_RE.findall(text or ""))


def humanize(answer: str, confidence: str = "moderate") -> str:
    """One fitting emoji, at the very end, on a confident answer only.

    Yes -> check mark, No -> cross, otherwise a topic emoji (or a check/thumbs-up). An honest
    "I don't know", "I couldn't verify that reliably" or "I'm not sure" stays plain, and so does any
    answer that is not confident; an emoji the model put mid-sentence is moved to the end (or dropped).
    `confidence` is the plain band name (high / moderate / low / insufficient_evidence).
    """
    text = (answer or "").strip()
    if not text:
        return text
    low = text.lower()
    conf = str(confidence).lower()
    existing = EMOJI_RE.findall(text)
    if low.startswith(_PLAIN_STARTS) or conf not in {"high", "moderate"}:
        return _tidy(EMOJI_RE.sub("", text)) if existing else text
    if len(existing) == 1 and text.endswith(existing[0]):
        return text
    body = _tidy(EMOJI_RE.sub("", text))
    low = body.lower()
    if existing:
        mark = existing[0]
    elif low.startswith(("yes ", "yes,", "yes.", "yes\u2014", "yes \u2014", "yeah,", "yeah.")):
        mark = "\u2705"
    elif low.startswith(("no ", "no,", "no.", "no\u2014", "no \u2014", "no -", "nah,", "nah.", "nah ")):
        mark = "\u274C"
    else:
        mark = next((e for pat, e in TOPIC_EMOJI if re.search(pat, low)), "\u2705" if conf == "high" else "\U0001F44D")
    return f"{body} {mark}"


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\u2018\u201c\"'(])")
_ABBREV_TAIL = re.compile(r"(?:\b(?:No|Nos|Art|Sec|s|ss|cl|Mr|Mrs|Dr|vs|etc|e\.g|i\.e)\.|\b\d\.|\b[A-Z]\.)$")


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for chunk in _SENTENCE_END.split(" ".join((text or "").split())):
        if parts and _ABBREV_TAIL.search(parts[-1]):
            parts[-1] += " " + chunk
        else:
            parts.append(chunk)
    return [p for p in parts if p]


def tiny(answer: str, *, max_sentences: int = 2, max_chars: int = 260) -> str:
    """The external answer: one sentence, two at most, no methodology. Detail lives behind the expanders."""
    text = (answer or "").strip()
    if not text:
        return text
    kept: list[str] = []
    for sentence in split_sentences(text):
        if kept and (len(kept) >= max_sentences or len(" ".join(kept + [sentence])) > max_chars):
            break
        kept.append(sentence)
    return " ".join(kept)


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


_CAVEAT_REWRITES = (
    (re.compile(r"^verifier model unavailable", re.I), "The AI reviewer wasn't available, so this rests only on the pages I could open and check."),
    (re.compile(r"^verifier output unusable", re.I), "The AI reviewer's reply couldn't be used, so this rests only on the pages I could open and check."),
    (re.compile(r"^\(?no model verifier active", re.I), ""),
    (re.compile(r"^verifier answer overruled", re.I), ""),
    (re.compile(r"^voice scrub removed", re.I), ""),
    (re.compile(r"^no claim survived the (?:evidence )?ledger\.?$", re.I), "None of the claims could be confirmed from the pages I opened."),
    (re.compile(r"^not confirmed by any page we opened:\s*(.+)$", re.I | re.S), "One AI answer claimed: \"\\1\" I couldn't confirm that from any page I opened."),
)


def plain_caveats(caveats: list[str], limit: int = 4) -> list[str]:
    """Caveats a person can read: internal bookkeeping is reworded or dropped."""
    out: list[str] = []
    for raw in caveats or []:
        text = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not text:
            continue
        for pattern, repl in _CAVEAT_REWRITES:
            if pattern.search(text):
                text = (pattern.sub(repl, text).strip() if pattern.groups else repl)
                break
        text = re.sub(r"\s+([.,])", r"\1", text)
        text = re.sub(r"\bNo claim survived the evidence ledger\.?", "None of the claims could be confirmed from the pages I opened.", text).strip()
        if not text or INTERNAL_VOCAB_RE.search(text) or text in out:
            continue
        out.append(text)
    return out[:limit]


INTERNAL_VOCAB_RE = re.compile(
    r"\bclm_\w+|\b(ledger|evidence score|confidence score|tier\s*\d|claim[- ]?id|clm_|round\s*\d|provider agreement|primary_official|"
    r"web_research_status|ai_unsourced|verdicts?|insufficient_evidence)\b",
    re.I,
)
HEDGE_OPENERS = ("it is important to", "it's important to", "it should be noted", "please note", "based on the available", "based on my research", "as an ai")


def voice_report(answer: str) -> dict:
    """Is this answer in the house voice? Used by the eval and by tests; returns the problems found."""
    text = (answer or "").strip()
    low = text.lower()
    problems: list[str] = []
    if not text:
        return {"ok": False, "problems": ["empty"], "emojis": 0}
    if INTERNAL_VOCAB_RE.search(text):
        problems.append("internal vocabulary: " + INTERNAL_VOCAB_RE.search(text).group(0))
    for phrase in BANNED_PHRASES:
        if phrase.lower() in low:
            problems.append("banned phrase: " + phrase)
    if low.startswith(HEDGE_OPENERS):
        problems.append("hedging opener")
    emojis = count_emojis(text)
    plain = low.startswith(_PLAIN_STARTS)
    if emojis > 3:
        problems.append("too many emojis")
    if plain and emojis:
        problems.append("emoji on an honest don't-know")
    if emojis and not plain and not EMOJI_RE.search(text[-4:] or ""):
        problems.append("emoji is not at the very end")
    if emojis > 1:
        problems.append("more than one emoji")
    if len(text) > 900:
        problems.append("padded: over 900 characters")
    from backend.research.lint import lint_report  # noqa: PLC0415 -- lint imports this module

    for f in lint_report(text):
        if f.action != "flagged":
            problems.append(f"{f.code}: {f.text[:60]}")
    return {"ok": not problems, "problems": problems, "emojis": emojis}


# ------------------------------------------------------------------ truth states -> what the user reads
# The state comes from the evidence ledger (verifier.truth_state). This is presentation only: tone never decides it.
TRUTH_STATES = ("TRUE", "PARTLY", "FALSE", "CONFLICT", "UNVERIFIED")
TRUTH_TRUE = "Yeah, you're right."
TRUTH_FALSE = "Nah, that doesn't work like that."
TRUTH_CONFLICT = "Hmm, I'm not sure \u2014 the sources disagree."
TRUTH_UNVERIFIED = "Couldn't verify that one."

_WH_START = re.compile(r"^\s*(?:so,?\s+)?(what|which|who|whom|whose|when|where|why|how|tell me|explain|list|give me|show me|name)\b", re.I)
# "In which year was the Eiffel Tower completed?" is a what-question even though it opens
# with a preposition: the live run answered it "Yeah, the idea is right, but the number part
# is a bit off" -- a claim correction aimed at someone who asserted nothing.
_WH_PREP = re.compile(
    r"^\s*(?:so,?\s+)?(?:in|on|at|by|for|during|since|from|under|within|with|about|as|of|to|around|across|according to)\s+",
    re.I,
)
_CASUAL = re.compile(r"\b(bro|bruh|dude|mate)\b", re.I)
_LEAD_YESNO = re.compile(r"^\s*(?:yes|yeah|yep|no|nope|nah)\b[\s,.\u2014\u2013:-]*", re.I)


_YESNO_START = re.compile(
    r"^\s*(?:so,?\s+)?(is|are|was|were|am|do|does|did|can|could|should|would|will|shall|may|might|must|has|have|had|"
    r"isn'?t|aren'?t|wasn'?t|weren'?t|don'?t|doesn'?t|didn'?t|can'?t|won'?t|is it true)\b", re.I)


def _wh_head(text: str) -> bool:
    """Does this clause ask for content, once any opening preposition is stepped over?"""
    return bool(_WH_START.match(_WH_PREP.sub("", text or "", count=1)))


def is_claim_check(question: str) -> bool:
    """A yes/no question or a statement the user wants checked ("is X true", "X works like Y, right?") -- not "what is X".

    Live (uk-dpa-age, 2026-10-08): "Under the Data Protection Act 2018, what is the minimum age ...?" opens with a phrase,
    not the wh-word, and was answered "Yeah, you're right." A question whose LAST clause is a wh-question is a "what is X"
    question too, unless the sentence itself opens as a yes/no question ("Is X taller than Y, which is in Paris?").
    """
    q = (question or "").strip()
    if not q or _wh_head(q):
        return False
    last = re.split(r"[,;:\u2014\u2013]", q)[-1]
    if _wh_head(last) and not _YESNO_START.match(q):
        return False
    return True


_DENIAL_RE = re.compile(
    r"\b(not|never|no longer|cannot|can'?t|doesn'?t|does not|didn'?t|did not|isn'?t|is not|aren'?t|are not|wasn'?t|weren'?t|myth|false)\b", re.I
)
_LEAD_NO = re.compile(r"^\s*(?:no|nope|nah)\b", re.I)


def premise_state(state: str, question: str, answer: str) -> str:
    """A TRUE ledger verdict means the ANSWER's claim is supported, not that the user's claim is.

    Live (res-ten-percent-brain): "Is it true that humans only use 10 percent of their brains?" got a supported
    answer "humans do not use only 10%" and was shown as "Yeah, you're right." -- agreement with the opposite of
    what the answer says. When the question states something without a negation and the answer denies it, the
    user's claim is FALSE. Only TRUE flips; every other state, and every case the polarity is unclear, is untouched.
    """
    if (state or "").upper() != "TRUE" or not is_claim_check(question):
        return state
    if _DENIAL_RE.search(question or ""):
        return state
    lead = (answer or "").strip()
    first = _first_sentence(lead)
    if _LEAD_NO.match(lead) or (first and _DENIAL_RE.search(first)):
        return "FALSE"
    return state


def _first_sentence(text: str) -> str:
    parts = split_sentences(_LEAD_YESNO.sub("", (text or "").strip()))
    return parts[0].strip() if parts else ""


def _ends_clean(sentence: str) -> str:
    sentence = sentence.strip()
    # a fact appended after "Nah, ..." / "Yeah, ..." starts a new sentence (live: "Nah, that doesn't work like that. the claim ...")
    first = sentence.split(" ", 1)[0] if sentence else ""
    if first and first[0].islower() and first.isalpha():
        sentence = sentence[0].upper() + sentence[1:]
    return sentence if not sentence or sentence[-1] in ".!?" else sentence + "."


def render_truth(state: str, *, question: str = "", answer: str = "", aspect: str = "", correction: str = "") -> str:
    """The user-facing answer for a truth state: 1-3 casual sentences, then stop. Detail lives behind "why?".

    TRUE / FALSE / PARTLY apply to claim checks; for "what is X" questions the content answer (already tiny) is kept,
    since "Yeah, you're right" would answer a question nobody asked. CONFLICT and UNVERIFIED always use the fixed line --
    those admit uncertainty and are never reworded into something that sounds surer.
    """
    state = (state or "").upper()
    claim = is_claim_check(question)
    bro = ", bro" if _CASUAL.search(question or "") else ""
    fact = _ends_clean(_first_sentence(correction or answer))
    if state == "UNVERIFIED":
        return TRUTH_UNVERIFIED
    if state == "CONFLICT":
        return TRUTH_CONFLICT
    if not claim:
        return tiny(answer, max_sentences=3, max_chars=280)
    if state == "TRUE":
        base = TRUTH_TRUE[:-1] + bro + "."
        extra = fact if fact and len(fact) > 12 and fact.lower() not in {"it is.", "that is correct."} else ""
        return tiny(f"{base} {extra}".strip(), max_sentences=2, max_chars=240)
    if state == "FALSE":
        base = TRUTH_FALSE[:-1] + bro + "." if bro else TRUTH_FALSE
        return tiny(f"{base} {fact}".strip(), max_sentences=2, max_chars=240)
    if state == "PARTLY":
        part = f"the {aspect.strip()} part" if aspect.strip() else "one detail"
        fix = fact.rstrip(".") if fact else ""
        line = f"Yeah, the idea is right, but {part} is a bit off" + (f" \u2014 {fix}." if fix else ".")
        return tiny(line, max_sentences=3, max_chars=260)
    return tiny(answer, max_sentences=3, max_chars=280)