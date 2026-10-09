"""Final-answer lint: REPORT EVIDENCE, DO NOT INVENT OR LECTURE.

The rule (AGENTS.md, non-negotiable): say what the sources establish; keep documented facts and unknowns apart; an
unknown is never turned into a yes or a no; no unrequested lectures, advice or speculation; no boilerplate.

This module is the last internal check before an answer reaches the user. It
  * REMOVES  D - an invented explanation for a gap ("perhaps because...", "one possible explanation...")
             E - unrequested advice / epistemic lecture ("you shouldn't assume...", "it's fair to conclude...", "doesn't prove...")
             F - boilerplate and useless disclaimers ("It's important to note", "That being said", "we cannot definitively say"...)
  * FLAGS    C - an unknown turned into a negative ("probably didn't pay") - removed too, because it is worse than silence
             B - documented / undocumented structure broken (a documented fact lost, an unknown stated as fact)
             G - longer than the question needs
It never touches honest uncertainty: "I couldn't find that documented.", "The sources I found don't say.",
"That's not publicly documented.", "I couldn't verify that one.", "The sources disagree." and real factual corrections stay.
If the user asks for inference, an assessment or advice, those parts are allowed - the lint follows the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- natural uncertainty wording
UNKNOWN_PHRASES = (
    "I couldn't find that documented.",
    "The sources I found don't say.",
    "That's not publicly documented.",
    "I couldn't verify that one.",
    "The sources disagree.",
)
NOT_FOUND_PUBLIC = "I couldn't find a public source documenting that."
NOT_SPECIFIED = "The available sources don't specify that."

# a sentence that honestly reports an unknown / conflict - never removed, never rewritten
HONEST_RE = re.compile(
    r"couldn'?t (?:find|verify|confirm)|could not (?:find|verify|confirm)|can'?t (?:find|verify|confirm)|don'?t (?:say|specify|state|mention|show|know)|"
    r"doesn'?t (?:say|specify|state|mention)|not (?:publicly )?(?:documented|specified|stated|known|disclosed|reported|recorded)|no (?:public )?(?:source|record|report)s? "
    r"(?:documents?|says?|states?|shows?|specif\w+|mentions?)|(?:sources?|records?) (?:disagree|conflict)|i don'?t know|unverified|undocumented|not available publicly",
    re.I,
)

# ---------------------------------------------------------------------------- what the user asked for
ASKS_ADVICE_RE = re.compile(r"\b(should i|what should|advice|advise|recommend|suggest|what do i do|how (?:do|can|should) i|what would you do|help me decide|tips?)\b", re.I)
ASKS_INFERENCE_RE = re.compile(
    r"\b(infer|assess|assessment|likely|likelihood|probabl\w+|what do you think|do you think|your (?:take|view|opinion|read)|judge|opinion|speculate|"
    r"does (?:that|this|it) mean|can we conclude|is it fair to|why might|what might|what could explain|what would explain|explain why|theor\w+|guess)\b", re.I)
WHAT_DO_WE_KNOW_RE = re.compile(r"\bwhat (?:do|can) we (?:actually |really |currently )?know\b|\bwhat(?:'s| is) (?:actually |really )?(?:known|documented|on record)\b|\bwhat do (?:the )?(?:sources|records|documents) (?:say|show)\b", re.I)


def asks_for_advice(question: str) -> bool:
    return bool(ASKS_ADVICE_RE.search(question or ""))


def asks_for_inference(question: str) -> bool:
    return bool(ASKS_INFERENCE_RE.search(question or ""))


def is_what_do_we_know(question: str) -> bool:
    return bool(WHAT_DO_WE_KNOW_RE.search(question or ""))


# ------------------------------------------------------------------------------------- violation patterns
# F - boilerplate. LEADINS are stripped (the fact after them stays); FILLER sentences are dropped whole.
F_LEADIN_RE = re.compile(
    r"^(?:it(?:'s| is) (?:important|worth|useful|helpful) (?:to )?(?:note|noting|mention|mentioning|remember|stress|emphasi[sz]e)(?: that)?[,:]?\s+"
    r"|it(?:'s| is) worth (?:noting|mentioning)(?: that)?[,:]?\s+|worth (?:noting|mentioning)(?: that)?[,:]?\s+"
    r"|that being said[,:]?\s+|having said that[,:]?\s+|ultimately[,:]?\s+|keep in mind(?: that)?[,:]?\s+|bear in mind(?: that)?[,:]?\s+|importantly[,:]?\s+"
    r"|please note(?: that)?[,:]?\s+|it should be noted(?: that)?[,:]?\s+)",
    re.I,
)
F_FILLER_RE = re.compile(
    r"we (?:can(?:not|'t)|could not|couldn'?t) (?:definitively|conclusively|say for certain|say with certainty)|cannot definitively say|"
    r"(?:there are|there may be|there could be) (?:several|many|various|multiple) (?:possible )?(?:explanations|reasons|factors|interpretations)|"
    r"highlights? the importance of|underscores? the importance of|(?:it(?:'s| is)|would be) prudent to|this is a (?:complex|nuanced|multifaceted)|"
    r"(?:the situation|this|it) is (?:complex|nuanced|multifaceted)|more research is (?:needed|required|warranted)|results may vary|as with (?:any|all) (?:research|information|sources)",
    re.I,
)
# E - unrequested advice (E_ADVICE) and epistemic lecture (E_EPI); advice is allowed when asked for, a lecture when inference/advice is asked for
E_EPI_RE = re.compile(
    r"\byou (?:should(?:n'?t| not)?|ought(?: not)? to|need to|must) (?:assume|conclude|jump|read too much|treat|rely|take|think)\b"
    r"|\b(?:it(?:'s| is)|that(?:'s| is)) (?:fair|reasonable|safe|natural|tempting|wise|sensible|prudent|advisable) to (?:conclude|assume|say|infer|think|believe|read|jump)\b"
    r"|\b(?:this|that|it|which)\s+(?:does(?:n'?t| not)|do(?:n'?t| not)|can(?:'t|not)|isn'?t|is not|would(?:n'?t| not))\s+(?:necessarily\s+|by itself\s+|in itself\s+)?(?:prove|imply|establish|demonstrate|show that)\b"
    r"|\b(?:shouldn'?t|should not|can'?t|cannot|must not) be (?:taken|read|interpreted|assumed|treated|seen|construed) as\b"
    r"|\bdon'?t (?:assume|jump to|read too much)\b",
    re.I,
)
E_ADVICE_RE = re.compile(
    r"\bi(?:'d| would)? (?:strongly )?(?:recommend|suggest|advise)\b|\bconsider (?:consulting|speaking|talking|reaching out|checking|verifying|seeking)\b"
    r"|\b(?:be careful|be cautious|be wary|beware)\b|\bgrain of salt\b|\b(?:verify|double-check|cross-check) (?:this|that|it|these|the)\b[^.]{0,40}\b(?:yourself|independently|before)\b"
    r"|\b(?:the )?(?:safest|best|wisest) (?:course|approach|thing|bet) (?:is|would be)\b|\byou (?:may|might|could) (?:want|wish) to\b"
    r"|\byou (?:should|ought to|may want to|might want to|need to) (?:consider|verify|double-check|check|consult|be careful|be cautious|keep|remember)\b",
    re.I,
)
# D - an invented explanation for a gap in the evidence
D_RE = re.compile(
    r"\b(?:possibly|perhaps|maybe|conceivably|presumably) (?:because|due to|since|they|he|she|it|the|there|this|that|their|his|her)\b"
    r"|\b(?:it|this|that|there)\s+(?:may|might|could|would)\s+(?:well\s+)?(?:be|have been|simply be|just be)\s+(?:because|due to|that|a result|explained|the case|related)\b"
    r"|\b(?:may|might|could) (?:have )?(?:been )?(?:because|due to|since)\b"
    r"|\b(?:one|another|a|an|the) (?:possible|likely|plausible|obvious|alternative|other) (?:explanation|reason|possibility|scenario|interpretation)s?\b|\bseveral possible (?:explanations|reasons)\b"
    r"|\b(?:likely|probably|presumably|possibly) (?:because|due to|since|owing to)\b"
    r"|\bit(?:'s| is) (?:possible|conceivable|plausible|likely|probable) that\b|\bit could be that\b|\bit may be that\b|\bit might be that\b|\bit seems (?:likely|probable|that)\b"
    r"|\b(?:suggests|implies|indicates|hints) (?:that )?(?:they|he|she|the \w+) (?:probably|likely|may|might|could|didn'?t|did not|never)\b",
    re.I,
)
# C - the unknown turned into a negative (or a positive)
NEG = r"(?:did not|didn'?t|never|wasn'?t|weren'?t|was not|were not|isn'?t|is not|aren'?t|are not|hasn'?t|haven'?t|has not|have not|no one|nobody|not)"
C_RE = re.compile(
    rf"\b(?:probably|likely|presumably|apparently|seemingly|most likely|almost certainly|very likely|surely)\s+(?:\w+\s+){{0,2}}{NEG}\b"
    rf"|\b(?:so|therefore|thus|hence)\s+(?:\w+\s+){{0,3}}{NEG}\b"
    rf"|\b(?:which|this|that) (?:suggests|implies|means|indicates|shows) (?:that )?(?:\w+\s+){{0,3}}{NEG}\b"
    rf"|\b(?:suggests|implies|indicates) (?:that )?(?:\w+\s+){{0,4}}{NEG}\b",
    re.I,
)
NEG_RE = re.compile(rf"\b{NEG}\b", re.I)

_STOP = frozenset("the a an of to in on for and or is was were are be been by with as at it its that this who whom what which from their his her they he she".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP and len(w) > 1}


@dataclass
class Finding:
    code: str  # B_structure | C_unknown_made_negative | D_invented_explanation | E_unrequested_advice | F_boilerplate | G_longer_than_needed
    text: str
    action: str  # removed | stripped | rewritten | flagged

    def __str__(self) -> str:
        return f"{self.code}: {self.action}: {self.text[:90]}"


@dataclass
class LintResult:
    text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(f.action != "flagged" for f in self.findings)

    @property
    def codes(self) -> set[str]:
        return {f.code for f in self.findings}


def _split(text: str) -> list[str]:
    from backend.research.style import split_sentences  # local import: style also uses this module

    return split_sentences(text)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def lint_answer(text: str, *, question: str = "", unknowns: list[str] | tuple[str, ...] = ()) -> LintResult:
    """Remove D/E/F, neutralise C. Returns the cleaned text and what was found. Never returns an empty string for non-empty input."""
    original = text or ""
    if not original.strip():
        return LintResult(original)
    advice_ok = asks_for_advice(question)
    inference_ok = asks_for_inference(question)
    unk = [(u, _tokens(u)) for u in unknowns if u and _tokens(u)]
    findings: list[Finding] = []
    out_lines: list[str] = []
    for line in original.split("\n"):
        if not line.strip():
            out_lines.append(line)
            continue
        m = re.match(r"^(\s*(?:[-*\u2022]|\d+[.)])\s+|\s*[A-Za-z ]{3,24}:\s+)?(.*)$", line)
        prefix, body = (m.group(1) or ""), m.group(2)
        if prefix and not re.match(r"\s*(?:[-*\u2022]|\d)", prefix) and F_LEADIN_RE.match(prefix.strip() + " "):
            prefix, body = "", line  # "Worth mentioning: ..." is a boilerplate lead-in, not a label
        kept: list[str] = []
        for sent in _split(body):
            s = sent.strip()
            if not s:
                continue
            honest = bool(HONEST_RE.search(s))
            # F: lead-in boilerplate is stripped, the fact stays
            stripped = F_LEADIN_RE.sub("", s)
            if stripped != s and stripped.strip():
                findings.append(Finding("F_boilerplate", s, "stripped"))
                s = _cap(stripped.strip())
            if not honest:
                if F_FILLER_RE.search(s):
                    findings.append(Finding("F_boilerplate", s, "removed"))
                    continue
                if (not advice_ok and E_ADVICE_RE.search(s)) or (not (advice_ok or inference_ok) and E_EPI_RE.search(s)):
                    findings.append(Finding("E_unrequested_advice", s, "removed"))
                    continue
                if not inference_ok and D_RE.search(s):
                    findings.append(Finding("D_invented_explanation", s, "removed"))
                    continue
                if C_RE.search(s) and (not inference_ok or unk):
                    findings.append(Finding("C_unknown_made_negative", s, "removed"))
                    continue
                if unk and NEG_RE.search(s):
                    toks = _tokens(s)
                    hit = next((u for u, ut in unk if len(ut & toks) >= max(2, int(0.6 * len(ut)))), None)
                    if hit:
                        findings.append(Finding("C_unknown_made_negative", s, "rewritten"))
                        kept.append(f"{hit.rstrip('.')}: I couldn't find that documented.")
                        continue
            kept.append(s)
        if kept:
            out_lines.append(prefix + " ".join(kept))
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()
    if not cleaned:  # the answer was nothing but lecture: keep the original rather than hand back silence
        return LintResult(original.strip(), findings + [Finding("G_longer_than_needed", "everything was removable; original kept", "flagged")])
    return LintResult(cleaned, findings)


def lint_report(text: str, *, question: str = "", unknowns: list[str] | tuple[str, ...] = ()) -> list[Finding]:
    """What would be changed, without changing it (used by the voice check and the eval)."""
    return lint_answer(text, question=question, unknowns=unknowns).findings


# ------------------------------------------------------------------ documented / undocumented structure
def evidence_report(documented: list[tuple[str, str]], undocumented: list[str], disputed: list[str] | tuple[str, ...] = ()) -> str:
    """'What do we actually know?' Documented facts with provenance, then what is undocumented. No advice, no judgment.

    documented = [(fact, provenance)], e.g. ("Parents spent about INR 4 lakh on coaching", "the family's own statement, 2026").
    Unknowns stay unknown: they are listed as undocumented, never as probably true or probably false.
    """
    lines: list[str] = []
    if documented:
        lines.append("Documented:")
        for fact, prov in documented:
            lines.append(f"- {fact.rstrip('.')}" + (f" ({prov})." if prov else "."))
    else:
        lines.append(NOT_FOUND_PUBLIC)
    if disputed:
        lines.append("The sources disagree on:")
        lines.extend(f"- {d.rstrip('.')}." for d in disputed)
    if undocumented:
        lines.append("Not documented:")
        lines.extend(f"- {u.rstrip('.')} \u2014 I couldn't find that documented." for u in undocumented)
    return "\n".join(lines)


_CONJ_HEAD = re.compile(r"^\s*(?:and|or|plus|along with|as well as|but)\s+", re.I)
# A clause opened by one of these after a comma describes the thing already named; it does not
# ask a second question ("the tower, which is in Paris, ..."). "who" is left out: after a comma
# it is usually relative, but "who does it apply to" is a real second question.
_RELATIVE_HEAD = re.compile(r"^\s*(?:which|that|because)\b", re.I)
_QPART_SPLIT = re.compile(r"\s*[,;]\s*|\s+(?:and|or|plus|along with|as well as)\s+", re.I)


def question_parts(question: str) -> list[str]:
    """The separate things a question asks for, in the order it asks them.

    "When did the Act receive Royal Assent, and when did it come into force?" asks two, and one
    of them may be documented while the other is not -- so a single refusal over it would hide
    what we actually established. Returns one element for a question that asks one thing.
    """
    from backend.research.style import is_question_clause  # local import: style also uses this module

    text = re.sub(r"\s+", " ", (question or "").strip())
    if not text:
        return []
    # ", and when..." splits on the comma first, so the conjunction is left glued to the next
    # clause; it is filler either way and comes off before the clause is judged.
    segments = [_CONJ_HEAD.sub("", s.strip(" ,;"), count=1) for s in _QPART_SPLIT.split(text)]
    segments = [s for s in segments if s]
    parts = [s for s in segments if not _RELATIVE_HEAD.match(s) and is_question_clause(s)]
    return parts if len(parts) > 1 else [text]


_TEMPORAL_ASK = re.compile(r"\b(when|what year|which year|what date|which date|since when|how old)\b", re.I)
_MONTH_WORD = re.compile(
    r"\b(january|february|march|april|june|july|august|september|october|november|december)\b", re.I
)
_WRITTEN_DATE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+\d{4}\b|\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\b")
_A_YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")


def states_a_time(text: str) -> bool:
    """Does this sentence actually date something -- a year, a month, or a written day?"""
    return bool(_A_YEAR.search(text or "") or _MONTH_WORD.search(text or "") or _WRITTEN_DATE.search(text or ""))


def _shared_words(left: str, right: str) -> set[str]:
    """Content words the two share, tolerant of endings: "open" reaches "opened", "force" doesn't reach "forces"."""
    a, b = _tokens(left), _tokens(right)
    out: set[str] = set()
    for w in a:
        for v in b:
            if w == v or (min(len(w), len(v)) >= 4 and (w.startswith(v) or v.startswith(w))):
                out.add(w)
                break
    return out


def covers_part(part: str, claim: str) -> bool:
    """Does this claim state the thing this part of the question asks for?

    Two shared content words is the floor, because the subject alone is not an answer: a page
    saying "the Eiffel Tower is 1083 feet tall" shares "Eiffel Tower" with "in which year was
    the Eiffel Tower completed" and answers nothing about it. So a part that asks when also
    needs the claim to actually carry a date.
    """
    if len(_shared_words(part, claim)) < 2:
        return False
    if _TEMPORAL_ASK.search(part or "") and not states_a_time(claim):
        return False
    return True


def check_structure(text: str, documented: list[str], undocumented: list[str]) -> list[Finding]:
    """Fact A documented / B documented / C undocumented must keep exactly that structure."""
    findings: list[Finding] = []
    sents = [s for line in (text or "").split("\n") for s in _split(line)]
    for fact in documented:
        ft = _tokens(fact)
        if ft and not any(len(ft & _tokens(s)) >= max(1, int(0.5 * len(ft))) for s in sents):
            findings.append(Finding("B_structure", f"documented fact missing: {fact}", "flagged"))
    for item in undocumented:
        it = _tokens(item)
        if not it:
            continue
        mentions = [s for s in sents if len(it & _tokens(s)) >= max(1, int(0.6 * len(it)))]
        for s in mentions:
            if not HONEST_RE.search(s) and not re.search(r"\bnot documented\b|:\s*$", s, re.I):
                findings.append(Finding("B_structure", f"undocumented item stated as a claim: {s}", "flagged"))
    return findings


@dataclass
class FinalCheck:
    ok: bool
    text: str
    issues: list[str]
    answered: bool = True
    facts_and_unknowns_separated: bool = True
    unknown_turned_negative: bool = False
    invented_explanation: bool = False
    unrequested_advice: bool = False
    useless_disclaimer: bool = False
    shorter_possible: bool = False


def final_check(text: str, *, question: str = "", documented: list[str] = (), undocumented: list[str] = (), max_chars: int = 700) -> FinalCheck:
    """The internal pre-send checklist, as code. Returns the cleaned text plus one boolean per question."""
    res = lint_answer(text, question=question, unknowns=list(undocumented))
    structure = check_structure(res.text, list(documented), list(undocumented))
    codes = res.codes
    long = len(res.text) > max_chars and not is_what_do_we_know(question) and not re.search(r"\bwhy\b|\bsources?\b|\bexplain\b", question or "", re.I)
    chk = FinalCheck(
        ok=False, text=res.text, issues=[str(f) for f in res.findings + structure],
        answered=bool(res.text.strip()),
        facts_and_unknowns_separated=not structure,
        unknown_turned_negative="C_unknown_made_negative" in codes,
        invented_explanation="D_invented_explanation" in codes,
        unrequested_advice="E_unrequested_advice" in codes,
        useless_disclaimer="F_boilerplate" in codes,
        shorter_possible=bool(long),
    )
    chk.ok = chk.answered and chk.facts_and_unknowns_separated and not (chk.unknown_turned_negative or chk.invented_explanation or chk.unrequested_advice or chk.useless_disclaimer or chk.shorter_possible)
    return chk