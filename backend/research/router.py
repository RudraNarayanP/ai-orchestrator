"""Classifier / router.

Decides how much investigation a question has earned. Two failure modes matter
here, in opposite directions: burning eight browser sessions on "2 + 2", and
letting a confidently-worded wrong answer count as "reliably answered".

Level 0 is therefore granted by only two things: something we can compute, or a
configured local model that explicitly swears the answer is stable. Time-sensitive
questions can never be answered at level 0 by a model, because that is precisely
the case where a confident model is confidently stale.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Iterable

from backend.models import (
    ConversationTurn,
    EscalationLevel,
    ProviderResponse,
    QuestionAnalysis,
    ResearchMode,
    StakesDomain,
)
from backend.research import memory

# ------------------------------------------------------------------ level 0

ARITHMETIC_RE = re.compile(
    r"^\s*(?:what(?:'s| is)?|calculate|compute|eval(?:uate)?|solve|=)?\s*"
    r"[-+*/%^()$\d.,\s]+(?:and|plus|minus|times|multiplied by|divided by|over|to the power of|squared|mod)?\s*"
    r"[-+*/%^()$\d.,\s]*(?:=|\?|equals?|is|please)?\s*$",
    re.I,
)
PURE_MATH_RE = re.compile(r"^[\s\d+\-*/%.^()]+$")
WORD_OPS = {
    "plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "over": "/", "divided by": "/",
    "into": "/", "modulo": "%", "mod": "%", "to the power of": "**", "raise to": "**",
}

CONVERSION_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(km|kg|lb|lbs|miles?|meters?|metres?|feet|ft|celsius|c|fahrenheit|f|gb|mb|tb|ounces?|oz)\b", re.I)

CHITCHAT_RE = re.compile(
    r"\b(joke|funny|make me laugh|banter|roast|hot take|what do you think about|do you like"
    r"|would you rather|imagine|hypothetical|sup|hey there|hi\b|hello\b|thanks|thank you"
    r"|who are you|what are you|how are you|tell me about yourself|write a (poem|haiku|story|tweet|lyric)"
    r"|lol|haha|brutal|sarcastic)\b",
    re.I,
)
OPINION_ASK_RE = re.compile(r"^\s*(in your opinion|personally|do you prefer|which do you like better|what'?s your favourite)\b", re.I)

STABLE_KNOWLEDGE_RE = re.compile(
    r"\b(syntax|how do i write|what does .* (?:mean|do)|difference between (?:list|tuple|dict|set|array|string)"
    r"|define\b|definition of|what is a \w+ in (?:python|javascript|java|c\+\+|go|rust|sql)"
    r"|reverse|uppercase|lowercase|camel case|snake case)\b",
    re.I,
)

CURRENT_RE = re.compile(
    r"\b(latest|recent|current|today|now|20\d{2}|price of|how much does|as of|news|just released"
    r"|announced|upcoming|who (?:is|won|owns|leads)|version|score|election|weather|stock|market)\b",
    re.I,
)

HIGH_STAKES = {
    StakesDomain.LEGAL: re.compile(r"\b(law|legal|lawsuit|court|ruling|verdict|contract|liab(?:ility|le)|patent|trademark|copyright|criminal|deport|fine|regulation|antitrust|compliance|gdpr|licence agreement)\b", re.I),
    StakesDomain.MEDICAL: re.compile(r"\b(dosage|dose|symptom|diagnos|medication|drug|treatment|cancer|disease|illness|hospital|therapy|vaccine|side effect|medical|health|risk of dying|pregnan)\b", re.I),
    StakesDomain.FINANCIAL: re.compile(r"\b(invest|investing|portfolio|tax\b|taxes|mortgage|loan|interest rate|retirement|401k|pension|buy or sell|dividend|crypto price|savings plan|debt)\b", re.I),
    StakesDomain.IMMIGRATION: re.compile(r"\b(visa|immigration|passport|citizenship|residence permit|work permit|study permit|pte|ielts|toefl|green card|asylum|border|customs)\b", re.I),
    StakesDomain.ADMISSIONS: re.compile(r"\b(admission|admissions|accept(?:ed|ance)|application deadline|entrance exam|sat|gre|gpa|university requirements|college (?:requirement|application)|offer letter)\b", re.I),
    StakesDomain.REGULATORY: re.compile(r"\b(fda|fcc|sec filing|eu ai act|hipaa|sox|iso standard|certification required|regulatory approval|authorised by)\b", re.I),
    StakesDomain.SAFETY: re.compile(r"\b(safe to|dangerous|toxic|hazard|explosive|structural|load bearing|evacuation|recalled|malpractice|lifethreatening|wall thickness)\b", re.I),
}

CAPABILITY_PATTERNS = {
    "web_search": CURRENT_RE,
    "official_documents": re.compile(r"\b(official|primary source|filing|paper|study|documentation|announcement|press release|dataset|specification)\b", re.I),
    "reasoning": re.compile(r"\b(why|prove|derive|compare|evaluate|step by step|which is better|trade-?offs|analyse|analyze|predict)\b", re.I),
    "code": re.compile(r"\b(code|function|script|regex|api|compile|debug|python|javascript|typescript|sql|rust|golang)\b", re.I),
    "long_context": re.compile(r"\b(this file|attached|pasted|the whole|entire document|all of these|table of \d+)\b", re.I),
}

# Provider families: agents that plausibly share training data or the same
# headline source should not be mistaken for independent verification.
FAMILY = {
    "chatgpt": "openai",
    "gemini": "google",
    "google_ai": "google",
    "copilot": "microsoft",
    "meta_ai": "meta",
    "le_chat": "mistral",
    "pi": "inflection",
    "qwen": "alibaba",
    "deepseek": "deepseek",
    "search": "web",
    "perplexity": "perplexity",
    "grok": "xai",
}

PRIMARY_FOR: dict[str, list[str]] = {
    "current_events": ["google_ai", "chatgpt", "copilot", "gemini"],
    "product": ["chatgpt", "google_ai", "gemini", "copilot"],
    "technical": ["deepseek", "chatgpt", "qwen", "gemini"],
    "scientific": ["gemini", "chatgpt", "perplexity", "google_ai"],
    "legal": ["chatgpt", "copilot", "google_ai", "gemini"],
    "medical": ["gemini", "chatgpt", "copilot", "perplexity"],
    "ranking": ["chatgpt", "gemini", "copilot", "qwen"],
    "comparison": ["chatgpt", "gemini", "deepseek", "copilot"],
    "definition": ["chatgpt", "gemini", "copilot"],
    "factual": ["chatgpt", "gemini", "google_ai", "copilot"],
    "visa_immigration": ["google_ai", "chatgpt", "copilot", "gemini"],
    "admissions": ["google_ai", "chatgpt", "gemini", "copilot"],
}

SECONDARY_ORDER = ["gemini", "chatgpt", "copilot", "google_ai", "le_chat", "qwen", "deepseek", "meta_ai", "pi"]

SAFE_FUNCS = {}


class SafeEvalError(ValueError):
    pass


def try_arithmetic(question: str) -> tuple[str | None, str]:
    """Compute arithmetic instead of asking a model that once saw it on the internet."""
    text = re.sub(r"^(what(?:'s| is)|calculate|compute|eval(?:uate)?|solve)\s+", "", question.strip().rstrip("?"), flags=re.I)
    normalised = text.lower()
    for word, sym in WORD_OPS.items():
        normalised = normalised.replace(word, sym)
    normalised = re.sub(r"[,]", "", normalised)
    normalised = normalised.replace("^", "**")
    normalised = re.sub(r"(squared|to the square)", "**2", normalised)
    normalised = re.sub(r"[a-z$?=%]\s*$", "", normalised).strip()
    if not PURE_MATH_RE.search(normalised) or not re.search(r"\d", normalised):
        return None, ""
    if not re.search(r"[-+*/%]|\*\*", normalised):
        return None, ""
    try:
        value = _safe_eval(normalised)
    except SafeEvalError:
        return None, ""
    if value is None:
        return None, ""
    pretty = int(value) if isinstance(value, float) and float(value).is_integer() else round(float(value), 6)
    return f"{pretty}", normalised


def _safe_eval(expr: str) -> float:
    """AST-whitelisted arithmetic. Never ``eval`` -- the question text is user input."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise SafeEvalError(str(exc)) from exc
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise SafeEvalError(f"disallowed node {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise SafeEvalError("non-numeric constant")
        if isinstance(node, ast.Pow):
            try:
                left, right = float(node.left.value), float(node.right.value)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                raise SafeEvalError("unsupported power") from None
            if abs(right) > 12 or abs(left) > 1e10:
                raise SafeEvalError("exponent too large")
    try:
        return float(eval(compile(ast.Expression(body=tree.body), "<arith>", "eval"), {"__builtins__": {}}, {}))  # noqa: S307
    except ZeroDivisionError:
        raise SafeEvalError("division by zero") from None
    except Exception as exc:  # noqa: BLE001
        raise SafeEvalError(str(exc)) from None


STATUTE_RE = re.compile(
    r"\b(?:section|s\.|article|clause|schedule)\s*\d+|\bact\s+(?:of\s+)?(?:19|20)\d\d\b|\b(?:statute|statutory|constitution|legislation|ordinance|"
    r"directive|family code|criminal code|civil code|bill of rights|regulations?\s+(?:19|20)\d\d)\b",
    re.I,
)
"""A question about the text of a law: never answered from the model's memory, however sure it sounds."""


def classify(
    question: str,
    *,
    mode: ResearchMode = ResearchMode.STANDARD,
    enabled: Iterable[str] = (),
    llm_stable_answer: str | None = None,
    history: Iterable[ConversationTurn] = (),
) -> QuestionAnalysis:
    enabled = list(enabled)
    text = (question or "").strip()
    low = text.lower()

    # Conversation memory: a follow-up is routed on what it follows, not on its
    # (usually tiny) own text -- "and how did they do it?" inherits the stakes and
    # time-sensitivity of the question before it.
    turns = list(history or [])
    prior = turns[-1] if turns else None
    follow_up = memory.is_follow_up(text, prior) and not CHITCHAT_RE.search(text)
    signal_text = f"{prior.question} {text}" if follow_up and prior else text

    computed, expr = try_arithmetic(text)
    entities = re.findall(r"\b([A-Z][A-Za-z0-9.'-]{2,})\b", text)
    proper_nouns = [e for e in entities if e not in ("I",)]
    time_sensitive = bool(CURRENT_RE.search(signal_text)) and not bool(re.search(r"\b(20\d{2})\b", low) and re.search(r"\b(history|past|in 20\d{2})\b", low))

    stakes = StakesDomain.NONE
    for domain, pattern in HIGH_STAKES.items():
        if pattern.search(signal_text):
            stakes = domain
            break
    high_stakes = stakes != StakesDomain.NONE

    intents: list[tuple[str, re.Pattern[str]]] = [
        ("arithmetic", re.compile(r"", re.I)),
        ("current_events", re.compile(r"\b(news|latest|announced|released|today|this week|as of|breaking)\b", re.I)),
        ("product", re.compile(r"\b(price|review|specs?|release date|model|gadget|laptop|phone|subscription|plan)\b", re.I)),
        ("legal", re.compile(r"\b(law|legal|court|verdict|regulation|licence|permit|allowed to|illegal)\b", re.I)),
        ("medical", re.compile(r"\b(dose|dosage|symptom|diagnosis|medication|treatment|disease|drug)\b", re.I)),
        ("scientific", re.compile(r"\b(study|research|paper|experiment|molecula|gene|climate|physics|chemistry|biology)\b", re.I)),
        ("technical", re.compile(r"\b(code|api|function|error|compile|algorithm|database|docker|kubernetes)\b", re.I)),
        ("ranking", re.compile(r"\b(top|best|ranked|#\d|largest|fastest|most popular)\b", re.I)),
        ("comparison", re.compile(r"\b(vs\.?|versus|compare|difference between|which is (?:better|faster|cheaper))\b", re.I)),
        ("definition", re.compile(r"\b(what (?:is|are|does)|define|definition|meaning of|what does .* mean)\b", re.I)),
    ]
    intent = "factual"
    for name, pattern in intents:
        if name == "arithmetic":
            continue
        if pattern.search(signal_text):
            intent = name
            break
    if computed is not None:
        intent = "arithmetic"

    chatty = bool(CHITCHAT_RE.search(text) or OPINION_ASK_RE.search(text))
    statutory = bool(STATUTE_RE.search(text))
    stable_knowledge = bool(STABLE_KNOWLEDGE_RE.search(text)) and not time_sensitive and not follow_up

    analysis = QuestionAnalysis(
        question=text,
        intent=intent,
        key_entities=proper_nouns[:8],
        time_sensitivity="high" if time_sensitive else ("low" if stable_knowledge else "medium"),
        needs_current_data=time_sensitive,
        needs_web_research=not (computed is not None or ((chatty or stable_knowledge) and not statutory)),
        difficulty="high" if (high_stakes or text.count("?") > 2 or len(text.split()) > 45) else ("low" if len(text.split()) < 8 else "medium"),
        classifier_source="heuristic",
    )
    analysis.capabilities = [name for name, pattern in CAPABILITY_PATTERNS.items() if pattern.search(signal_text)]
    if follow_up and prior is not None:
        analysis.follow_up = True
        analysis.inherited_entities = memory.inherited_entities(prior)
        analysis.standalone_question = memory.standalone_question(text, prior)
        analysis.key_entities = list(dict.fromkeys(analysis.key_entities + analysis.inherited_entities))[:10]
    analysis.stakes = stakes
    analysis.high_stakes = high_stakes
    analysis.sub_questions = _sub_questions(text)
    if len(analysis.sub_questions) > 2:
        analysis.difficulty = "high"

    if computed is not None:
        analysis.trivial = True
        analysis.can_answer_directly = True
        analysis.direct_answer = f"{computed}"
        analysis.rationale = f"computed locally from {expr}; no model or browser needed"
        analysis.starting_level = EscalationLevel.DIRECT
    elif chatty and not time_sensitive and not high_stakes and not statutory:
        analysis.trivial = True
        analysis.can_answer_directly = bool(llm_stable_answer)
        analysis.direct_answer = llm_stable_answer
        analysis.starting_level = EscalationLevel.DIRECT if llm_stable_answer else EscalationLevel.PRIMARY
        analysis.rationale = (
            "banter/creative ask -- answered as itself, no research" if llm_stable_answer
            else "banter/creative ask; no model configured, so one provider handles it cheaply"
        )
    elif stable_knowledge and llm_stable_answer and not statutory:
        analysis.trivial = True
        analysis.can_answer_directly = True
        analysis.direct_answer = llm_stable_answer
        analysis.starting_level = EscalationLevel.DIRECT
        analysis.rationale = "stable knowledge the configured model claims to know, cross-checked as non-time-sensitive"
    else:
        analysis.starting_level = EscalationLevel.DEEP if high_stakes or mode == ResearchMode.DEEP_RESEARCH else EscalationLevel.PRIMARY
        if mode == ResearchMode.QUICK:
            analysis.starting_level = EscalationLevel.PRIMARY
        if mode == ResearchMode.DEEP_RESEARCH:
            analysis.starting_level = EscalationLevel.PARALLEL
    return analysis


def _sub_questions(text: str) -> list[str]:
    """Decompose only genuinely multi-part questions.

    Echoing the single original question back as its own "sub-question" makes an
    unanswerable coverage check out of nothing and triggers escalation that the
    evidence does not warrant.
    """
    explicit = [q.strip(" ?") for q in re.findall(r"([^?]{15,140}\?)", text)]
    explicit = [q for q in explicit if q and q.lower() != text.strip("? ").lower()]
    if len(explicit) >= 2:
        return explicit[:4]
    parts = [p.strip(" ?") for p in re.split(r"\b(?:and also|additionally|separately|finally|second)\b", text) if len(p.strip()) > 20]
    parts = [p for p in parts if p.lower() != text.strip("? ").lower()]
    if len(parts) >= 2:
        return parts[:4]
    clauses = [c.strip() for c in re.split(r",\s+(?:and|how|what|where|when)\s+", text) if len(c.strip()) > 18]
    clauses = [c for c in clauses if c.lower() != text.strip("? ").lower()]
    return clauses[:4] if len(clauses) >= 2 else []


_TOPIC_STOP = {"which", "what", "when", "where", "does", "under", "about", "there", "their", "these", "those", "with", "from", "that", "this", "have", "will", "would", "could", "should", "into", "than", "then", "also"}


def question_from_prompt(prompt: str) -> str:
    """The user's question inside a provider prompt (round 1 is the question itself; later rounds quote it)."""
    text = (prompt or "").strip()
    m = re.search(r"Original question:\s*\n?(.+?)(?:\n\s*\n|\Z)", text, re.S)
    if m:
        return m.group(1).strip()
    return text.split("\n\n", 1)[0].strip()


def off_topic(question: str, answer: str) -> bool:
    """True when an answer shares nothing distinctive with the question.

    Live eval defect: a reused chat window answered the *previous* question (an
    Employment Rights Act reply to a Ukrainian Family Code question) and the claims
    from it flowed into the ledger. Capitalised names (Ukraine, Family Code, ...) are
    the strongest signal; failing that, no overlap at all on four or more content words.
    """
    q, a = (question or "").strip(), (answer or "").lower()
    if len(a) < 80 or len(q) < 12:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", q)
    proper = [w.lower() for i, w in enumerate(words) if i > 0 and w[0].isupper() and len(w) >= 4 and w.lower() not in _TOPIC_STOP]
    if proper:
        return not any(w in a for w in proper)
    content = [w.lower() for w in words if len(w) >= 5 and w.lower() not in _TOPIC_STOP]
    return len(content) >= 4 and not any(w in a for w in content)


_ASK_STOP = _TOPIC_STOP | {"university", "people", "number", "being", "explain", "describe", "define", "defines", "guidance", "students", "please", "known", "using", "regarding", "current", "currently", "year", "years"}


def future_year(question: str, *, now_year: int | None = None) -> str | None:
    """A year later than today that the question asks about ("... during 2028"), else None."""
    import datetime as _dt

    this_year = now_year or _dt.date.today().year
    for y in re.findall(r"\b(20\d{2}|21\d{2})\b", question or ""):
        if int(y) > this_year:
            return y
    return None


def addresses_question(question: str, claim: str) -> bool:
    """Does this claim speak to what was asked, not merely to the same subject?

    Live eval defect: asked for the share of Manchester PhD vivas that failed, the
    model-free path stated, with high confidence, that the regulations were last
    modified on a given date. It was well sourced and beside the point. A claim is on
    point if it shares a number or an asked-for content word (not a proper name) with
    the question; a question with no such words accepts everything.
    """
    q = question or ""
    c = (claim or "").lower()
    future = future_year(q)
    if future and future not in c:
        return False  # asked about a year that has not happened; a claim about other years is beside the point
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", q)
    asked = [
        w.lower()
        for i, w in enumerate(words)
        if len(w) >= 5 and w.lower() not in _ASK_STOP and not (i > 0 and w[0].isupper())
    ]
    figures = set(re.findall(r"\b\d{2,}\b", q))
    if not asked and not figures:
        return True
    if any(f in c for f in figures):
        return True
    return any(w[:5] in c for w in asked)


def is_failure_phrase(text: str) -> list[str]:
    """Section 4: an admitted inability is a routing signal, not a shrug."""
    patterns = [
        r"\bi (?:don'?t|do not|can'?t|cannot|couldn'?t|was unable|am unable|wasn'?t able) (?:verify|confirm|find|locate|determine|answer|know|guarantee|establish|check)\b",
        r"\bi'?m not (?:certain|sure|able|confident)\b",
        r"\b(?:the )?information (?:is|appears|seems) (?:unclear|inconclusive|limited|unavailable)\b",
        r"\bsources? (?:conflict|disagree|contradict|are unclear)\b",
        r"\bi have (?:no|insufficient|limited) (?:information|data|evidence)\b",
        r"\b(?:there(?:'s| is) no (?:reliable|authoritative|public(?:ly)? available) (?:data|source|information|record))\b",
        r"\b(?:this )?(?:may|could|might) be outdated\b",
        r"\bnot(?:able)? to (?:access|verify|retrieve)\b",
        r"\bcould (?:not|n'?t) (?:locate|establish|verify|confirm|access)\b",
        r"\bno (?:reliable|verifiable|published) source\b",
        r"\bi'?d need (?:further|additional) (?:research|verification|checking)\b",
        r"\bmy knowledge (?:cutoff|cut-off)|training data (?:cutoff|cut-off)|as of my (?:knowledge|training)\b",
        r"\bi cannot (?:browse|search|access the web)\b",
        r"\b(?:i )?(?:couldn'?t|could not|can'?t|cannot|unable to|wasn'?t able to) find\b",
        r"\bnot (?:entirely |fully |completely )?(?:certain|sure)\b",
        r"\b(?:don'?t|do not) have enough (?:information|data|evidence)\b",
        r"\bsources? (?:are |is )?(?:unclear|ambiguous)\b",
        r"\bi cannot establish this\b",
    ]
    hits: list[str] = []
    low = (text or "").lower()
    for pattern in patterns:
        match = re.search(pattern, low)
        if match:
            hits.append(match.group(0)[:80])
    return hits


BLOCKED_STATES = {"logged_out", "broken", "rate_limited", "failed"}


def _prior_blocked(prior: dict[str, str] | None) -> set[str]:
    """Sites that were walled in an EARLIER run. They go last, never off the list: access changes."""
    return {name for name, state in (prior or {}).items() if state in BLOCKED_STATES}


def select_primary(
    analysis: QuestionAnalysis,
    enabled: list[str],
    health: dict[str, str] | None = None,
    prior: dict[str, str] | None = None,
) -> str | None:
    health = health or {}
    late = _prior_blocked(prior)

    def usable(name: str) -> bool:
        state = health.get(name, "unknown")
        return state not in {"logged_out", "broken", "rate_limited", "failed"} and name in enabled

    key = analysis.stakes.value if analysis.high_stakes and analysis.stakes.value in PRIMARY_FOR else analysis.intent
    order = PRIMARY_FOR.get(key, []) + PRIMARY_FOR["factual"] + SECONDARY_ORDER + list(enabled)
    # stable: previously walled sites only after every site that worked last time
    order = [n for n in order if n not in late] + [n for n in order if n in late]
    for candidate in order:
        if usable(candidate) and candidate != "search":
            return candidate
    return enabled[0] if enabled else None


def select_secondaries(
    analysis: QuestionAnalysis,
    enabled: list[str],
    *,
    exclude: Iterable[str] = (),
    count: int = 4,
    health: dict[str, str] | None = None,
    search_always: bool = False,
    prior: dict[str, str] | None = None,
    reprobe: bool = False,
) -> list[str]:
    """Pick *different families*, not just different names.

    Two models from the same lab citing the same wire story is one source seen
    twice; the point of expanding is independence.

    ``search`` is deliberately not a researcher: it is the evidence transport the
    pool already uses for every round, so counting it as a council member would
    make escalation look cheaper than it is.
    """
    health = health or {}
    exclude = set(exclude) | (set() if search_always else {"search"})
    chosen: list[str] = []
    families: set[str] = set()
    late = _prior_blocked(prior)
    ordered = SECONDARY_ORDER + [n for n in enabled if n not in SECONDARY_ORDER]
    ordered = [n for n in ordered if n not in late] + [n for n in ordered if n in late]
    for name in ordered:
        if name in exclude or name not in enabled:
            continue
        if (health.get(name, "unknown")) in BLOCKED_STATES:
            continue
        fam = FAMILY.get(name, name)
        if fam in families and len(chosen) < count:
            continue
        chosen.append(name)
        families.add(fam)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for name in SECONDARY_ORDER + enabled:
            if name in chosen or name in exclude or name not in enabled:
                continue
            chosen.append(name)
            if len(chosen) >= count:
                break
    chosen = chosen[: max(count, 2)]
    if reprobe and len(chosen) >= 3:
        # one slot per escalation goes to a site that was walled last time: if access changed, we learn it now
        again = next((n for n in ordered if n in late and n not in chosen and n not in exclude and n in enabled and health.get(n, "unknown") not in BLOCKED_STATES), None)
        if again:
            chosen[-1] = again
    return chosen


def failure_payload(response: ProviderResponse) -> dict[str, Any]:
    """What escalation prompts are built from (section 6): hand the next
    researcher the failure context, not a fresh copy of the same question."""
    return {
        "provider": response.provider,
        "status": response.status.value,
        "failure_signals": response.failure_signals,
        "web_research": response.web_research_status.value,
        "citations": [c.url for c in response.citations][:12],
        "answer_excerpt": (response.answer_text or "")[:1500],
    }
