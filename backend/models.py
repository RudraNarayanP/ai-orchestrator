"""Typed domain model for OmniBrain.

Everything the pipeline produces is one of these objects, so the storage layer,
the API, the UI and the verifier all speak the same language.
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"


class ResearchMode(str, enum.Enum):
    QUICK = "QUICK"
    STANDARD = "STANDARD"
    DEEP_RESEARCH = "DEEP_RESEARCH"


class EscalationLevel(int, enum.Enum):
    """Adaptive escalation: parallelism is earned by failure, not assumed.

    Objective is the highest reliable accuracy for the least investigation, so
    every step up must be justified by something concrete -- unresolved evidence,
    a conflict, or consequence.
    """

    DIRECT = 0
    PRIMARY = 1
    PARALLEL = 2
    DEEP = 3


class StakesDomain(str, enum.Enum):
    NONE = "none"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    IMMIGRATION = "visa_immigration"
    ADMISSIONS = "admissions"
    REGULATORY = "regulatory"
    SAFETY = "safety_critical"
    OTHER = "other"


class EscalationStep(BaseModel):
    level: EscalationLevel
    reason: str
    providers: list[str] = Field(default_factory=list)
    round: int = 1
    triggered_by: list[str] = Field(default_factory=list)
    """Concrete signals: no_sources, citation_mismatch, failure_phrase,
    contradiction, high_stakes, sub_question_unanswered, outdated, ..."""

    at: float = Field(default_factory=time.time)


class ProviderStatus(str, enum.Enum):
    IDLE = "idle"
    LAUNCHING = "launching"
    CONNECTED = "connected"
    SEARCHING = "searching"
    RESPONDING = "responding"
    COMPLETED = "completed"
    LOGGED_OUT = "logged_out"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    FAILED = "failed"
    # UI changed underneath us and no selector strategy worked. Reported as
    # broken rather than worked around -- we never fight a site's controls.
    BROKEN = "broken"

    @property
    def terminal(self) -> bool:
        return self in {
            ProviderStatus.COMPLETED,
            ProviderStatus.LOGGED_OUT,
            ProviderStatus.RATE_LIMITED,
            ProviderStatus.TIMEOUT,
            ProviderStatus.FAILED,
            ProviderStatus.BROKEN,
        }


class ClaimStatus(str, enum.Enum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTESTED = "contested"
    REFUTED = "refuted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Confidence(str, enum.Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NONE = "insufficient_evidence"


class SourceTier(str, enum.Enum):
    PRIMARY_OFFICIAL = "primary_official"
    ORIGINAL_RESEARCH = "original_research"
    GOVERNMENT = "government"
    JOURNALISM = "journalism"
    TECHNICAL = "technical_publication"
    EXPERT = "expert_commentary"
    COMMUNITY = "community"
    SOCIAL = "social_media"
    AI_UNSOURCED = "ai_unsourced"
    UNKNOWN = "unknown"


class SourceCheckStatus(str, enum.Enum):
    NOT_CHECKED = "not_checked"
    CONFIRMED = "confirmed"
    MISMATCH = "mismatch"
    IRRELEVANT = "irrelevant"
    OUTDATED = "outdated"
    BROKEN_URL = "broken_url"
    UNREACHABLE = "unreachable"
    HALLUCINATED = "hallucinated"
    BLOCKED = "blocked"


class WebResearchStatus(str, enum.Enum):
    """Section 17: a provider that should have searched but did not gets
    down-weighted instead of being trusted for sounding confident."""

    PERFORMED = "performed"
    FAILED_OR_UNCLEAR = "failed_or_unclear"
    NOT_REQUIRED = "not_required"
    UNKNOWN = "unknown"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    RESEARCHING = "researching"
    EXTRACTING = "extracting"
    VERIFYING = "verifying"
    FOLLOWUP = "followup"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Citation(BaseModel):
    url: str
    title: str | None = None
    snippet: str | None = None
    published: str | None = None
    provider: str | None = None
    """Where this citation was seen: a provider answer, or a fetched page."""

    marker: str | None = None
    """The in-text marker it was attached to, e.g. '1' for [1]."""

    ai_opened: bool | None = None
    """True: the AI said it opened the page. False: it only saw it mentioned.
    None: cited, opening not confirmed. Never promoted without the AI saying so."""

    audited: bool | None = None
    """OmniBrain opened this URL itself to check what the AI cited (not to find new sources)."""

    claim_ids: list[str] = Field(default_factory=list)
    """Claims this source supports (or refutes) in the ledger."""

    cited_by: list[str] = Field(default_factory=list)
    """Which AIs cited it."""

    origin: str = "native"
    """native: a link the chat site itself rendered (source chip / anchor). text: a URL the AI wrote in its
    answer. Either way it is the AI's citation; OmniBrain never adds one."""

    omnibrain_opened: bool = False
    """OmniBrain's own audit fetched this page. Says nothing about whether the AI did."""

    provenance: str | None = None
    """MENTIONED -> OPENED -> INSPECTED -> CITED -> CLAIM_SUPPORTED (highest reached). None: no AI cited it."""


class ProviderResponse(BaseModel):
    """One provider, one prompt, one round -- fully auditable (section 19)."""

    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(default_factory=lambda: new_id("resp"))
    job_id: str
    round: int = 1
    provider: str
    prompt: str
    raw_text: str = ""
    answer_text: str = ""
    citations: list[Citation] = Field(default_factory=list)

    status: ProviderStatus = ProviderStatus.IDLE
    error: str | None = None
    detail: str | None = None

    web_research_status: WebResearchStatus = WebResearchStatus.UNKNOWN
    web_research_signals: list[str] = Field(default_factory=list)

    started_at: float | None = None
    finished_at: float | None = None
    duration_s: float | None = None

    ui_url: str | None = None
    """Direct link to the conversation in the provider's own web UI."""

    conversation_url: str | None = None
    """The chat's URL after the answer arrived (where a follow-up continues)."""

    thread_id: str | None = None
    """research id + provider: one conversation per AI per research."""

    turn: int = 1
    """Position in that conversation (1 = first prompt)."""

    continued: bool = False
    """True when this prompt went into an existing conversation (a follow-up)."""

    fingerprint: str | None = None
    """Stable hash of the captured answer, used to detect duplicated content."""

    model_label: str | None = None
    """Best-effort read of which model the site says it is serving. Informational
    only -- the UI changes this constantly, so nothing downstream trusts it."""

    role: str = "primary"
    """primary | secondary | targeted -- how this researcher entered the job"""

    escalation_reason: str | None = None
    """Why this agent was spawned, so a redundant swarm is visible in the audit."""

    pages_visited: list[str] = Field(default_factory=list)
    """URLs the researcher cited or that we opened while checking it (section 14)."""

    failure_signals: list[str] = Field(default_factory=list)
    """Detected uncertainty admissions and structural failures that justify
    escalating past this answer rather than accepting it."""

    superseded: bool = False
    """The same AI corrected this answer in a follow-up. The transcript is kept
    for the curator; its claims no longer count."""

    def note(self, status: ProviderStatus, error: str | None = None, detail: str | None = None) -> None:
        self.status = status
        if error:
            self.error = error
        if detail:
            self.detail = detail


class SelfCorrection(BaseModel):
    """An AI's answer before and after a same-conversation re-investigation."""

    id: str = Field(default_factory=lambda: new_id("corr"))
    job_id: str
    provider: str
    thread_id: str | None = None
    round: int = 1
    initial_claim: str = ""
    follow_up_result: str = ""
    verdict: str = "unclear"
    """confirmed | corrected | cannot_establish | unclear"""

    correction_reason: str = ""
    final_position: str = ""
    first_response_id: str | None = None
    follow_up_response_id: str | None = None


class Claim(BaseModel):
    """An atomic, independently verifiable factual assertion (section 11)."""

    id: str = Field(default_factory=lambda: new_id("claim"))
    job_id: str
    round: int = 1
    claim: str
    kind: str = "fact"
    """fact | statistic | date | causal | ranking | definition | opinion"""

    topic: str | None = None
    """Short label so claims about the same thing can be clustered."""

    provider_sources: list[str] = Field(default_factory=list)
    web_sources: list[str] = Field(default_factory=list)

    supporting: list[str] = Field(default_factory=list)
    contradicting: list[str] = Field(default_factory=list)

    status: ClaimStatus = ClaimStatus.UNVERIFIED
    confidence: Confidence | None = None
    rationale: str | None = None

    @property
    def support_count(self) -> int:
        return len(self.provider_sources)


class Evidence(BaseModel):
    """A piece of independently gathered evidence attached to a claim."""

    id: str = Field(default_factory=lambda: new_id("ev"))
    job_id: str
    round: int = 1
    claim_id: str | None = None

    url: str | None = None
    title: str | None = None
    domain: str | None = None
    snippet: str | None = None
    published: str | None = None
    retrieved_at: float = Field(default_factory=time.time)

    tier: SourceTier = SourceTier.UNKNOWN
    polarity: str = "support"
    """support | refute | neutral"""

    check_status: SourceCheckStatus = SourceCheckStatus.NOT_CHECKED
    check_notes: str | None = None

    origin: str = "provider"
    """provider | search | fetch | verifier"""

    verbatim_excerpt: str | None = None
    """Text actually found on the page, proving the source says what we claim."""

    ai_opened: bool | None = None
    """What the citing AI said about this URL: opened / mentioned only / not stated."""

    cited_by: list[str] = Field(default_factory=list)
    """AIs that cited this URL. Empty means OmniBrain found it itself (optional discovery)."""

    omnibrain_opened: bool = False
    """OmniBrain's audit fetched the page. Not the AI's research."""

    provenance: str | None = None
    """MENTIONED -> OPENED -> INSPECTED -> CITED -> CLAIM_SUPPORTED; None = no AI cited it (OmniBrain-only)."""


class Disagreement(BaseModel):
    id: str = Field(default_factory=lambda: new_id("dis"))
    job_id: str
    round: int = 1
    topic: str
    description: str
    positions: dict[str, list[str]] = Field(default_factory=dict)
    """provider -> claim ids / claim texts that hold this position"""

    severity: str = "minor"
    """material | minor -- only material ones reach the final answer"""

    claim_ids: list[str] = Field(default_factory=list)
    resolution: str | None = None
    resolved_by_round: int | None = None


class FollowUp(BaseModel):
    """A targeted question generated from an unresolved conflict (section 14)."""

    id: str = Field(default_factory=lambda: new_id("fu"))
    job_id: str
    question: str
    reason: str
    target_providers: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    disagreement_id: str | None = None
    round: int = 1


class ClaimVerdict(BaseModel):
    claim_id: str
    claim: str
    verdict: ClaimStatus
    confidence: Confidence
    reasoning: str
    strong_evidence: list[str] = Field(default_factory=list)
    weak_or_bad_evidence: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    """Detected issues: citation_mismatch, outdated, unsupported_inference,
    exaggeration, hallucinated_citation, secondary_misrepresents_primary..."""


class VerifierReport(BaseModel):
    """Output of the adversarial engine. It argues with the evidence, not with
    the other models -- majority votes carry no weight here."""

    job_id: str
    round: int = 1
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    answer: str = ""
    why: str = ""
    important_disagreement: str | None = None
    confidence: Confidence = Confidence.LOW
    confidence_note: str | None = None
    sources: list[Citation] = Field(default_factory=list)

    needs_more_research: bool = False
    follow_ups: list[FollowUp] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    raw_output: str = ""
    verifier_model: str | None = None

    reviewer_status: str = "NOT_RUN"
    """COMPLETED (the model answered usably) | UNAVAILABLE (error, 429, timeout) | INVALID_OUTPUT (answered, unusable) |
    NOT_RUN (the evidence settled it; no reviewer was needed)."""

    synthesis_status: str = "DETERMINISTIC"
    """CURATED (the model wrote the verdicts) | FALLBACK (the model failed, ledger wrote them) | DETERMINISTIC (by design)."""

    fallback_reason: str = ""
    """Why a fallback happened, in the words of the failure. Empty when there was none."""


class ConversationTurn(BaseModel):
    """One earlier question in the same thread: what was asked, what we answered,
    and the claims our own ledger confirmed -- never what the providers merely said."""

    job_id: str
    question: str
    answer: str = ""
    confirmed_claims: list[str] = Field(default_factory=list)
    confidence: str = ""
    at: float = 0.0


class QuestionAnalysis(BaseModel):
    question: str
    intent: str = "factual"
    """factual | current_events | comparison | technical | product | legal |
    scientific | ranking | opinion | creative | chitchat"""

    needs_web_research: bool = True
    needs_current_data: bool = False
    time_sensitivity: str = "medium"
    difficulty: str = "medium"
    key_entities: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    ambiguity: str | None = None
    first_principles: list[str] = Field(default_factory=list)
    rationale: str | None = None

    # Escalation routing. A stable, self-contained question must stop at level 0
    # with zero browser sessions: the classifier answers it and we are done.
    trivial: bool = False
    can_answer_directly: bool = False
    direct_answer: str | None = None
    high_stakes: bool = False
    stakes: StakesDomain = StakesDomain.NONE
    starting_level: EscalationLevel = EscalationLevel.PRIMARY
    capabilities: list[str] = Field(default_factory=list)
    """what the primary agent needs: web_search | images | code | long_context |
    reasoning | official_documents"""

    classifier_source: str = "heuristic"
    """heuristic | llm | user -- records how the routing decision was made"""

    follow_up: bool = False
    """True when the question only makes sense against an earlier turn of the thread"""

    standalone_question: str | None = None
    """The follow-up rewritten so it can be researched without the thread"""

    inherited_entities: list[str] = Field(default_factory=list)
    """Names carried over from the earlier turn that this question refers to"""


class SufficiencyAssessment(BaseModel):
    """The stopping test: is the remaining uncertainty material to the answer?

    A confident *tone* is never a reason to stop. Only confirmed evidence plus
    the absence of unresolved conflicts is.
    """

    job_id: str
    round: int = 1
    sufficient: bool = False
    signals: list[str] = Field(default_factory=list)
    failure_signals: list[str] = Field(default_factory=list)
    established: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    """material claims no opened source confirmed"""

    unanswered_subquestions: list[str] = Field(default_factory=list)
    """parts of a multi-part question that no answer addressed at all"""
    unanswered_subquestions: list[str] = Field(default_factory=list)
    confirmed_sources: int = 0
    independent_domains: int = 0
    has_primary_source: bool = False
    contradictions: int = 0
    coverage: float = 0.0
    strong_primary: bool = False
    """Every material claim is backed by a primary page the AI itself opened (provenance CLAIM_SUPPORTED)."""

    reason: str = ""
    recommends_level: EscalationLevel = EscalationLevel.PRIMARY


class ResearchPlan(BaseModel):
    job_id: str
    analysis: QuestionAnalysis
    providers: list[str] = Field(default_factory=list)
    prompts: dict[str, str] = Field(default_factory=dict)
    """provider -> its own tailored research instruction (section 9: never send
    the identical prompt blindly to everyone)"""

    search_queries: list[str] = Field(default_factory=list)
    verification_priorities: list[str] = Field(default_factory=list)
    max_rounds: int = 3


class RoundRecord(BaseModel):
    number: int
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    kind: str = "initial"
    """initial | follow_up"""
    response_ids: list[str] = Field(default_factory=list)
    follow_up_ids: list[str] = Field(default_factory=list)
    summary: str | None = None


class JobEvent(BaseModel):
    ts: float = Field(default_factory=time.time)
    kind: str
    """status | provider | round | claim | evidence | disagreement | verifier |
    final | error | log"""

    message: str = ""
    provider: str | None = None
    round: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(BaseModel):
    """Section 18 -- concise by default, full detail behind expanders."""

    answer: str = ""
    why: str = ""
    important_disagreement: str | None = None
    confidence: Confidence = Confidence.LOW
    confidence_label: str = "Low confidence"
    sources: list[Citation] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    rounds_run: int = 1
    providers_used: list[str] = Field(default_factory=list)
    providers_failed: list[str] = Field(default_factory=list)

    truth_state: str = ""
    """TRUE | PARTLY | FALSE | CONFLICT | UNVERIFIED, from the evidence ledger (empty when nothing was checked)."""

    research_status: str = "COMPLETED"
    """NOT_NEEDED (answered at level 0) | COMPLETED (settled) | UNRESOLVED (researched, not settled) | BLOCKED (no AI answered)."""

    reviewer_status: str = "NOT_RUN"
    synthesis_status: str = "DETERMINISTIC"
    fallback_reason: str = ""


class Complaint(BaseModel):
    """A complaint that recurs across independent review pages (community tier, never a fact)."""

    theme: str
    mentions: int = 0
    sources: int = 0
    domains: list[str] = Field(default_factory=list)
    example: str = ""
    example_url: str = ""


class ReviewFindings(BaseModel):
    """What owners say, gathered separately from the fact ledger (product/service questions)."""

    attempted: bool = False
    subject: str = ""
    queries: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    pages_read: int = 0
    complaints: list[Complaint] = Field(default_factory=list)
    note: str = ""


class Job(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    question: str
    conversation_id: str | None = None
    history: list[ConversationTurn] = Field(default_factory=list)
    """Earlier turns of this conversation (oldest first), loaded when the job starts."""

    mode: ResearchMode = ResearchMode.STANDARD
    status: JobStatus = JobStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    finished_at: float | None = None

    project: str | None = None
    """Optional project namespace: project memories are only offered to questions asked in that project."""
    memory_used: list[dict[str, Any]] = Field(default_factory=list)
    """Which memories were offered to the AIs for this question and why (never evidence)."""
    memory_kind: str = ""
    memory_context: str = Field(default="", exclude=True)
    thread_id: str | None = None
    """The OmniBrain thread this question belongs to (context only; see backend/thread)."""
    thread_used: dict[str, Any] = Field(default_factory=dict)
    max_rounds: int = 3
    stop_note: str = ""
    """Why the loop ended early when a stop rule (not the round limit) ended it."""
    rounds_run: int = 0
    active_round: int = 0

    plan: ResearchPlan | None = None
    reviews: ReviewFindings | None = None
    analysis: QuestionAnalysis | None = None
    level: EscalationLevel = EscalationLevel.DIRECT
    escalation_log: list[EscalationStep] = Field(default_factory=list)
    assessments: list[SufficiencyAssessment] = Field(default_factory=list)
    stop_reason: str | None = None
    verifier_calls: int = 0
    browser_sessions_used: int = 0
    rounds: list[RoundRecord] = Field(default_factory=list)
    responses: list[ProviderResponse] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    follow_ups: list[FollowUp] = Field(default_factory=list)
    corrections: list[SelfCorrection] = Field(default_factory=list)
    reports: list[VerifierReport] = Field(default_factory=list)
    final: FinalAnswer | None = None
    error: str | None = None
