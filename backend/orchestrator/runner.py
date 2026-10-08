"""Adaptive research runner.

Sequential first, parallel on failure. The objective is the highest reliable
accuracy for the *least* necessary investigation, so every escalation has to be
justified by a concrete signal: an unresolved claim, a citation that failed
checking, a real conflict, or consequence (legal/medical/financial/visa).

Nothing here runs the verifier just because it can, and nothing stops just
because an answer sounded confident.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import traceback
from typing import Any, Awaitable, Callable, Protocol

from backend.evidence.pool import build_pool
from backend.models import (
    Claim,
    ClaimStatus,
    Confidence,
    Disagreement,
    EscalationLevel,
    EscalationStep,
    Evidence,
    FinalAnswer,
    FollowUp,
    Job,
    JobStatus,
    ProviderResponse,
    ProviderStatus,
    ResearchMode,
    RoundRecord,
    SourceCheckStatus,
    SufficiencyAssessment,
    VerifierReport,
    WebResearchStatus,
)
from backend.research import claims as claim_ops
from backend.research import router
from backend.cancel import CancelToken
from backend.evidence.pool import browser_fetch_factory
from backend.orchestrator.politeness import PolitenessGate
from backend.evidence.reviews import caveat_lines as review_caveats
from backend.evidence.reviews import gather_reviews, subject_for
from backend.research import memory
from backend.research import citations as citation_ops
from backend.research import corrections as correction_ops
from backend.research.prompts import angle_for, escalation_prompt, follow_up_prompt, research_prompt, thread_follow_up_prompt
from backend.research.style import style_prompt
from backend.verification.llm import Endpoint, LLMClient
from backend.verification.verifier import Verifier, build_final_answer, confidence_label
from backend.settings import Settings

MATERIAL_KINDS = {"statistic", "date", "legal", "scientific", "product", "ranking", "causal", "contested"}

STABLE_KNOWLEDGE_ASK = """You are the level-0 classifier for a research system.

Decide ONE thing: can you answer this question reliably right now, with no web
access and no browser research?

Answer directly only if the facts involved are stable and you are genuinely
certain (standard definitions, programming syntax you use routinely, simple
transformations, arithmetic, or a question that is banter/opinion/creative and
needs no facts at all).

Answer UNRESEARCHABLE if the question involves anything current, contested,
numeric-and-recent, niche, or anything where you would be reconstructing rather
than knowing. Do not guess. Do not produce a confident answer to look useful.

Reply with strict JSON: {"can_answer": true|false, "answer": "..." , "why": "..."}"""


class Adapter(Protocol):
    provider: str

    async def ask(self, job_id: str, prompt: str, round_no: int = 1, emit: Any = None) -> ProviderResponse: ...


class EventBus(Protocol):
    async def emit(self, kind: str, message: str, provider: str | None = None, round_no: int | None = None, **payload: Any) -> None: ...


class NullBus:
    async def emit(self, kind: str, message: str, provider: str | None = None, round_no: int | None = None, **payload: Any) -> None:
        return None


class ResearchRunner:
    def __init__(
        self,
        settings: Settings,
        adapters: dict[str, Adapter],
        *,
        engine: Any = None,
        bus: EventBus | None = None,
        health: dict[str, str] | None = None,
        verifier: Verifier | None = None,
        analysis_endpoint: Endpoint | None = None,
        store: Any = None,
        cancel: CancelToken | None = None,
        politeness: PolitenessGate | None = None,
        memory: Any = None,
        threads: Any = None,
    ) -> None:
        self.settings = settings
        self.threads = threads  # ThreadService | None -- the canonical thread; its packet is context only, never evidence
        self.memory = memory  # MemoryService | None -- personalisation context only; never evidence
        self.adapters = adapters
        self.engine = engine
        self.bus = bus or NullBus()
        # what earlier runs saw: it orders the candidates but never removes one (a wall can lift)
        self.prior_health = dict(health or {})
        self.health: dict[str, str] = {}
        self.verifier = verifier
        self.analysis_endpoint = analysis_endpoint
        self.store = store
        self._sem = asyncio.Semaphore(max(1, settings.research.max_workers))
        self._provider_sems: dict[str, asyncio.Semaphore] = {}
        self._turns: dict[tuple[str, str], int] = {}
        self.cancel = cancel or CancelToken()
        self.politeness = politeness or PolitenessGate(settings.research)
        # Set when the live browser driver itself is not usable (e.g. the extension is not connected in the chosen
        # browser). Every provider would fail the same way: stop asking and tell the user what to fix.
        self._driver_down: str | None = None
        for adapter in adapters.values():
            if hasattr(adapter, "cancel_token"):
                adapter.cancel_token = self.cancel

    # ------------------------------------------------------------------ entry

    async def run(self, job: Job) -> Job:
        try:
            finished = await self._run(job)
            await self._learn(finished)
            await self._record_thread(finished)
            return finished
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.stop_reason = "cancelled by user"
            await self._emit("status", "cancelled", job=job)
            raise
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            # An unexpected crash must leave its traceback behind: a bare "IndexError: list index out of range" in an
            # event row cannot be debugged (live: ua-closed-session failed this way and left nothing to follow).
            tb = traceback.format_exc()
            logging.getLogger("omnibrain.runner").error("research job %s failed:\n%s", job.id, tb)
            await self._emit("error", f"research job failed: {job.error}", job=job, traceback=tb[-3000:])
            return job

    async def _run(self, job: Job) -> Job:
        enabled = [name for name, cfg in self.settings.providers.items() if cfg.enabled and name in self.adapters]
        if not enabled:
            job.status = JobStatus.FAILED
            job.error = "no providers enabled"
            return job

        job.status = JobStatus.ANALYZING
        await self._emit("status", "classifying question", job=job)
        analysis = await self._analyze(job.question, job.mode, enabled, history=job.history)
        job.analysis = analysis
        job.level = analysis.starting_level

        if analysis.can_answer_directly and analysis.direct_answer:
            job.status = JobStatus.COMPLETED
            job.level = EscalationLevel.DIRECT
            job.escalation_log.append(
                EscalationStep(level=EscalationLevel.DIRECT, reason=analysis.rationale or "level 0", triggered_by=["trivial"])
            )
            await self._emit("escalation", f"level 0 -- answered without research: {analysis.rationale}", job=job)
            job.final = FinalAnswer(
                answer=analysis.direct_answer,
                why=None or "",
                confidence=Confidence.HIGH if analysis.classifier_source == "computed" else Confidence.MODERATE,
                confidence_label="High confidence" if analysis.classifier_source == "computed" else "Moderate confidence",
                caveats=[] if analysis.classifier_source == "computed" else ["Answered from stable knowledge without research -- ask 'why' to have it checked."],
                rounds_run=0,
                providers_used=[],
                providers_failed=[],
                research_status="NOT_NEEDED",
                reviewer_status="NOT_RUN",
                synthesis_status="DIRECT",
            )
            job.stop_reason = "question answered at level 0; no investigation earned"
            await self._emit("final", job.final.answer, job=job)
            return job

        await self._prepare_memory(job)
        await self._prepare_thread(job)

        if not analysis.needs_web_research and analysis.trivial:
            # banter with no model configured: one provider, no research framing.
            primary = router.select_primary(analysis, enabled, self.health, self.prior_health)
            if primary:
                response = await self._ask(primary, self._banter_prompt(job, analysis), 1, role="primary", needs_web=False, job_id=job.id)
                if response and response.status.value == "completed":
                    job.status = JobStatus.COMPLETED
                    job.final = FinalAnswer(
                        answer=response.answer_text[:1200],
                        confidence=Confidence.MODERATE,
                        confidence_label="Moderate confidence",
                        rounds_run=1,
                        providers_used=[primary],
                        providers_failed=[],
                    )
                    job.browser_sessions_used = 1
                    job.stop_reason = "non-factual ask handled by one provider; no research required"
                    await self._emit("final", job.final.answer, job=job)
                    return job

        return await self._investigate(job, analysis, enabled)

    # --------------------------------------------------------------- level 1+

    async def _driver_down_fail(self, job: Job) -> Job:
        job.status = JobStatus.FAILED
        job.error = f"the live browser is not ready: {self._driver_down}"
        job.stop_reason = "live browser not ready; no provider could be asked"
        await self._emit("error", job.error, job=job)
        return job

    async def _investigate(self, job: Job, analysis: Any, enabled: list[str]) -> Job:
        if self._driver_down:
            return await self._driver_down_fail(job)
        max_rounds = max(1, job.max_rounds)
        job.status = JobStatus.RESEARCHING
        # "Answer now" shortcuts skip the provider's own research; acceptable in
        # QUICK, never in a round whose job is to establish evidence.
        for adapter in self.adapters.values():
            setattr(adapter, "quick_answer_allowed", job.mode == ResearchMode.QUICK)
        primary = router.select_primary(analysis, enabled, self.health, self.prior_health)
        if not primary:
            job.status = JobStatus.FAILED
            job.error = "no healthy provider available"
            return job

        await self._emit(
            "escalation",
            f"level {int(job.level.value) if job.level else 1} -- primary researcher: {primary}",
            provider=primary,
            job=job,
        )
        job.escalation_log.append(
            EscalationStep(
                level=EscalationLevel.PRIMARY,
                reason=f"question needs external evidence ({analysis.intent}); {primary} gets first shot alone",
                providers=[primary],
                triggered_by=["needs_web_research"] if analysis.needs_web_research else ["non_factual"],
            )
        )

        round_no = 1
        record = RoundRecord(number=round_no, kind="initial")
        job.rounds.append(record)
        job.active_round = round_no

        prompt = research_prompt(
            job.question,
            provider=primary,
            analysis=analysis,
            round_no=round_no,
            needs_web=analysis.needs_web_research,
            history=job.history if analysis.follow_up else None,
            context=job.memory_context,
        )
        response = await self._ask(primary, prompt, round_no, role="primary", job_id=job.id)
        if self._driver_down:
            job.responses.extend([r for r in [response] if r])
            return await self._driver_down_fail(job)
        responses: list[ProviderResponse] = [r for r in [response] if r]
        job.responses.extend(responses)
        record.response_ids = [r.id for r in responses]
        job.browser_sessions_used += len({r.provider for r in responses})

        claims = await self._extract(responses, job)
        evidence, trace = await self._pool(job, claims, responses, round_no)
        await self._gather_reviews(job, analysis)
        disagreements = await self._disagreements(job, claims, round_no)
        assessment = self._assess(job, analysis, claims, evidence, disagreements, round_no, focus="primary")
        job.assessments.append(assessment)
        await self._emit(
            "assessment",
            f"primary {assessment.reason}",
            provider=primary,
            round_no=round_no,
            job=job,
            sufficient=assessment.sufficient,
        )

        verifier_enabled = self.verifier is not None and self.settings.verifier.provider != "disabled"

        # Enough already? Then stop. Do not spawn a swarm to feel thorough.
        if self._settled(assessment, disagreements) and not analysis.high_stakes:
            report = await self._lightweight_report(job, analysis, claims, evidence, responses, round_no)
            return self._complete(job, report, responses, rounds=round_no, stop="evidence sufficient after the primary researcher; no escalation earned")

        # FIRST escalation: the same AI, in the same conversation, looks again.
        # Parallelism comes only after that has failed to settle it.
        if not assessment.sufficient:
            before = len(job.responses)
            (round_no, responses, claims, evidence, disagreements, assessment) = await self._thread_follow_up(
                job, analysis, primary, round_no, responses, claims, evidence, disagreements, assessment
            )
            if len(job.responses) > before:
                max_rounds += 1  # the follow-up is a step of its own, not a round taken from the budget
            if self._settled(assessment, disagreements) and not analysis.high_stakes:
                report = await self._lightweight_report(job, analysis, claims, evidence, responses, round_no)
                return self._complete(job, report, responses, rounds=round_no, stop="settled by a same-conversation follow-up with the primary researcher")

        # Level 2 -- expansion, earned by failure, never assumed.
        job.level = EscalationLevel.PARALLEL
        partial = bool(assessment.established) and bool(assessment.unresolved)
        if partial:
            # Most of it is settled. Investigate only what is not -- re-running
            # every provider on the whole question wastes sessions and re-asks
            # claims the ledger already confirmed.
            (
                round_no,
                responses,
                claims,
                evidence,
                disagreements,
                assessment,
            ) = await self._targeted_expansion(job, analysis, enabled, primary, round_no, responses, claims, evidence, disagreements, assessment)

        secondaries = router.select_secondaries(
            analysis,
            enabled,
            exclude=[primary],
            count=2 if job.mode == ResearchMode.QUICK else self.settings.research.swarm_providers,
            health=self.health, prior=self.prior_health, reprobe=True,
        )
        # A full swarm is for the case where nothing was established at all.
        # After a targeted expansion, what remains is a conflict for the verifier,
        # not a gap more researchers would fill.
        # Sufficient but high-stakes goes to the curator, not to a swarm: parallel research is for an unresolved claim.
        if secondaries and not partial and not assessment.sufficient:
            context = self._failure_context(responses, claims, evidence, disagreements, assessment)
            job.escalation_log.append(
                EscalationStep(
                    level=EscalationLevel.PARALLEL,
                    reason=f"primary could not close it: {assessment.reason}",
                    providers=secondaries,
                    triggered_by=assessment.failure_signals[:8],
                    round=round_no,
                )
            )
            await self._emit(
                "escalation",
                f"level 2 -- escalating to {len(secondaries)} independent researchers: {', '.join(secondaries)}",
                round_no=round_no,
                job=job,
            )
            extra = await self._ask_many(secondaries, job, analysis, context, round_no, exclude=primary)
            job.browser_sessions_used += len(extra)
            responses.extend(extra)
            job.responses = responses
            record.response_ids.extend(r.id for r in extra)
            claims = await self._extract(responses, job)
            more_evidence, more_trace = await self._pool(job, claims, extra or responses, round_no)
            evidence = _merge_evidence(evidence, more_evidence)
            disagreements = await self._disagreements(job, claims, round_no)
            assessment = self._assess(job, analysis, claims, evidence, disagreements, round_no, focus="swarm")
            job.assessments.append(assessment)
            await self._emit(
                "assessment",
                f"after parallel research: {assessment.reason}",
                round_no=round_no,
                job=job,
                sufficient=assessment.sufficient,
            )
            if (
                assessment.sufficient
                and not analysis.high_stakes
                and not disagreements
                and not verifier_enabled
            ):
                report = await self._lightweight_report(job, analysis, claims, evidence, responses, round_no)
                return self._complete(job, report, responses, rounds=round_no, stop="sufficient evidence without needing the verifier")

        # Enough now, and nothing conflicts? Skip the expensive engine entirely.
        if self._settled(assessment, disagreements) and not analysis.high_stakes:
            report = await self._lightweight_report(job, analysis, claims, evidence, responses, round_no)
            return self._complete(
                job, report, responses, rounds=round_no,
                stop="evidence sufficient after expansion; running the verifier would have added cost, not certainty",
            )

        # Level 3 -- adversarial verification, then targeted rounds.
        report: VerifierReport | None = None
        progress = (len(assessment.established), assessment.confirmed_sources)
        stalled = 0
        while round_no <= max_rounds:
            job.status = JobStatus.VERIFYING
            job.level = EscalationLevel.DEEP if (analysis.high_stakes or disagreements) else job.level
            if job.mode == ResearchMode.QUICK and round_no > 1:
                break
            report = await self._verify(job, analysis, claims, evidence, responses, disagreements, round_no)
            job.reports.append(report)
            job.verifier_calls += 1
            await self._emit(
                "verifier",
                f"round {round_no}: {report.confidence_label if hasattr(report, 'confidence_label') else report.confidence.value}"
                + (" -- requesting more research" if report.needs_more_research else ""),
                round_no=round_no,
                job=job,
            )

            actionable = [f for f in report.follow_ups if f.question]
            if not report.needs_more_research or not actionable or round_no >= max_rounds:
                break
            # Stop rules: the curator may ask, but it does not get to keep asking.
            if assessment.strong_primary:
                job.stop_note = "the AI opened a primary source that supports every claim; the curator's request for more research was not followed"
                break
            if stalled >= 2:
                job.stop_note = "the last two targeted rounds added no new confirmed claim or source; stopped instead of asking again"
                break

            # Target only what is unresolved. Never restart the whole thing.
            next_round = round_no + 1
            job.status = JobStatus.FOLLOWUP
            job.level = EscalationLevel.DEEP
            job.escalation_log.append(
                EscalationStep(
                    level=EscalationLevel.DEEP,
                    reason=f"verifier found {len(actionable)} unresolved point(s); researching those specifically",
                    providers=[],
                    triggered_by=[u[:60] for u in report.unresolved[:4]] or ["needs_more_research"],
                    round=next_round,
                )
            )
            await self._emit(
                "escalation",
                f"level 3 round {next_round} -- targeted research on {len(actionable)} unresolved point(s), not a re-ask",
                round_no=next_round,
                job=job,
            )
            # A site that already hit a login wall, a block or a broken page in this job will do so again; do not
            # spend another round (up to 90 s for Google AI Mode) finding that out.
            unusable = self._unusable_this_job(responses)
            # Researchers nobody has asked yet come first: a minority holding the evidence must get a turn.
            asked = {r.provider for r in responses}
            targets = router.select_secondaries(
                analysis,
                [p for p in enabled if p not in unusable],
                exclude=sorted(asked),
                count=max(2, self.settings.research.follow_up_providers),
                health=self.health, prior=self.prior_health,
            )
            if len(targets) < 2 or not any(t not in asked for t in targets):
                targets = router.select_secondaries(
                    analysis,
                    [p for p in enabled if p not in unusable],
                    exclude=[primary] if round_no == 1 else [],
                    count=max(2, self.settings.research.follow_up_providers),
                    health=self.health, prior=self.prior_health,
                )
            follow_round = RoundRecord(number=next_round, kind="follow_up")
            job.rounds.append(follow_round)
            job.active_round = next_round
            job.follow_ups.extend(actionable)
            follow_round.follow_up_ids = [f.id for f in actionable]

            targeted = await self._targeted_round(job, actionable, targets, round_no=next_round)
            responses.extend(targeted)
            job.responses = responses
            follow_round.response_ids = [r.id for r in targeted]
            job.browser_sessions_used += len({r.provider for r in targeted})

            claims = await self._extract(responses, job)
            more_evidence, _ = await self._pool(job, claims, targeted or responses, next_round)
            evidence = _merge_evidence(evidence, more_evidence)
            disagreements = await self._disagreements(job, claims, next_round)
            assessment = self._assess(job, analysis, claims, evidence, disagreements, next_round, focus="targeted")
            job.assessments.append(assessment)
            follow_round.finished_at = time.time()
            now = (len(assessment.established), assessment.confirmed_sources)
            stalled = stalled + 1 if (now[0] <= progress[0] and now[1] <= progress[1]) else 0
            progress = (max(progress[0], now[0]), max(progress[1], now[1]))
            round_no = next_round
            if self._settled(assessment, disagreements) and not analysis.high_stakes:
                report = await self._verify(job, analysis, claims, evidence, responses, disagreements, round_no)
                job.reports.append(report)
                job.verifier_calls += 1
                break

        if report is None:
            report = await self._lightweight_report(job, analysis, claims, evidence, responses, round_no)
        return self._complete(
            job,
            report,
            responses,
            rounds=round_no,
            stop=self._stop_reason(assessment, disagreements, analysis, round_no, max_rounds, job.stop_note),
        )

    # ----------------------------------------------- same-conversation follow-up

    async def _thread_follow_up(
        self,
        job: Job,
        analysis: Any,
        primary: str,
        round_no: int,
        responses: list[ProviderResponse],
        claims: list[Claim],
        evidence: list[Evidence],
        disagreements: list[Disagreement],
        assessment: SufficiencyAssessment,
    ):
        """Ask the primary AI, in its own conversation, to re-investigate what is unsettled."""
        unchanged = (round_no, responses, claims, evidence, disagreements, assessment)
        first = next((r for r in responses if r.provider == primary and r.status.value == "completed" and not r.superseded), None)
        if first is None or self._turns.get((job.id, primary), 0) < 1:
            return unchanged
        points = [p for p in assessment.unresolved[:3] if p] or [correction_ops.headline(first) or self._q(job)]
        claim_text = "\n".join(f"- {p}" for p in points)
        next_round = round_no + 1
        record = RoundRecord(number=next_round, kind="thread_follow_up")
        job.rounds.append(record)
        job.active_round = next_round
        job.escalation_log.append(
            EscalationStep(
                level=EscalationLevel.PRIMARY,
                reason=f"first escalation: {primary} re-investigates in the same conversation ({assessment.reason})",
                providers=[primary],
                triggered_by=(assessment.failure_signals[:6] or ["not_sufficient"]),
                round=next_round,
            )
        )
        await self._emit(
            "escalation",
            f"follow-up in the same {primary} conversation -- asking it to re-check: {points[0][:80]}",
            provider=primary,
            round_no=next_round,
            job=job,
        )
        prompt = thread_follow_up_prompt(claim_text, f"Why I am asking: {assessment.reason}.")
        follow = await self._ask(
            primary, prompt, next_round, role="thread_follow_up", job_id=job.id, continue_thread=True,
            escalation_reason="same-conversation follow-up",
        )
        if follow is None:
            return unchanged
        job.responses.append(follow)
        responses = list(job.responses)
        record.response_ids = [follow.id]
        if follow.status.value != "completed":
            record.finished_at = time.time()
            return (next_round, responses, claims, evidence, disagreements, assessment)
        correction = correction_ops.build_record(first, follow, round_no=next_round)
        job.corrections.append(correction)
        if correction.verdict == "corrected":
            first.superseded = True
        await self._emit(
            "correction",
            f"{primary} {correction.verdict.replace('_', ' ')} its earlier answer",
            provider=primary,
            round_no=next_round,
            job=job,
        )
        claims = await self._extract(responses, job)
        more_evidence, _ = await self._pool(job, claims, [follow], next_round)
        evidence = _merge_evidence(evidence, more_evidence)
        disagreements = await self._disagreements(job, claims, next_round)
        assessment = self._assess(job, analysis, claims, evidence, disagreements, next_round, focus="follow-up")
        job.assessments.append(assessment)
        record.finished_at = time.time()
        await self._emit("assessment", f"after the follow-up {assessment.reason}", provider=primary, round_no=next_round, job=job, sufficient=assessment.sufficient)
        return (next_round, responses, claims, evidence, disagreements, assessment)

    # ------------------------------------------------------------ components

    def _provider_slot(self, provider: str) -> asyncio.Semaphore:
        """``research.per_provider_concurrency`` simultaneous questions to one site (default 1)."""
        slot = self._provider_sems.get(provider)
        if slot is None:
            slot = self._provider_sems[provider] = asyncio.Semaphore(max(1, int(self.settings.research.per_provider_concurrency)))
        return slot

    @staticmethod
    def _q(job: Job) -> str:
        """The question as research should see it: a follow-up with its references resolved."""
        analysis = job.analysis
        if analysis is not None and getattr(analysis, "follow_up", False) and analysis.standalone_question:
            return analysis.standalone_question
        return job.question

    async def _gather_reviews(self, job: Job, analysis: Any) -> None:
        """Product/service questions: what owners say, kept apart from the fact ledger (spec 8)."""
        self.cancel.raise_if_cancelled()
        if (
            job.reviews is not None
            or not self.settings.search.own_discovery
            or getattr(analysis, "intent", "") != "product"
            or job.mode == ResearchMode.QUICK
            or int(self.settings.search.review_queries) <= 0
        ):
            return
        await self._emit("status", "product question -- reading owner reviews (kept separate from the facts)", job=job)
        browser_fetch = browser_fetch_factory(self.engine, self.settings) if self.engine is not None else None
        job.reviews = await gather_reviews(
            subject_for(job.question, list(getattr(analysis, "key_entities", []) or [])),
            self.settings,
            deep=job.mode == ResearchMode.DEEP_RESEARCH,
            browser_fetch=browser_fetch,
            emit=self._adapter_emit,
        )

    async def _prepare_memory(self, job: Job) -> None:
        """Offer relevant things the user told us to every AI that starts a conversation -- as prompt context only.

        The block lives on `job.memory_context` and in prompts. It is never added to claims, evidence, the verifier's
        payload, sources or the answer: memory is not evidence.
        """
        job.memory_context, job.memory_used, job.memory_kind = "", [], ""
        if self.memory is None:
            return
        try:
            block, res = self.memory.context_for(job.question, project=job.project, conversation=job.conversation_id)
        except Exception as exc:  # noqa: BLE001 -- memory must never break research
            await self._emit("status", f"memory unavailable ({type(exc).__name__}); researching without it", job=job)
            return
        job.memory_kind = res.kind
        job.memory_context = block
        job.memory_used = [h.public() for h in res.hits]
        if res.hits:
            await self._emit("memory", f"using {len(res.hits)} remembered thing(s) as context (not evidence)", job=job, memory_kind=res.kind, ids=[h.memory.memory_id for h in res.hits])

    async def _prepare_thread(self, job: Job) -> None:
        """Questions asked inside an OmniBrain thread start their fresh provider chats with the thread's continuation context.

        Same rules as memory: prompt context only, never a claim, evidence, source or verifier input. A job without a thread_id
        gets nothing, and a thread's context is built from that thread's own messages only.
        """
        job.thread_used = {}
        if self.threads is None or not job.thread_id:
            return
        try:
            packet = self.threads.context_for_job(job.thread_id, job.question, with_user_lines=False, include_task=False)
        except Exception as exc:  # noqa: BLE001 -- a broken thread store must never break research
            await self._emit("status", f"thread context unavailable ({type(exc).__name__}); researching without it", job=job)
            return
        if packet is None:
            return
        job.memory_context = packet.text + (("\n\n" + job.memory_context) if job.memory_context else "")
        job.thread_used = {"tokens": packet.tokens, "sections": packet.sections}
        await self._emit("memory", "continuing the thread: earlier conversation supplied as context (not evidence)", job=job, thread_tokens=packet.tokens)

    async def _record_thread(self, job: Job) -> None:
        """The user's message always joins the canonical thread; the final answer joins it when there is one."""
        if self.threads is None or not job.thread_id:
            return
        try:
            answer = job.final.answer if (job.final and job.status == JobStatus.COMPLETED) else ""
            self.threads.record_job(job.thread_id, job.question, answer, job_id=job.id)
        except Exception:  # noqa: BLE001
            pass

    async def _learn(self, job: Job) -> None:
        """After the conversation: keep what the USER said (never what an AI answered), subject to their settings."""
        if self.memory is None or job.status not in (JobStatus.COMPLETED,):
            return
        try:
            learned = self.memory.learn(job.question, project=job.project, conversation=job.conversation_id)
        except Exception:  # noqa: BLE001
            return
        kept = [r for r in learned.added if r.memory is not None and r.action in {"created", "superseded"}]
        if kept or learned.forgotten:
            await self._emit("memory", f"memory updated: {len(kept)} saved, {len(learned.forgotten)} forgotten", job=job)

    @staticmethod
    def _with_context(job: Job, prompt: str) -> str:
        return f"{job.memory_context}\n\n{prompt}" if job.memory_context else prompt

    @staticmethod
    def _banter_prompt(job: Job, analysis: Any) -> str:
        ctx = f"{job.memory_context}\n\n" if job.memory_context else ""
        if getattr(analysis, "follow_up", False) and job.history:
            return f"{ctx}{memory.history_block(job.history)}\n\n{job.question}"
        return f"{ctx}{job.question}"

    async def _rewrite_follow_up(self, question: str, prior: Any) -> str | None:
        """Ask the analysis model to resolve "they"/"it"; the heuristic rewrite stands if it can't."""
        endpoint = self.analysis_endpoint
        if not (endpoint and endpoint.enabled):
            return None
        try:
            parsed, _reply = await LLMClient(endpoint).complete_json(memory.rewrite_messages(question, prior), temperature=0.0)
        except Exception:  # noqa: BLE001 -- the heuristic rewrite is a fine fallback
            return None
        return memory.accept_rewrite((parsed or {}).get("standalone"), question)

    async def _analyze(self, question: str, mode: ResearchMode, enabled: list[str], history: list[Any] | None = None) -> Any:
        hint: str | None = None
        endpoint = self.analysis_endpoint
        prior = history[-1] if history else None
        follow_up = memory.is_follow_up(question, prior)
        if endpoint and endpoint.enabled and not follow_up and not _obviously_arithmetic(question):
            client = LLMClient(endpoint)
            parsed, reply = await client.complete_json(
                [
                    {"role": "system", "content": STABLE_KNOWLEDGE_ASK},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                temperature=0.0,
            )
            if parsed and parsed.get("can_answer") and str(parsed.get("answer") or "").strip():
                hint = str(parsed["answer"]).strip()
        analysis = router.classify(question, mode=mode, enabled=enabled, llm_stable_answer=hint, history=history or ())
        if analysis.follow_up and prior is not None:
            rewritten = await self._rewrite_follow_up(question, prior)
            if rewritten:
                analysis.standalone_question = rewritten
        if hint:
            analysis.classifier_source = "llm"
        computed = analysis.starting_level == EscalationLevel.DIRECT and analysis.trivial and analysis.classifier_source == "heuristic"
        if computed and analysis.direct_answer:
            analysis.classifier_source = "computed"
        return analysis

    async def _ask(
        self,
        provider: str,
        prompt: str,
        round_no: int,
        *,
        role: str = "secondary",
        needs_web: bool = True,
        escalation_reason: str | None = None,
        job_id: str = "job",
        continue_thread: bool = False,
    ) -> ProviderResponse | None:
        adapter = self.adapters.get(provider)
        if adapter is None:
            return None
        self.cancel.raise_if_cancelled()
        # One conversation per AI per research id. A follow-up goes into the thread
        # that AI already has for this research; everything else opens a new chat.
        thread_key = (job_id, provider)
        # per-provider first, then the global pool: waiting on one busy site must not hold a global slot
        async with self._provider_slot(provider), self._sem:
            # numbered inside the slot: two requests to one AI in the same round are two turns, not one
            turn = self._turns.get(thread_key, 0) + 1
            continue_thread = bool(continue_thread and turn > 1)
            declined = None if self._driver_down else await self.politeness.before(provider, sleep=self.cancel.sleep)
            if self._driver_down:
                response = ProviderResponse(id=f"resp_skip_{provider}_{round_no}", job_id="", round=round_no, provider=provider, prompt=prompt)
                response.status = ProviderStatus.FAILED
                response.error = f"LiveChromeUnavailable: {self._driver_down}"
                response.answer_text = ""
            elif declined:
                response = ProviderResponse(id=f"resp_skip_{provider}_{round_no}", job_id="", round=round_no, provider=provider, prompt=prompt)
                response.status = ProviderStatus.RATE_LIMITED
                response.error = declined
                response.answer_text = ""
            else:
                await self._emit("provider", f"{provider}: starting ({role})", provider=provider, round_no=round_no)
                try:
                    extra = {"continue_thread": True} if continue_thread else {}
                    response = await adapter.ask(job_id, prompt, round_no, emit=self._adapter_emit, **extra)
                except Exception as exc:  # noqa: BLE001
                    self.cancel.raise_if_cancelled()
                    response = ProviderResponse(id=f"resp_err_{provider}_{round_no}", job_id="", round=round_no, provider=provider, prompt=prompt)
                    response.status = ProviderStatus.FAILED
                    response.error = f"{type(exc).__name__}: {exc}"
                    response.answer_text = ""
                self.politeness.after(provider, response.status.value)
        if (response.error or "").startswith("LiveChromeUnavailable:") and not self._driver_down:
            self._driver_down = response.error.split(":", 1)[1].strip()
        if (
            response.status.value == "completed"
            and not continue_thread
            and router.off_topic(router.question_from_prompt(prompt), response.answer_text or "")
        ):
            # a reused chat answered an earlier question: never let it into the ledger
            response.status = ProviderStatus.FAILED
            response.error = "off_topic: the reply does not address this question"
            # keep nothing of the stale text: it belongs to another question and must not sit in this research's record
            response.detail = f"off-topic reply discarded ({len(response.answer_text or '')} characters)"
            response.raw_text = ""
            response.answer_text = ""
            response.citations = []
        response.role = role
        response.thread_id = f"{job_id}:{provider}"
        response.turn = turn
        response.continued = continue_thread
        if response.status.value in {"completed", "timeout"}:
            self._turns[thread_key] = turn
        response.escalation_reason = escalation_reason
        citation_ops.text_citations(response)
        citation_ops.mark_opened(response)
        response.pages_visited = [c.url for c in response.citations if c.url][:20]
        response.failure_signals = router.is_failure_phrase(response.answer_text or response.raw_text or "")
        if response.status.value == "completed":
            self.health[provider] = "completed"
        elif (response.error or "").startswith("LiveChromeUnavailable:"):
            pass  # the browser driver was down, not this provider: say nothing about the provider's health
        elif response.status.value in {"logged_out", "rate_limited", "broken", "failed", "timeout"}:
            self.health[provider] = response.status.value
        await self._emit(
            "provider",
            f"{provider}: {response.status.value}" + (f" -- {response.error}" if response.error else ""),
            provider=provider,
            round_no=round_no,
        )
        return response

    async def _adapter_emit(self, kind: str, message: str, provider: str | None = None, round_no: int | None = None) -> None:
        await self._emit(kind, message, provider=provider, round_no=round_no)

    async def _ask_many(
        self,
        providers: list[str],
        job: Job,
        analysis: Any,
        context: dict[str, Any],
        round_no: int,
        *,
        exclude: str,
    ) -> list[ProviderResponse]:
        tasks = []
        for index, provider in enumerate(providers):
            prompt = escalation_prompt(
                self._q(job),
                provider=provider,
                primary=context["primary"],
                established=context["established"],
                unresolved=context["unresolved"],
                contradictions=context["contradictions"],
                sources=context["sources"],
                needs_web=analysis.needs_web_research,
            )
            prompt = self._with_context(job, prompt)
            if index >= 2:
                angle = angle_for(provider, index)
                if angle:
                    prompt = f"{prompt}\n\nApproach it from this angle: {angle}"
            tasks.append(
                self._ask(
                    provider,
                    prompt,
                    round_no,
                    role="secondary",
                    escalation_reason=f"primary ({exclude}) left {len(context['unresolved'])} point(s) unresolved",
                    job_id=job.id,
                )
            )
        done = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[ProviderResponse] = []
        for provider, item in zip(providers, done):
            if isinstance(item, BaseException):
                job.error = f"{provider} crashed: {item}"
                continue
            if item is not None:
                out.append(item)  # type: ignore[arg-type]
        return out

    @staticmethod
    def _unusable_this_job(responses: list[ProviderResponse]) -> set[str]:
        """Providers whose every response so far was a login wall, a block or a broken page."""
        dead_status = {ProviderStatus.LOGGED_OUT, ProviderStatus.BROKEN}
        by_provider: dict[str, list[ProviderResponse]] = {}
        for response in responses:
            by_provider.setdefault(response.provider, []).append(response)
        dead: set[str] = set()
        for provider, items in by_provider.items():
            if all(
                r.status in dead_status or (r.status == ProviderStatus.FAILED and (r.error or "").startswith(("readiness=blocked", "off_topic", "answer captured but empty")))
                for r in items
            ):
                dead.add(provider)
        return dead

    async def _targeted_round(self, job: Job, follow_ups: list[FollowUp], targets: list[str], *, round_no: int) -> list[ProviderResponse]:
        """Give each unresolved point to a specific agent. One agent, one point."""
        tasks = []
        for index, follow_up in enumerate(follow_ups):
            # the curator's preferred researcher, if it is one we can actually reach; otherwise whoever is available
            wanted = [p for p in follow_up.target_providers if p in self.adapters and p != "search" and self.health.get(p) not in {"logged_out", "broken", "rate_limited", "failed"}]
            pool = wanted or targets
            if not pool:
                break
            provider = pool[index % len(pool)]
            prompt = self._with_context(job, follow_up_prompt(self._q(job), follow_up))
            tasks.append(
                self._ask(
                    provider, prompt, round_no, role="targeted", escalation_reason=follow_up.reason, job_id=job.id,
                    continue_thread=self._turns.get((job.id, provider), 0) > 0,
                )
            )
        done = await asyncio.gather(*tasks, return_exceptions=True)
        return [item for item in done if isinstance(item, ProviderResponse)]

    async def _extract(self, responses: list[ProviderResponse], job: Job) -> list[Claim]:
        self.cancel.raise_if_cancelled()
        job.status = JobStatus.EXTRACTING
        await self._emit("status", "extracting atomic claims", round_no=job.active_round, job=job)
        extracted = await claim_ops.extract_claims(
            [r for r in responses if r.status.value in {"completed", "timeout"} and not r.superseded],
            job.id,
            endpoint=self.analysis_endpoint if self.analysis_endpoint and self.analysis_endpoint.enabled else None,
            batch_size=self.settings.research.claim_batch_size,
            question=job.question,
        )
        extracted = claim_ops.reconcile_with_prior(extracted, list(job.claims or []))
        job.claims = extracted
        return extracted

    async def _pool(self, job: Job, claims: list[Claim], responses: list[ProviderResponse], round_no: int) -> tuple[list[Evidence], dict[str, Any]]:
        self.cancel.raise_if_cancelled()
        await self._emit("status", "opening cited sources and gathering independent evidence", round_no=round_no, job=job)
        search_adapter = self.adapters.get("search") if self.settings.providers.get("search") and self.settings.providers["search"].enabled else None
        evidence, trace = await build_pool(
            job_id=job.id,
            question=self._q(job),
            claims=claims,
            responses=responses,
            mode=job.mode,
            settings=self.settings,
            engine=self.engine,
            search_adapter=search_adapter,
            round_no=round_no,
            emit=self._adapter_emit,
            run_searches=job.mode != ResearchMode.QUICK or round_no > 1,
        )
        job.evidence = _merge_evidence(job.evidence, evidence)
        return evidence, trace

    async def _disagreements(self, job: Job, claims: list[Claim], round_no: int) -> list[Disagreement]:
        self.cancel.raise_if_cancelled()
        conflicts = claim_ops.find_contradictions(claims)
        existing = {d.topic for d in job.disagreements}
        found: list[Disagreement] = []
        for conflict in conflicts:
            left, right = conflict["left"], conflict["right"]
            topic = left.topic or right.topic or left.claim[:40]
            key = f"{topic}|{conflict['kind']}"
            if key in existing:
                continue
            disagreement = Disagreement(
                job_id=job.id,
                round=round_no,
                topic=key[:60],
                description=(
                    f"{conflict['kind']} conflict at topic overlap {conflict['topic_overlap']}: "
                    f"{left.provider_sources} said \"{left.claim[:120]}\" vs "
                    f"{right.provider_sources} said \"{right.claim[:120]}\""
                ),
                positions={
                    ",".join(left.provider_sources): [left.claim],
                    ",".join(right.provider_sources): [right.claim],
                },
                severity="material" if conflict["material"] else "minor",
                claim_ids=[left.id, right.id],
            )
            found.append(disagreement)
        job.disagreements = _merge_disagreements(job.disagreements, found)
        material = [d for d in found if d.severity == "material"]
        for index, disagreement in enumerate(material, start=1):
            await self._emit(
                "disagreement",
                f"checking disagreement #{index}: {disagreement.topic} ({disagreement.description.split(':')[0]})",
                round_no=round_no,
            )
        return job.disagreements

    def _assess(
        self,
        job: Job,
        analysis: Any,
        claims: list[Claim],
        evidence: list[Evidence],
        disagreements: list[Disagreement],
        round_no: int,
        *,
        focus: str,
    ) -> SufficiencyAssessment:
        # A page that was opened and says the OPPOSITE is confirmed evidence too, but
        # it never counts toward establishing a claim.
        confirmed = [e for e in evidence if e.check_status == SourceCheckStatus.CONFIRMED and e.polarity != "refute"]
        refuting = [e for e in evidence if e.check_status == SourceCheckStatus.CONFIRMED and e.polarity == "refute"]
        domains = {e.domain for e in confirmed if e.domain}
        has_primary = any(e.tier.value in {"primary_official", "original_research", "government"} for e in confirmed)
        material = [c for c in claims if c.kind in MATERIAL_KINDS or _has_figure(c.claim)]
        contradicted = {e.claim_id for e in refuting if e.claim_id}
        established = [
            c for c in material
            if c.id not in contradicted and any(e.claim_id == c.id for e in confirmed)
        ]
        # Contradicted and nothing backs it: that question is settled (it is false),
        # so it is not "unresolved" -- but it also establishes nothing.
        refuted_only = [c for c in material if c.id in contradicted and not any(e.claim_id == c.id for e in confirmed)]
        unresolved = [c.claim for c in material if c not in established and c not in refuted_only]
        failure_signals = sorted(
            {s for r in job.responses if r.role in {"primary", "secondary", "thread_follow_up"} and not r.superseded for s in r.failure_signals}
        )
        no_sources = [r.provider for r in job.responses if r.status.value == "completed" and not r.citations and not r.superseded]
        mismatch = [e for e in evidence if e.check_status in {SourceCheckStatus.MISMATCH, SourceCheckStatus.HALLUCINATED, SourceCheckStatus.BROKEN_URL}]
        stale = [e for e in evidence if e.check_status == SourceCheckStatus.OUTDATED]
        unanswered = _unanswered_subquestions(analysis, claims)
        weak_only = bool(confirmed) and not has_primary and analysis.high_stakes
        material_conflicts = [d for d in disagreements if d.severity == "material"]
        contradictions = len(material_conflicts)

        signals: list[str] = []
        if established:
            signals.append(f"{len(established)}/{len(material)} material claim(s) confirmed by an opened source")
        if len(domains) >= self.settings.research.min_independent_sources:
            signals.append(f"{len(domains)} independent domain(s) in agreement")
        if refuted_only:
            signals.append(f"{len(refuted_only)} claim(s) contradicted by an opened source")
        min_sources = self.settings.research.min_independent_sources
        open_material = [c for c in material if c not in refuted_only]
        coverage = (len(established) / len(open_material)) if open_material else (0.0 if material else (1.0 if confirmed else 0.0))
        # Evidence only establishes something if it attaches to a claim. Eight
        # opened pages that answer nothing are eight pages, not an answer -- without
        # this, a provider that produced no text at all could look "sufficiently
        # verified" and the run would stop while saying "I don't know."
        attached = [e for e in confirmed if e.claim_id]

        # Only the claims that answer the question that was asked can be blocked by a conflict. A disagreement about
        # something beside the point (an enactment date, a section number) is reported but never keeps research going.
        question = self._q(job)
        on_point = [c for c in open_material if router.addresses_question(question, c.claim)] or open_material
        best = max((router.asked_overlap(question, c.claim) for c in on_point), default=0)
        key = [c for c in on_point if best >= 2 and router.asked_overlap(question, c.claim) >= max(2, best - 1)] or on_point
        key_ids = {c.id for c in key}
        key_conflicts = [d for d in material_conflicts if set(d.claim_ids) & key_ids]
        if key:
            contradictions = len(key_conflicts)

        # "Enough" is a property of the evidence, never of how the answer was
        # phrased.
        ledger_ok = (
            bool(attached)
            and bool(claims)
            and coverage >= 0.8
            and len({e.domain for e in attached}) >= min_sources
            and contradictions == 0
            and not unresolved
            and not unanswered
        )
        # Strong primary evidence ends the hunt: every open claim rests on a primary/official page that the AI
        # itself opened (and OmniBrain's check of the same page agrees). More rounds could only repeat it.
        strong = {
            e.claim_id for e in attached
            if e.tier.value in {"primary_official", "original_research", "government"} and e.provenance == "CLAIM_SUPPORTED"
        }
        # "Every claim" would mean the side remarks too (a guidance page's update date) -- the test is the claims
        # that answer the question that was asked, and only conflicts about THOSE can block it.
        strong_primary = bool(key) and all(c.id in strong for c in key) and not key_conflicts and not unanswered
        ledger_ok = ledger_ok or (strong_primary and bool(claims))
        sufficient = ledger_ok
        if analysis.high_stakes:
            sufficient = sufficient and has_primary and not weak_only
        # An admitted inability is a demand for evidence, not a life sentence: once
        # the ledger independently confirms the material claims, the model's own
        # hedge stops being the controlling signal. Before that point it blocks.
        if failure_signals and not ledger_ok:
            sufficient = False
        if unanswered:
            sufficient = False
        if mismatch and len(mismatch) > len(confirmed):
            sufficient = False
        # No figure-bearing claims, but sources that clearly speak to the question:
        # still needs an actual claim to hang them on.
        if not material and attached and claims and len({e.domain for e in attached}) >= min_sources:
            sufficient = sufficient and not failure_signals and contradictions == 0
        if not claims:
            sufficient = False

        recommends = EscalationLevel.PRIMARY
        reasons: list[str] = []
        if not confirmed:
            recommends = EscalationLevel.PARALLEL
            reasons.append("no opened source confirms anything yet")
        elif not attached:
            recommends = EscalationLevel.PARALLEL
            reasons.append(f"{len(confirmed)} source(s) opened but none attach to a claim the researchers actually made")
        elif coverage < 0.8:
            recommends = EscalationLevel.PARALLEL
            reasons.append(f"{len(unresolved)} material claim(s) still unconfirmed")
        if refuted_only:
            recommends = max(recommends, EscalationLevel.PARALLEL, key=int)
            reasons.append(f"{len(refuted_only)} claim(s) contradicted by an opened source; the true answer is still open")
        if contradictions:
            recommends = EscalationLevel.DEEP
            reasons.append(f"{contradictions} material conflict(s)")
        if failure_signals:
            recommends = max(recommends, EscalationLevel.PARALLEL, key=int)
            reasons.append("an explicit inability to verify appeared in the answers")
        if analysis.high_stakes and not has_primary:
            recommends = EscalationLevel.DEEP
            reasons.append("high-stakes question with no primary source yet")
        if stale and not any(fresh_enough(e) for e in confirmed) and analysis.needs_current_data:
            recommends = max(recommends, EscalationLevel.PARALLEL, key=int)
            reasons.append("only aging sources so far for a current question")

        if sufficient:
            reason = f"evidence sufficient after {focus}: {len(established) or len(confirmed)} confirmed claim(s) across {len(domains)} domain(s), no material conflict"
        else:
            reason = "; ".join(reasons) or f"{focus} did not establish the material claims"

        return SufficiencyAssessment(
            job_id=job.id,
            round=round_no,
            sufficient=sufficient,
            signals=signals,
            failure_signals=failure_signals + [f"no_sources:{p}" for p in no_sources] + [f"mismatch:{len(mismatch)}"] if mismatch or no_sources else failure_signals,
            established=[c.claim for c in established],
            unresolved=unresolved,
            unanswered_subquestions=unanswered,
            confirmed_sources=len(confirmed),
            independent_domains=len(domains),
            has_primary_source=has_primary,
            contradictions=contradictions,
            coverage=round(coverage, 2),
            strong_primary=strong_primary and bool(claims),
            reason=reason,
            recommends_level=recommends,
        )

    def _failure_context(
        self,
        responses: list[ProviderResponse],
        claims: list[Claim],
        evidence: list[Evidence],
        disagreements: list[Disagreement],
        assessment: SufficiencyAssessment,
    ) -> dict[str, Any]:
        primary = responses[0] if responses else None
        confirmed_urls = [e.url for e in evidence if e.check_status == SourceCheckStatus.CONFIRMED and e.url]
        return {
            "primary": primary.provider if primary else "none",
            "established": assessment.established or [c.claim for c in claims if c.status == ClaimStatus.SUPPORTED][:6],
            "unresolved": assessment.unresolved or [c.claim for c in claims][:6],
            "contradictions": [d.description for d in disagreements if d.severity == "material"][:4],
            "sources": confirmed_urls[:8] or (list(primary.citations[:6]) if primary else []),
        }

    async def _targeted_expansion(
        self,
        job: Job,
        analysis: Any,
        enabled: list[str],
        primary: str,
        round_no: int,
        responses: list[ProviderResponse],
        claims: list[Claim],
        evidence: list[Evidence],
        disagreements: list[Disagreement],
        assessment: SufficiencyAssessment,
    ) -> tuple[int, list[ProviderResponse], list[Claim], list[Evidence], list[Disagreement], SufficiencyAssessment]:
        follow_ups = self._open_follow_ups(job, assessment, round_no + 1)
        chosen = router.select_secondaries(
            analysis, enabled, exclude=[primary],
            count=min(3, max(2, len(enabled) - 1)), health=self.health, prior=self.prior_health,
        )
        next_round = round_no + 1
        job.escalation_log.append(
            EscalationStep(
                level=EscalationLevel.PARALLEL,
                reason=(
                    f"{len(assessment.established)} claim(s) already established, {len(assessment.unresolved)} open -- "
                    "investigating only the open ones"
                ),
                providers=chosen,
                triggered_by=["partial_success"] + assessment.failure_signals[:4],
                round=next_round,
            )
        )
        await self._emit(
            "escalation",
            f"level 2 (targeted) -- {len(follow_ups)} open point(s) to {', '.join(chosen)}; "
            f"the {len(assessment.established)} settled claim(s) were deliberately not re-asked",
            round_no=next_round,
            job=job,
        )
        record = RoundRecord(number=next_round, kind="follow_up")
        job.rounds.append(record)
        job.active_round = next_round
        job.follow_ups.extend(follow_ups)
        record.follow_up_ids = [f.id for f in follow_ups]

        targeted = await self._targeted_round(job, follow_ups, chosen, round_no=next_round)
        job.browser_sessions_used += len({r.provider for r in targeted})
        responses = responses + targeted
        job.responses = responses
        record.response_ids = [r.id for r in targeted]

        claims = await self._extract(responses, job)
        more_evidence, _ = await self._pool(job, claims, targeted or responses, next_round)
        evidence = _merge_evidence(evidence, more_evidence)
        disagreements = await self._disagreements(job, claims, next_round)
        assessment = self._assess(job, analysis, claims, evidence, disagreements, next_round, focus="targeted expansion")
        job.assessments.append(assessment)
        record.finished_at = time.time()
        await self._emit("assessment", f"after targeted research: {assessment.reason}", round_no=next_round, job=job)
        return next_round, responses, claims, evidence, disagreements, assessment

    @staticmethod
    def _open_follow_ups(job: Job, assessment: SufficiencyAssessment, next_round: int) -> list[FollowUp]:
        """Turn each unresolved material claim into its own research task."""
        out: list[FollowUp] = []
        for item in assessment.unresolved[:5]:
            clean = item.strip()
            if not clean:
                continue
            out.append(
                FollowUp(
                    job_id=job.id,
                    question=(
                        f"Establish whether this is true: {clean[:300]} "
                        "Find the primary or official source that settles it, give the exact figure or date, "
                        "the publisher, and the source's publication date. If nothing solid exists, say so."
                    ),
                    reason="The primary researcher could not back this with a source we were able to confirm.",
                    claim_ids=[],
                    round=next_round,
                )
            )
        for item in assessment.unanswered_subquestions[:2]:
            clean = " ".join(item.split()).lstrip("-: ").strip()
            if not clean:
                continue
            out.append(
                FollowUp(
                    job_id=job.id,
                    question=(
                        f"Answer this specific part of a larger question: {clean[:280]} "
                        "Nothing so far addressed it. Cite the source that answers it."
                    ),
                    reason="A part of the question has gone untouched.",
                    claim_ids=[],
                    round=next_round,
                )
            )
        return out

    async def _verify(
        self,
        job: Job,
        analysis: Any,
        claims: list[Claim],
        evidence: list[Evidence],
        responses: list[ProviderResponse],
        disagreements: list[Disagreement],
        round_no: int,
    ) -> VerifierReport:
        self.cancel.raise_if_cancelled()
        if self.verifier is None:
            endpoint = Endpoint.from_config(self.settings.verifier)
            self.verifier = Verifier(endpoint, min_independent_sources=self.settings.research.min_independent_sources, deep=self.settings.verifier.deep_verification)
        job.status = JobStatus.VERIFYING
        await self._emit("status", "adversarial verification (evidence, not votes)", round_no=round_no, job=job)
        return await self.verifier.verify(
            job_id=job.id,
            question=self._q(job),
            round_no=round_no,
            claims=claims,
            evidence=evidence,
            responses=responses,
            disagreements=disagreements,
            corrections=list(job.corrections),
        )

    async def _lightweight_report(
        self,
        job: Job,
        analysis: Any,
        claims: list[Claim],
        evidence: list[Evidence],
        responses: list[ProviderResponse],
        round_no: int,
    ) -> VerifierReport:
        """Skip the expensive engine when the ledger already settles it (section 9)."""
        endpoint = Endpoint.from_config(self.settings.verifier)
        verifier = self.verifier or Verifier(endpoint, min_independent_sources=self.settings.research.min_independent_sources)
        report = verifier.deterministic(
            job_id=job.id,
            question=self._q(job),
            round_no=round_no,
            claims=claims,
            evidence=evidence,
            responses=responses,
            disagreements=[],
            reason="escalation rules did not require the adversarial pass",
        )
        report.verifier_model = "deterministic ledger (no verifier call)"
        job.reports.append(report)
        return report

    def _complete(self, job: Job, report: VerifierReport | None, responses: list[ProviderResponse], *, rounds: int, stop: str) -> Job:
        job.status = JobStatus.SYNTHESIZING
        if report is None:
            job.final = FinalAnswer(
                answer="I don't know.",
                confidence=Confidence.NONE,
                confidence_label="Insufficient evidence",
                caveats=["No researcher returned anything usable."],
                rounds_run=rounds,
                providers_used=[],
                providers_failed=sorted({r.provider for r in responses}),
                research_status="BLOCKED",
                reviewer_status="NOT_RUN",
                synthesis_status="DETERMINISTIC",
            )
            job.stop_reason = stop
            return job
        final = build_final_answer(report, responses, rounds, self._q(job))
        # community-sourced owner reports are a labelled caveat of their own, never part of the answer
        final.caveats = [c for c in final.caveats if c][:5] + review_caveats(job.reviews)
        if not any(r.status.value == "completed" for r in responses):
            final.research_status = "BLOCKED"
        elif final.confidence.value in {"high", "moderate"}:
            final.research_status = "COMPLETED"
        else:
            final.research_status = "UNRESOLVED"
        job.final = final
        job.status = JobStatus.COMPLETED
        job.stop_reason = stop
        job.finished_at = time.time()
        for record in job.rounds:
            record.finished_at = record.finished_at or time.time()
        return job

    @staticmethod
    def _settled(assessment: SufficiencyAssessment, disagreements: list[Disagreement]) -> bool:
        """Sufficient, and nothing that matters conflicts. Strong primary evidence already accounts for conflicts that touch the asked claims."""
        return assessment.sufficient and (assessment.strong_primary or assessment.contradictions == 0)

    def _stop_reason(self, assessment: SufficiencyAssessment, disagreements: list[Disagreement], analysis: Any, round_no: int, max_rounds: int, note: str = "") -> str:
        if note:
            return note
        if assessment.sufficient and (assessment.strong_primary or assessment.contradictions == 0):
            return f"remaining uncertainty is not material to the question (round {round_no})"
        if round_no >= max_rounds:
            return f"stopped at max rounds ({max_rounds}) with material uncertainty still open"
        material = [d for d in disagreements if d.severity == "material"]
        if material:
            return f"sources still conflict on {material[0].topic}; reported as unresolved rather than averaged"
        return assessment.reason

    async def _emit(
        self,
        kind: str,
        message: str,
        provider: str | None = None,
        round_no: int | None = None,
        job: Job | None = None,
        **payload: Any,
    ) -> None:
        if job is not None:
            job.updated_at = time.time()
        try:
            await self.bus.emit(kind, message, provider=provider, round_no=round_no, **payload)
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------ utilities


def _has_figure(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def _merge_evidence(old: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    # One row per (page, claim): a single page can legitimately settle several claims.
    by_url = {(e.url, e.claim_id): e for e in old if e.url}
    for ev in new:
        key = (ev.url, ev.claim_id)
        if ev.url and not ev.claim_id and key not in by_url:
            # an unattributed sighting of a page already filed under a claim
            key = next((k for k in by_url if k[0] == ev.url), key)
        if ev.url and ev.claim_id and key not in by_url and (ev.url, None) in by_url:
            # a page first filed under no claim is now known to belong to this one
            by_url[(ev.url, None)].claim_id = ev.claim_id
            by_url[key] = by_url.pop((ev.url, None))
        if ev.url and key in by_url:
            existing = by_url[key]
            rank = {SourceCheckStatus.CONFIRMED: 3, SourceCheckStatus.OUTDATED: 2, SourceCheckStatus.NOT_CHECKED: 1}
            if rank.get(ev.check_status, 0) > rank.get(existing.check_status, 0):
                existing.check_status = ev.check_status
                existing.polarity = ev.polarity
                existing.verbatim_excerpt = ev.verbatim_excerpt or existing.verbatim_excerpt
                existing.check_notes = ev.check_notes or existing.check_notes
            continue
        old.append(ev)
        if ev.url:
            by_url[key] = ev
    return old


def _merge_disagreements(existing: list[Disagreement], found: list[Disagreement]) -> list[Disagreement]:
    topics = {d.topic for d in existing}
    for dis in found:
        if dis.topic in topics:
            for d in existing:
                if d.topic == dis.topic:
                    for provider, claims in dis.positions.items():
                        d.positions.setdefault(provider, [])
                        for claim in claims:
                            if claim not in d.positions[provider]:
                                d.positions[provider].append(claim)
            continue
        existing.append(dis)
    return existing


def _unanswered_subquestions(analysis: Any, claims: list[Claim]) -> list[str]:
    if not getattr(analysis, "sub_questions", None):
        return []
    corpus = " ".join(c.claim for c in claims).lower()
    out = []
    for question in analysis.sub_questions:
        tokens = [t for t in re.findall(r"[a-z0-9']+", question.lower()) if len(t) > 3][:5]
        if tokens and not any(t in corpus for t in tokens):
            out.append(f"sub-question untouched: {question[:80]}")
    return out


def fresh_enough(evidence: Evidence) -> bool:
    from backend.evidence.sources import freshness

    return freshness(evidence.published)["verdict"] in {"fresh", "unknown"}


def _obviously_arithmetic(question: str) -> bool:
    computed, _ = router.try_arithmetic(question or "")
    return computed is not None
