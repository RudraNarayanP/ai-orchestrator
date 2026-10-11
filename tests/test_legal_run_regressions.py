"""Regressions from the live legal run of 2026-10-11 (each case below is a real record from that run).

A verdict with its reason deleted ("Nah, that doesn't work like that. X"), "I couldn't find that documented" pasted onto a
supported answer, a settled "no" shown as "Yeah, the idea is right", and a round that waited on the same model three times in
a row. Nothing here is special-cased to a question: each test pins a mechanism."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from backend.evidence.sources import is_bot_challenge, point_in_time_date
from backend.models import ClaimStatus, ClaimVerdict, Confidence, FollowUp, VerifierReport
from backend.orchestrator.runner import ResearchRunner
from backend.research.lint import lint_answer
from backend.research.style import fit_fact, premise_state, render_truth, split_sentences
from backend.verification.verifier import build_final_answer, truth_state

S = ClaimStatus


def v(status, claim, strong=1):
    return ClaimVerdict(claim_id=claim[:8], claim=claim, verdict=status, confidence=Confidence.MODERATE, reasoning="r", strong_evidence=["https://x.test/a"] * strong)


def rep(answer, *verdicts, why="", confidence=Confidence.MODERATE):
    return VerifierReport(job_id="j", verdicts=list(verdicts), answer=answer, why=why, confidence=confidence)


BASIC_Q = "Can Parliament use Article 368 to amend the basic structure of the Indian Constitution?"
BASIC_A = (
    "No, Parliament cannot use Article 368 to amend or destroy the basic structure of the Indian Constitution; any amendment "
    "that damages or destroys the basic structure is subject to judicial review and can be declared unconstitutional."
)
DPDP_Q = "Under India's Digital Personal Data Protection Act 2023, must a data fiduciary obtain verifiable parental consent before processing a 17-year-old's personal data?"
DPDP_A = (
    "Yes, under the Digital Personal Data Protection Act, 2023, a 17-year-old is defined as a child (Section 2(f)) and Section 9(1) "
    "requires a data fiduciary to obtain verifiable parental consent before processing a child's personal data, although the "
    "substantive requirements of Section 9 are not yet in force (commencement scheduled for May 2027)."
)
NONCOMPETE_Q = "Is a post-termination non-compete covenant in an employment contract enforceable under section 27 of the Indian Contract Act, 1872?"
NONCOMPETE_A = (
    "Generally, a post-termination non-compete covenant in an employment contract is void and unenforceable under Section 27 of the "
    "Indian Contract Act, 1872, unless it falls within the statutory exception for the sale of business goodwill."
)


# ---- 1. a verdict keeps its reason (live: "Nah, that doesn't work like that. \u274c" and "Yeah, you're right. \u2705")
def test_a_long_reason_is_shortened_never_dropped():
    for state, lead in (("FALSE", "Nah"), ("TRUE", "Yeah"), ("PARTLY", "Yeah")):
        shown = render_truth(state, question=BASIC_Q, answer=BASIC_A, aspect="detail")
        assert shown.startswith(lead)
        assert "basic structure" in shown, shown
        assert len(shown) <= 320


def test_fit_fact_cuts_at_a_clause_and_never_returns_nothing_for_a_long_reason():
    fact = "Parliament cannot amend the basic structure of the Constitution; any amendment that destroys it can be struck down by the courts on review."
    assert fit_fact(fact, 400) == fact
    cut = fit_fact(fact, 80)
    assert cut and len(cut) <= 80 and cut.endswith(".")
    assert "basic structure" in cut
    assert fit_fact("", 80) == ""


def test_a_case_citation_is_not_split_at_v_or_co():
    text = "Under Erie Railroad Co. v. Tompkins, 304 U.S. 64 (1938), a federal court applies state law. Next sentence here."
    assert split_sentences(text) == [
        "Under Erie Railroad Co. v. Tompkins, 304 U.S. 64 (1938), a federal court applies state law.",
        "Next sentence here.",
    ]
    assert len(split_sentences("See para. 702 of the notes. Then Smith v. Jones Ltd. held otherwise.")) == 2


# ---- 2. "I couldn't find that documented" is never pasted onto a statement the ledger settled
def test_unjudged_claims_never_rewrite_a_settled_answer_or_its_why():
    verdicts = [
        v(S.SUPPORTED, "Section 9(1) of the Digital Personal Data Protection Act, 2023 addresses verifiable parental consent.", 2),
        v(S.INSUFFICIENT_EVIDENCE, "A 17-year-old falls within the DPDP Act's definition of a child."),
        v(S.INSUFFICIENT_EVIDENCE, "Section 2(f) of the Digital Personal Data Protection Act, 2023 defines a child as an individual who has not completed the age of eighteen years."),
    ]
    why = "Section 9(1) mandates verifiable parental consent before processing a child's personal data. A 17-year-old is a child, and the substantive requirements of Section 9 are not yet in force."
    final = build_final_answer(rep(DPDP_A, *verdicts, why=why), [], 1, DPDP_Q)
    assert "couldn't find that" not in final.answer.lower()
    assert "couldn't find that" not in (final.why or "").lower()
    assert final.truth_state == "TRUE"
    assert final.answer.startswith("Yeah")  # "not yet in force" is a qualifier of a YES, not a denial


def test_lint_leaves_a_sentence_that_states_a_settled_claim_alone():
    sentence = "The Act does not apply to a 17-year-old falls within the definition of a child under Section 2(f) of the DPDP Act."
    unknown = ["A 17-year-old falls within the definition of a child under Section 2(f) of the DPDP Act"]
    assert "couldn't find" in lint_answer(sentence, unknowns=unknown).text
    assert "couldn't find" not in lint_answer(sentence, unknowns=unknown, settled=[unknown[0]]).text


# ---- 3. direction comes from the answer's own verdict, not from a clause that qualifies it
def test_a_yes_lead_is_never_flipped_by_a_later_qualifier():
    assert premise_state("TRUE", DPDP_Q, DPDP_A) == "TRUE"
    assert premise_state("TRUE", "Is the DPDP Act in force?", "Yes, it is, although some parts are not yet in force.") == "TRUE"


def test_void_or_unenforceable_is_a_denial_of_an_enforceable_premise():
    assert premise_state("TRUE", NONCOMPETE_Q, NONCOMPETE_A) == "FALSE"
    # a qualifier after the verdict does not decide it
    assert premise_state("TRUE", "Is the clause enforceable?", "Yes, it is enforceable, unless it is void for uncertainty.") == "TRUE"
    assert premise_state("TRUE", "Is section 9 of the DPA 2018 still in force?", "Section 9 was omitted by the 2026 Act, so no longer in force.") == "FALSE"


def test_a_refuted_web_page_is_not_a_claim_about_the_question():
    verdicts = [
        v(S.SUPPORTED, "A post-termination non-compete covenant in an employment contract is generally void and unenforceable under Section 27.", 2),
        v(S.SUPPORTED, "Limiting a post-termination non-compete to a reasonable duration does not, by itself, make it enforceable.", 2),
        v(S.REFUTED, "The URL https://order.law/library/courts/supreme-court/2006/mar/x reproduces the Supreme Court judgment of 2006.", 0),
    ]
    report = rep(NONCOMPETE_A, *verdicts, confidence=Confidence.HIGH)
    assert truth_state(report, NONCOMPETE_Q) == "TRUE"
    final = build_final_answer(report, [], 1, NONCOMPETE_Q)
    assert final.answer.startswith("Nah"), final.answer
    assert "void and unenforceable" in final.answer


def test_page_claims_alone_still_count_when_nothing_else_was_judged():
    only = rep("x", v(S.REFUTED, "The URL https://a.test/x reproduces the judgment.", 1), confidence=Confidence.MODERATE)
    assert truth_state(only, "Is it true?") == "FALSE"


# ---- 4. evidence handling: a bot-check page is a gate, an old legislation version is old
def test_a_bot_check_answer_is_a_gate_not_a_missing_page():
    assert is_bot_challenge(202, {"x-amzn-waf-action": "challenge"}, "")
    assert is_bot_challenge(202, {}, "")
    assert is_bot_challenge(429, {}, "   ")
    assert not is_bot_challenge(200, {}, "Article 28 ...")
    assert not is_bot_challenge(404, {}, "Not found page with words")


def test_point_in_time_legislation_urls_carry_their_date():
    assert point_in_time_date("https://www.legislation.gov.uk/ukpga/2018/12/section/9/2018-05-25") == "2018-05-25T00:00:00+00:00"
    assert point_in_time_date("https://www.legislation.gov.uk/ukpga/2018/12/section/9/2018-05-25?view=plain") == "2018-05-25T00:00:00+00:00"
    assert point_in_time_date("https://www.legislation.gov.uk/ukpga/2018/12/section/9") is None
    assert point_in_time_date("https://example.com/blog/2024-01-05-post-title") is None


# ---- 5. speed: independent model calls go out together; a round asks each researcher once
def test_claims_are_extracted_from_all_answers_at_once(monkeypatch):
    from backend.models import ProviderResponse, ProviderStatus
    from backend.research import claims as claim_ops

    class SlowClient:
        def __init__(self, endpoint):
            pass

        async def complete_json(self, messages, temperature=None):
            await asyncio.sleep(0.3)
            return {"claims": [{"claim": "Section 9 was omitted by the 2026 Act.", "kind": "legal", "topic": "s9"}]}, None

    monkeypatch.setattr(claim_ops, "LLMClient", SlowClient)
    endpoint = SimpleNamespace(enabled=True)
    responses = [
        ProviderResponse(id=f"r{i}", job_id="j", round=1, provider=p, prompt="q", status=ProviderStatus.COMPLETED, answer_text="Section 9 was omitted by the 2026 Act.")
        for i, p in enumerate(("chatgpt", "gemini", "qwen"))
    ]
    started = time.perf_counter()
    claims = asyncio.run(claim_ops.extract_claims(responses, "j", endpoint=endpoint, question="Is section 9 in force?"))
    assert claims
    assert time.perf_counter() - started < 0.8  # three calls in a row would take 0.9s+


def test_a_site_that_could_not_send_is_not_asked_again_in_the_same_job():
    from backend.models import ProviderResponse, ProviderStatus

    def failed(provider, error):
        r = ProviderResponse(id=provider, job_id="j", round=1, provider=provider, prompt="q")
        r.status = ProviderStatus.FAILED
        r.error = error
        return r

    dead = ResearchRunner._unusable_this_job([
        failed("gemini", "submit_failed: the prompt stayed in the composer"),
        failed("copilot", "answer captured but empty after cleaning"),
        failed("qwen", "TimeoutError: slow"),
    ])
    assert dead == {"gemini", "copilot"}


def test_a_targeted_round_asks_each_researcher_once():
    asked: list[str] = []

    async def fake_ask(provider, prompt, round_no, **kw):
        asked.append(provider)
        return None

    runner = object.__new__(ResearchRunner)
    runner.adapters = {"gemini": object(), "qwen": object(), "copilot": object()}
    runner.health = {}
    runner._turns = {}
    runner._with_context = lambda job, prompt: prompt
    runner._q = lambda job: "question"
    runner._ask = fake_ask
    points = [FollowUp(job_id="j", question=f"point {i}", reason="open") for i in range(5)]
    asyncio.run(runner._targeted_round(SimpleNamespace(id="j"), points, ["gemini", "qwen", "copilot"], round_no=3))
    assert sorted(asked) == ["copilot", "gemini", "qwen"]


# ---- 6. a site's own "limit reached" message is a rate limit, not an empty answer (live: Copilot 11x, Le Chat 7x, ~1900 s of retries)
def test_a_limit_notice_in_the_chat_is_recognised():
    from browser.adapters.base import limit_notice

    copilot = "Copilot said:\nYou've reached your daily limit. Get more usage now or check back at 5:30 AM.\nUpgrade\nMessage Copilot"
    lechat = "Messages limit reached\n\nYou have reached your message limit. Your limit will reset in 6 minutes. Upgrade to Pro right now"
    assert "daily limit" in limit_notice(copilot)
    assert "reset in 6 minutes" in limit_notice(lechat)
    assert limit_notice("The Court reached its decision on the daily rate of interest.") is None
    assert limit_notice("") is None


def test_a_send_that_failed_in_the_background_is_not_retried_for_a_while(monkeypatch):
    from backend.browser import live_chrome
    from backend.models import ProviderStatus
    from browser.adapters.base import ChatAdapter

    engine = object.__new__(live_chrome.LiveChromeEngine)
    engine._send_failed = {}
    assert not engine.send_blocked("gemini")
    engine.note_send_failed("gemini")
    assert engine.send_blocked("gemini") and not engine.send_blocked("qwen")
    engine.note_send_ok("gemini")
    assert not engine.send_blocked("gemini")
    engine.note_send_failed("gemini")
    clock = {"t": live_chrome.time.monotonic() + engine.SEND_FAIL_TTL_S + 1}
    monkeypatch.setattr(live_chrome.time, "monotonic", lambda: clock["t"])
    assert not engine.send_blocked("gemini")  # forgotten after the TTL: access can change

    engine.note_send_failed("gemini")
    clock["t"] += 5
    adapter = object.__new__(ChatAdapter)
    adapter.provider = "gemini"
    adapter.cfg = SimpleNamespace(max_retries=1)
    adapter.engine = SimpleNamespace(
        live=True, send_blocked=engine.send_blocked, front_allowed=lambda p: False,
        settings=SimpleNamespace(browser=SimpleNamespace(chrome_use_background=True, chrome_use_front_providers=[])),
    )

    async def emit(*a, **k):
        return None

    response = asyncio.run(adapter._ask_dom("j", "prompt", 1, emit))
    assert response.status == ProviderStatus.FAILED
    assert response.error.startswith("needs_focus: skipped in silent mode")
    assert response.duration_s is not None and response.duration_s < 1
