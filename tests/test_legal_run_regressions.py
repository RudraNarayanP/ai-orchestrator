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

        async def complete_json(self, messages, temperature=None, **_kw):
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


# ---- 7. a question about NOW needs a settled claim about now (general rule, live: s9 "still in force" answered "the idea is right")
NOW_Q = "Is section 9 of the Data Protection Act 2018 still in force as the provision that sets the UK's age of consent for online services?"


def test_an_unreviewed_current_status_claim_is_never_the_idea_is_right():
    verdicts = [
        v(S.NOT_REVIEWED, "Section 9 of the Data Protection Act 2018 remains in force as of 11 October 2026.", 0),
        v(S.PARTIALLY_SUPPORTED, "Section 9 of the Data Protection Act 2018 modifies Article 8(1) of the GDPR by substituting 13 for 16.", 1),
        v(S.PARTIALLY_SUPPORTED, "Section 9 sets the age threshold for children's consent at 13.", 2),
    ]
    report = rep("Section 9 of the Data Protection Act 2018 modifies Article 8(1) of the GDPR by substituting 13 for 16.", *verdicts)
    assert truth_state(report, NOW_Q) == "UNVERIFIED"
    final = build_final_answer(report, [], 1, NOW_Q)
    assert final.answer == "Couldn't verify that one."
    assert "idea is right" not in final.answer


def test_a_settled_current_status_claim_keeps_the_answer_and_other_questions_are_untouched():
    settled = [v(S.SUPPORTED, "Section 9 of the Data Protection Act 2018 remains in force as of 11 October 2026.", 2)]
    assert truth_state(rep("It is.", *settled), NOW_Q) == "TRUE"
    # outdated evidence does not settle a present-tense claim
    old = ClaimVerdict(claim_id="o", claim="Section 9 remains in force today.", verdict=S.PARTIALLY_SUPPORTED, confidence=Confidence.MODERATE, reasoning="r", problems=["outdated"])
    assert truth_state(rep("It is.", old), NOW_Q) == "UNVERIFIED"
    # a question that does not ask about the present is judged exactly as before
    past = [v(S.NOT_REVIEWED, "Section 9 remains in force as of today.", 0), v(S.SUPPORTED, "Section 9 lowered the age to 13.", 2)]
    assert truth_state(rep("Yes.", *past), "Did section 9 of the Data Protection Act 2018 lower the age of consent to 13?") == "TRUE"
    # no present-tense claim at all: nothing to be unreviewed, the ledger state stands
    assert truth_state(rep("Yes.", v(S.SUPPORTED, "Section 9 lowered the age to 13.", 2)), NOW_Q) == "TRUE"


# ---- 8. Le Chat's Work (Vibe) UI: the reply lives in div[data-message-author-role="assistant"]; its time stamp is not the answer
def test_le_chat_selectors_find_the_work_ui_reply_and_the_time_stamp_is_stripped():
    from browser.adapters.base import ChatAdapter
    from browser.adapters.selectors import selectors_for

    roots = selectors_for("le_chat").assistant_message
    assert '[data-message-author-role="assistant"]' in roots and '[data-message-part-type="answer"]' in roots
    adapter = object.__new__(ChatAdapter)
    assert adapter._clean("The capital of France is Paris.\n\n10:56pm") == "The capital of France is Paris."
    assert adapter._clean("It ended at 10:56pm sharp.\nSee 3:45 for the cut.") == "It ended at 10:56pm sharp.\nSee 3:45 for the cut."


# ---- 9. the curator and the whole question have a clock (live: curator 90 s, three rounds 410 s)
def test_a_curator_that_is_too_slow_is_dropped_for_the_next_model(monkeypatch):
    from backend.verification import verifier as vmod
    from backend.verification.llm import Endpoint, LLMReply

    calls: list[str] = []

    class FakeClient:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        async def complete_json(self, messages, temperature=None, **_kw):
            calls.append(self.endpoint.model)
            if self.endpoint.model == "slow-reasoner":
                await asyncio.sleep(5)
            return {"verdicts": []}, LLMReply(text="{}", ok=True, model=self.endpoint.model)

    monkeypatch.setattr(vmod, "LLMClient", FakeClient)
    endpoint = Endpoint(provider="openrouter", model="slow-reasoner", base_url="http://x", fallback_models=("fast-model",))
    curator = vmod.Verifier(endpoint, time_cap_s=1.6)
    curator.FIRST_SHARE_MIN_S, curator.MIN_LEFT_S = 0.3, 0.1
    curator._deadline = time.monotonic() + 1.6
    started = time.perf_counter()
    parsed, reply = asyncio.run(curator._curate([{"role": "user", "content": "x"}]))
    assert calls == ["slow-reasoner", "fast-model"] and reply.model == "fast-model" and parsed == {"verdicts": []}
    assert time.perf_counter() - started < 2.5

    # nothing answers in time: no parsed reply (the evidence ledger then judges), never an invented one
    calls.clear()
    only = vmod.Verifier(Endpoint(provider="openrouter", model="slow-reasoner", base_url="http://x"), time_cap_s=0.5)
    only.MIN_LEFT_S = 0.1
    only._deadline = time.monotonic() + 0.5
    parsed, reply = asyncio.run(only._curate([{"role": "user", "content": "x"}]))
    assert parsed is None and not reply.ok and "no answer in" in reply.error


def test_no_cap_means_the_plain_client(monkeypatch):
    from backend.verification import verifier as vmod
    from backend.verification.llm import Endpoint, LLMReply

    class FakeClient:
        def __init__(self, endpoint):
            pass

        async def complete_json(self, messages, temperature=None, **_kw):
            return {"ok": 1}, LLMReply(text="{}", ok=True)

    monkeypatch.setattr(vmod, "LLMClient", FakeClient)
    curator = vmod.Verifier(Endpoint(provider="openrouter", model="m", base_url="http://x"))
    assert asyncio.run(curator._curate([]))[0] == {"ok": 1}


def test_a_question_out_of_time_starts_no_further_round():
    runner = object.__new__(ResearchRunner)
    runner.settings = SimpleNamespace(research=SimpleNamespace(time_budget_s=120))
    job = SimpleNamespace(stop_note="")
    runner._started = time.monotonic() - 30
    assert runner._over_budget(job) is False and job.stop_note == ""
    runner._started = time.monotonic() - 130
    assert runner._over_budget(job) is True
    assert "stopped researching early" in job.stop_note and "settled" not in job.stop_note
    runner.settings.research.time_budget_s = 0
    assert runner._over_budget(SimpleNamespace(stop_note="")) is False


# ---- 11. a present-tense claim about ANOTHER provision does not answer "is section 9 still in force?" (live run B)
def test_a_present_claim_about_another_provision_does_not_settle_the_present_question():
    from backend.research.style import answers_the_present, reference_anchors

    assert reference_anchors("Is s. 9 of the Act and Article 8(1) of the GDPR still in force?") == {"section 9", "article 8"}
    assert not answers_the_present(NOW_Q, "Article 8 of the UK GDPR currently establishes the 13-year threshold for consent.")
    assert answers_the_present(NOW_Q, "As of 12 October 2026, section 9 of the Data Protection Act 2018 is no longer in force.")
    assert not answers_the_present("Did section 9 lower the age to 13?", "Section 9 remains in force today.")
    # no numbered provision in the question: any present-tense claim about it counts, as before
    assert answers_the_present("Is the UK currently a party to the Lugano Convention?", "The UK is not currently a party.")

    verdicts = [
        v(S.NOT_REVIEWED, "As of 12 October 2026, section 9 of the Data Protection Act 2018 is no longer in force.", 0),
        v(S.NOT_REVIEWED, "Section 9 of the Data Protection Act 2018 was omitted with effect from 31 December 2020.", 0),
        v(S.PARTIALLY_SUPPORTED, "Article 8 of the UK GDPR currently establishes the 13-year threshold for consent.", 1),
        v(S.PARTIALLY_SUPPORTED, "The UK's 13-year-old threshold for children's consent is now provided for by Article 8.", 1),
    ]
    report = rep("Article 8 of the UK GDPR currently establishes the 13-year threshold for consent.", *verdicts)
    assert truth_state(report, NOW_Q) == "UNVERIFIED"
    assert build_final_answer(report, [], 1, NOW_Q).answer == "Couldn't verify that one."


# ---- 12. Le Chat's "New Connectors" announcement is closed with its own "Later" button, not reported as a consent page (live run B)
def test_a_later_button_on_an_announcement_is_dismissed_but_accept_is_never_clicked():
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "browser" / "adapters" / "dom_library.py").read_text(encoding="utf-8")
    want = next(line for line in js.splitlines() if "const want = " in line)
    assert r"^\s*later\s*$" in want, "the bare 'Later' button of a product announcement must be a dismiss control"
    assert "accept all" in want and "unsafe = /(accept|agree|allow|consent" in js, "the safe-dismiss refusal of accept / agree / allow stays"


def test_the_curator_asks_a_cooling_model_last_and_never_waits_out_a_rate_limit_on_the_way(monkeypatch):
    from backend.verification import verifier as vmod
    from backend.verification.llm import Endpoint, LLMReply, cool_model

    seen: list[tuple[str, bool]] = []

    class FakeClient:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        async def complete_json(self, messages, temperature=None, fail_fast=False):
            seen.append((self.endpoint.model, fail_fast))
            return {"verdicts": []}, LLMReply(text="{}", ok=True, model=self.endpoint.model)

    monkeypatch.setattr(vmod, "LLMClient", FakeClient)
    endpoint = Endpoint(provider="openrouter", model="primary", base_url="http://x", fallback_models=("second",))
    cool_model("http://x", "primary", 60)
    curator = vmod.Verifier(endpoint, time_cap_s=30)
    curator._deadline = time.monotonic() + 30
    asyncio.run(curator._curate([{"role": "user", "content": "x"}]))
    assert seen == [("second", True)], "the cooling primary is behind the healthy fallback, and a non-last model fails fast"


# ---- 13. a long reason is cut at a clause, never inside a statute's name (live run C: "...2020 by the Data Protection. \u274c")
def test_a_reason_is_never_cut_inside_a_title_with_a_comma():
    from backend.research.style import fit_fact, render_truth

    fact = (
        "Section 9 of the Data Protection Act 2018 is no longer in force as the provision setting the UK's age of consent for online "
        "services; it was omitted with effect from 31 December 2020 by the Data Protection, Privacy and Electronic Communications "
        "(Amendments etc) (EU Exit) Regulations 2019."
    )
    cut = fit_fact(fact, 230)
    assert cut.endswith("31 December 2020.") and "Data Protection." not in cut
    shown = render_truth("FALSE", question="Is section 9 of the Data Protection Act 2018 still in force?", answer=fact)
    assert "Data Protection." not in shown and "31 December 2020" in shown
    assert fit_fact("It is a rule, and it applies.", 200) == "It is a rule, and it applies."


# ---- 14. under-sourced is not partly wrong (live run C: "Yeah, the idea is right, but the date part is a bit off" over a supported answer)
def test_a_single_source_claim_beside_a_supported_one_does_not_make_the_answer_partly_wrong():
    ok = v(S.SUPPORTED, "Section 9(1) of the DPDP Act, 2023 requires a Data Fiduciary to obtain verifiable parental consent before processing a child's personal data.", 2)
    thin = v(S.PARTIALLY_SUPPORTED, "The Digital Personal Data Protection Act, 2023 is Act No. 22 of 2023.", 1)
    thin.problems = ["single_source"]
    wrong = v(S.PARTIALLY_SUPPORTED, "Rule 10 of the DPDP Rules, 2025 was notified in 2024.", 1)
    wrong.problems = ["outdated"]
    assert truth_state(rep("Yes.", ok, thin), DPDP_Q) == "TRUE"
    assert truth_state(rep("Yes.", ok, thin, wrong), DPDP_Q) == "PARTLY", "a claim that is partly WRONG still makes it partly"
    only_thin = v(S.PARTIALLY_SUPPORTED, "Act No. 22 of 2023 is the DPDP Act.", 1)
    only_thin.problems = ["single_source"]
    assert truth_state(rep("Yes.", only_thin), DPDP_Q) == "PARTLY", "with nothing supported, thin evidence alone is still not a flat yes"
    final = build_final_answer(rep("Section 9(1) requires verifiable parental consent for a child's data.", ok, thin), [], 1, DPDP_Q)
    assert final.answer.startswith("Yeah, you're right.") and "date part" not in final.answer


def test_a_still_in_force_question_answered_no_longer_in_force_leads_with_no():
    """Live s9 re-check (2026-10-12): the verdict was PARTLY, and the reply opened "Yeah, the idea is right, but the date
    part is a bit off - Section 9 ... is no longer in force" -- a yes in front of a no."""
    from backend.research.style import render_truth

    q = "Is section 9 of the Data Protection Act 2018 still in force as the provision that sets the UK's age of consent?"
    out = render_truth("PARTLY", question=q, answer="Section 9 of the Data Protection Act 2018 is no longer in force.", aspect="date")
    assert out.lower().startswith("nah") and "no longer in force" in out, out
    # a PARTLY whose wrong part is not the asked-about status keeps the partly line
    kept = render_truth("PARTLY", question=q, answer="It sets the age at 13, not 16.", aspect="number")
    assert kept.startswith("Yeah, the idea is right"), kept
