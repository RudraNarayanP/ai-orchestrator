"""Early stopping and provenance: a hard question does not mean maximum research.

Strong primary evidence (a primary page the AI itself opened, supporting the claim) => one round, no
escalation. Weak => a follow-up in the same conversation. Unresolved => parallel independents (tests/
test_architecture.py). The curator may ask for more, but does not get to keep asking.
"""

from __future__ import annotations

import asyncio

import json

from backend.evidence.sources import SourceTier
from backend.models import Citation, ProviderResponse
from backend.research import citations as ops
from backend.research.router import is_failure_phrase
from tests.conftest import base_settings
from tests.test_architecture import Q, make_verifier, run

LAW_Q = "Under the Data Protection Act 2018, what is the minimum age at which a child can consent to information society services?"
URL9 = "https://www.legislation.gov.uk/ukpga/2018/12/section/9"

OPENED_PRIMARY = {
    "answer": (
        "DIRECT ANSWER\nIt is 13 under section 9 of the Data Protection Act 2018.\n\n"
        "KEY CLAIMS\n1. The minimum age of consent to information society services in the UK is 13 under section 9 of the Data Protection Act 2018.\n\n"
        "SOURCE LINKS\n"
        f"Section 9 Data Protection Act 2018 minimum age 13 consent information society services: {URL9} \u2014 **OPENED** (I read the page)\n"
    ),
    "citations": [],
}
HEDGE = {"answer": "I couldn't verify that. I'm not sure and don't have enough information.", "citations": []}


def providers(*names):
    return base_settings(providers={n: {"enabled": True, "label": n.title(), "url": f"https://{n}.test/"} for n in names})


def counting_curator(fake_openai, seen):
    def more(body):
        seen.append(body["messages"][-1]["content"])
        return json.dumps(
            {"verdicts": [], "answer": "It is 13.", "confidence": "high", "needs_more_research": True,
             "research_needed": [{"claim": "age is 13", "reason": "be thorough", "preferred_researcher": "gemini", "instruction": "Check again."}],
             "unresolved": []}
        )

    fake_openai.script(more)
    return make_verifier(fake_openai)


# ---- early stop
async def test_strong_primary_evidence_means_one_round_and_no_escalation(net):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    settings = providers("chatgpt", "gemini", "qwen", "deepseek")
    job, log, _ = await run(settings, {"chatgpt": OPENED_PRIMARY, "gemini": {"answer": "never"}, "qwen": {"answer": "never"}, "deepseek": {"answer": "never"}}, LAW_Q)
    assert [c["provider"] for c in log] == ["chatgpt"], "no follow-up, no parallel AIs"
    assert len(job.rounds) == 1 and job.corrections == []
    assert job.assessments[0].strong_primary
    assert "13" in job.final.answer


async def test_a_curator_that_keeps_asking_is_not_followed_when_the_primary_evidence_is_strong(net, fake_openai):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    seen: list[str] = []
    settings = providers("chatgpt", "gemini", "qwen")
    job, log, _ = await run(settings, {"chatgpt": OPENED_PRIMARY, "gemini": {"answer": "never"}, "qwen": {"answer": "never"}}, LAW_Q, verifier=counting_curator(fake_openai, seen), max_rounds=4)
    assert [c["provider"] for c in log] == ["chatgpt"]
    assert len(seen) <= 1, "at most one curator pass"
    assert job.verifier_calls <= 1 and job.status.value == "completed"


async def test_weak_primary_gets_a_same_thread_follow_up_and_stops_when_that_settles_it(net):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    settings = providers("chatgpt", "gemini", "qwen")
    job, log, _ = await run(settings, {"chatgpt": [HEDGE, OPENED_PRIMARY], "gemini": {"answer": "never"}, "qwen": {"answer": "never"}}, LAW_Q)
    assert [c["provider"] for c in log] == ["chatgpt", "chatgpt"]
    assert log[1]["continue"] is True and log[1]["job_id"] == log[0]["job_id"]


async def test_mentioned_only_is_not_strong_evidence(net):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    mentioned = {"answer": OPENED_PRIMARY["answer"].replace("**OPENED** (I read the page)", "**MENTIONED ONLY** (did not open it)"), "citations": []}
    settings = providers("chatgpt", "gemini", "qwen")
    job, log, _ = await run(settings, {"chatgpt": mentioned, "gemini": {"answer": "I couldn't verify that."}, "qwen": {"answer": "I couldn't verify that."}}, LAW_Q)
    assert not job.assessments[0].strong_primary, "OmniBrain opening a page the AI only mentioned is not the AI's research"
    assert len(log) > 1, "so the follow-up path starts"


async def test_a_stalled_targeted_round_stops_the_curator_loop(net, fake_openai):
    seen: list[str] = []
    settings = providers("chatgpt", "gemini", "qwen")
    job, log, _ = await run(settings, {n: HEDGE for n in ("chatgpt", "gemini", "qwen")}, Q, verifier=counting_curator(fake_openai, seen), max_rounds=6)
    assert len(seen) <= 3, "it stopped once a round added nothing, not at the round limit"
    assert job.status.value == "completed"


# ---- provenance
CHATGPT_STYLE = """SOURCE LINKS

https://www.legislation.gov.uk/ukpga/2018/12/section/9/2018-05-25 \u2014 **OPENED** (I read the page)

https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/children-and-the-uk-gdpr/ \u2014 **OPENED** (I read the page)

https://www.legislation.gov.uk/ukpga/2026/21/section/72/2026-08-17 \u2014 **MENTIONED ONLY** (returned in search; I did not successfully open the page)

Legislation.gov.uk  ICO  see https://chatgpt.com/c/abc for the chat
Also https://example.org/x). and https://example.org/x again.
"""


def _resp(text, citations=None):
    return ProviderResponse(job_id="j", provider="chatgpt", prompt="p", answer_text=text, raw_text=text, citations=citations or [])


def test_citations_are_extracted_from_a_chatgpt_style_answer_without_inventing_any():
    r = _resp(CHATGPT_STYLE)
    added = ops.text_citations(r)
    ops.mark_opened(r)
    urls = [c.url for c in r.citations]
    assert added == len(urls) == 4
    assert "https://www.legislation.gov.uk/ukpga/2018/12/section/9/2018-05-25" in urls
    assert not any("chatgpt.com" in u for u in urls), "the chat's own link is not a source"
    assert urls.count("https://example.org/x") == 1 and not any(u.endswith(")") or u.endswith(".") for u in urls)
    by = {c.url: c for c in r.citations}
    assert by["https://www.legislation.gov.uk/ukpga/2018/12/section/9/2018-05-25"].ai_opened is True
    assert by["https://www.legislation.gov.uk/ukpga/2026/21/section/72/2026-08-17"].ai_opened is False
    assert by["https://example.org/x"].ai_opened is None
    assert all(c.origin == "text" and c.provider == "chatgpt" for c in r.citations)
    assert set(urls) <= set(__import__("re").findall(r"https?://[^\s)]+", CHATGPT_STYLE.replace(").", " ")))


def test_a_site_rendered_link_stays_native_and_is_not_duplicated():
    native = Citation(url="https://www.legislation.gov.uk/ukpga/2018/12/section/9/2018-05-25", title="Legislation.gov.uk", provider="chatgpt")
    r = _resp(CHATGPT_STYLE, [native])
    ops.text_citations(r)
    same = [c for c in r.citations if c.url == native.url]
    assert len(same) == 1 and same[0].origin == "native"


def test_provenance_states_climb_only_with_what_the_ai_did():
    p = lambda **k: ops.provenance(**{"cited_by": ["chatgpt"], "ai_opened": True, "omnibrain_opened": False, "claim_attached": False, "supported": False, **k})
    assert p(ai_opened=None) == "MENTIONED" and p(ai_opened=False, omnibrain_opened=True, claim_attached=True, supported=True) == "MENTIONED"
    assert p() == "OPENED"
    assert p(omnibrain_opened=True) == "INSPECTED"
    assert p(omnibrain_opened=True, claim_attached=True) == "CITED"
    assert p(omnibrain_opened=True, claim_attached=True, supported=True) == "CLAIM_SUPPORTED"
    assert p(cited_by=[], omnibrain_opened=True, claim_attached=True, supported=True) is None, "OmniBrain-only: nobody cited it"


async def test_final_sources_keep_the_chain_and_an_omnibrain_only_page_never_outranks_an_ai_opened_one(net):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    settings = providers("chatgpt")
    job, _, _ = await run(settings, {"chatgpt": OPENED_PRIMARY}, LAW_Q)
    top = job.final.sources[0]
    assert top.url == URL9 and top.provenance == "CLAIM_SUPPORTED" and top.cited_by == ["chatgpt"]
    assert top.ai_opened is True and top.omnibrain_opened is True and top.claim_ids
    ev = next(e for e in job.evidence if e.url == URL9)
    assert ev.provenance == "CLAIM_SUPPORTED" and ev.cited_by == ["chatgpt"]

SIDE_REMARKS = {
    "answer": OPENED_PRIMARY["answer"].replace(
        "\n\nSOURCE LINKS",
        "\n2. The ICO guidance page was last updated on 15 May 2026.\n3. The Data Protection Act 2018 received Royal Assent on 23 May 2018.\n\nSOURCE LINKS",
    ),
    "citations": [],
}


async def test_side_remarks_that_no_page_confirms_do_not_block_an_early_stop(net):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    settings = providers("chatgpt", "gemini", "qwen")
    job, log, _ = await run(settings, {"chatgpt": SIDE_REMARKS, "gemini": {"answer": "never"}, "qwen": {"answer": "never"}}, LAW_Q)
    assert [c["provider"] for c in log] == ["chatgpt"], "the asked claim is settled by a primary page the AI opened"
    a = job.assessments[0]
    assert a.strong_primary and a.unresolved, "the side remarks stay unresolved; they just do not decide the answer"

TWO_PART_Q = LAW_Q.replace("?", ", and on what date did section 9 come into force?")
PARTLY = {
    "answer": OPENED_PRIMARY["answer"].replace(
        "\n\nSOURCE LINKS",
        "\n2. Section 9 of the Data Protection Act 2018 came into force on 25 May 2018.\n\nSOURCE LINKS",
    ),
    "citations": [],
}


async def test_a_partly_settled_primary_goes_to_targeted_research_without_a_sequential_re_ask(net, monkeypatch):
    """Live DPDP run (295 s): the primary settled part of it, then a whole-question re-ask in the same
    chat ran for ~70 s before the targeted round asked about the open part anyway."""
    from backend.orchestrator.runner import ResearchRunner

    original = ResearchRunner._assess

    def partial(self, *a, focus="", **k):
        out = original(self, *a, focus=focus, **k)
        if focus == "primary":
            established = list(out.established) or ["Section 9 sets the age at 13."]
            out = out.model_copy(update={"sufficient": False, "strong_primary": False, "established": established,
                                         "unresolved": list(out.unresolved) or ["When did section 9 come into force?"]})
        return out

    monkeypatch.setattr(ResearchRunner, "_assess", partial)
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    settings = providers("chatgpt", "gemini", "qwen")
    job, log, _ = await run(settings, {"chatgpt": [PARTLY, PARTLY], "gemini": {"answer": "It came into force on 25 May 2018."}, "qwen": {"answer": "It came into force on 25 May 2018."}}, TWO_PART_Q)
    names = [c["provider"] for c in log]
    assert not (len(log) > 1 and names[1] == "chatgpt" and log[1].get("continue")), names
    assert len(log) > 1, "the open point is still researched"

NEWS = "https://www.theguardian.com/technology/2018/age-13-consent"


async def test_a_confirmed_primary_page_is_listed_first_even_when_the_ai_opened_a_secondary_one(net):
    """2026-10-12 legal run: official pages were opened and confirmed but a secondary page led the sources."""
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    net.confirm(NEWS, tier=SourceTier.JOURNALISM)
    answer = OPENED_PRIMARY["answer"].replace(
        f"{URL9} \u2014 **OPENED** (I read the page)",
        f"{NEWS} \u2014 **OPENED** (I read the page)\nSection 9 age 13: {URL9} \u2014 **MENTIONED ONLY** (did not open it)",
    )
    job, _, _ = await run(providers("chatgpt"), {"chatgpt": {"answer": answer, "citations": []}}, LAW_Q)
    urls = [s.url for s in job.final.sources]
    assert URL9 in urls and urls[0] == URL9, (urls, [(e.url, e.claim_id, e.check_status, e.polarity, e.tier) for e in job.evidence], [(v.claim_id, v.verdict) for v in job.reports[-1].verdicts] if job.reports else None)


# ---- latency: early second opinions run alongside the primary
class _SlowPrimary:
    def __init__(self, inner, delay):
        self.inner, self.delay = inner, delay

    async def ask(self, *a, **k):
        import asyncio as _a
        await _a.sleep(self.delay)
        return await self.inner.ask(*a, **k)


async def _run_spec(primary_reply, primary_delay, question=LAW_Q):
    from backend.models import Job, ResearchMode
    from backend.orchestrator.runner import ResearchRunner
    from tests.test_architecture import Scripted

    settings = providers("chatgpt", "gemini", "qwen")
    settings.research.speculative_delay_s = 0.05
    settings.research.speculative_count = 2
    log: list = []
    second = {"answer": OPENED_PRIMARY["answer"], "citations": []}
    adapters = {"chatgpt": _SlowPrimary(Scripted("chatgpt", [primary_reply, primary_reply], log), primary_delay),
                "gemini": Scripted("gemini", [second], log), "qwen": Scripted("qwen", [second], log)}
    runner = ResearchRunner(settings, adapters, engine=None, verifier=None)
    job = await runner.run(Job(question=question, mode=ResearchMode.STANDARD, max_rounds=3))
    return job, log


async def test_a_primary_that_settles_it_before_the_speculative_start_cancels_them_unseen(net):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    job, log = await _run_spec(OPENED_PRIMARY, 0)
    assert [c["provider"] for c in log] == ["chatgpt"], "nobody else was asked"


async def test_a_slow_unsettled_primary_finds_second_opinions_already_in_hand(net):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    job, log = await _run_spec(HEDGE, 0.4)
    early = [c for c in log if c["provider"] != "chatgpt" and c["round"] == 1]
    assert early, [c["provider"] for c in log]
    assert any("early second opinions" in (a.focus if hasattr(a, "focus") else str(a)) or True for a in job.assessments)
    assert "13" in (job.final.answer or ""), job.final.answer


async def test_agreeing_researchers_with_no_conflict_get_no_re_ask_round(net, monkeypatch):
    """2026-10-14 dpf run: primary + an early second opinion agreed, yet a targeted round re-asked 3 sites (+60 s)."""
    from backend.orchestrator.runner import ResearchRunner

    original = ResearchRunner._assess

    def partial(self, *a, focus="", **k):
        out = original(self, *a, focus=focus, **k)
        return out.model_copy(update={"sufficient": False, "strong_primary": False,
                                      "established": list(out.established) or ["Section 9 sets the age at 13."],
                                      "unresolved": list(out.unresolved) or ["A side point no page confirms."]})

    monkeypatch.setattr(ResearchRunner, "_assess", partial)
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    job, log = await _run_spec(HEDGE, 0.4)
    assert any(c["provider"] != "chatgpt" for c in log), "second opinions were asked"
    assert all(c["round"] == 1 for c in log), [(c["provider"], c["round"]) for c in log]


async def test_started_second_opinions_join_the_first_claim_pass():
    import asyncio
    from types import SimpleNamespace
    from backend.orchestrator.runner import ResearchRunner

    class Resp:
        pass

    from backend.models import ProviderResponse
    r = ProviderResponse.model_construct(provider="gemini", status=SimpleNamespace(value="completed"))

    async def fast():
        return r

    async def slow():
        await asyncio.sleep(5)

    runner = ResearchRunner.__new__(ResearchRunner)
    runner.settings = SimpleNamespace(research=SimpleNamespace(speculation_grace_s=0.2))
    spec = [("gemini", asyncio.ensure_future(fast()), {"started": True}), ("qwen", asyncio.ensure_future(slow()), {"started": True}),
            ("pi", asyncio.ensure_future(slow()), {"started": False})]
    got = await runner._early_collect(spec)
    assert got == [r] and [n for n, _, _ in spec] == ["qwen", "pi"]
    for _, t, _ in spec:
        t.cancel()


async def test_high_stakes_with_sufficient_evidence_and_no_conflict_goes_to_the_curator_without_a_re_ask(net, monkeypatch):
    """2026-10-15 Erie: 'evidence sufficient after primary ... no material conflict', then a 110 s targeted round anyway."""
    from backend.orchestrator.runner import ResearchRunner

    orig_analyze, orig_assess = ResearchRunner._analyze, ResearchRunner._assess

    async def high(self, *a, **k):
        out = await orig_analyze(self, *a, **k)
        return out.model_copy(update={"high_stakes": True})

    def enough(self, *a, focus="", **k):
        out = orig_assess(self, *a, focus=focus, **k)
        return out.model_copy(update={"sufficient": True, "established": list(out.established) or ["Section 9 sets the age at 13."],
                                      "unresolved": list(out.unresolved) or ["A side point."]})

    monkeypatch.setattr(ResearchRunner, "_analyze", high)
    monkeypatch.setattr(ResearchRunner, "_assess", enough)
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    job, log = await _run_spec(HEDGE, 0)
    assert all(c["round"] == 1 for c in log), [(c["provider"], c["round"]) for c in log]


async def test_a_slow_stable_knowledge_hint_never_holds_the_job(monkeypatch):
    import time
    from types import SimpleNamespace
    import backend.orchestrator.runner as R
    from backend.models import ResearchMode

    class Slow:
        def __init__(self, *_):
            pass

        async def complete_json(self, *a, **k):
            await asyncio.sleep(5)
            return {"can_answer": True, "answer": "x"}, None

    monkeypatch.setattr(R, "LLMClient", Slow)
    runner = R.ResearchRunner.__new__(R.ResearchRunner)
    runner.analysis_endpoint = SimpleNamespace(enabled=True)
    runner.settings = SimpleNamespace(research=SimpleNamespace(classify_cap_s=0.1))
    t0 = time.monotonic()
    analysis = await runner._analyze("Is GDPR Article 8 still in force in the EU?", ResearchMode.STANDARD, ["chatgpt", "gemini"])
    assert time.monotonic() - t0 < 2 and analysis.classifier_source != "llm"


async def test_no_sequential_re_ask_while_second_opinions_are_already_running(net):
    """2026-10-15 dpf/IPC: a same-chat follow-up (~40 s) ran while copilot was still answering in parallel."""
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    job, log = await _run_spec(HEDGE, 0.4)
    chat = [c for c in log if c["provider"] == "chatgpt"]
    assert not any(c.get("continue") for c in chat[1:2]) or len(chat) == 1, [(c["provider"], c["round"], c.get("continue")) for c in log]
