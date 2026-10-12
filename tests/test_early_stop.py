"""Early stopping and provenance: a hard question does not mean maximum research.

Strong primary evidence (a primary page the AI itself opened, supporting the claim) => one round, no
escalation. Weak => a follow-up in the same conversation. Unresolved => parallel independents (tests/
test_architecture.py). The curator may ask for more, but does not get to keep asking.
"""

from __future__ import annotations

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
