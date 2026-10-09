"""A provider's own quoted passage is recorded, and it proves nothing on its own.

Live (Eiffel run, 2026-10-09): DeepSeek's UI showed *"The record-breaking structure was
completed on 31 March 1889 and the Tower opened to the public on 15 May"* from a page our
fetcher was blocked from. Citing a URL, quoting a source, and OmniBrain reading the page are
three different acts. The quote goes into the ledger as `not_checked` so the curator and the
audit trail can see it -- and so it can never establish a claim or lift confidence by itself.
"""

from __future__ import annotations

from backend.evidence.pool import provider_quote_rows
from backend.models import (
    Citation,
    Claim,
    ClaimStatus,
    Confidence,
    ProviderResponse,
    ProviderStatus,
    SourceCheckStatus,
)
from backend.storage.db import Store
from tests.conftest import base_settings

QUOTE = (
    "The record-breaking structure was completed on 31 March 1889 and the Tower opened to "
    "the public on 15 May"
)
SOURCE = "https://bie.example.org/site/en/1889-paris"


def response(provider: str, snippet: str, *, url: str = SOURCE, chat: str = "", ai_opened: bool | None = None) -> ProviderResponse:
    return ProviderResponse(
        job_id="j",
        provider=provider,
        round=1,
        prompt="",
        answer_text="KEY CLAIMS\n1. The Eiffel Tower was completed on 31 March 1889.",
        raw_text="x",
        status=ProviderStatus.COMPLETED,
        citations=[Citation(url=url, title="Expo 1889 Paris", snippet=snippet, ai_opened=ai_opened)],
        conversation_url=chat or f"https://{provider}.example/chat/abc123",
    )


def claim(text: str = "The Eiffel Tower was completed on 31 March 1889.", cid: str = "clm_1") -> Claim:
    return Claim(id=cid, job_id="j", claim=text, kind="date", provider_sources=["deepseek"])


# --------------------------------------------------------------------- the row itself


def test_a_provider_quote_is_stored_verbatim_and_unverified():
    rows = provider_quote_rows("j", [response("deepseek", f'The official Eiffel Tower website states: “{QUOTE}”')], [claim()], 1)
    assert len(rows) == 1
    ev = rows[0]
    assert ev.origin == "provider_quote"
    assert ev.check_status == SourceCheckStatus.NOT_CHECKED, "a quote never counts as a check"
    assert ev.verbatim_excerpt == QUOTE, "the passage is preserved exactly"
    assert ev.url == SOURCE and ev.domain == "bie.example.org"
    assert "provider-quoted, not opened by us" in ev.check_notes
    assert "quoted by deepseek" in ev.check_notes and "deepseek.example/chat/abc123" in ev.check_notes
    assert ev.omnibrain_opened is False, "we did not open it"
    assert ev.provenance == "MENTIONED", ev.provenance
    # the live quote says "the record-breaking structure was completed on 31 March 1889":
    # the clause never names the tower, so it is recorded, shown and left unattached.
    assert ev.claim_id is None and "does not state this claim" in ev.check_notes, ev.check_notes


def test_a_quote_that_names_the_subject_and_the_date_is_filed_under_that_claim():
    named = "The Eiffel Tower's record-breaking structure was completed on 31 March 1889 after two years of work"
    rows = provider_quote_rows("j", [response("deepseek", f'The official website states: “{named}”', ai_opened=True)], [claim()], 1)
    assert len(rows) == 1
    ev = rows[0]
    assert ev.claim_id == "clm_1", ev.check_notes
    assert ev.verbatim_excerpt == named
    assert "'" in ev.verbatim_excerpt, "an apostrophe inside the quote is part of the quote"
    assert ev.check_status == SourceCheckStatus.NOT_CHECKED, "filed under the claim, still not a confirmation"
    assert ev.provenance == "CITED", ev.provenance


def test_a_quote_that_does_not_state_the_claim_is_left_unattached():
    rows = provider_quote_rows(
        "j",
        [response("deepseek", "The official Eiffel Tower website states: “The tower remains the most visited monument in the world, drawing millions of visitors each year.”")],
        [claim()],
        1,
    )
    assert len(rows) == 1
    assert rows[0].claim_id is None, "a passage that never gives the date is not evidence for this claim"
    assert rows[0].check_status == SourceCheckStatus.NOT_CHECKED


def test_the_export_never_calls_a_provider_quote_a_page_we_opened(tmp_path):
    from backend.export import job_markdown
    from backend.models import Evidence
    from tests.test_history_export import finished_job

    job = finished_job()
    job.evidence.append(
        Evidence(
            job_id=job.id, claim_id=job.claims[0].id, url=SOURCE, title="Expo 1889 Paris",
            domain="bie.example.org", check_status=SourceCheckStatus.NOT_CHECKED,
            origin="provider_quote", verbatim_excerpt=QUOTE,
            check_notes="provider-quoted, not opened by us; quoted by deepseek",
        )
    )
    settings = base_settings(storage={"db_path": str(tmp_path / "q.db")})
    store = Store(settings)
    store.save_job(job)
    md = job_markdown(store.job_snapshot(job.id))

    opened, quoted = md.split("## Evidence we opened", 1)[1].split("## Quoted by a provider", 1)
    assert SOURCE not in opened, "the quote row must not sit under the pages we opened"
    assert "found on page" not in quoted, "we never saw that page"
    assert "Quoted by a provider, not opened by us (1)" in md
    assert QUOTE[:50] in quoted


def test_no_quote_text_no_row():
    rows = provider_quote_rows("j", [response("chatgpt", f"{SOURCE} — OPENED (official site; read the page)")], [claim()], 1)
    assert rows == [], "a self-declared 'I opened it' with no quoted content is not a quote"


# ------------------------------------------------------- through a real research run


async def test_a_quoted_blocked_source_leaves_the_answer_uncertain(net, fake_openai, tmp_path):
    """Nothing we opened confirms the date; two providers quote pages we could not read.
    The quotes reach the curator and the audit trail, and still establish nothing."""
    from tests.test_ledger_adjudication import ask, curator, make_curator

    net.fail(SOURCE, status=SourceCheckStatus.BLOCKED)
    scripts = {
        "chatgpt": {
            "answer": "KEY CLAIMS\n1. The Eiffel Tower was completed on 31 March 1889.",
            "citations": [{"url": SOURCE, "title": "Expo 1889 Paris", "snippet": f"The official Eiffel Tower website says: “{QUOTE}”"}],
        },
        "gemini": {
            "answer": "KEY CLAIMS\n1. The Eiffel Tower was completed on 31 March 1889.",
            "citations": [{"url": SOURCE, "title": "Expo 1889 Paris", "snippet": f"Quotation from the Eiffel Tower page: “{QUOTE}”"}],
        },
        "search": {"answer": "results", "citations": []},
    }
    decide = lambda text: {
        "verdict": "insufficient_evidence",
        "confidence": "low",
        "reasoning": "no page we opened states it",
        "strong_evidence": [],
        "problems": [],
    }
    fn = curator(decide=decide, answer="", confidence="low")
    db = str(tmp_path / "quote_check.db")
    job = await ask(base_settings(storage={"db_path": db}), scripts, question="In which year was the Eiffel Tower completed?", verifier=make_curator(fake_openai.script(fn)))

    quotes = [e for e in job.evidence if e.origin == "provider_quote"]
    assert quotes, "the provider's quoted passage must survive into the ledger"
    assert all(e.check_status == SourceCheckStatus.NOT_CHECKED for e in quotes)
    assert QUOTE in quotes[0].verbatim_excerpt
    assert not any(e.check_status == SourceCheckStatus.CONFIRMED for e in job.evidence), "a quote cannot become a confirmation"
    assert all(c.status != ClaimStatus.SUPPORTED for c in job.claims), [(c.claim, c.status) for c in job.claims]
    assert job.final.confidence in {Confidence.LOW, Confidence.NONE}, job.final.confidence
    assert job.final.answer.lower().startswith(("i don't know", "i couldn't", "i could not", "i can't", "couldn't verify")), job.final.answer

    sent = "".join(fn.seen)
    assert QUOTE[:60] in sent, "the curator must see what the provider quoted"
    assert "provider-quoted, not opened by us" in sent

    Store(base_settings(storage={"db_path": db})).save_job(job)
    rows = [e for e in Store(base_settings(storage={"db_path": db})).job_snapshot(job.id)["evidence"] if e["origin"] == "provider_quote"]
    assert rows and rows[0]["check_status"] == "not_checked", "the audit trail shows it as unverified"


def test_two_providers_quoting_the_same_passage_do_not_lift_confidence(net):
    """Agreement is bookkeeping. The same quote twice is still not a page we read."""
    shown = f'The official Eiffel Tower website states: “{QUOTE}”'
    rows = provider_quote_rows("j", [response("deepseek", shown), response("chatgpt", shown)], [claim()], 1)
    assert len(rows) == 2
    assert all(r.check_status == SourceCheckStatus.NOT_CHECKED for r in rows)
    assert all(r.provenance != "CLAIM_SUPPORTED" for r in rows)
    assert {r.cited_by[0] for r in rows} == {"deepseek", "chatgpt"}, "who displayed it is recorded per provider"
