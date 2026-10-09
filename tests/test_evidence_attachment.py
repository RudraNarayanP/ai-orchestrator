"""Evidence is matched against what the opened page says, not against its headline.

Live (eiffel runs, 2026-10-09): the official tower site's page "133 years and 330 metres"
states the completion date inside the body, and a canonical date key ("1887-01") was being
looked for in the page text instead of the words the page actually uses -- so a page that
settled the claim was filed as a mismatch and the run answered "Couldn't verify that one."
Matching stays strict: the date has to be on the page, in the page we opened.
"""

from __future__ import annotations

from backend.evidence import sources
from backend.evidence.sources import FetchedPage, check_support, gather_from_links
from backend.evidence.sources import SourceTier
from backend.models import Claim, ClaimStatus, Confidence, SourceCheckStatus
from backend.verification.verifier import VerifierReport, truth_state
from backend.export import job_markdown
from tests.conftest import base_settings

EIFFEL_BODY = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. "
    "Work on the foundations began in January 1887 and the construction of the tower "
    "itself was completed on 31 March 1889, taking two years, two months and five days. "
    "It was originally built as the entrance arch to the 1889 World's Fair."
)


def page(text: str = EIFFEL_BODY, *, title: str = "133 years and 330 metres", ok: bool = True, url: str = "https://www.toureiffel.paris/en/news/history-and-culture/133-years") -> FetchedPage:
    return FetchedPage(url=url, final_url=url, status=200 if ok else 0, title=title, text=text, ok=ok)


def outcome(claim: str, pg: FetchedPage) -> dict:
    return check_support(claim, pg)


# ------------------------------------------------- positive: the fact is in the body


def test_a_date_stated_in_the_body_confirms_even_when_the_title_omits_it():
    out = outcome("The Eiffel Tower was completed on 31 March 1889.", page())
    assert out["status"] == SourceCheckStatus.CONFIRMED, out
    assert out["missing"] == [], out
    assert "1889" in (out["excerpt"] or ""), "the recorded passage is the sentence carrying the date"
    assert "completed on 31 March 1889" in out["excerpt"]


def test_the_same_date_written_the_other_way_confirms_too():
    """Models answer in US order, European pages write it the other way round."""
    out = outcome("The Eiffel Tower was completed on March 31, 1889.", page())
    assert out["status"] == SourceCheckStatus.CONFIRMED, out
    assert "1889-03-31" not in out["missing"], "a canonical key is never what the page prints"


def test_a_paraphrase_of_the_body_still_confirms():
    out = outcome("The tower's construction finished on 31 March 1889.", page())
    assert out["status"] == SourceCheckStatus.CONFIRMED, out


def test_a_paraphrase_that_changes_the_subject_is_left_unverified():
    """Documented limit, in the safe direction: the page says "Work on the foundations
    began in January 1887" and the claim says "Construction of the Eiffel Tower began in
    January 1887". The cue and the date match, the subject noun does not, so this layer
    leaves it NOT_CHECKED for the curator instead of guessing they are the same event."""
    out = outcome("Construction of the Eiffel Tower began in January 1887.", page())
    assert out["status"] == SourceCheckStatus.NOT_CHECKED, out
    assert out["role"] and out["role"]["verdict"] == "unknown", out["role"]
    # the page's own words do bind the completion claim
    assert outcome("The Eiffel Tower was completed on 31 March 1889.", page())["status"] == SourceCheckStatus.CONFIRMED


# ------------------------------------------------------------ negative: not evidence


def test_a_page_that_never_gives_the_year_is_not_support():
    body = "The Eiffel Tower stands on the Champ de Mars. It was built as the entrance arch to a World's Fair."
    out = outcome("The Eiffel Tower was completed on 31 March 1889.", page(text=body))
    assert out["status"] != SourceCheckStatus.CONFIRMED, out
    assert "1889" in out["missing"], out


def test_a_start_year_cannot_evidence_a_completion_claim():
    """The dense sentence: both years, both verbs, one clause each. Matching the claim
    against the page as a whole used to accept this; the date now has to belong to the
    event the claim names."""
    out = outcome("The Eiffel Tower was completed in 1887.", page())
    assert out["status"] != SourceCheckStatus.CONFIRMED, out
    assert out["status"] == SourceCheckStatus.MISMATCH, out
    assert any("1887" in str(m) for m in out["missing"]), out


async def test_a_refuted_premise_is_answered_with_the_correction_not_a_refusal(net, fake_openai):
    """Live (eiffel premise run): the opened pages gave 31 March 1889, the curator refuted
    1887, and the run still said "Couldn't verify that one." A refutation backed by a page
    we opened is a finding, so the answer has to say the premise is wrong."""
    from tests.test_ledger_adjudication import ask, curator, make_curator

    history = "https://www.toureiffel.paris/en/the-monument/history"
    net.confirm(history, tier=SourceTier.PRIMARY_OFFICIAL, excerpt="completed on 31 March 1889")
    scripts = {
        "chatgpt": {"answer": "KEY CLAIMS\n1. The Eiffel Tower was completed in 1887.", "citations": [{"url": history, "title": "Eiffel Tower history 1887 completion"}]},
        "gemini": {"answer": "KEY CLAIMS\n1. The Eiffel Tower was completed on 31 March 1889.", "citations": [{"url": history, "title": "Eiffel Tower history completed 31 March 1889"}]},
        "search": {"answer": "results", "citations": [{"url": history, "title": "Eiffel Tower history completed 31 March 1889"}]},
    }

    def judge(text: str) -> dict:
        if "1887" in text:
            return {"verdict": "refuted", "confidence": "high", "reasoning": "the official history gives 31 March 1889", "strong_evidence": [history], "problems": []}
        return {"verdict": "supported", "confidence": "high", "reasoning": "the official history states it", "strong_evidence": [history], "problems": []}

    fn = curator(
        decide=judge,
        answer="The Eiffel Tower was completed on 31 March 1889, not 1887.",
        confidence="high",
    )
    job = await ask(base_settings(), scripts, question="Is it true that the Eiffel Tower was completed in 1887?", verifier=make_curator(fake_openai.script(fn)), max_rounds=2)

    wrong = [c for c in job.claims if "1887" in c.claim]
    assert wrong and wrong[0].status == ClaimStatus.REFUTED, [(c.claim, c.status) for c in job.claims]
    assert job.final.answer.startswith("Nah, that doesn't work like that."), job.final.answer
    assert "1889" in job.final.answer, job.final.answer
    assert not job.final.answer.lower().startswith("couldn't verify"), job.final.answer


def test_an_incidental_year_in_an_unrelated_page_is_not_support():
    body = "Our 1889 vintage port is still available. The cellar holds casks from 1889 and the tasting room opens at noon."
    out = outcome("The Eiffel Tower was completed on 31 March 1889.", page(text=body, title="Cellar door vintage port", url="https://port.example/vintages"))
    assert out["status"] != SourceCheckStatus.CONFIRMED, out


def test_a_different_subject_with_the_same_date_is_not_support():
    body = "Blackpool Tower is a tower in Blackpool, England. Its construction was completed on 31 March 1889 after six years of work."
    out = outcome("The Eiffel Tower was completed on 31 March 1889.", page(text=body, title="Blackpool Tower history", url="https://blackpool.example/tower"))
    assert out["status"] != SourceCheckStatus.CONFIRMED, out
    assert any("subject" in m for m in out["missing"]), out


def test_an_unopened_page_confirms_nothing(monkeypatch):
    """A source that was mentioned but never read cannot become evidence for a claim."""
    closed = FetchedPage(url="https://www.toureiffel.paris/en/the-monument/history", ok=False, status=0, text="", error="connection reset")
    out = _run_gather(monkeypatch, [closed], [("clm_a", "The Eiffel Tower was completed on 31 March 1889.")])
    assert out, "the unread page is still recorded, as unread"
    assert all(e.check_status != SourceCheckStatus.CONFIRMED for e in out), [(e.check_status, e.claim_id) for e in out]
    assert all(not (e.claim_id == "clm_a" and e.check_status == SourceCheckStatus.CONFIRMED) for e in out)


# ------------------------------------------------------ a blocked source says so

BLOCKED_BODY = "Please enable cookies. Sorry, you have been blocked. You are unable to access toureiffel.paris."


class _Response:
    def __init__(self, status: int, body: str, url: str = ""):
        self.status_code = status
        self.text = body
        self.content = body.encode()
        self.headers = {"content-type": "text/html"}
        self.url = url


def _stub_http(monkeypatch, status: int, body: str):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            return _Response(status, body, url)

    monkeypatch.setattr(sources.httpx, "AsyncClient", _Client)


def _gather(monkeypatch, url: str, browser_fetch=None):
    import asyncio

    links = [{"href": url, "title": "133 years and 330 metres", "claim_id": None, "claim_text": None}]
    return asyncio.run(gather_from_links("j", links, browser_fetch=browser_fetch, max_pages=1))


def test_a_security_gate_is_recorded_as_blocked_not_unreachable(monkeypatch):
    _stub_http(monkeypatch, 403, BLOCKED_BODY)
    out = _gather(monkeypatch, "https://www.toureiffel.paris/en/the-monument/history")
    assert len(out) == 1
    assert out[0].check_status == SourceCheckStatus.BLOCKED, (out[0].check_status, out[0].check_notes)
    assert out[0].claim_id is None, "a page we could not read is never evidence for a claim"


def test_a_failed_browser_re_read_is_written_into_the_record(monkeypatch):
    _stub_http(monkeypatch, 403, BLOCKED_BODY)

    async def nothing(url, **kwargs):
        return None

    out = _gather(monkeypatch, "https://www.toureiffel.paris/en/the-monument/history", browser_fetch=nothing)
    assert "browser re-read gave no readable text" in (out[0].check_notes or ""), out[0].check_notes


def test_a_refused_browser_re_read_is_written_into_the_record(monkeypatch):
    """The live driver reads provider sites only, so a source domain has no second path.
    That must show up in the audit trail, not vanish."""
    _stub_http(monkeypatch, 403, BLOCKED_BODY)

    async def refused(url, **kwargs):
        raise RuntimeError("url is not a provider site")

    out = _gather(monkeypatch, "https://www.toureiffel.paris/en/the-monument/history", browser_fetch=refused)
    assert "browser fetch failed: RuntimeError" in (out[0].check_notes or ""), out[0].check_notes


def _run_gather(monkeypatch, rows, claims):
    """Drive gather_from_links with a stub transport that returns the given page."""
    async def fake(url, **kwargs):
        return rows[0] if rows else FetchedPage(url=url, ok=False)

    monkeypatch.setattr(sources, "fetch_page", fake)
    import asyncio

    links = [{"href": rows[0].url, "title": rows[0].title, "claim_id": None, "claim_text": None}]
    return asyncio.run(gather_from_links("j", links, attribute_to=claims))


def test_a_body_match_is_filed_only_under_the_claim_it_actually_supports(monkeypatch):
    """attribute_to spreads one opened page across the claims its text really contains --
    and no further."""
    pg = page()
    supported = ("clm_yes", "The Eiffel Tower was completed on 31 March 1889.")
    unsupported = ("clm_no", "The Eiffel Tower was rebuilt in 1954.")
    out = _run_gather(monkeypatch, [pg], [supported, unsupported])
    attached = {e.claim_id for e in out if e.check_status == SourceCheckStatus.CONFIRMED}
    assert "clm_yes" in attached, [
        (e.claim_id, e.check_status, e.check_notes) for e in out
    ]
    assert "clm_no" not in attached, "a page that never mentions 1954 must not back that claim"
    row = next(e for e in out if e.claim_id == "clm_yes")
    assert "1889" in (row.verbatim_excerpt or ""), "the passage that carries the fact is the one recorded"
    assert "matched to this claim" in (row.check_notes or ""), row.check_notes


# ------------------------------------------------ refutation is a finding, not a gap


def _report(verdicts, confidence: Confidence) -> VerifierReport:
    return VerifierReport(job_id="j", round=1, verdicts=verdicts, confidence=confidence, answer="", why="")


def test_a_refutation_backed_by_an_opened_page_reads_as_false():
    from backend.models import ClaimVerdict

    verdict = ClaimVerdict(
        claim_id="clm_p", claim="The Eiffel Tower was completed in 1887.", verdict=ClaimStatus.REFUTED,
        confidence=Confidence.HIGH, reasoning="the official history gives 31 March 1889",
        strong_evidence=["https://www.toureiffel.paris/en/the-monument/history"],
    )
    # the ledger caps answer confidence when nothing was "supported"; that must not turn
    # a documented refutation into "I couldn't verify it"
    assert truth_state(_report([verdict], Confidence.LOW)) == "FALSE"


def test_a_refutation_with_no_opened_page_still_admits_it_does_not_know():
    """Never round uncertainty up: a refutation citing nothing is not a finding."""
    from backend.models import ClaimVerdict

    verdict = ClaimVerdict(
        claim_id="clm_p", claim="The Eiffel Tower was completed in 1887.", verdict=ClaimStatus.REFUTED,
        confidence=Confidence.HIGH, reasoning="I doubt it", strong_evidence=[],
    )
    assert truth_state(_report([verdict], Confidence.LOW)) == "UNVERIFIED"


def test_a_refuted_figure_the_user_asked_about_reads_as_false_not_partly():
    from backend.models import ClaimVerdict

    refuted = ClaimVerdict(
        claim_id="a", claim="The Eiffel Tower was completed in 1887.", verdict=ClaimStatus.REFUTED,
        confidence=Confidence.HIGH, reasoning="the official history gives 31 March 1889",
        strong_evidence=["https://www.toureiffel.paris/en/the-monument/history"],
    )
    supported = ClaimVerdict(
        claim_id="b", claim="The Eiffel Tower was completed on 31 March 1889.", verdict=ClaimStatus.SUPPORTED,
        confidence=Confidence.HIGH, reasoning="the official history states it",
        strong_evidence=["https://www.toureiffel.paris/en/the-monument/history"],
    )
    report = _report([refuted, supported], Confidence.HIGH)
    assert truth_state(report, "Is it true that the Eiffel Tower was completed in 1887?") == "FALSE"
    # refuting something else the model said stays a partial correction
    assert truth_state(report, "Was the Eiffel Tower built in iron?") == "PARTLY"


# ------------------------------------------------------------------------ the export


def test_the_export_does_not_print_the_same_word_twice(tmp_path):
    """Confidence.NONE is spelled "insufficient_evidence", the same word as the status."""
    from backend.storage.db import Store
    from tests.conftest import base_settings
    from tests.test_history_export import finished_job

    job = finished_job()
    job.claims[0].status = ClaimStatus.INSUFFICIENT_EVIDENCE
    job.claims[0].confidence = Confidence.NONE
    job.claims.append(
        Claim(id="clm_b", job_id=job.id, claim="Warfarin is a blood thinner.", kind="fact",
              provider_sources=["gemini"], status=ClaimStatus.SUPPORTED, confidence=Confidence.HIGH)
    )
    settings = base_settings(storage={"db_path": str(tmp_path / "x.db")})
    store = Store(settings)
    store.save_job(job)
    md = job_markdown(store.job_snapshot(job.id))

    row = next(line for line in md.splitlines() if "Ibuprofen raises" in line)
    assert row.count("insufficient") == 1, row
    assert "supported / high" in md, "a real confidence pair still shows"
