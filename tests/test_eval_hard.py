"""The eval's scoring logic (scripts/eval_hard.py): it must catch an invented figure and reward an honest don't-know."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_hard", ROOT / "scripts" / "eval_hard.py")
eval_hard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_hard)


def snapshot(answer, *, evidence=(), sources=(), claims=()):
    return {"live": {"final": {"answer": answer, "why": "", "sources": [{"url": u} for u in sources], "confidence_label": "Moderate confidence", "caveats": [], "rounds_run": 1,
                               "providers_used": ["chatgpt"], "providers_failed": []},
                     "evidence": list(evidence), "claims": [{"claim": c} for c in claims], "reports": [{"verdicts": [], "verifier_model": "m"}]}}


def ev(url, domain, text, status="confirmed", polarity="support"):
    return {"url": url, "domain": domain, "snippet": text, "verbatim_excerpt": text, "title": "", "check_status": status, "polarity": polarity}


def q(**kw):
    base = {"id": "t", "category": "uk-law", "kind": "answer", "question": "What is the minimum age for consent?", "primary_domains": ["legislation.gov.uk"], "expect": {"must_match": [r"\b13\b"]}}
    base.update(kw)
    return base


def test_an_answer_backed_by_a_cited_primary_page_passes():
    snap = snapshot("It's 13 in the UK \u2696\uFE0F", evidence=[ev("https://www.legislation.gov.uk/ukpga/2018/12/section/9", "www.legislation.gov.uk", "section 9 child aged 13 or over")],
                    sources=["https://www.legislation.gov.uk/ukpga/2018/12/section/9"])
    rec = eval_hard.analyse(q(), snap, 12.0)
    assert rec["judgement"]["verdict"] == "PASS", rec["judgement"]
    assert rec["primary_opened"] and rec["primary_cited"] and rec["unsupported_figures"] == []


def test_a_figure_no_opened_page_contains_is_caught_as_unsupported():
    """The key failure: a confident number that nothing we opened says."""
    snap = snapshot("It's 13, and the fine is up to 4,500 pounds.", evidence=[ev("https://www.legislation.gov.uk/x", "legislation.gov.uk", "child aged 13 or over")],
                    sources=["https://www.legislation.gov.uk/x"])
    rec = eval_hard.analyse(q(), snap, 5)
    assert "4500" in rec["unsupported_figures"] and "13" not in rec["unsupported_figures"]
    assert rec["judgement"]["verdict"] in {"WARN", "FAIL"} and "figures" in " ".join(rec["judgement"]["reasons"])


def test_a_figure_from_the_question_is_not_flagged():
    assert eval_hard.unsupported_figures("The Online Safety Act 2023 has 241 sections.", "the act has 241 sections", "Under the 2023 Act") == []
    assert eval_hard.unsupported_figures("Section 9 applies.", "see section 9 of the act") == []


def test_an_unanswerable_question_needs_an_honest_idk_with_a_reason():
    uq = q(kind="unanswerable", expect={"must_match": ["don't know"]})
    ok = eval_hard.analyse(uq, snapshot("I don't know. Parliament hasn't passed or published those amendments."), 3)
    assert ok["judgement"]["verdict"] == "PASS"
    invented = eval_hard.analyse(uq, snapshot("Parliament will amend section 4 to add a new notice period \U0001F44D"), 3)
    assert invented["judgement"]["verdict"] == "FAIL"
    bare = eval_hard.analyse(uq, snapshot("I don't know."), 3)
    assert bare["judgement"]["verdict"] == "WARN" and "without a reason" in " ".join(bare["judgement"]["reasons"])


def test_a_false_premise_answer_that_asserts_the_premise_fails():
    fq = q(kind="false_premise", expect={"must_match": [r"\b(no|not)\b"], "must_not_match": [r"section \d+ bans"]})
    bad = eval_hard.analyse(fq, snapshot("Section 8 bans VPNs in the UK."), 2)
    assert bad["judgement"]["verdict"] == "FAIL"
    good = eval_hard.analyse(fq, snapshot("No - the Act doesn't ban VPNs. \u2696\uFE0F"), 2)
    assert good["judgement"]["verdict"] != "FAIL"


def test_a_wrong_voice_is_reported():
    snap = snapshot("It is important to note that the ledger score for round 2 is 13.", evidence=[ev("https://www.legislation.gov.uk/x", "legislation.gov.uk", "13 round 2")], sources=["https://www.legislation.gov.uk/x"])
    rec = eval_hard.analyse(q(), snap, 1)
    assert not rec["voice"]["ok"] and any("voice" in r for r in rec["judgement"]["reasons"])


def test_the_question_file_is_valid_and_covers_every_trap_type_and_category():
    data = yaml.safe_load((ROOT / "scripts" / "eval_questions.yaml").read_text(encoding="utf-8"))["questions"]
    assert len({d["id"] for d in data}) == len(data), "ids must be unique"
    assert {d["kind"] for d in data} == {"answer", "false_premise", "myth", "unanswerable"}
    assert {d["category"] for d in data} == {"uk-law", "ua-law", "university", "research", "trivial"}
    for d in data:
        assert d["question"].strip() and d.get("expect") is not None and d.get("notes")


def test_markdown_report_has_a_table_row_per_question():
    rec = eval_hard.analyse(q(), snapshot("It's 13 \u2696\uFE0F", evidence=[ev("https://www.legislation.gov.uk/x", "legislation.gov.uk", "13")], sources=["https://www.legislation.gov.uk/x"]), 4)
    md = eval_hard.render_markdown("20261003_000000", [rec], {"providers": ["chatgpt"], "verifier": "ready", "mode": "STANDARD", "max_rounds": 2})
    assert "| t | answer | **PASS** |" in md and "Totals: PASS 1" in md

def test_temp_config_never_contains_the_api_key(tmp_path, monkeypatch):
    fake = "sk-or-v1-" + "0123456789abcdef" * 4
    src = tmp_path / "settings.yaml"
    src.write_text(yaml.safe_dump({"providers": {"chatgpt": {"enabled": True}}, "verifier": {"api_key": fake, "model": "m"}, "analysis": {"api_key": fake}}), encoding="utf-8")
    real_root = eval_hard.ROOT
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(eval_hard, "ROOT", tmp_path)
    out = tmp_path / "cfg.yaml"
    eval_hard.trimmed_config([], out)
    assert fake not in out.read_text(encoding="utf-8")
    assert eval_hard.SECRET_FOR_CHILD["key"] == fake, "the key is passed to the child through the environment instead"
    eval_hard.SECRET_FOR_CHILD["key"] = ""
    assert real_root.exists()

def test_a_computed_level_zero_answer_is_not_flagged_for_having_no_pages():
    q = {"id": "t", "category": "trivial", "kind": "answer", "question": "What is 2 + 2?", "expect": {"must_match": ["\\b4\\b"]}}
    snap = {"final": {"answer": "4"}, "browser_sessions": 0, "verifier_calls": 0, "stop_reason": "question answered at level 0; no investigation earned", "evidence": []}
    rec = eval_hard.analyse(q, snap, 1.0)
    assert rec["unsupported_figures"] == [] and rec["judgement"]["verdict"] == "PASS" and rec["browser_sessions"] == 0