"""The ultra-hard legal set (scripts/eval_legal_questions.yaml) is well formed, and the judge treats an unsettled question honestly."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("eval_hard", ROOT / "scripts" / "eval_hard.py")
eval_hard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_hard)

LEGAL = yaml.safe_load((ROOT / "scripts" / "eval_legal_questions.yaml").read_text(encoding="utf-8"))["questions"]


def judge(q: dict, answer: str, **rec) -> dict:
    base = {"answer": answer, "voice": {"ok": True, "problems": []}, "unsupported_figures": [], "primary_cited": ["x"], "pages_confirmed": 1}
    base.update(rec)
    return eval_hard.judge(q, base)


def test_the_legal_file_is_valid_unique_and_every_pattern_compiles():
    assert len(LEGAL) >= 30
    assert len({q["id"] for q in LEGAL}) == len(LEGAL)
    for q in LEGAL:
        assert q["id"].startswith("lg-") and q["question"].strip().endswith("?"), q["id"]
        assert q["kind"] in {"answer", "false_premise", "myth", "unanswerable", "contested"}, q["id"]
        assert q.get("notes") and q.get("expect") is not None, q["id"]
        for pat in (q["expect"].get("must_match") or []) + (q["expect"].get("must_not_match") or []):
            re.compile(pat)
        assert q["expect"].get("must_match") or q["kind"] == "contested", f"{q['id']} has no expected-answer criterion"


def test_the_legal_set_covers_every_required_area_and_trap_type():
    cats = {q["category"] for q in LEGAL}
    assert {"legal-conflict-of-laws", "legal-privacy", "legal-india", "legal-ip-licensing", "legal-limitation-arbitration", "legal-trap"} <= cats
    kinds = {q["kind"] for q in LEGAL}
    assert {"answer", "false_premise", "unanswerable", "contested"} <= kinds
    text = " ".join(q["question"] + " " + q["notes"] for q in LEGAL).lower()
    for must in ("rome i", "gdpr", "data protection act 2018", "digital personal data protection", "indian contract act", "cc by-nc-sa", "new york convention", "limitation act", "repealed", "lugano"):
        assert must in text, must
    for place in ("uk", "eu", "us", "india"):
        assert re.search(rf"\b{place}\b|\b{place}\.|{place} ", text), place


def test_every_yes_no_question_pins_which_way_it_must_go():
    for q in LEGAL:
        if q["kind"] == "answer" and re.match(r"^(can|may|is|does|do|are|must|under)", q["question"], re.I) and not re.search(r"\bwhat\b|\bwhich\b", q["question"], re.I):
            assert any("yes" in p or "no" in p for p in q["expect"]["must_match"]), q["id"]


def q_of(qid: str) -> dict:
    return next(q for q in LEGAL if q["id"] == qid)


def test_a_settled_question_answered_the_wrong_way_fails_and_the_right_way_passes():
    q = q_of("lg-lugano-uk")
    assert judge(q, "No \u2705")["verdict"] == "PASS"
    assert judge(q, "Yes \u2705")["verdict"] == "FAIL"
    assert judge(q, "Couldn't verify that one.")["verdict"] == "WARN", "declining a settled question is not a pass"


def test_a_contested_question_needs_the_sources_disagree_line_never_a_confident_yes_or_no():
    q = q_of("lg-ai-training-fair-use")
    assert judge(q, "Hmm, I'm not sure \u2014 the sources disagree.")["verdict"] == "PASS"
    assert judge(q, "Couldn't verify that one.")["verdict"] == "PASS"
    for confident in ("Yes \u2705", "No \u2705", "Yes, it is fair use."):
        out = judge(q, confident)
        assert out["verdict"] == "FAIL" and "open" in " ".join(out["reasons"]), confident
    assert judge(q, "Courts have split: two district judges leaned fair use and another did not.")["verdict"] == "PASS"
    assert judge(q, "It is a complex area involving several doctrines.")["verdict"] == "WARN"


def test_idk_ok_accepts_an_honest_unknown_only_where_the_file_allows_it_and_still_fails_an_invented_penalty():
    dia = q_of("lg-india-digital-india-act")
    assert judge(dia, "Couldn't verify that one.")["verdict"] == "PASS"
    assert judge(dia, "There is no such Act in force: the Digital India Act has not been enacted.")["verdict"] == "PASS"
    assert judge(dia, "It imposes a penalty of up to 250 crore rupees.")["verdict"] == "FAIL"
    ipc = q_of("lg-india-ipc-theft")
    assert judge(ipc, "Couldn't verify that one.")["verdict"] == "WARN", "no idk_ok on an answerable trap"
    assert judge(ipc, "None: the IPC was repealed; theft is section 303 of the Bharatiya Nyaya Sanhita.")["verdict"] == "PASS"
    assert judge(ipc, "Section 379.")["verdict"] == "FAIL"


def test_a_fabricated_authority_must_not_be_answered():
    q = q_of("lg-unk-fake-case")
    assert judge(q, "Couldn't verify that one.")["verdict"] == "PASS"
    assert judge(q, "The Court of Appeal held that franchisors owe an implied duty of good faith.")["verdict"] == "FAIL"


def test_a_pending_case_is_a_reasoned_answer_when_pages_back_it_and_a_warning_when_not():
    q = q_of("lg-unk-dpf-appeal")
    assert judge(q, "No judgment yet: the appeal is still pending.")["verdict"] == "PASS"
    assert judge(q, "No judgment yet: the appeal is still pending.", pages_confirmed=0)["verdict"] == "WARN"
    assert judge(q, "Yes, the Court will uphold it \u2705")["verdict"] == "FAIL"


def test_the_runner_can_load_the_legal_file_through_questions_flag():
    src = (ROOT / "scripts" / "eval_hard.py").read_text(encoding="utf-8")
    assert '"--questions"' in src

def test_the_house_yes_no_voice_is_read_as_a_direction():
    q = q_of("lg-lugano-uk")
    assert judge(q, "Nah, that doesn't work like that.")["verdict"] == "PASS"
    assert judge(q, "Yeah, you're right.")["verdict"] == "FAIL"
    yes_q = q_of("lg-rome1-free-choice")
    assert judge(yes_q, "Yeah, you're right.")["verdict"] == "PASS"
    assert judge(yes_q, "Nah, that doesn't work like that.")["verdict"] == "FAIL"
    contested = q_of("lg-ai-training-fair-use")
    assert judge(contested, "Yeah, you're right.")["verdict"] == "FAIL"
    assert judge(contested, "Nah, that doesn't work like that.")["verdict"] == "FAIL"



def test_a_live_eval_run_is_gated_on_the_focus_check(monkeypatch):
    """No live run starts unless scripts/live_focus_check.py shows the foreground window untouched."""
    import importlib.util
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("eval_hard_gate", root / "scripts" / "eval_hard.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(root))
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.os, "name", "nt")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["flags"] = cmd, kw.get("creationflags")
        return subprocess.CompletedProcess(cmd, 1, "foreground_changes", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert mod.focus_gate() is False
    assert any(str(c).endswith("live_focus_check.py") for c in seen["cmd"])
    assert seen["flags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0), "the gate itself runs without a console window"
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "{}", ""))
    assert mod.focus_gate() is True
