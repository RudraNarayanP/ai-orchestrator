"""Every configured provider is in the pool; a wall seen earlier lowers a site's rank, never removes it."""

from __future__ import annotations

from backend.models import QuestionAnalysis
from backend.research import router

ALL = ["chatgpt", "gemini", "google_ai", "copilot", "meta_ai", "le_chat", "pi", "qwen", "deepseek", "search"]


def test_nothing_is_hard_coded_out_of_the_pool():
    a = QuestionAnalysis(question="q")
    seen = set()
    health: dict[str, str] = {}
    for _ in range(len(ALL)):
        pick = router.select_secondaries(a, ALL, exclude=sorted(seen), count=3, health=health)
        seen.update(pick)
    assert seen >= set(ALL) - {"search"}, "every chat AI must be reachable as a researcher"
    assert "search" not in seen, "Google Search is the evidence transport, not an AI researcher"


def test_a_site_walled_last_run_goes_last_but_is_still_tried_again():
    a = QuestionAnalysis(question="q")
    prior = {"gemini": "logged_out", "qwen": "broken"}
    assert router.select_primary(a, ALL, {}, prior) != "gemini"
    first_three = router.select_secondaries(a, ALL, exclude=["chatgpt"], count=3, health={}, prior=prior)
    assert "gemini" not in first_three and "qwen" not in first_three
    probing = router.select_secondaries(a, ALL, exclude=["chatgpt"], count=3, health={}, prior=prior, reprobe=True)
    assert set(probing) & {"gemini", "qwen"}, "one escalation slot re-probes a site that was walled before"
    only_blocked = router.select_secondaries(a, ["chatgpt", "gemini"], exclude=["chatgpt"], count=2, health={}, prior=prior)
    assert only_blocked == ["gemini"], "with nobody else left the walled site is still attempted"


def test_a_site_that_hit_a_wall_in_this_job_is_skipped_for_this_job():
    a = QuestionAnalysis(question="q")
    pick = router.select_secondaries(a, ALL, exclude=["chatgpt"], count=3, health={"gemini": "logged_out"})
    assert "gemini" not in pick

def test_google_ai_targets_ai_mode_with_udm_50_and_reuses_the_dedicated_window():
    import yaml
    from pathlib import Path

    raw = yaml.safe_load(Path("config/settings.example.yaml").read_text(encoding="utf-8"))
    cfg = raw["providers"]["google_ai"]
    assert "google.com/search" in cfg["url"] and "udm=50" in cfg["url"]
    assert "udm=50" in cfg["new_chat_url"], "each new research opens AI Mode itself, not plain Search"
    assert raw["browser"].get("reuse_tabs", True) is True