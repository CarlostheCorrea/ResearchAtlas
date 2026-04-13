"""
Tests for the competitive Analyze Paper summary selector.
These tests avoid LLM calls and verify the deterministic best-answer-wins policy.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DB_PATH", "/tmp/test_summary_competition.db")
os.environ.setdefault("VECTORSTORE_DIR", "/tmp/test_vectorstore_summary_competition")

from app.agents.summary_agent import build_summary_competition, select_summary_candidate


def _metric(status: str, score: float) -> dict:
    return {"status": status, "score": score, "note": ""}


def _candidate(
    candidate_id: str,
    order: int,
    *,
    faithfulness: str = "pass",
    faithfulness_score: float = 0.8,
    specificity_score: float = 0.8,
    completeness_score: float = 0.8,
    section_accuracy_score: float = 0.8,
) -> dict:
    return {
        "id": candidate_id,
        "name": f"{candidate_id.title()} Agent",
        "order": order,
        "summary": {"overview": candidate_id},
        "evaluation": {
            "overall_status": "passed" if faithfulness == "pass" else "needs_review",
            "faithfulness": _metric(faithfulness, faithfulness_score),
            "specificity": _metric("pass", specificity_score),
            "completeness": _metric("pass", completeness_score),
            "section_accuracy": _metric("pass", section_accuracy_score),
        },
    }


def test_faithfulness_pass_beats_higher_scoring_failure():
    faithful = _candidate(
        "conservative",
        0,
        faithfulness="pass",
        faithfulness_score=0.7,
        specificity_score=0.5,
        completeness_score=0.5,
        section_accuracy_score=0.5,
    )
    unfaithful = _candidate(
        "technical",
        1,
        faithfulness="fail",
        faithfulness_score=1.0,
        specificity_score=1.0,
        completeness_score=1.0,
        section_accuracy_score=1.0,
    )

    assert select_summary_candidate([unfaithful, faithful])["id"] == "conservative"


def test_highest_average_score_wins_among_faithful_candidates():
    lower = _candidate("conservative", 0, specificity_score=0.7, completeness_score=0.7)
    higher = _candidate("technical", 1, specificity_score=0.9, completeness_score=0.9)

    assert select_summary_candidate([lower, higher])["id"] == "technical"


def test_tie_breaker_prefers_lower_order_candidate():
    first = _candidate("conservative", 0)
    second = _candidate("teaching", 2)

    assert select_summary_candidate([second, first])["id"] == "conservative"


def test_competition_metadata_marks_winner():
    candidates = [
        _candidate("conservative", 0, specificity_score=0.7),
        _candidate("technical", 1, specificity_score=0.9),
    ]
    winner = select_summary_candidate(candidates)
    competition = build_summary_competition(
        candidates=candidates,
        winner=winner,
        repair_status="not_needed",
    )

    assert competition["winner_id"] == "technical"
    assert competition["mode"] == "competitive_best_answer_wins"
    assert sum(1 for card in competition["scorecards"] if card["selected"]) == 1
