"""Unit tests for the rating-runner builder.

These tests do not call the LLM; they exercise the form-building
logic that produces the rater-fillable JSONL.
"""

from __future__ import annotations

from eval.rating_runner import build_records


def test_build_records_includes_each_corpus_task() -> None:
    corpus = [
        {"id": "t1", "task": "draft a thing", "category": "communication", "expected_stage": "fts"},
        {"id": "t2", "task": "another task",  "category": "git",            "expected_stage": "miss"},
    ]
    rows = build_records(corpus, {})
    assert {r["task_id"] for r in rows} == {"t1", "t2"}
    assert rows[0]["observed_stage"] == "miss"
    assert rows[0]["observed_node"] is None
    assert rows[0]["llm_rating"] is None
    assert rows[0]["human_rating"] is None


def test_build_records_uses_skg_run_when_present() -> None:
    corpus = [
        {"id": "t1", "task": "draft a thing", "category": "communication", "expected_stage": "fts"},
    ]
    skg_run = {
        "t1": {"stage": "fts", "node_id": "reviewer-ping-draft"},
    }
    rows = build_records(corpus, skg_run)
    assert rows[0]["observed_stage"] == "fts"
    assert rows[0]["observed_node"] == "reviewer-ping-draft"


def test_build_records_carries_through_corpus_metadata() -> None:
    corpus = [
        {"id": "t1", "task": "draft", "category": "communication", "expected_stage": "fts"},
    ]
    rows = build_records(corpus, {})
    assert rows[0]["category"] == "communication"
    assert rows[0]["expected_stage"] == "fts"
    assert "human_rating" in rows[0]
    assert rows[0]["human_rating"] is None
