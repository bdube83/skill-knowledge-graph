"""Tests for eval.corpus_builder_large.

Covers:
  - Deterministic generation under a fixed seed.
  - Task count and category distribution.
  - Per-task context size floor (proxy for token volume).
"""

from __future__ import annotations

import json

from eval.corpus_builder_large import (
    CATEGORIES,
    TASKS_PER_CATEGORY,
    generate_large_corpus,
)


def test_generation_is_deterministic_under_fixed_seed() -> None:
    a = generate_large_corpus(seed=42)
    b = generate_large_corpus(seed=42)
    assert json.dumps(a) == json.dumps(b)


def test_corpus_has_200_tasks_with_40_per_category() -> None:
    corpus = generate_large_corpus(seed=42)
    assert len(corpus) == 200

    counts: dict[str, int] = {}
    for t in corpus:
        counts[t["category"]] = counts.get(t["category"], 0) + 1

    assert set(counts) == set(CATEGORIES)
    for cat in CATEGORIES:
        assert counts[cat] == TASKS_PER_CATEGORY


def test_each_context_body_is_at_least_500_chars() -> None:
    corpus = generate_large_corpus(seed=42)
    for t in corpus:
        body = json.dumps(t["context"])
        assert len(body) >= 500, f"task {t['id']} context too small: {len(body)} chars"


def test_task_schema_matches_corpus_jsonl_shape() -> None:
    corpus = generate_large_corpus(seed=42)
    expected_keys = {"id", "task", "category", "context", "expected_stage"}
    for t in corpus:
        assert set(t.keys()) == expected_keys
        assert t["id"].startswith("t")
        assert isinstance(t["task"], str) and t["task"]
        assert isinstance(t["context"], dict)
