"""Unit tests for the inter-rater agreement metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.rating_agreement import (
    cohen_kappa,
    disagreement_counts,
    krippendorff_alpha,
    label_distribution,
    report,
)


def test_cohen_kappa_perfect_agreement() -> None:
    a = {"t1": "correct", "t2": "false_negative", "t3": "correct"}
    b = {"t1": "correct", "t2": "false_negative", "t3": "correct"}
    kappa, n = cohen_kappa(a, b)
    assert n == 3
    assert kappa == pytest.approx(1.0)


def test_cohen_kappa_disagreement_with_some_overlap() -> None:
    """Kappa is negative when observed agreement is below chance."""
    a = {"t1": "correct", "t2": "correct",        "t3": "false_negative", "t4": "false_negative"}
    b = {"t1": "correct", "t2": "false_negative", "t3": "correct",        "t4": "correct"}
    kappa, n = cohen_kappa(a, b)
    assert n == 4
    assert kappa < 0


def test_cohen_kappa_no_overlap_returns_nan() -> None:
    a = {"t1": "correct"}
    b = {"t2": "correct"}
    kappa, n = cohen_kappa(a, b)
    assert n == 0
    assert math.isnan(kappa)


def test_krippendorff_alpha_perfect_agreement() -> None:
    a = {"t1": "correct", "t2": "false_negative"}
    b = {"t1": "correct", "t2": "false_negative"}
    alpha, n = krippendorff_alpha([a, b])
    assert n == 2
    assert alpha == pytest.approx(1.0)


def test_krippendorff_alpha_skips_units_rated_by_only_one() -> None:
    a = {"t1": "correct", "t2": "correct"}
    b = {"t1": "correct"}
    alpha, n = krippendorff_alpha([a, b])
    assert n == 1


def test_label_distribution_counts() -> None:
    m = {"t1": "correct", "t2": "correct", "t3": "false_positive"}
    counts = label_distribution(m)
    assert counts["correct"] == 2
    assert counts["false_positive"] == 1
    assert counts["false_negative"] == 0
    assert counts["unsure"] == 0


def test_disagreement_counts_only_returns_disagreed_tasks() -> None:
    a = {"t1": "correct", "t2": "correct"}
    b = {"t1": "correct", "t2": "false_negative"}
    out = disagreement_counts([a, b])
    assert out == {"t2": 2}


def test_report_structure_with_two_raters(tmp_path: Path) -> None:
    p1 = tmp_path / "alice.jsonl"
    p2 = tmp_path / "bob.jsonl"
    p1.write_text(json.dumps({"task_id": "t1", "human_rating": "correct"}) + "\n")
    p2.write_text(json.dumps({"task_id": "t1", "human_rating": "correct"}) + "\n")
    out = report([p1, p2])
    assert out["raters"] == ["alice", "bob"]
    assert "alice__bob" in out["pair_cohen_kappa"]
    assert out["krippendorff_alpha"] == pytest.approx(1.0)
