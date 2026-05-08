"""Tests for the Baselines B, C, D aggregate runner.

These tests verify that `eval.baseline_bcd_runner` produces non empty
aggregate reports for each baseline against the project's 200 task
corpus, and that the reports satisfy basic invariants the paper's
tables rely on.

The tests run the actual runners over the actual corpus. Baselines B
and C are pure Python and run instantly. Baseline D loads three
.wasm artifacts and runs them through Wasmtime. Tests that touch
Baseline D are skipped if any artifact is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.baseline_bcd_runner import (
    NODE_WASM,
    load_corpus,
    main,
    run_baseline_b,
    run_baseline_c,
    run_baseline_d,
)


REPO_ROOT  = Path(__file__).resolve().parent.parent
CORPUS     = REPO_ROOT / "eval" / "corpus.jsonl"


def _require_corpus() -> None:
    if not CORPUS.exists():
        pytest.skip(f"Corpus not present: {CORPUS}")


def _require_wasm_artifacts() -> None:
    missing = [str(p) for p in NODE_WASM.values() if not p.exists()]
    if missing:
        pytest.skip(f"Missing WASM artifacts: {missing}")


def _check_report_shape(report: dict, expected_model: str) -> None:
    """Common invariants every B, C, D report must satisfy."""
    expected_fields = {
        "task_count",
        "input_tokens_total",
        "output_tokens_total",
        "input_tokens_mean",
        "output_tokens_mean",
        "latency_p50_ms",
        "latency_p95_ms",
        "cost_usd",
        "hit_count",
        "hit_rate",
        "model",
    }
    assert expected_fields.issubset(report.keys())
    assert report["model"]      == expected_model
    assert report["cost_usd"]   == 0.0
    assert report["task_count"] > 0
    assert 0 <= report["hit_count"] <= report["task_count"]
    assert 0.0 <= report["hit_rate"] <= 1.0


# ---- Baseline B -------------------------------------------------------------

def test_baseline_b_report_non_empty_and_consistent() -> None:
    """B's report covers the full corpus with a non negative hit rate."""
    _require_corpus()
    corpus = load_corpus(CORPUS)
    report = run_baseline_b(corpus)

    _check_report_shape(report, expected_model="baseline_b")
    assert report["task_count"]       == len(corpus)
    assert report["hit_count"]        <= report["task_count"]
    assert report["hit_rate"]         >= 0.0
    assert report["input_tokens_total"]  > 0
    assert report["output_tokens_total"] > 0


# ---- Baseline C -------------------------------------------------------------

def test_baseline_c_report_non_empty_and_consistent() -> None:
    """C's report covers the full corpus and bounds the hit count."""
    _require_corpus()
    corpus = load_corpus(CORPUS)
    report = run_baseline_c(corpus)

    _check_report_shape(report, expected_model="baseline_c")
    assert report["task_count"] == len(corpus)
    assert report["hit_count"]  <= report["task_count"]


# ---- Baseline D -------------------------------------------------------------

def test_baseline_d_report_non_empty_and_consistent() -> None:
    """D's report covers the full corpus when the .wasm artifacts exist."""
    _require_corpus()
    _require_wasm_artifacts()
    corpus = load_corpus(CORPUS)
    report = run_baseline_d(corpus)

    _check_report_shape(report, expected_model="baseline_d")
    assert report["task_count"] == len(corpus)
    assert report["hit_count"]  <= report["task_count"]


# ---- End to end main() ------------------------------------------------------

def test_main_writes_three_reports(tmp_path: Path) -> None:
    """`main()` writes one JSON report per baseline to the output dir."""
    _require_corpus()
    _require_wasm_artifacts()
    reports = main(CORPUS, tmp_path)

    assert set(reports.keys()) == {"b", "c", "d"}
    for key, name in (("b", "baseline_b_report.json"),
                      ("c", "baseline_c_report.json"),
                      ("d", "baseline_d_report.json")):
        out = tmp_path / name
        assert out.exists() and out.stat().st_size > 0
        assert reports[key]["task_count"] > 0
