"""Tests for eval.seeded_split."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.seeded_split import split


def _write_corpus(path: Path, n: int) -> None:
    lines = []
    for i in range(n):
        lines.append(json.dumps({"id": f"t{i:04d}", "task": f"task {i}"}))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_same_seed_same_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, 200)

    train_a, holdout_a = split(corpus, seed=7)
    train_b, holdout_b = split(corpus, seed=7)

    assert [r["id"] for r in train_a]   == [r["id"] for r in train_b]
    assert [r["id"] for r in holdout_a] == [r["id"] for r in holdout_b]


def test_different_seeds_different_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, 200)

    _, holdout_a = split(corpus, seed=1)
    _, holdout_b = split(corpus, seed=2)

    ids_a = [r["id"] for r in holdout_a]
    ids_b = [r["id"] for r in holdout_b]
    assert ids_a != ids_b


def test_holdout_frac_default_yields_40_of_200(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, 200)

    train, holdout = split(corpus, seed=1)

    assert len(holdout) == 40
    assert len(train)   == 160


def test_holdout_frac_custom(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, 100)

    train, holdout = split(corpus, seed=1, holdout_frac=0.10)

    assert len(holdout) == 10
    assert len(train)   == 90


def test_train_and_holdout_are_disjoint(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, 200)

    train, holdout = split(corpus, seed=3)
    train_ids   = {r["id"] for r in train}
    holdout_ids = {r["id"] for r in holdout}

    assert train_ids.isdisjoint(holdout_ids)
    assert len(train_ids | holdout_ids) == 200


def test_invalid_holdout_frac_raises(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, 10)

    with pytest.raises(ValueError):
        split(corpus, seed=1, holdout_frac=0.0)
    with pytest.raises(ValueError):
        split(corpus, seed=1, holdout_frac=1.0)
