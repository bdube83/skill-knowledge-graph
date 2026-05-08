"""Deterministic train/holdout split for the SKG evaluation corpus.

The seeded splitter shuffles the corpus with a fixed seed and slices off
a holdout fraction. The same seed always yields the same split. Different
seeds yield different splits.

The mechanism backs the held-out corpus protocol referenced in paper
Section 7.5: confidence intervals require multiple repetitions over
different seeded splits.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def _load_corpus(corpus_path: Path) -> list[dict]:
    """Load a JSONL corpus file. Each non-empty line is one task dict."""
    records: list[dict] = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def split(
    corpus_path: Path,
    seed: int,
    holdout_frac: float = 0.20,
) -> tuple[list[dict], list[dict]]:
    """Split the corpus into training and holdout records.

    The function loads the JSONL file at corpus_path, shuffles a copy of
    the records with random.Random(seed), then slices off the first
    holdout_frac proportion as the holdout set. The remainder is the
    training set.

    Determinism. Identical seed and corpus always produce identical
    splits. The seed isolates randomness from the global RNG.

    Returns (training_records, holdout_records).
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(
            f"holdout_frac must be in (0, 1); got {holdout_frac}",
        )

    records = _load_corpus(corpus_path)
    if not records:
        return [], []

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    holdout_size = int(round(len(shuffled) * holdout_frac))
    holdout = shuffled[:holdout_size]
    training = shuffled[holdout_size:]
    return training, holdout
