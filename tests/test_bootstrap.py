"""Tests for eval.bootstrap."""

from __future__ import annotations

import random
import statistics

import pytest

from eval.bootstrap import bootstrap_ci


def test_empty_input_returns_zeros() -> None:
    mean, lower, upper = bootstrap_ci([])
    assert mean  == 0.0
    assert lower == 0.0
    assert upper == 0.0


def test_single_value_returns_value() -> None:
    mean, lower, upper = bootstrap_ci([4.2])
    assert mean  == 4.2
    assert lower == 4.2
    assert upper == 4.2


def test_mean_close_to_sample_mean() -> None:
    rng = random.Random(123)
    values = [rng.gauss(10.0, 2.0) for _ in range(200)]
    expected = statistics.fmean(values)

    mean, _, _ = bootstrap_ci(values, iterations=1000, seed=0)

    assert abs(mean - expected) < 1e-9


def test_ci_brackets_mean() -> None:
    rng = random.Random(7)
    values = [rng.gauss(5.0, 1.0) for _ in range(150)]

    mean, lower, upper = bootstrap_ci(values, iterations=1000, seed=0)

    assert lower <= mean <= upper


def test_ci_coverage_on_gaussian() -> None:
    """Repeated trials: CI envelope contains the true mean roughly 95% of the time."""
    true_mean = 0.0
    true_sd   = 1.0
    n         = 60
    trials    = 100
    contained = 0

    rng = random.Random(2024)
    for trial in range(trials):
        sample = [rng.gauss(true_mean, true_sd) for _ in range(n)]
        _, lower, upper = bootstrap_ci(sample, iterations=400, seed=trial)
        if lower <= true_mean <= upper:
            contained += 1

    coverage = contained / trials
    # Percentile bootstrap on a Gaussian sample should sit near 0.95.
    # Allow a generous window since trials=100 is small.
    assert 0.85 <= coverage <= 1.0


def test_invalid_confidence_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0, 3.0], confidence=0.0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0, 3.0], confidence=1.0)


def test_invalid_iterations_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0, 3.0], iterations=0)


def test_deterministic_with_same_seed() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    a = bootstrap_ci(values, iterations=500, seed=42)
    b = bootstrap_ci(values, iterations=500, seed=42)
    assert a == b
