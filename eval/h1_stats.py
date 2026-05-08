"""Hypothesis test for H1 on the larger-context corpus.

H1 from paper Section 7.4: SKG cuts input tokens by 50%+ for recurring
tasks, with threshold T < 0.50 * A using a one-sided t-test, p < 0.05.
This script runs the SKG router on every task in `corpus_large.jsonl`,
pairs each task's outcome with the matching Baseline A per-task input
measurement from `baseline_a_large_tasks.jsonl`, and reports:

  - Per-task SKG estimated input tokens (120 on hit, A_input[i] on miss)
  - Paired sample t-test, one-sided (T < 0.50 * A)
  - Cohen's d effect size on the difference T - A
  - Mean and 95% bootstrap CI for the saved-token delta

Output JSON to `eval/results/h1_stats.json`.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Iterable


HEADER_COST_TOKENS = 120


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _route_skg(task_text: str) -> tuple[bool, str]:
    """Run the SKG router on `task_text`; return (hit, stage_name)."""
    from skg.graph import SKG
    skg = SKG()
    res = skg.route(task_text, {})
    return bool(res.hit), str(getattr(res, "stage", "miss"))


def _pair_token_counts(corpus: list[dict], baseline: list[dict]) -> list[tuple[int, int]]:
    """Return [(skg_tokens, a_tokens), ...] one per task, in corpus order."""
    by_id = {row["task_id"]: row for row in baseline}
    out: list[tuple[int, int]] = []
    hits = 0
    for task in corpus:
        a_row = by_id.get(task["id"])
        if a_row is None:
            continue
        a_tokens = int(a_row["input_tokens"])
        hit, _stage = _route_skg(task["task"])
        if hit:
            hits += 1
            skg_tokens = HEADER_COST_TOKENS
        else:
            skg_tokens = a_tokens
        out.append((skg_tokens, a_tokens))
    print(f"SKG routed {len(out)} tasks; {hits} hits, {len(out) - hits} misses")
    return out


def _one_sided_t_test_t_lt_half_a(pairs: list[tuple[int, int]]) -> tuple[float, float, int]:
    """Paired t-test for H1: T - 0.5 * A < 0. Returns (t, p, df)."""
    diffs = [t - 0.5 * a for (t, a) in pairs]
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), 0
    mean_d = statistics.fmean(diffs)
    var_d  = statistics.variance(diffs)
    se     = math.sqrt(var_d / n)
    if se == 0.0:
        return float("nan"), 0.0 if mean_d < 0 else 1.0, n - 1
    t = mean_d / se
    p = _t_cdf(t, df=n - 1)
    return t, p, n - 1


def _t_cdf(t: float, df: int) -> float:
    """Lower-tail CDF of Student's t (df) at value t.

    Uses the regularised incomplete beta function so we can avoid scipy.
    """
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    cdf_betainc = _regularised_incomplete_beta(x, a, b)
    if t < 0:
        return 0.5 * cdf_betainc
    return 1.0 - 0.5 * cdf_betainc


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta I_x(a, b) via continued fraction.

    Numerical Recipes recipe, sufficient for a couple of decimal places
    on practical t-test values.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(x, a, b) / a
    return 1.0 - bt * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iters: int = 200, tol: float = 1e-10) -> float:
    """Continued fraction for the incomplete beta."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            return h
    return h


def _cohens_d(values_t: list[int], values_a: list[int]) -> float:
    """Cohen's d for paired samples on the difference T - A."""
    diffs = [t - a for t, a in zip(values_t, values_a)]
    if len(diffs) < 2:
        return float("nan")
    return statistics.fmean(diffs) / statistics.stdev(diffs)


def _bootstrap_ci(values: list[float], iterations: int = 1000, seed: int = 0,
                  confidence: float = 0.95) -> tuple[float, float, float]:
    import random
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - confidence) / 2 * iterations)]
    hi = means[int((1 + confidence) / 2 * iterations) - 1]
    return statistics.fmean(values), lo, hi


def main() -> None:
    repo  = Path(__file__).parent.parent
    corpus    = _load_jsonl(repo / "eval" / "corpus_large.jsonl")
    baseline  = _load_jsonl(repo / "eval" / "results" / "baseline_a_large_tasks.jsonl")
    pairs     = _pair_token_counts(corpus, baseline)

    skg_tokens = [t for t, _ in pairs]
    a_tokens   = [a for _, a in pairs]
    diffs      = [t - a for t, a in pairs]
    saved      = [-d for d in diffs]

    t, p, df = _one_sided_t_test_t_lt_half_a(pairs)
    d_eff    = _cohens_d(skg_tokens, a_tokens)
    saved_mean, saved_lo, saved_hi = _bootstrap_ci(saved, iterations=1000, seed=0)

    skg_mean = statistics.fmean(skg_tokens)
    a_mean   = statistics.fmean(a_tokens)

    report = {
        "task_count":               len(pairs),
        "skg_input_mean":           skg_mean,
        "baseline_a_input_mean":    a_mean,
        "ratio_t_over_a":           skg_mean / a_mean if a_mean else float("nan"),
        "h1_threshold":             "T < 0.50 * A",
        "one_sided_t_statistic":    t,
        "one_sided_p_value":        p,
        "degrees_of_freedom":       df,
        "cohens_d":                 d_eff,
        "saved_tokens_mean":        saved_mean,
        "saved_tokens_ci_lower":    saved_lo,
        "saved_tokens_ci_upper":    saved_hi,
        "h1_supported":             p < 0.05 and skg_mean < 0.5 * a_mean,
    }

    out = repo / "eval" / "results" / "h1_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
