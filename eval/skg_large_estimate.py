"""Compute SKG token estimate against Baseline A on the larger-context corpus.

The arithmetic mirrors the formula used in paper.md:

    skg_input_tokens = hit_count * HEADER_COST + miss_count * BASELINE_A_INPUT_MEAN

Hits pay only the routing-header cost. Misses pay the full Baseline A
input cost because the router falls through to the LLM. This script
reads the Baseline A measurement on the larger corpus and writes the
estimate to eval/results/skg_large_estimate.json.

Run after eval.baseline_a_large has produced its report.
"""

from __future__ import annotations

import json
from pathlib import Path


HEADER_COST_TOKENS = 120
SKG_HIT_RATE       = 0.80


def compute(report_path: Path, out_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    n      = int(report["task_count"])
    base_mean = float(report["input_tokens_mean"])

    hit_count  = round(n * SKG_HIT_RATE)
    miss_count = n - hit_count

    skg_input_total = hit_count * HEADER_COST_TOKENS + miss_count * base_mean
    skg_input_mean  = skg_input_total / n
    saved_tokens    = base_mean - skg_input_mean
    saved_pct       = (saved_tokens / base_mean) * 100.0 if base_mean > 0 else 0.0

    out = {
        "task_count":                    n,
        "skg_hit_rate":                  SKG_HIT_RATE,
        "skg_input_tokens_mean":         round(skg_input_mean, 2),
        "baseline_a_input_tokens_mean":  round(base_mean, 2),
        "saved_tokens":                  round(saved_tokens, 2),
        "saved_pct":                     round(saved_pct, 2),
        "header_cost_tokens":            HEADER_COST_TOKENS,
        "header_breakeven_input_tokens": HEADER_COST_TOKENS,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    repo = Path(__file__).parent.parent
    rep  = repo / "eval" / "results" / "baseline_a_large_report.json"
    out  = repo / "eval" / "results" / "skg_large_estimate.json"
    print(json.dumps(compute(rep, out), indent=2))
