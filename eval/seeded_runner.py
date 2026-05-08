"""Seeded held-out corpus runner with bootstrap confidence intervals.

Paper Section 7.5 calls for held-out corpus runs across at least 5
seeds. This module executes that protocol:

  1. For each seed in [1..5], deterministically split the 200-task
     corpus into 80% training and 20% holdout.
  2. Run Baseline A (gpt-4o-mini, real OpenAI) on the holdout.
  3. Run the SKG router on the holdout. Routing only, no node
     execution. Token accounting reuses the constants in
     eval.baseline_runner.
  4. Persist per-seed records to eval/results/seeded_runs/seed_<n>.json.
  5. Aggregate per-metric mean and 95% bootstrap CI across seeds.
  6. Persist to eval/results/seeded_aggregated.json.

Cost. Five seeds at 40 holdout tasks each is 200 LLM calls. At the
gpt-4o-mini list price the run lands well under $0.10.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from eval.baseline_a_runner import run_one as baseline_a_run_one
from eval.baseline_runner import AVG_HEADER_TOKENS, AVG_LLM_TOKENS
from eval.bootstrap import bootstrap_ci
from eval.seeded_split import split


SEEDS:         list[int] = [1, 2, 3, 4, 5]
HOLDOUT_FRAC:  float     = 0.20


def _read_key() -> str:
    key_path = Path.home() / ".agent-proxy" / "openai-key"
    return key_path.read_text(encoding="utf-8").strip()


def _ensure_openai_client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _read_key()
    return OpenAI()


def _run_baseline_a_on_holdout(
    client: OpenAI,
    holdout: list[dict],
) -> list[dict]:
    """Issue one Chat Completions call per holdout task. Return per-task records."""
    records: list[dict] = []
    for task_obj in holdout:
        rec = baseline_a_run_one(
            client,
            task_obj["task"],
            task_obj.get("context", {}),
        )
        rec["task_id"] = task_obj["id"]
        records.append(rec)
    return records


def _run_skg_routing_on_holdout(
    skg: Any,
    holdout: list[dict],
) -> list[dict]:
    """Route each holdout task. Record stage, hit, latency, and tokens.

    Token accounting follows eval.baseline_runner: a hit consumes
    AVG_HEADER_TOKENS, a miss consumes AVG_LLM_TOKENS. Routing only,
    no node execution.
    """
    records: list[dict] = []
    for task_obj in holdout:
        task_id   = task_obj["id"]
        task_text = task_obj["task"]
        context   = task_obj.get("context", {})

        started = time.monotonic()
        try:
            result = skg.route(task_text, context)
            route_ms = round((time.monotonic() - started) * 1000, 2)
            records.append({
                "task_id":      task_id,
                "stage":        result.stage.value if hasattr(result.stage, "value") else str(result.stage),
                "hit":          bool(result.hit),
                "route_ms":     route_ms,
                "tokens_used": (
                    AVG_HEADER_TOKENS if result.hit else AVG_LLM_TOKENS
                ),
                "error":        "",
            })
        except Exception as exc:
            route_ms = round((time.monotonic() - started) * 1000, 2)
            records.append({
                "task_id":     task_id,
                "stage":       "miss",
                "hit":         False,
                "route_ms":    route_ms,
                "tokens_used": AVG_LLM_TOKENS,
                "error":       str(exc),
            })
    return records


def _summarize_seed(
    seed: int,
    holdout: list[dict],
    baseline_a_records: list[dict],
    skg_records: list[dict],
) -> dict:
    """Compute per-seed aggregates. Pure arithmetic; no I/O."""
    n = len(holdout)
    a_input  = [r["input_tokens"]  for r in baseline_a_records]
    a_output = [r["output_tokens"] for r in baseline_a_records]
    a_lat    = [r["latency_ms"]    for r in baseline_a_records]

    skg_hits = sum(1 for r in skg_records if r["hit"])
    skg_tokens_total = sum(r["tokens_used"] for r in skg_records)

    return {
        "seed":              seed,
        "holdout_size":      n,
        "holdout_ids":       [r["id"] for r in holdout],
        "baseline_a": {
            "input_tokens_mean":  sum(a_input)  / n if n else 0.0,
            "output_tokens_mean": sum(a_output) / n if n else 0.0,
            "latency_mean_ms":    sum(a_lat)    / n if n else 0.0,
            "input_tokens_total":  sum(a_input),
            "output_tokens_total": sum(a_output),
        },
        "skg": {
            "hit_rate":     skg_hits / n if n else 0.0,
            "hit_count":    skg_hits,
            "miss_count":   n - skg_hits,
            "tokens_total": skg_tokens_total,
            "tokens_mean":  skg_tokens_total / n if n else 0.0,
        },
        "baseline_a_records": baseline_a_records,
        "skg_records":        skg_records,
    }


def _aggregate_across_seeds(seed_summaries: list[dict]) -> dict:
    """Compute mean and 95% bootstrap CI per metric across seeds."""
    skg_hit_rate     = [s["skg"]["hit_rate"]                for s in seed_summaries]
    skg_tokens_mean  = [s["skg"]["tokens_mean"]             for s in seed_summaries]
    a_input_mean     = [s["baseline_a"]["input_tokens_mean"]  for s in seed_summaries]
    a_output_mean    = [s["baseline_a"]["output_tokens_mean"] for s in seed_summaries]
    a_latency_mean   = [s["baseline_a"]["latency_mean_ms"]    for s in seed_summaries]

    def _ci(values: list[float], seed: int = 0) -> dict[str, float]:
        mean, lower, upper = bootstrap_ci(values, iterations=1000, seed=seed)
        return {
            "mean":     round(mean,  6),
            "ci_lower": round(lower, 6),
            "ci_upper": round(upper, 6),
        }

    return {
        "skg_hit_rate":           _ci(skg_hit_rate,    seed=1),
        "skg_tokens_mean":        _ci(skg_tokens_mean, seed=2),
        "baseline_a_input_mean":  _ci(a_input_mean,    seed=3),
        "baseline_a_output_mean": _ci(a_output_mean,   seed=4),
        "baseline_a_latency_ms":  _ci(a_latency_mean,  seed=5),
    }


def main(corpus_path: Path, out_dir: Path) -> dict:
    """Run all seeds, persist per-seed records, persist aggregate. Return aggregate."""
    seeds_dir = out_dir / "seeded_runs"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    client = _ensure_openai_client()

    from skg.graph import SKG

    seed_summaries: list[dict] = []
    holdout_size_observed: int | None = None

    for seed in SEEDS:
        _, holdout = split(corpus_path, seed=seed, holdout_frac=HOLDOUT_FRAC)
        if holdout_size_observed is None:
            holdout_size_observed = len(holdout)

        print(f"[seed {seed}] holdout={len(holdout)} tasks; running Baseline A and SKG router")

        baseline_a_records = _run_baseline_a_on_holdout(client, holdout)

        skg = SKG()
        skg_records = _run_skg_routing_on_holdout(skg, holdout)

        summary = _summarize_seed(
            seed,
            holdout,
            baseline_a_records,
            skg_records,
        )
        seed_summaries.append(summary)

        seed_path = seeds_dir / f"seed_{seed}.json"
        seed_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"[seed {seed}] saved {seed_path.name}; "
            f"a_input_mean={summary['baseline_a']['input_tokens_mean']:.2f} "
            f"skg_hit_rate={summary['skg']['hit_rate']:.2%}"
        )

    metrics = _aggregate_across_seeds(seed_summaries)
    aggregate = {
        "seeds":         SEEDS,
        "holdout_size":  holdout_size_observed or 0,
        "metrics":       metrics,
    }

    aggregate_path = out_dir / "seeded_aggregated.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\nAggregate written to {aggregate_path}")
    return aggregate


if __name__ == "__main__":
    repo = Path(__file__).parent.parent
    main(repo / "eval" / "corpus.jsonl", repo / "eval" / "results")
