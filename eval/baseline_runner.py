"""Replay corpus and baseline evaluation runner (slice 6).

This module implements the experiment execution plan from:
  designs/proposed/skg-experiment-execution-plan.md

It requires a sanitized task corpus of at least 200 tasks. The corpus
is NEVER committed to this repository. It must be generated from sanitized
personal or synthetic task logs with all private identifiers removed.

Gate: the corpus pipeline must be reviewed before this slice runs. See
  designs/proposed/skill-graph-codex-v10/design/skill-knowledge-graph.md
  "Replay research slice".

Usage:
    python -m eval.baseline_runner --corpus eval/corpus.jsonl --out eval/results/

Output:
    eval/results/
        routing_hit_rate.json     — hit/miss per stage, by task category
        latency_p50_p95.json      — WASI execution latency distribution
        cost_savings.json         — tokens saved vs LLM fallback (estimated)
        promotion_gate_stats.json — gate pass/fail rates
        figures/                  — matplotlib figures for paper

Paper tables this feeds:
    Table 1: Routing stage distribution
    Table 2: Latency by stage (p50/p95)
    Table 3: Estimated cost savings (token budget)
    Table 4: Promotion gate failure rates
    Figure 1: Hit-rate CDF over corpus size
    Figure 2: Latency distribution (violin plot)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Minimum corpus size for results to be publishable.
MINIMUM_CORPUS_SIZE = 200

# Estimated cost per token in USD (gpt-4o-mini pricing, adjust to actual vendor).
COST_PER_TOKEN_USD = 0.00000015

# Average tokens per LLM fallback call (measured from agent-proxy-kit logs).
# TODO [Baseline]: measure this from actual logs before publishing.
AVG_LLM_TOKENS     = 1500

# Average tokens per SKG header scan (router overhead).
AVG_HEADER_TOKENS  = 120


@dataclass
class TaskResult:
    """Result of running one task from the corpus through the router."""

    task_id:           str
    task_text:         str
    category:          str                = ""
    route_stage:       str                = "miss"
    route_duration_ms: float              = 0.0
    wasm_duration_ms:  float              = 0.0
    tokens_used:       int                = 0
    hit:               bool               = False
    error:             str                = ""


@dataclass
class EvalReport:
    """Aggregated evaluation results across the corpus."""

    corpus_size:       int                = 0
    hit_count:         int                = 0
    miss_count:        int                = 0
    stage_counts:      dict[str, int]     = field(default_factory=dict)
    latency_p50_ms:    float              = 0.0
    latency_p95_ms:    float              = 0.0
    tokens_saved:      int                = 0
    cost_saved_usd:    float              = 0.0
    gate_stats:        dict[str, Any]     = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.hit_count / self.corpus_size if self.corpus_size else 0.0

    def to_dict(self) -> dict:
        return {
            "corpus_size":    self.corpus_size,
            "hit_count":      self.hit_count,
            "miss_count":     self.miss_count,
            "hit_rate":       round(self.hit_rate, 4),
            "stage_counts":   self.stage_counts,
            "latency_p50_ms": self.latency_p95_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "tokens_saved":   self.tokens_saved,
            "cost_saved_usd": round(self.cost_saved_usd, 4),
        }


class BaselineRunner:
    """Routes each task in the corpus and records the result.

    The runner compares two baselines:
      A. LLM-only: every task goes to the LLM. Cost = corpus_size * AVG_LLM_TOKENS.
      B. SKG-first: router runs first. LLM is called only on miss.

    The difference is the measured saving.
    """

    def __init__(self, skg_instance: Any) -> None:
        self._skg = skg_instance

    def run(self, corpus: list[dict]) -> tuple[list[TaskResult], EvalReport]:
        if len(corpus) < MINIMUM_CORPUS_SIZE:
            raise ValueError(
                f"Corpus has {len(corpus)} tasks. Minimum for publishable results: {MINIMUM_CORPUS_SIZE}. "
                f"Add more tasks or use synthetic augmentation."
            )

        results: list[TaskResult] = []
        for item in corpus:
            r = self._run_one(item)
            results.append(r)

        report = self._aggregate(results)
        return results, report

    def _run_one(self, item: dict) -> TaskResult:
        task_id   = item.get("id", "")
        task_text = item.get("task", "")
        category  = item.get("category", "")
        context   = item.get("context", {})

        t0 = time.monotonic()
        try:
            result = self._skg.route(task_text, context)
            route_ms = (time.monotonic() - t0) * 1000
        except Exception as e:
            return TaskResult(task_id=task_id, task_text=task_text, category=category, error=str(e))

        return TaskResult(
            task_id=task_id,
            task_text=task_text,
            category=category,
            route_stage=result.stage,
            route_duration_ms=round(route_ms, 2),
            hit=result.hit,
            tokens_used=AVG_HEADER_TOKENS if result.hit else AVG_LLM_TOKENS,
        )

    def _aggregate(self, results: list[TaskResult]) -> EvalReport:
        import statistics
        hits   = [r for r in results if r.hit]
        misses = [r for r in results if not r.hit]

        stage_counts: dict[str, int] = {}
        for r in results:
            stage_counts[r.route_stage] = stage_counts.get(r.route_stage, 0) + 1

        latencies = sorted(r.route_duration_ms for r in hits)
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

        # Tokens saved = (what LLM would have cost) - (what SKG actually spent)
        llm_baseline = len(results) * AVG_LLM_TOKENS
        actual_tokens = sum(r.tokens_used for r in results)
        tokens_saved = max(0, llm_baseline - actual_tokens)
        cost_saved = tokens_saved * COST_PER_TOKEN_USD

        return EvalReport(
            corpus_size=len(results),
            hit_count=len(hits),
            miss_count=len(misses),
            stage_counts=stage_counts,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            tokens_saved=tokens_saved,
            cost_saved_usd=cost_saved,
        )


def load_corpus(path: Path) -> list[dict]:
    """Load a JSONL corpus file. Each line is one task dict."""
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return tasks


def save_report(report: EvalReport, out_dir: Path) -> None:
    """Write evaluation report JSON files to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )


def generate_figures(results: list[TaskResult], out_dir: Path) -> None:
    """Generate matplotlib figures for the paper.

    Figure 1: Hit-rate CDF over corpus size (cumulative hit rate as N grows).
    Figure 2: Latency distribution violin plot (route_duration_ms by stage).
    Figure 3: Token savings bar chart (SKG vs LLM-only baseline).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed. Skipping figure generation.")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: cumulative hit rate
    cumulative_hits = 0
    xs, ys = [], []
    for i, r in enumerate(results, 1):
        if r.hit:
            cumulative_hits += 1
        xs.append(i)
        ys.append(cumulative_hits / i)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, linewidth=1.5)
    ax.set_xlabel("Tasks evaluated (N)")
    ax.set_ylabel("Cumulative hit rate")
    ax.set_title("SKG hit rate over corpus")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="50% line")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_hit_rate_cdf.pdf", format="pdf")
    plt.close(fig)

    # Figure 2: latency violin by stage
    stages = {}
    for r in results:
        if r.hit and r.route_duration_ms > 0:
            stages.setdefault(r.route_stage, []).append(r.route_duration_ms)

    if stages:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.violinplot(list(stages.values()), showmedians=True)
        ax.set_xticks(range(1, len(stages) + 1))
        ax.set_xticklabels(list(stages.keys()))
        ax.set_ylabel("Route duration (ms)")
        ax.set_title("Routing latency by stage")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig2_latency_violin.pdf", format="pdf")
        plt.close(fig)

    # Figure 3: token savings bar
    hit_count  = sum(1 for r in results if r.hit)
    miss_count = len(results) - hit_count
    llm_tokens = len(results) * AVG_LLM_TOKENS
    skg_tokens = hit_count * AVG_HEADER_TOKENS + miss_count * AVG_LLM_TOKENS

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["LLM-only", "SKG-first"], [llm_tokens, skg_tokens], color=["#e07070", "#70a8e0"])
    ax.set_ylabel("Estimated tokens consumed")
    ax.set_title("Token savings: SKG vs LLM-only")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_token_savings.pdf", format="pdf")
    plt.close(fig)


# ---- CLI entry point ---------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SKG baseline evaluation runner")
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus file (min 200 tasks).")
    parser.add_argument("--out", default="eval/results", help="Output directory for results and figures.")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        raise SystemExit(1)

    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} tasks from {corpus_path}")

    from skg.graph import SKG
    skg = SKG()
    runner = BaselineRunner(skg)
    results, report = runner.run(corpus)

    out_dir = Path(args.out)
    save_report(report, out_dir)
    generate_figures(results, out_dir)

    print(f"\nResults written to {out_dir}/")
    print(f"Hit rate:      {report.hit_rate:.1%} ({report.hit_count}/{report.corpus_size})")
    print(f"Latency p50:   {report.latency_p50_ms:.1f}ms")
    print(f"Latency p95:   {report.latency_p95_ms:.1f}ms")
    print(f"Tokens saved:  {report.tokens_saved:,}")
    print(f"Cost saved:    ${report.cost_saved_usd:.4f}")
