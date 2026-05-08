"""Baselines B, C, D runner for the SKG paper.

Runs three baselines against the 200-task synthetic corpus and writes
one aggregate report per baseline. The reports feed the paper's per
system comparison tables. None of the baselines call a real LLM. B and
C use the deterministic stubs already inside their runtime modules. D
runs the promoted .wasm artifacts under the full WASI surface but
skips graph composition.

Baseline B: flow registry. String equality match on the lower cased
task. Hits return a stored response. Misses go to the stub LLM and
contribute zero to the hit count.

Baseline C: semantic cache. Feature hashed cosine match against three
seed pairs at the default threshold of 0.85. Hits replay the seeded
response. Misses go to the stub LLM and seed the cache for later
calls in the same run.

Baseline D: flat tool library. Maps the task to one of three promoted
nodes by keyword match, then runs the .wasm artifact under the full
WASI surface. Token accounting mirrors the SKG runner: a hit costs
AVG_HEADER_TOKENS, a miss costs the measured Baseline A per task input
mean.

Output. Writes one report per baseline:
  - eval/results/baseline_b_report.json
  - eval/results/baseline_c_report.json
  - eval/results/baseline_d_report.json

Each report has the same shape:
  task_count, input_tokens_total, output_tokens_total,
  input_tokens_mean, output_tokens_mean, latency_p50_ms,
  latency_p95_ms, cost_usd (always 0.0), hit_count, hit_rate, model.

Usage:
    python -m eval.baseline_bcd_runner --corpus eval/corpus.jsonl --out eval/results/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from skg.baselines.flat_library  import FlatToolLibraryRuntime
from skg.baselines.flow_registry import FlowRegistryRuntime
from skg.baselines.semantic_cache import SemanticCacheRuntime


# ---- Constants --------------------------------------------------------------

# Per task input token cost on a Baseline A miss. Mirrors the figure the
# SKG runner uses for token accounting and matches the measured mean
# from eval/results/baseline_a_report.json (90.32, rounded to int).
AVG_LLM_TOKENS    = 90

# Per task token cost when a baseline returns a hit. The number stands
# for the routing header the agent has to read to decide which tool to
# call. Same value the SKG runner uses.
AVG_HEADER_TOKENS = 120

# Output tokens per stub response. The paper's runners report a
# response excerpt; we take a small fixed number per call so the
# baseline output column is non zero and comparable to A.
STUB_OUTPUT_TOKENS = 32

# Three known nodes plus their match keywords. Matching is case
# insensitive substring. The first matching keyword wins.
NODE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("reviewer-ping-draft", ("reviewer", "ping", "draft for pr")),
    ("git-summary",         ("git", "commit", "log")),
    ("doc-update",          ("doc", "documentation")),
]

# One representative task and one canned response per node. B uses the
# task as the registry key; C uses it as a seed embedding.
NODE_SEED_PAIRS: dict[str, tuple[str, str]] = {
    "reviewer-ping-draft": (
        "draft a reviewer ping for pr review",
        "Hi reviewer, please take a look at the PR when you get a chance.",
    ),
    "git-summary": (
        "summarise recent git commits",
        "Recent commits cover bug fixes and minor refactors.",
    ),
    "doc-update": (
        "update documentation for the new flag",
        "Documentation updated to describe the new flag and its default.",
    ),
}

# Wasm artifact paths. The reviewer node uses an underscore filename;
# the other two match their node id verbatim. Resolved relative to the
# repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_WASM: dict[str, Path] = {
    "reviewer-ping-draft": (
        REPO_ROOT / "nodes" / "reviewer-ping-draft" / "target"
        / "wasm32-wasip1" / "release" / "reviewer_ping_draft.wasm"
    ),
    "git-summary": (
        REPO_ROOT / "nodes" / "git-summary" / "target"
        / "wasm32-wasip1" / "release" / "git-summary.wasm"
    ),
    "doc-update": (
        REPO_ROOT / "nodes" / "doc-update" / "target"
        / "wasm32-wasip1" / "release" / "doc-update.wasm"
    ),
}


# ---- Helpers ----------------------------------------------------------------

def match_node(task: str) -> str | None:
    """Return the node id whose first keyword matches the task, or None."""
    lowered = task.lower()
    for node_id, keywords in NODE_KEYWORDS:
        for kw in keywords:
            if kw in lowered:
                return node_id
    return None


def percentile(values: list[float], pct: float) -> float:
    """Return the requested percentile from a list of floats."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


def aggregate(
    records: list[dict],
    model_name: str,
) -> dict[str, Any]:
    """Build the per system aggregate report from per task records."""
    in_total  = sum(r["input_tokens"]  for r in records)
    out_total = sum(r["output_tokens"] for r in records)
    hit_count = sum(1 for r in records if r["hit"])
    n         = len(records)
    latencies = [r["latency_ms"] for r in records]

    return {
        "model":               model_name,
        "task_count":          n,
        "input_tokens_total":  in_total,
        "output_tokens_total": out_total,
        "input_tokens_mean":   round(in_total  / max(n, 1), 2),
        "output_tokens_mean": round(out_total / max(n, 1), 2),
        "latency_p50_ms":      round(percentile(latencies, 0.50), 2),
        "latency_p95_ms":      round(percentile(latencies, 0.95), 2),
        "cost_usd":            0.0,
        "hit_count":           hit_count,
        "hit_rate":            round(hit_count / max(n, 1), 4),
    }


# ---- Per baseline runners ---------------------------------------------------

def _registry_pairs() -> list[tuple[str, str]]:
    """Lower cased seed task to canned response pairs for B."""
    return [(task.lower(), resp) for task, resp in NODE_SEED_PAIRS.values()]


def run_baseline_b(corpus: list[dict]) -> dict[str, Any]:
    """Run Baseline B (flow registry) over the corpus."""
    runtime = FlowRegistryRuntime(registry=_registry_pairs())
    records: list[dict] = []
    for item in corpus:
        task     = item["task"]
        node_id  = match_node(task) or "miss"
        started  = time.monotonic()
        result   = runtime.execute(
            wasm_path="/dev/null",
            node_id=node_id,
            task=task.lower(),
            context=item.get("context", {}),
            granted_effects=[],
        )
        latency_ms = round((time.monotonic() - started) * 1000, 2)
        hit        = result.output.get("source") == "registry"
        in_tokens  = AVG_HEADER_TOKENS if hit else AVG_LLM_TOKENS
        out_tokens = STUB_OUTPUT_TOKENS
        records.append({
            "task_id":       item.get("id", ""),
            "input_tokens":  in_tokens,
            "output_tokens": out_tokens,
            "latency_ms":    latency_ms,
            "hit":           hit,
        })
    return aggregate(records, model_name="baseline_b")


def run_baseline_c(corpus: list[dict]) -> dict[str, Any]:
    """Run Baseline C (semantic cache) over the corpus."""
    runtime = SemanticCacheRuntime(seed_pairs=list(NODE_SEED_PAIRS.values()))
    records: list[dict] = []
    for item in corpus:
        task    = item["task"]
        node_id = match_node(task) or "miss"
        started = time.monotonic()
        result  = runtime.execute(
            wasm_path="/dev/null",
            node_id=node_id,
            task=task,
            context=item.get("context", {}),
            granted_effects=[],
        )
        latency_ms = round((time.monotonic() - started) * 1000, 2)
        hit        = result.output.get("source") == "cache"
        in_tokens  = AVG_HEADER_TOKENS if hit else AVG_LLM_TOKENS
        out_tokens = STUB_OUTPUT_TOKENS
        records.append({
            "task_id":       item.get("id", ""),
            "input_tokens":  in_tokens,
            "output_tokens": out_tokens,
            "latency_ms":    latency_ms,
            "hit":           hit,
        })
    return aggregate(records, model_name="baseline_c")


def run_baseline_d(corpus: list[dict]) -> dict[str, Any]:
    """Run Baseline D (flat tool library) over the corpus.

    Each task picks at most one node by keyword match. Hits invoke the
    matching .wasm artifact under the full WASI surface. Misses
    contribute the Baseline A per task input cost. Wasm execution
    latency is included in the hit latency.
    """
    runtime = FlatToolLibraryRuntime()
    records: list[dict] = []
    for item in corpus:
        task    = item["task"]
        node_id = match_node(task)
        started = time.monotonic()
        if node_id is None:
            latency_ms = round((time.monotonic() - started) * 1000, 2)
            records.append({
                "task_id":       item.get("id", ""),
                "input_tokens":  AVG_LLM_TOKENS,
                "output_tokens": STUB_OUTPUT_TOKENS,
                "latency_ms":    latency_ms,
                "hit":           False,
            })
            continue

        wasm_path = NODE_WASM[node_id]
        result    = runtime.execute(
            wasm_path=wasm_path,
            node_id=node_id,
            task=task,
            context=item.get("context", {}),
            granted_effects=["text.generate"],
        )
        latency_ms = round((time.monotonic() - started) * 1000, 2)
        hit        = bool(result.success and result.output)
        in_tokens  = AVG_HEADER_TOKENS if hit else AVG_LLM_TOKENS
        out_tokens = STUB_OUTPUT_TOKENS
        records.append({
            "task_id":       item.get("id", ""),
            "input_tokens":  in_tokens,
            "output_tokens": out_tokens,
            "latency_ms":    latency_ms,
            "hit":           hit,
        })
    return aggregate(records, model_name="baseline_d")


# ---- IO ---------------------------------------------------------------------

def load_corpus(path: Path) -> list[dict]:
    """Load a JSONL corpus. One task dict per non empty line."""
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def write_report(report: dict[str, Any], out_path: Path) -> None:
    """Write a single aggregate report as pretty printed JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


# ---- CLI --------------------------------------------------------------------

def main(corpus_path: Path, out_dir: Path) -> dict[str, dict[str, Any]]:
    """Run B, C, D and write one report per system. Return the reports."""
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} tasks from {corpus_path}")

    reports: dict[str, dict[str, Any]] = {}

    print("Running Baseline B (flow registry)")
    reports["b"] = run_baseline_b(corpus)
    write_report(reports["b"], out_dir / "baseline_b_report.json")

    print("Running Baseline C (semantic cache)")
    reports["c"] = run_baseline_c(corpus)
    write_report(reports["c"], out_dir / "baseline_c_report.json")

    print("Running Baseline D (flat tool library)")
    reports["d"] = run_baseline_d(corpus)
    write_report(reports["d"], out_dir / "baseline_d_report.json")

    for key, report in reports.items():
        print(f"--- baseline_{key} ---")
        print(json.dumps(report, indent=2))

    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baselines B, C, D over a JSONL corpus.")
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus file.")
    parser.add_argument("--out",    default="eval/results", help="Output directory for reports.")
    args = parser.parse_args()

    main(Path(args.corpus), Path(args.out))
