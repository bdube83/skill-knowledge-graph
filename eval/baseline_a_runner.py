"""Baseline A (LLM-only) runner for the SKG paper.

Produces measured token counts and cost for the LLM-only baseline on
the 200-task synthetic corpus. The measurements feed Tables 2 and 3 in
the paper. Replaces the prior "estimated" Baseline A numbers in the
paper's partial-results blockquote.

Cost. Uses gpt-4o-mini at the published rate ($0.15 / 1M input,
$0.60 / 1M output). 200 tasks at ~1500 input tokens and ~500 output
tokens each lands around $0.10 total.

Output. Writes:
  - eval/results/baseline_a_report.json: aggregate counts and cost
  - eval/results/baseline_a_tasks.jsonl: per-task record (tokens,
    latency, response excerpt, model name)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from openai import OpenAI


MODEL          = "gpt-4o-mini"
INPUT_PRICE_M  = 0.15   # USD per million input tokens
OUTPUT_PRICE_M = 0.60   # USD per million output tokens


def _read_key() -> str:
    key_path = Path.home() / ".agent-proxy" / "openai-key"
    return key_path.read_text(encoding="utf-8").strip()


def _system_prompt() -> str:
    return (
        "You are an LLM agent assisting with software engineering tasks. "
        "Given a task description and a small JSON context, produce a "
        "concise plan or draft output. Reply only with the result; "
        "no preamble."
    )


def _user_prompt(task: str, context: dict) -> str:
    body = {"task": task, "context": context}
    return json.dumps(body, indent=2)


def run_one(client: OpenAI, task: str, context: dict) -> dict:
    """Issue one Chat Completions call. Return the structured record."""
    started = time.monotonic()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",  "content": _system_prompt()},
            {"role": "user",    "content": _user_prompt(task, context)},
        ],
        temperature=0.0,
    )
    latency_ms = round((time.monotonic() - started) * 1000, 2)
    usage = resp.usage
    body  = resp.choices[0].message.content or ""
    return {
        "input_tokens":  int(usage.prompt_tokens),
        "output_tokens": int(usage.completion_tokens),
        "latency_ms":    latency_ms,
        "model":         MODEL,
        "response_excerpt": body[:200],
    }


def main(corpus_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENAI_API_KEY"] = _read_key()
    client = OpenAI()

    corpus_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    records:    list[dict] = []
    in_total:   int        = 0
    out_total:  int        = 0
    lat_p50_ms: list[float] = []

    print(f"Running Baseline A ({MODEL}) over {len(corpus_lines)} tasks")
    for i, line in enumerate(corpus_lines, 1):
        task_obj = json.loads(line)
        rec = run_one(client, task_obj["task"], task_obj.get("context", {}))
        rec["task_id"] = task_obj["id"]
        records.append(rec)
        in_total  += rec["input_tokens"]
        out_total += rec["output_tokens"]
        lat_p50_ms.append(rec["latency_ms"])
        if i % 25 == 0:
            print(f"  done {i}/{len(corpus_lines)}")

    cost_usd = (in_total * INPUT_PRICE_M + out_total * OUTPUT_PRICE_M) / 1_000_000
    lat_sorted = sorted(lat_p50_ms)
    p50 = lat_sorted[len(lat_sorted) // 2]
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)]

    report = {
        "model":              MODEL,
        "task_count":         len(records),
        "input_tokens_total": in_total,
        "output_tokens_total": out_total,
        "input_tokens_mean":  round(in_total  / max(len(records), 1), 2),
        "output_tokens_mean": round(out_total / max(len(records), 1), 2),
        "latency_p50_ms":     p50,
        "latency_p95_ms":     p95,
        "cost_usd":           round(cost_usd, 6),
        "input_price_per_m":  INPUT_PRICE_M,
        "output_price_per_m": OUTPUT_PRICE_M,
    }

    (out_dir / "baseline_a_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    with (out_dir / "baseline_a_tasks.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print("---")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    repo = Path(__file__).parent.parent
    main(repo / "eval" / "corpus.jsonl", repo / "eval" / "results")
