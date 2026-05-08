"""Rating runner for SKG routing precision (false-positive / false-negative).

Builds a per-task rating record that a human reviewer can fill in. To
seed default values and to calibrate the form, the runner can also run
a single LLM pass that proposes a rating for each task. The LLM pass
is a stand-in, not multi-rater consensus.

Why not call this multi-rater agreement. Inter-rater statistics (Cohen
kappa, Krippendorff alpha) require independent raters with real
disagreement. A single LLM at low temperature produces near-identical
ratings across runs and inflates apparent agreement. The paper's
multi-rater claim therefore stays pending real humans; this module
provides the form they fill out.

Output format. JSONL records with these fields:
  task_id              str
  task                 str
  category             str
  expected_stage       str       (from corpus)
  observed_stage       str       (from SKG run; may be miss)
  observed_node        str|null
  llm_rating           str       ("correct" / "false_positive" / "false_negative" / "unsure")
  llm_reason           str       (short)
  human_rating         str|null  (filled by the reviewer; default null)
  human_reason         str|null

Usage (stand-alone):
  .venv/bin/python eval/rating_runner.py \
    --corpus eval/corpus.jsonl \
    --skg-run eval/results/report.json \
    --out eval/results/rating_pass.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI


MODEL = "gpt-4o-mini"


SYSTEM_PROMPT = (
    "You rate the correctness of an LLM agent's routing decision for a "
    "given task. Reply with a single JSON object containing exactly two "
    "fields: 'rating' and 'reason'. Rating must be one of: 'correct', "
    "'false_positive' (a wrong node was returned), 'false_negative' (a "
    "miss when a known node should have matched), or 'unsure'. Reason "
    "must be one short sentence."
)


def _read_key() -> str:
    return (Path.home() / ".agent-proxy" / "openai-key").read_text(
        encoding="utf-8",
    ).strip()


def _user_prompt(task: dict, observed_stage: str, observed_node: str | None) -> str:
    body = {
        "task":           task["task"],
        "category":       task.get("category"),
        "expected_stage": task.get("expected_stage"),
        "observed_stage": observed_stage,
        "observed_node":  observed_node,
    }
    return json.dumps(body, indent=2)


def rate_one(client: OpenAI, task: dict, observed_stage: str, observed_node: str | None) -> dict:
    """Call gpt-4o-mini and return {'rating': ..., 'reason': ...}.

    Caller is responsible for any retry policy. On parse failure,
    returns `unsure` with the raw error in the reason field.
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _user_prompt(task, observed_stage, observed_node)},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    body = resp.choices[0].message.content or "{}"
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as e:
        return {"rating": "unsure", "reason": f"JSON parse failure: {e}"}
    rating = decoded.get("rating", "unsure")
    reason = decoded.get("reason", "")
    if rating not in {"correct", "false_positive", "false_negative", "unsure"}:
        rating = "unsure"
    return {"rating": rating, "reason": str(reason)[:300]}


def build_records(
    corpus_records:  list[dict],
    skg_run_per_task: dict[str, dict],
) -> list[dict]:
    """Build the rating-form record set without calling the LLM."""
    rows: list[dict] = []
    for task in corpus_records:
        task_id        = task["id"]
        run            = skg_run_per_task.get(task_id) or {}
        observed_stage = run.get("stage", "miss")
        observed_node  = run.get("node_id")
        rows.append({
            "task_id":        task_id,
            "task":           task["task"],
            "category":       task.get("category"),
            "expected_stage": task.get("expected_stage"),
            "observed_stage": observed_stage,
            "observed_node":  observed_node,
            "llm_rating":     None,
            "llm_reason":     None,
            "human_rating":   None,
            "human_reason":   None,
        })
    return rows


def fill_llm_ratings(rows: list[dict]) -> None:
    """Mutate `rows` in place, filling llm_rating and llm_reason for each."""
    os.environ["OPENAI_API_KEY"] = _read_key()
    client = OpenAI()
    for i, row in enumerate(rows, 1):
        task = {
            "id":             row["task_id"],
            "task":           row["task"],
            "category":       row["category"],
            "expected_stage": row["expected_stage"],
        }
        rating = rate_one(client, task, row["observed_stage"], row["observed_node"])
        row["llm_rating"] = rating["rating"]
        row["llm_reason"] = rating["reason"]
        if i % 25 == 0:
            print(f"  rated {i}/{len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a rating form for SKG routing decisions.")
    parser.add_argument("--corpus",   required=True, type=Path)
    parser.add_argument("--skg-run",  required=False, type=Path,
                        help="Optional per-task run records keyed by task_id; otherwise rows record observed_stage='miss'.")
    parser.add_argument("--out",      required=True, type=Path)
    parser.add_argument("--no-llm",   action="store_true",
                        help="Build the form without LLM-stand-in ratings.")
    args = parser.parse_args()

    corpus_records = [
        json.loads(line)
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    skg_run_per_task: dict[str, dict] = {}
    if args.skg_run and args.skg_run.exists():
        skg_obj = json.loads(args.skg_run.read_text(encoding="utf-8"))
        for rec in skg_obj.get("per_task", []):
            skg_run_per_task[rec["task_id"]] = rec

    rows = build_records(corpus_records, skg_run_per_task)
    if not args.no_llm:
        print(f"Filling LLM ratings for {len(rows)} tasks at model {MODEL}")
        fill_llm_ratings(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rating records to {args.out}")


if __name__ == "__main__":
    main()
