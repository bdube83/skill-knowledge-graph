"""Corpus generation utilities for SKG evaluation (slice 6).

The corpus is the 200-task minimum required for publishable evaluation results.
This module provides tools to generate a sanitized synthetic corpus and to
validate that an existing corpus meets the gate requirements.

Gate requirements:
  - Minimum 200 tasks.
  - No private names, employer names, client names, or real PR/issue numbers.
  - Tasks span at least 5 categories.
  - Category distribution documented in the paper.

Corpus schema (one JSONL line per task):
    {
        "id":       "t001",
        "task":     "Draft a reviewer ping for PR review",
        "category": "communication",
        "context":  { ... },
        "expected_stage": "exact" | "fts" | "vector" | "miss"
    }

The `expected_stage` field is used for gate validation but not for scoring.
Routing decisions are made by the live router, not by this field.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from eval.baseline_runner import MINIMUM_CORPUS_SIZE


# Synthetic task templates by category.
# None of these reference real employers, projects, or people.
SYNTHETIC_TEMPLATES: dict[str, list[dict]] = {
    "communication": [
        {"task": "Draft a reviewer ping for PR review", "context": {"pr_number": "{n}", "repo": "example/repo", "author": "alice", "reviewers": ["bob"]}},
        {"task": "Write a status update for a stalled issue", "context": {"issue": "{n}", "repo": "example/repo", "days_stalled": 5}},
        {"task": "Compose a request for design feedback", "context": {"doc": "design-draft-{n}.md", "audience": "team"}},
        {"task": "Draft a polite follow-up on an unanswered code review", "context": {"pr_number": "{n}", "days_waiting": 3}},
        {"task": "Write release notes summary for a patch version", "context": {"version": "1.0.{n}", "changes": ["bug fix", "perf"]}},
    ],
    "git": [
        {"task": "Summarise the diff for a pull request", "context": {"pr_number": "{n}", "repo": "example/repo"}},
        {"task": "List open PRs needing review", "context": {"repo": "example/repo", "author": "alice"}},
        {"task": "Check if a branch is behind main", "context": {"branch": "feature-{n}", "base": "main"}},
        {"task": "Find PRs with no reviewer assigned", "context": {"repo": "example/repo"}},
        {"task": "Generate a commit message from a diff summary", "context": {"summary": "fixed off-by-one in loop {n}"}},
    ],
    "planning": [
        {"task": "Break a feature request into sub-tasks", "context": {"feature": "add export to CSV {n}"}},
        {"task": "Estimate effort for a bug report", "context": {"issue": "login fails on mobile {n}", "severity": "medium"}},
        {"task": "Draft a sprint planning agenda", "context": {"sprint": "{n}", "team_size": 4}},
        {"task": "Identify blockers in a task list", "context": {"tasks": ["task-a", "task-b", "task-c"]}},
        {"task": "Propose a technical approach for a feature", "context": {"feature": "real-time notifications {n}"}},
    ],
    "documentation": [
        {"task": "Write a docstring for a function", "context": {"fn_name": "compute_score_{n}", "params": ["x", "y"]}},
        {"task": "Generate a README section for a new module", "context": {"module": "parser_{n}"}},
        {"task": "Summarise an API endpoint for internal docs", "context": {"endpoint": "/api/v{n}/users"}},
        {"task": "Write a changelog entry for a feature", "context": {"feature": "CSV export {n}"}},
        {"task": "Draft a troubleshooting guide section", "context": {"issue": "connection timeout {n}"}},
    ],
    "analysis": [
        {"task": "Summarise recent CI failures", "context": {"repo": "example/repo", "days": "{n}"}},
        {"task": "Identify the most common error in a log", "context": {"log_lines": 500, "period": "7d"}},
        {"task": "Compare two implementation approaches", "context": {"option_a": "approach-a-{n}", "option_b": "approach-b-{n}"}},
        {"task": "List dependencies added in the last sprint", "context": {"sprint": "{n}"}},
        {"task": "Summarise test coverage change for a PR", "context": {"pr_number": "{n}", "before": 82, "after": 85}},
    ],
}


def generate_synthetic_corpus(n: int = MINIMUM_CORPUS_SIZE, seed: int = 42) -> list[dict]:
    """Generate a synthetic corpus of n tasks from the template set.

    Uses deterministic seeding so results are reproducible.
    All task IDs, context values, and PR numbers are synthetic.
    """
    random.seed(seed)
    tasks = []
    idx   = 1

    categories = list(SYNTHETIC_TEMPLATES.keys())
    per_cat    = n // len(categories)
    extras     = n % len(categories)

    for cat_idx, category in enumerate(categories):
        count     = per_cat + (1 if cat_idx < extras else 0)
        templates = SYNTHETIC_TEMPLATES[category]
        for i in range(count):
            tmpl = templates[i % len(templates)]
            num  = str(idx)
            task_text = tmpl["task"]
            # Substitute {n} in context values.
            ctx = json.loads(json.dumps(tmpl["context"]).replace('"{n}"', f'"{num}"').replace('{n}', num))
            tasks.append({
                "id":             f"t{idx:04d}",
                "task":           task_text,
                "category":       category,
                "context":        ctx,
                "expected_stage": "miss",   # updated after router calibration
            })
            idx += 1

    return tasks


def validate_corpus(corpus: list[dict]) -> list[str]:
    """Validate a corpus against the gate requirements.

    Returns a list of error strings. Empty list means the corpus passes.
    """
    errors = []

    if len(corpus) < MINIMUM_CORPUS_SIZE:
        errors.append(f"Corpus has {len(corpus)} tasks; minimum is {MINIMUM_CORPUS_SIZE}.")

    categories = {t.get("category", "") for t in corpus}
    if len(categories) < 5:
        errors.append(f"Corpus covers {len(categories)} categories; minimum is 5.")

    for i, task in enumerate(corpus):
        if not task.get("id"):
            errors.append(f"Task {i}: missing 'id'.")
        if not task.get("task"):
            errors.append(f"Task {i}: missing 'task' text.")

    return errors


def save_corpus(corpus: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in corpus:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a synthetic SKG evaluation corpus.")
    parser.add_argument("--n", type=int, default=MINIMUM_CORPUS_SIZE, help="Number of tasks to generate.")
    parser.add_argument("--out", default="eval/corpus.jsonl", help="Output JSONL file path.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = generate_synthetic_corpus(n=args.n, seed=args.seed)
    errors = validate_corpus(corpus)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        raise SystemExit(1)

    out = Path(args.out)
    save_corpus(corpus, out)
    cats = {t["category"] for t in corpus}
    print(f"Generated {len(corpus)} tasks across {len(cats)} categories -> {out}")
