"""Interactive rater CLI for SKG routing decisions.

Walks the rater through every row of `rating_pass.jsonl` (or any
JSONL produced by `rating_runner.py`) and writes a new file with the
`human_rating` field filled. Saves progress on every keystroke so the
rater can quit and resume.

Usage:
  .venv/bin/python eval/rating_review_cli.py \
    --input eval/results/rating_pass.jsonl \
    --rater-id alice \
    --output eval/results/rating_alice.jsonl

  # Resume an interrupted session: re-run the same command. Rows
  # already filled in --output are skipped automatically.

Reviewer keys:
  c = correct
  p = false positive (a wrong node was returned)
  n = false negative (a miss when a known node should have matched)
  u = unsure
  s = skip (do not rate this row; come back later)
  q = quit (saves and exits)

The reviewer is asked for a one-sentence reason on every rating
other than skip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_KEY_TO_RATING: dict[str, str] = {
    "c": "correct",
    "p": "false_positive",
    "n": "false_negative",
    "u": "unsure",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _format_row(row: dict, idx: int, total: int) -> str:
    return (
        f"\n[{idx + 1}/{total}] task_id={row['task_id']}\n"
        f"  category:        {row.get('category')}\n"
        f"  task:            {row['task']}\n"
        f"  expected_stage:  {row.get('expected_stage')}\n"
        f"  observed_stage:  {row.get('observed_stage')}\n"
        f"  observed_node:   {row.get('observed_node')}\n"
        f"  llm_rating:      {row.get('llm_rating')}  "
        f"({row.get('llm_reason') or 'no reason'})\n"
    )


def _prompt_rating() -> tuple[str, str] | None:
    """Return (rating, reason) tuple, or None on quit."""
    while True:
        choice = input("  rating [c/p/n/u/s/q]: ").strip().lower()
        if choice == "q":
            return None
        if choice == "s":
            return ("__skip__", "")
        if choice in _KEY_TO_RATING:
            rating = _KEY_TO_RATING[choice]
            reason = input("  one-sentence reason: ").strip()
            return (rating, reason)
        print("  unknown key; please enter c, p, n, u, s, or q.")


def review(
    input_path:  Path,
    output_path: Path,
    rater_id:    str,
) -> int:
    """Walk the rater through unrated rows. Returns count of new ratings."""
    in_rows  = _load_jsonl(input_path)
    out_rows = _load_jsonl(output_path)
    by_id    = {row["task_id"]: row for row in out_rows}

    # Carry forward fields from input for any rows not in output yet.
    merged: list[dict] = []
    for src in in_rows:
        merged.append(dict(by_id.get(src["task_id"], src)))

    new_ratings = 0
    for i, row in enumerate(merged):
        if row.get("human_rating") not in (None, "", "__skip__"):
            continue
        print(_format_row(row, i, len(merged)))
        result = _prompt_rating()
        if result is None:
            print("\nSaving and exiting.")
            break
        rating, reason = result
        if rating == "__skip__":
            print("  skipped.")
            continue
        row["human_rating"]  = rating
        row["human_reason"]  = reason
        row["rater_id"]      = rater_id
        new_ratings += 1
        _save_jsonl(output_path, merged)
        print(f"  saved.")
    else:
        print("\nAll rows reviewed.")
    _save_jsonl(output_path, merged)
    return new_ratings


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive rater CLI for SKG routing.")
    parser.add_argument("--input",     required=True,  type=Path,
                        help="Path to the rating-pass JSONL produced by rating_runner.py.")
    parser.add_argument("--output",    required=True,  type=Path,
                        help="Path to the per-rater output JSONL. Will be created or resumed.")
    parser.add_argument("--rater-id",  required=True,  type=str,
                        help="Identifier for this rater (e.g. 'alice').")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    new_count = review(args.input, args.output, args.rater_id)
    print(f"\nWrote {new_count} new ratings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
