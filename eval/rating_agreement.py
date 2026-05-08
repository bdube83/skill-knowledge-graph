"""Compute inter-rater agreement metrics for the SKG rating pass.

Takes two or more rater output files (each is a JSONL produced by
`rating_review_cli.py`) and reports:
  - Per-pair Cohen kappa (for every unordered pair of raters)
  - Krippendorff alpha across all raters (nominal data)
  - Per-rater label distribution
  - Per-task disagreement count (for spot-checking high-disagreement
    rows that may need a third reviewer)

Cohen kappa interpretation (Landis & Koch 1977 conventions):
  < 0.00  poor
  0.00-0.20 slight
  0.21-0.40 fair
  0.41-0.60 moderate
  0.61-0.80 substantial
  0.81-1.00 almost perfect

Krippendorff alpha interpretation (Krippendorff 2004):
  alpha >= 0.800 reliable
  alpha >= 0.667 acceptable for tentative conclusions
  alpha <  0.667 not acceptable

Usage:
  .venv/bin/python eval/rating_agreement.py \
    eval/results/rating_alice.jsonl \
    eval/results/rating_bob.jsonl \
    --output eval/results/rating_agreement.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import krippendorff


_VALID_LABELS: set[str] = {
    "correct",
    "false_positive",
    "false_negative",
    "unsure",
}


def _load(path: Path) -> dict[str, str]:
    """Return {task_id: human_rating} for rows with a valid rating."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rating = row.get("human_rating")
        if rating in _VALID_LABELS:
            out[row["task_id"]] = rating
    return out


def cohen_kappa(a: dict[str, str], b: dict[str, str]) -> tuple[float, int]:
    """Pairwise Cohen kappa over the intersection of rated task ids.

    Returns (kappa, n) where n is the number of co-rated tasks. Returns
    (nan, 0) when there are no co-rated tasks.
    """
    common = sorted(set(a) & set(b))
    n = len(common)
    if n == 0:
        return float("nan"), 0
    labels = sorted(_VALID_LABELS)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix: list[list[int]] = [[0] * len(labels) for _ in labels]
    for tid in common:
        i = label_to_idx[a[tid]]
        j = label_to_idx[b[tid]]
        matrix[i][j] += 1
    po = sum(matrix[i][i] for i in range(len(labels))) / n
    row_totals = [sum(matrix[i]) for i in range(len(labels))]
    col_totals = [sum(matrix[i][j] for i in range(len(labels))) for j in range(len(labels))]
    pe = sum((row_totals[i] * col_totals[i]) / (n * n) for i in range(len(labels)))
    if pe == 1.0:
        return float("nan"), n
    return (po - pe) / (1 - pe), n


def krippendorff_alpha(rater_maps: list[dict[str, str]]) -> tuple[float, int]:
    """Krippendorff alpha across all raters on nominal labels.

    Returns (alpha, n_units) where n_units is the count of task ids
    rated by at least two raters.
    """
    all_ids = sorted({tid for m in rater_maps for tid in m})
    label_to_idx = {label: i for i, label in enumerate(sorted(_VALID_LABELS))}
    units: list[list[int | None]] = []
    for tid in all_ids:
        row = []
        rated_by = 0
        for m in rater_maps:
            label = m.get(tid)
            if label is None:
                row.append(None)
            else:
                row.append(label_to_idx[label])
                rated_by += 1
        if rated_by >= 2:
            units.append(row)
    if not units:
        return float("nan"), 0
    distinct = {v for unit in units for v in unit if v is not None}
    if len(distinct) < 2:
        return 1.0, len(units)
    transposed = list(map(list, zip(*units)))
    alpha = krippendorff.alpha(
        reliability_data=transposed,
        level_of_measurement="nominal",
    )
    return float(alpha), len(units)


def label_distribution(m: dict[str, str]) -> dict[str, int]:
    counts = Counter(m.values())
    return {k: counts.get(k, 0) for k in sorted(_VALID_LABELS)}


def disagreement_counts(rater_maps: list[dict[str, str]]) -> dict[str, int]:
    """For each task id rated by >=2 raters, count distinct labels."""
    out: dict[str, int] = {}
    all_ids = sorted({tid for m in rater_maps for tid in m})
    for tid in all_ids:
        labels = {m[tid] for m in rater_maps if tid in m}
        if len(labels) >= 2:
            out[tid] = len(labels)
    return out


def report(rater_paths: list[Path]) -> dict:
    rater_maps = [_load(p) for p in rater_paths]
    rater_ids  = [p.stem for p in rater_paths]

    pair_kappa: dict[str, dict] = {}
    for (i, a), (j, b) in itertools.combinations(enumerate(rater_maps), 2):
        kappa, n = cohen_kappa(a, b)
        pair_kappa[f"{rater_ids[i]}__{rater_ids[j]}"] = {
            "cohen_kappa":  kappa,
            "common_tasks": n,
        }

    alpha, n_units = krippendorff_alpha(rater_maps)

    return {
        "raters":              rater_ids,
        "rater_label_distribution": {
            rater_ids[i]: label_distribution(rater_maps[i])
            for i in range(len(rater_ids))
        },
        "pair_cohen_kappa":    pair_kappa,
        "krippendorff_alpha":  alpha,
        "krippendorff_n_units": n_units,
        "disagreement_task_ids": disagreement_counts(rater_maps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inter-rater agreement for SKG rating files.")
    parser.add_argument("rater_files", nargs="+", type=Path,
                        help="Two or more rater JSONL files.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON output path. Defaults to stdout.")
    args = parser.parse_args()

    if len(args.rater_files) < 2:
        print("Need at least two rater files for agreement metrics.")
        return 1

    result = report(args.rater_files)
    body   = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
        print(f"Wrote agreement report to {args.output}")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
