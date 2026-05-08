"""Run Baseline A (LLM-only) over the larger-context corpus.

Reuses the runner in eval.baseline_a_runner. Points it at
eval/corpus_large.jsonl and writes its outputs to a parallel set of
result files so the original Baseline A measurement on eval/corpus.jsonl
is not overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.baseline_a_runner import main as run_baseline_a


def run() -> dict:
    repo        = Path(__file__).parent.parent
    corpus_path = repo / "eval" / "corpus_large.jsonl"
    out_dir     = repo / "eval" / "results"

    # The runner writes baseline_a_report.json and baseline_a_tasks.jsonl
    # by default. Stash any existing originals, run, and rename outputs.
    report_path  = out_dir / "baseline_a_report.json"
    tasks_path   = out_dir / "baseline_a_tasks.jsonl"
    backup_report = out_dir / "_orig_baseline_a_report.json"
    backup_tasks  = out_dir / "_orig_baseline_a_tasks.jsonl"

    if report_path.exists() and not backup_report.exists():
        backup_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    if tasks_path.exists() and not backup_tasks.exists():
        backup_tasks.write_text(tasks_path.read_text(encoding="utf-8"), encoding="utf-8")

    run_baseline_a(corpus_path, out_dir)

    # Rename to the _large_ variants so the original results stay intact.
    large_report = out_dir / "baseline_a_large_report.json"
    large_tasks  = out_dir / "baseline_a_large_tasks.jsonl"
    large_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    large_tasks.write_text(tasks_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Restore original files from backup.
    if backup_report.exists():
        report_path.write_text(backup_report.read_text(encoding="utf-8"), encoding="utf-8")
        backup_report.unlink()
    if backup_tasks.exists():
        tasks_path.write_text(backup_tasks.read_text(encoding="utf-8"), encoding="utf-8")
        backup_tasks.unlink()

    return json.loads(large_report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    rep = run()
    print(json.dumps(rep, indent=2))
