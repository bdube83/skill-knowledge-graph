# Multi-rater workflow for SKG routing precision

This is the workflow for filling in the `human_rating` column on the
SKG routing-decision form and computing inter-rater agreement
(Cohen kappa, Krippendorff alpha). One human-rater pass populates the
form; two or more passes give the agreement metrics the paper's
Section 7.7 needs.

## Files

- `eval/rating_runner.py`: builds the form. Runs an LLM stand-in pass
  that fills `llm_rating` per row. Output: `eval/results/rating_pass.jsonl`.
- `eval/rating_review_cli.py`: interactive CLI. One rater walks
  through every row, presses a key per task, and writes a per-rater
  output file with `human_rating` filled.
- `eval/rating_agreement.py`: takes 2+ rater output files and reports
  per-pair Cohen kappa, overall Krippendorff alpha, per-rater label
  distribution, and the list of disagreed task ids.

## End-to-end

### Step 1, build the form (already done; rebuilds on demand)

```bash
.venv/bin/python eval/rating_runner.py \
  --corpus eval/corpus.jsonl \
  --out eval/results/rating_pass.jsonl
```

This reads the corpus, optionally fills `llm_rating` per row via
gpt-4o-mini, and writes `rating_pass.jsonl`. The `human_rating` field
is `null` on every row.

### Step 2, each rater walks the CLI

Rater 1, called `alice`:

```bash
.venv/bin/python eval/rating_review_cli.py \
  --input  eval/results/rating_pass.jsonl \
  --rater-id alice \
  --output eval/results/rating_alice.jsonl
```

Rater 2, called `bob`:

```bash
.venv/bin/python eval/rating_review_cli.py \
  --input  eval/results/rating_pass.jsonl \
  --rater-id bob \
  --output eval/results/rating_bob.jsonl
```

Per-task keys:
- `c` correct
- `p` false positive (a wrong node was returned)
- `n` false negative (a miss when a known node should have matched)
- `u` unsure
- `s` skip (do not rate this row right now)
- `q` quit (saves and exits)

After every rating, the CLI saves to disk, so a rater can quit any
time and resume by re-running the same command.

### Step 3, compute agreement

```bash
.venv/bin/python eval/rating_agreement.py \
  eval/results/rating_alice.jsonl \
  eval/results/rating_bob.jsonl \
  --output eval/results/rating_agreement.json
```

The output includes:
- `pair_cohen_kappa`: kappa for every pair of raters.
- `krippendorff_alpha`: alpha across all raters (nominal data).
- `rater_label_distribution`: count of each label per rater.
- `disagreement_task_ids`: tasks where >=2 raters used different
  labels. Useful for routing those rows to a third reviewer.

## Interpretation

Cohen kappa (Landis and Koch 1977):
- < 0.00 poor
- 0.00 to 0.20 slight
- 0.21 to 0.40 fair
- 0.41 to 0.60 moderate
- 0.61 to 0.80 substantial
- 0.81 to 1.00 almost perfect

Krippendorff alpha (Krippendorff 2004):
- alpha >= 0.800 reliable
- alpha >= 0.667 acceptable for tentative conclusions
- alpha <  0.667 not acceptable

The paper's Section 7.4 plan targets inter-rater kappa above 0.7. If
you land below that, the disagreed-task list in the agreement report
is the right place to send a third reviewer.

## Practical notes

- 200 tasks at ~10 seconds per rating is roughly 35 minutes per
  reviewer. The `s` (skip) key lets a reviewer batch-skip uncertain
  rows and revisit them at the end.
- The CLI auto-saves after every rating, so an interrupted session
  resumes cleanly. No work is lost if the terminal closes.
- The output JSONL files are gitignored under `eval/results/`. To
  publish the agreement report (not the raw labels), copy
  `eval/results/rating_agreement.json` to a tracked path before
  committing, or paste the numbers into the paper.
