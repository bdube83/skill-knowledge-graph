# Skill Knowledge Graph: Capability-Token Enforcement for LLM-Synthesized Procedures

Reproducibility guide for the SKG paper. Clone, install, run the listed
commands, and the numbers in `paper.md` come back.

## Table of contents

1. [What is in the repo](#what-is-in-the-repo)
2. [Quick install](#quick-install)
3. [Reproducing the measurements](#reproducing-the-measurements)
4. [Reproducing the figures](#reproducing-the-figures)
5. [Multi-rater workflow](#multi-rater-workflow)
6. [Test suite](#test-suite)
7. [Code layout](#code-layout)
8. [License and citation](#license-and-citation)
9. [Status notes and gotchas](#status-notes-and-gotchas)

## What is in the repo

`skg/` is the system: router, policy, Wasmtime launcher, and the four
baseline runtimes (B, C, D, E) used in the paper's containment matrix.
`paper.md` (also rendered as `paper.pdf` and `paper.html`) is the writeup.
`figures/` holds the architecture diagram source and the rendered PNGs.

## Quick install

Python 3.13 in a virtualenv. Python 3.11 and 3.12 are also supported per
`pyproject.toml`.

```bash
git clone https://github.com/bdube83/skill-knowledge-graph
cd skill-knowledge-graph
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies pulled by `pyproject.toml`: `pyyaml`, `wasmtime`,
`qdrant-client`, `openai`, `krippendorff`, `numpy`. Dev extras add
`pytest` and `pytest-cov`.

OpenAI key. The runners that call `gpt-4o-mini` read the key from
`~/.agent-proxy/openai-key` (a one-line file containing the secret),
not from `OPENAI_API_KEY`. Create the file before running any
LLM-touching command:

```bash
mkdir -p ~/.agent-proxy
printf 'sk-...' > ~/.agent-proxy/openai-key
chmod 600 ~/.agent-proxy/openai-key
```

Rust toolchain (only for the WASI sample node used in `test_wasmtime.py`
and `tests/test_local_wasi.py`):

```bash
rustup target add wasm32-wasip1
cd nodes/reviewer-ping-draft && cargo build --release --target wasm32-wasip1 && cd ../..
```

## Reproducing the measurements

Every runner writes to `eval/results/`. That directory is gitignored;
the runners recreate it.

### Step 0: build the synthetic corpora

`eval/corpus.jsonl` and `eval/corpus_large.jsonl` are gitignored. Generate
them deterministically:

```bash
python eval/corpus_builder.py --n 200 --out eval/corpus.jsonl --seed 42
python eval/corpus_builder_large.py
```

Both builders use a fixed seed. Identical inputs always produce identical
output.

### Step 1: SKG hit rate on the small corpus

Routes every task in `eval/corpus.jsonl` through the SKG router and writes
`eval/results/report.json` with hit rate, p50/p95 latency, and the
estimated token-savings bar.

```bash
python -m eval.baseline_runner --corpus eval/corpus.jsonl --out eval/results
```

Paper anchor: Section 7 Tables 1 and 3, the 80% hit-rate plateau on the
3-node corpus.

### Step 2: real Baseline A on the small corpus

Issues one `gpt-4o-mini` call per task with `temperature=0`, records token
counts and latency, and writes `eval/results/baseline_a_report.json` plus
`baseline_a_tasks.jsonl`. Costs about USD 0.02 at list pricing.

```bash
python eval/baseline_a_runner.py
```

Paper anchor: Section 7 Table 2 row A. The measured 90.32 mean input
tokens per task is what the small-corpus blockquote in the abstract
quotes.

### Step 3: held-out 5-seed bootstrap CIs

Splits the corpus 80/20 across seeds 1 through 5, runs Baseline A and the
SKG router on each holdout, and writes per-seed records under
`eval/results/seeded_runs/seed_{1..5}.json` plus the aggregate at
`eval/results/seeded_aggregated.json`. The aggregate carries the
percentile bootstrap (1000 iterations) 95% CIs from `eval/bootstrap.py`.

```bash
python -m eval.seeded_runner
```

Paper anchor: Section 7.5 statistical analysis plan.

### Step 4: larger-context corpus (the H1 result)

Run Baseline A on `eval/corpus_large.jsonl`, then compute the SKG estimate
against it:

```bash
python eval/baseline_a_large.py
python eval/skg_large_estimate.py
```

Outputs `eval/results/baseline_a_large_report.json` and
`eval/results/skg_large_estimate.json`. The estimate uses the formula in
Section 6.1: hits pay 120 routing-header tokens, misses pay the full
Baseline A input mean. With an 80% hit rate that lands at 387.62 mean
input tokens per SKG task against 1458.12 for Baseline A: 73.4% input-token
reduction. This is the H1 result that holds at the larger-context scale.

### Step 5: adversarial-corpus differential

The differential between SKG (treatment T) and the declared-capability
baseline (E) is exercised by the test suite, not a runner:

```bash
pytest tests/test_containment_matrix.py -q
pytest tests/test_adversarial_corpus.py -q
pytest tests/test_confused_deputy.py -q
```

Paper anchor: Section 7.4 Table 4. T contains 13/13 attacks across 3 classes;
E contains 5/13.

## Reproducing the figures

Figure 1 (architecture) is a Graphviz dot source. Render the vector copy
that ships in the paper:

```bash
dot -Tpdf figures/fig1_architecture.dot -o figures/fig1_architecture.pdf
dot -Tsvg figures/fig1_architecture.dot -o figures/fig1_architecture.svg
dot -Tpng figures/fig1_architecture.dot -o figures/fig1_architecture.png
```

Figures 2 through 4 come from the baseline runner. Step 1 above writes
the PDFs to `eval/results/figures/` and the existing PNGs in `figures/`
were taken from a prior run on the same corpus seed.

| Paper figure | Source file in repo | Generated by |
|---|---|---|
| Figure 1 | `figures/fig1_architecture.dot` | Graphviz `dot` |
| Figure 2 | `figures/fig1_hit_rate_cdf.png` | `eval.baseline_runner` |
| Figure 3 | `figures/fig3_token_savings.png` | `eval.baseline_runner` |
| Figure 4 | `figures/fig2_latency_violin.png` | `eval.baseline_runner` |

## Multi-rater workflow

The Cohen kappa and Krippendorff alpha numbers in Section 7.7 come from
the rater CLI. Three steps: build the form, walk it once per rater,
compute agreement. Full instructions and key bindings are in
[`eval/RATING_WORKFLOW.md`](eval/RATING_WORKFLOW.md).

## Test suite

```bash
pytest tests/ -q
```

209 tests at commit `622bbc8`. The Wasmtime tests skip cleanly if the
`reviewer-ping-draft` node has not been built. Untracked work in
progress on `main` adds extra files; the 209 number is the clean-tree
count.

## Code layout

| Directory | One-liner |
|---|---|
| `skg/` | The system: router, policy, Wasmtime launcher, baseline runtimes, CLI. |
| `nodes/` | Trusted Rust-to-WASI node sources (`reviewer-ping-draft`, `git-summary`, `doc-update`). |
| `eval/` | Corpus builders, runners, rater workflow, bootstrap, seeded splits. |
| `tests/` | 209 pytest tests covering kernel, router, runtime, host adapters, and the adversarial differential. |
| `figures/` | Architecture diagram source plus the three rendered evaluation PNGs. |

## License and citation

Apache 2.0. See `LICENSE`.

```bibtex
@misc{dube2026skg,
  title  = {Skill Knowledge Graph: Capability-Token Enforcement for LLM-Synthesized Procedures},
  author = {Dube, Bongani},
  year   = {2026},
  note   = {Paystack. Draft},
  url    = {https://github.com/bdube83/skill-knowledge-graph}
}
```

## Status notes and gotchas

`eval/results/` is gitignored. Every runner above recreates what it needs.
Do not expect the directory to exist on a fresh clone.

The seeded runner's reported SKG hit rate is conditional on a populated
local node store at `~/.skg/skg.db`. On a fresh machine that store is
empty and the routing hit rate comes back at 0%. Provision the three
WASI nodes (build them, then add their manifests via `skg node add`)
before relying on the hit-rate number. Baseline A measurements are
independent of the local store.

The 73.4% token-reduction headline holds on the larger-context corpus
(`eval/corpus_large.jsonl`, 500 to 1500 tokens of context per task), not
on the small synthetic corpus. The crossover is the 120-token routing
header: when per-task LLM input falls below the header cost, SKG burns
more input tokens than LLM-only. The small corpus sits below that
crossover by design (mean 90.32 tokens). The paper reports both regimes.
