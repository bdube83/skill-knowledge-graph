# Contributing to Skill Knowledge Graph

## Before you start

Read the design document before adding any new feature. SKG has precise
security guarantees that are easy to break with well-intentioned changes.
The invariant: a node cannot access anything it did not explicitly request,
and policy grants always outrank manifest declarations.

Trusted nodes must be Rust-to-WASI. Do not add Python functions as node
implementations. The security property lives in the Wasmtime linker, not in
policy checks at call time.

## Setup

```bash
git clone https://github.com/bdube83/skill-knowledge-graph.git
cd skill-knowledge-graph
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Rust toolchain (for node compilation)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add wasm32-wasip1
```

## Running tests

```bash
# Compile the sample WASI node first
cd nodes/reviewer-ping-draft
cargo build --release --target wasm32-wasip1
cd ../..

python -m pytest tests/ -v
```

All tests must pass before submitting a pull request.

## Pull request checklist

- [ ] Tests pass (`python -m pytest tests/ -v`)
- [ ] New behaviour has corresponding tests
- [ ] No Paystack, employer, or private project references
- [ ] Security properties preserved: policy denial is never bypassed

## Reporting issues

Open a GitHub issue. Include the node manifest, the task string, and the
full traceback if applicable.
