# Skill Knowledge Graph (SKG)

A capability-governed procedure reuse system for LLM agents.

SKG stores reusable agent procedures as nodes in a typed graph. When an agent receives a task, the router checks the graph first — exact match, full-text search, vector similarity — before falling back to an LLM call. Matched nodes carry a policy-issued grant that limits what the procedure can do. Trusted nodes run as WASI components under Wasmtime; the runtime enforces granted handles at the import level, not at call time.

## The core rule

```
nodes request capabilities
policy grants capabilities
runtime enforces granted handles only
```

A node cannot access the network, filesystem, secrets, or external services by declaring them in its own manifest. Policy grants a run-scoped subset as handles. Wasmtime enforces the handles. Nothing else does.

## Status

**Slices 1 and 2 are complete and tested.** Slices 3–6 are in progress.

| Slice | What it covers | Status |
|---|---|---|
| 1. Kernel | Manifest parsing, effect algebra, policy grants, node store, attestation | Done |
| 2. Router | Exact lookup, SQLite FTS5, Qdrant vector, graph expansion | Done (Qdrant: local-hash-v1 embedding) |
| 3. Runtime | Wasmtime execution, Rust-to-WASI nodes, grant handles | Done (sample node compiled and tested) |
| 4. Verifier | Promotion gates, dry-run contracts, stale handling | In progress |
| 5. Synthesis | LLM Rust source generation on miss | Planned |
| 6. Replay | Baseline evaluation against 200-task corpus | Planned |

## Quick start

```bash
# Install (Python 3.11+, Rust required for node compilation)
pip install skill-knowledge-graph

# Route a task
skg route "draft reviewer ping for PR #42"

# Add a node from a manifest file
skg node add path/to/manifest.yaml

# Inspect a node
skg node inspect reviewer-ping-draft

# Check store health
skg doctor
```

## Manifest format

```yaml
task_type: reviewer_ping_draft
header: "Draft a reviewer-ping message for a pull request."
tags: [git, review, communication]
requested_capabilities:
  - effect: text.generate
    adapter: local
forbidden_capabilities: []
preconditions: []
verifiers: []
```

## Building a trusted node (Rust-to-WASI)

Trusted nodes are Rust programs compiled to the `wasm32-wasip1` target. They read a JSON task context from stdin and write a JSON result to stdout. The Wasmtime launcher passes only the granted capability handles as imports; anything not granted is absent from the linker.

```bash
# Add the WASI target (one-time setup)
rustup target add wasm32-wasip1

# Build the example node
cd nodes/reviewer-ping-draft
cargo build --release --target wasm32-wasip1
```

Node input (stdin):
```json
{
  "task": "draft reviewer ping",
  "context": { "pr_number": 42, "repo": "example/repo", "author": "alice", "reviewers": ["bob"] },
  "grants": ["text.generate"]
}
```

Node output (stdout):
```json
{
  "output": { "message": "Hi @bob, could you review PR #42 in `example/repo`?" },
  "observed_effects": ["text.generate"]
}
```

## Policy

The default policy allows `text.generate`, `git.read`, and most local effects. It denies `external.send`, `production.write`, and `secret.read`. Override by placing a `policy.yaml` at `~/.skg/policy.yaml`:

```yaml
rules:
  - effect: git.write
    adapter: github
    allow: true
  - effect: external.send
    adapter: "*"
    allow: false
default: deny
```

## agent-proxy-kit integration

```python
try:
    from skg.integrations.agent_proxy import route_proposal
    _SKG_AVAILABLE = True
except ImportError:
    _SKG_AVAILABLE = False

# In your proposal handler:
if _SKG_AVAILABLE:
    result = route_proposal(proposal_text)
    if result.hit:
        # Use result.node and result.grant
        ...
```

## Running the tests

```bash
# Install deps
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Compile the sample WASI node (required for test_wasmtime.py)
cd nodes/reviewer-ping-draft
cargo build --release --target wasm32-wasip1
cd ../..

# Run all tests
python -m pytest tests/ -v
```

## Design and research

The full design is in `designs/proposed/skill-graph-codex-v10/` in the [agent-proxy-kit](https://github.com/bdube83/agent-proxy-kit) repository. The research paper draft and experiment execution plan are there too.

## License

Apache 2.0. See [LICENSE](LICENSE).
