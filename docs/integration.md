# Skill Knowledge Graph

Capability-token enforcement at the WASM import layer for
LLM-synthesized procedures. SKG sits between an LLM agent and the
operations it wants to perform, so the runtime, not the manifest,
decides what each call may do.

This README is the integration guide. The accompanying paper at
[`paper.md`](paper.md) covers the design, evaluation methodology,
and measurements; the reproducibility steps for those measurements
live at [`docs/paper-reproduction.md`](docs/paper-reproduction.md).

## Why use SKG

- You want an LLM agent to call host functions (HTTP, git, local
  files, secrets, external messages) under capability-token
  enforcement instead of trust-the-manifest semantics.
- You want a routing layer that picks a verified Rust-to-WASI
  procedure for known task families before falling back to the
  LLM, so recurring work runs faster and (above ~120 input tokens
  per task) cheaper.
- You want every external operation a node performs to be auditable
  via append-only attestation files under `~/.skg/`.

If your agent always falls back to the LLM, SKG adds latency without
saving tokens. The break-even per-task LLM input is the routing
header cost, ~120 tokens. See `paper.md` Section 7.4 for measured
numbers.

## Install

Python 3.13 is required (the package targets it; `pyproject.toml`
classifiers).

```bash
git clone https://github.com/bdube83/skill-knowledge-graph.git
cd skill-knowledge-graph
python3.13 -m venv .venv
.venv/bin/pip install -e .
```

Optional but useful at integration time: an OpenAI key (or another
LLM key your code uses) at `~/.agent-proxy/openai-key`. SKG itself
does not call an LLM; the integration helper at
`skg.integrations.agent_proxy` calls one only on a miss when the
caller passes `synthesize_on_miss=True`.

## Quick start

The simplest integration: route a task through SKG, prepend the
routing summary to your prompt, fall back to your LLM on a miss.

```python
from skg.integrations.agent_proxy import route_proposal, build_prompt_prefix

result = route_proposal("Draft a reviewer ping for PR review")

if result.hit:
    prefix = build_prompt_prefix(result)
    prompt = prefix + "\n" + your_existing_prompt
    response = your_llm.chat(prompt)
else:
    response = your_llm.chat(your_existing_prompt)
```

`RouteResult` carries `hit`, `node`, `stage`, `grant`. On a miss
the caller falls through; on a hit you can either use the node's
header as a prompt prefix (above) or execute the WASM artifact
directly (below).

## Direct execution under capability-token enforcement

```python
from pathlib import Path
from skg.wasmtime_launcher import WasmtimeRuntime

rt = WasmtimeRuntime()
result = rt.execute(
    wasm_path=Path("nodes/reviewer-ping-draft/target/wasm32-wasip1/release/reviewer_ping_draft.wasm"),
    node_id="reviewer-ping-draft",
    task="draft reviewer ping",
    context={"pr_number": 42, "repo": "x/y", "author": "alice", "reviewers": ["bob"]},
    granted_effects=["text.generate"],
)
print(result.success, result.output, result.observed_effects)
```

The runtime wires only the WASI and `skg.*` host imports the grant
set permits. A node that imports a host function it was not granted
fails at instantiate-time, not at call-time. See `skg/cap_to_imports.py`
for the effect-to-import mapping and `skg/host_imports.py` for the
per-call grant validation.

## Effect classes

The kernel understands twelve generic effect classes plus
`text.generate`. They live in `skg/effects.py`:

```
local.read   local.write
network.read network.write
external.draft external.send (approval-gated)
browser.read browser.write
git.read     git.write     (approval-gated)
secret.read
production.write           (approval-gated)
text.generate
```

Approval-gated effects require a non-zero approval token at call
time (`skg/cap_to_imports.py:APPROVAL_HOST`). The runtime returns
`ERRNO_DENIED` (13) when the token is zero.

## Adding a new node

A node is a Rust crate compiled to WASI plus a manifest and a header.
Skeleton:

```
nodes/<node-id>/
  Cargo.toml
  manifest.yaml         # task_type, header, tags, requested_capabilities
  src/main.rs           # reads task+context+grants from stdin, writes JSON to stdout
```

The three reference nodes under `nodes/` (reviewer-ping-draft,
git-summary, doc-update) are the working templates. Build with:

```bash
cd nodes/<node-id>
cargo build --release --target wasm32-wasip1
```

The .wasm artifact lands at
`nodes/<node-id>/target/wasm32-wasip1/release/<node>.wasm`. Place
the manifest at `~/.skg/nodes/<node-id>/manifest.yaml` (or pass
`store_path` to `SKG()` to use a different root) and call
`skg.add_node(...)`.

## Routing pipeline

The router's four stages are `exact`, `fts`, `vector`, and
`graph composition` (`skg/router.py`). On a miss the router
returns `RouteResult(hit=False, ...)`; the caller decides whether
to fall back to an LLM or to ask the user.

```python
from skg.graph import SKG

skg = SKG()
result = skg.route("Draft a reviewer ping", {"pr_number": 1})
print(result.hit, result.stage, result.node and result.node.id)
```

## Manifest scopes (Phase 3d)

Per-effect URL patterns and path scopes go in the manifest under
`requested_capabilities`. The launcher reads them when you pass
`manifest_path=` to `execute(...)`:

```yaml
requested_capabilities:
  - effect:        network.read
    url_pattern:   "https://api.allowed.example/*"
  - effect:        local.read
    path_scope:    "/workspace/data"
```

When `manifest_path` is omitted the launcher mints wildcard
scopes for backward compatibility.

## Adversarial test surface

`tests/test_security.py`, `tests/test_containment_matrix.py`,
`tests/test_scoped_enforcement.py`, `tests/test_confused_deputy.py`,
and `tests/test_local_wasi.py` together exercise four attack
classes (manifest lies, path escape, WASI introspection, confused
deputy) end-to-end through real WASM execution. Use them as the
contract specification for the runtime gate.

```bash
.venv/bin/python -m pytest tests/ -q
# 221 tests as of commit fa0644c
```

## Code layout

| Path | What lives there |
|---|---|
| `skg/`           | The package. Kernel, router, runtime, host imports, baselines. |
| `skg/integrations/agent_proxy.py` | Drop-in helper for agent-proxy-style callers. |
| `skg/baselines/` | The 4 baseline runtimes (declared, flow_registry, semantic_cache, flat_library). |
| `skg/host_adapters.py` | Real adapter implementations (HTTP, git, secrets, drafts, audit). |
| `nodes/`         | Rust crates that compile to WASI. Three reference nodes ship. |
| `eval/`          | Corpus, runners, and statistical scripts. Outputs land under `eval/results/` (gitignored). |
| `tests/`         | 221 pytest cases. |
| `figures/`       | Architecture diagram source (`.dot`) and rendered PNG/SVG/PDF. |
| `docs/`          | Reproducibility guide for the paper. |

## Configuration

SKG state lives at `~/.skg/`:

- `~/.skg/skg.db` (SQLite node store)
- `~/.skg/nodes/<node-id>/` (manifest, source, attestations per node)
- `~/.skg/secrets/<name>` (secret files for `skg.secret_read`)
- `~/.skg/drafts/`, `~/.skg/sent/`, `~/.skg/browser_requests/`,
  `~/.skg/browser_writes/`, `~/.skg/production_log/` (audit
  directories written by the host adapters)

To override the root, pass `store_path=Path(...)` to `SKG(...)`
and `wasm_path=...` to `WasmtimeRuntime.execute(...)`.

## Compatibility

- Python 3.13.
- Wasmtime Python bindings 40.x (pinned in `pyproject.toml`).
- WASI snapshot preview1 (the runtime gate's reduction argument
  cites this version; component model migration is future work).

## Paper

The accompanying paper (markdown source at [`paper.md`](paper.md),
PDF at [`paper.pdf`](paper.pdf)) describes the architecture, the
formal cost model, the evaluation methodology, and the measurements.
The H3 (capability-token enforcement) hypothesis is supported with
a 13/13 vs 5/13 adversarial-corpus differential against a
declared-capability baseline. The H1 (50% token reduction) hypothesis
is not supported at the measured scales; SKG saves ~36% on a
larger-context corpus, falling short of the 50% threshold.

## Citation

```bibtex
@misc{dube2026skg,
  title  = {Skill Knowledge Graph: Capability-Token Enforcement for LLM-Synthesized Procedures},
  author = {Dube, Bongani},
  year   = {2026},
  note   = {Paystack. Draft},
  url    = {https://github.com/bdube83/skill-knowledge-graph}
}
```

## License

Apache 2.0. See `LICENSE`.

## Reproducibility

See [`docs/paper-reproduction.md`](docs/paper-reproduction.md) for
how to recreate the paper's measurements end-to-end.
