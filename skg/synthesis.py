"""LLM synthesis of Rust WASI node source on router miss (slice 5).

When all routing stages return a miss, the synthesizer calls an LLM to
generate Rust source for a new node. The generated node:
  - Is stored as CANDIDATE status (never ACTIVE on first write).
  - Can only dry-run until it passes all six promotion gates.
  - Is compiled to wasm32-wasip1 by the build step before any execution.

Gate: synthesis + dry-run success rate on 20 test tasks must be >= 70%.
Gate: p95 WASI execution latency must be < 500ms.

Both gates are measured during the replay research slice (slice 6).

The synthesizer does NOT synthesize production nodes autonomously. Every
synthesized node is CANDIDATE until a human (or automated verifier suite)
promotes it via the PromotionEngine.

LLM provider: uses the agent_router from agent-proxy-kit when available.
Falls back to a direct API call via the configured vendor.

Reference:
  designs/proposed/skill-graph-codex-v10/design/skill-knowledge-graph.md
  "Synthesis slice"
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skg.node import CapabilityRequest, Manifest, Node, NodeStatus
from skg.store import NodeStore


_SYNTHESIS_PROMPT_TEMPLATE = """\
You are generating a Rust program that will be compiled to WASI and run as a
trusted SKG node. The program reads a JSON task context from stdin and writes
a JSON result to stdout. It has no network access, no filesystem access, and
no secrets access unless those effects appear in `grants`.

Task type: {task_type}
Header: {header}
Requested capabilities: {capabilities}

Requirements:
1. Read stdin to end. Parse as JSON with keys: task, context, grants.
2. Check that required capabilities are in the grants list. If not, write an
   error JSON and exit.
3. Perform the task using only the information in context.
4. Write to stdout: {{"output": {{...}}, "observed_effects": [...]}}
5. Use serde_json for JSON. No other dependencies unless stated.
6. The program must compile with: cargo build --release --target wasm32-wasip1

Write ONLY the contents of src/main.rs. No explanation, no markdown.
"""


@dataclass
class SynthesisResult:
    """Result of one LLM synthesis attempt."""

    task_type:     str
    success:       bool
    node:          Node | None       = None
    rust_source:   str               = ""
    error:         str               = ""
    llm_tokens:    int               = 0


class Synthesizer:
    """Generates Rust WASI node source from a task description using an LLM.

    The synthesizer does not execute or promote. It writes the generated
    source to the node store as a CANDIDATE node and returns the node ID.
    The caller is responsible for:
      1. Compiling the Rust source to WASM.
      2. Running the PromotionEngine to pass all six gates.
    """

    def __init__(
        self,
        store: NodeStore,
        llm_fn: Any | None = None,
        build_after_synthesis: bool = False,
        wasm_root: Path | None = None,
    ) -> None:
        """
        Parameters
        ----------
        store:
            Node store to persist the synthesized CANDIDATE node.
        llm_fn:
            Callable(prompt: str) -> str. If None, a placeholder is used that
            returns an error. Wire in your LLM provider before calling synthesize().
        build_after_synthesis:
            If True, attempt to compile the generated Rust source to WASM
            immediately. Requires cargo and wasm32-wasip1 target installed.
        wasm_root:
            Where to write the compiled artifact. Defaults to ~/.skg/nodes/.
        """
        self._store    = store
        self._llm_fn   = llm_fn
        self._build    = build_after_synthesis
        self._wasm_root = wasm_root or (Path.home() / ".skg" / "nodes")

    def synthesize(
        self,
        task: str,
        requested_capabilities: list[str] | None = None,
        tags: list[str] | None = None,
        node_id: str | None = None,
    ) -> SynthesisResult:
        """Generate Rust source for a node that handles the given task.

        Returns a SynthesisResult. On success, the node is in the store as
        CANDIDATE and node.id is set.
        """
        if self._llm_fn is None:
            return SynthesisResult(
                task_type=_normalize(task),
                success=False,
                error=(
                    "No LLM function configured. Pass llm_fn=callable to Synthesizer(). "
                    "The callable receives a prompt string and returns generated Rust source."
                ),
            )

        task_type = _normalize(task)
        caps = requested_capabilities or ["text.generate"]
        prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
            task_type=task_type,
            header=task,
            capabilities=", ".join(caps),
        )

        try:
            rust_source = self._llm_fn(prompt)
        except Exception as e:
            return SynthesisResult(task_type=task_type, success=False, error=f"LLM call failed: {e}")

        if not rust_source or not rust_source.strip():
            return SynthesisResult(task_type=task_type, success=False, error="LLM returned empty source.")

        manifest = Manifest(
            task_type=task_type,
            header=task,
            tags=tags or [],
            requested_capabilities=[
                CapabilityRequest(effect=e, adapter="local") for e in caps
            ],
            forbidden_capabilities=[],
            verifiers=[
                {"name": "output_present", "check": "bool(output)"},
            ],
            preconditions=[],
        )

        import uuid
        nid = node_id or f"synth-{uuid.uuid4().hex[:8]}"
        node = Node.new(id=nid, manifest=manifest, source=rust_source)
        node.status = NodeStatus.CANDIDATE
        self._store.put(node)

        compiled = False
        build_error = ""
        if self._build:
            compiled, build_error = self._compile(node, rust_source)

        if self._build and not compiled:
            return SynthesisResult(
                task_type=task_type,
                success=False,
                node=node,
                rust_source=rust_source,
                error=f"Compilation failed: {build_error}",
            )

        return SynthesisResult(
            task_type=task_type,
            success=True,
            node=node,
            rust_source=rust_source,
        )

    def _compile(self, node: Node, source: str) -> tuple[bool, str]:
        """Write source to a temp Cargo project and compile to WASM."""
        wasm_out = self._wasm_root / node.id / "artifact"
        wasm_out.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_dir = tmp_path / "src"
            src_dir.mkdir()
            (src_dir / "main.rs").write_text(source, encoding="utf-8")
            (tmp_path / "Cargo.toml").write_text(
                f'[package]\nname = "node"\nversion = "0.1.0"\nedition = "2021"\n'
                f'\n[[bin]]\nname = "node"\npath = "src/main.rs"\n'
                f'\n[dependencies]\nserde_json = "1"\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                ["cargo", "build", "--release", "--target", "wasm32-wasip1"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return False, result.stderr[-1000:]
            wasm_src = tmp_path / "target" / "wasm32-wasip1" / "release" / "node.wasm"
            if wasm_src.exists():
                import shutil
                shutil.copy(wasm_src, wasm_out / "node.wasm")
                return True, ""
            return False, "WASM artifact not produced."


def _normalize(task: str) -> str:
    import re
    return re.sub(r"\s+", "_", task.lower().strip())[:80]
