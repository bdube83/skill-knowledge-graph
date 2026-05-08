"""Baseline B: existing flow registry.

Section 7.2 of the paper defines Baseline B as a system that
pattern-matches the task string against a registry of prior successful
task to response pairs. On exact match the runtime replays the stored
response. On miss it falls through to a stub LLM that returns a
placeholder.

The class implements the same `execute(...)` signature as
`skg.wasmtime_launcher.WasmtimeRuntime` so the eval harness can call
every baseline through the same contract. The `wasm_path` argument is
accepted for signature parity and ignored. The registry is a per-runtime
dict initialised from the constructor's `registry` list.

Reference:
  designs/in-progress/skill-graph-codex-v10/paper-draft.md  Section 7.2
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..wasmtime_launcher import WasmRunResult


__all__ = ["FlowRegistryRuntime", "WasmRunResult"]


DEFAULT_TIMEOUT_MS = 5_000


def _stub_llm(task: str, context: dict[str, Any]) -> str:
    """Stand-in for a planning LLM call.

    The paper's Baseline B falls through to a planning LLM on a
    registry miss. The eval harness needs deterministic, offline output.
    The stub returns a fixed placeholder string that names the task and
    a hash of the context. No network call is made.
    """
    ctx_hash = hash(repr(sorted(context.items()))) & 0xFFFF
    return f"stub-llm-response for {task!r} ctx#{ctx_hash:04x}"


class FlowRegistryRuntime:
    """Pattern-matching baseline keyed on the task string.

    The constructor accepts a list of `(task, response)` tuples. The
    runtime stores them in a dict keyed by the exact task string. Each
    `execute()` call looks up the task. A hit returns the stored
    response with `success=True`. A miss invokes the stub LLM and
    returns the stub output with `success=True` and an empty
    `observed_effects` list.

    The `wasm_path`, `granted_effects`, and `dry_run` arguments are
    accepted for signature parity with `WasmtimeRuntime.execute` and
    ignored. They appear in the harness call site without per-baseline
    branching.
    """

    def __init__(
        self,
        registry: list[tuple[str, str]] | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        self._registry: dict[str, str] = dict(registry or [])
        self._timeout_ms = timeout_ms

    def execute(
        self,
        wasm_path: Path | str,
        node_id: str,
        task: str,
        context: dict[str, Any],
        granted_effects: list[str],
        dry_run: bool = False,
    ) -> WasmRunResult:
        """Look up the task in the registry; fall through to the stub LLM."""
        start = time.monotonic()

        if task in self._registry:
            response = self._registry[task]
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            return WasmRunResult(
                node_id=node_id,
                success=True,
                output={"response": response, "source": "registry"},
                error="",
                duration_ms=duration_ms,
                observed_effects=[],
            )

        response = _stub_llm(task, context)
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        return WasmRunResult(
            node_id=node_id,
            success=True,
            output={"response": response, "source": "stub_llm"},
            error="",
            duration_ms=duration_ms,
            observed_effects=[],
        )
