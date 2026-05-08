"""Differential matrix across all paper baselines.

The paper's Section 7.2 names six systems: A (direct LLM), B (flow
registry), C (semantic cache), D (flat tool library), E (declared
capability), and T (SKG treatment). This test feeds the same task
through every system and reports the per-system success flag, output
dict, and observed effects. Tables 2 and 3 in the paper consume that
shape.

The test asserts structural invariants only:

  * every system returns a `WasmRunResult`;
  * every result includes the six fields the harness expects.

Specific success rates and outputs are evaluation outcomes, not test
assertions. Keep this test green as the runtimes evolve.

Baseline A is stubbed inline. The paper's full A baseline calls a
planning LLM. That call is gated out of unit tests; the stub here
returns a deterministic placeholder with the same `WasmRunResult`
shape the other runtimes use.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pytest

from skg.baselines import (
    DeclaredCapabilityRuntime,
    FlatToolLibraryRuntime,
    FlowRegistryRuntime,
    SemanticCacheRuntime,
)
from skg.wasmtime_launcher import WasmRunResult, WasmtimeRuntime


PROJECT_ROOT = Path(__file__).parent.parent

DOC_UPDATE_ARTIFACT = (
    PROJECT_ROOT / "nodes" / "doc-update" / "target"
    / "wasm32-wasip1" / "release" / "doc-update.wasm"
)


TASK = "update doc section"
CONTEXT: dict[str, Any] = {
    "doc_path":           "README.md",
    "section_title":      "Setup",
    "existing_content":   "Run pip install.",
    "change_description": "Mention the venv.",
}
GRANTS = ["text.generate"]


def _baseline_a_stub_run(
    wasm_path: Path | str,
    node_id: str,
    task: str,
    context: dict[str, Any],
    granted_effects: list[str],
    dry_run: bool = False,
) -> WasmRunResult:
    """Direct LLM stub that mirrors the runtime contract.

    A real Baseline A would call a planning LLM per task. The unit
    suite cannot make network calls, so the stub returns a fixed
    placeholder with the standard result shape.
    """
    start = time.monotonic()
    response = f"direct-llm-stub for {task!r}"
    duration_ms = round((time.monotonic() - start) * 1000, 2)
    return WasmRunResult(
        node_id=node_id,
        success=True,
        output={"response": response, "source": "direct_llm_stub"},
        error="",
        duration_ms=duration_ms,
        observed_effects=[],
    )


SystemRunner = Callable[..., WasmRunResult]


def _build_systems() -> dict[str, SystemRunner]:
    seed_pairs = [(TASK, "cached doc-update response")]
    return {
        "A": _baseline_a_stub_run,
        "B": FlowRegistryRuntime(registry=seed_pairs).execute,
        "C": SemanticCacheRuntime(seed_pairs=seed_pairs).execute,
        "D": FlatToolLibraryRuntime().execute,
        "E": DeclaredCapabilityRuntime().execute,
        "T": WasmtimeRuntime().execute,
    }


@pytest.mark.parametrize("system_id", ["A", "B", "C", "D", "E", "T"])
def test_full_baseline_matrix_returns_uniform_shape(system_id: str) -> None:
    """Every system returns a `WasmRunResult` with the expected fields.

    Systems A, B, and C ignore `wasm_path`. Systems D, E, and T load
    the doc-update artifact when present. When the artifact is not on
    disk, D, E, and T still return a structured failure rather than
    raising.
    """
    systems = _build_systems()
    runner  = systems[system_id]

    wasm_path = (
        DOC_UPDATE_ARTIFACT
        if DOC_UPDATE_ARTIFACT.exists()
        else Path("/nonexistent/matrix.wasm")
    )

    result = runner(
        wasm_path=wasm_path,
        node_id="doc-update",
        task=TASK,
        context=CONTEXT,
        granted_effects=GRANTS,
    )

    assert isinstance(result, WasmRunResult), (
        f"System {system_id} returned {type(result)}, expected WasmRunResult"
    )
    fields = result.to_dict()
    expected = {
        "node_id",
        "success",
        "output",
        "error",
        "duration_ms",
        "observed_effects",
    }
    assert expected.issubset(fields.keys()), (
        f"System {system_id} result missing fields: {expected - fields.keys()}"
    )
    assert isinstance(result.success, bool)
    assert isinstance(result.output, dict)
    assert isinstance(result.observed_effects, list)


def test_full_baseline_matrix_collects_per_system_record() -> None:
    """Collect a per-system record so the harness has something to log.

    Tables 2 and 3 in the paper aggregate per-system outcomes across a
    corpus. This test does not assert specific numbers; it checks that
    each system produces a record with the fields the harness reads.
    """
    systems = _build_systems()
    wasm_path = (
        DOC_UPDATE_ARTIFACT
        if DOC_UPDATE_ARTIFACT.exists()
        else Path("/nonexistent/matrix.wasm")
    )

    records: list[dict[str, Any]] = []
    for system_id, runner in systems.items():
        result = runner(
            wasm_path=wasm_path,
            node_id="doc-update",
            task=TASK,
            context=CONTEXT,
            granted_effects=GRANTS,
        )
        records.append({
            "system":           system_id,
            "success":          result.success,
            "output":           result.output,
            "observed_effects": result.observed_effects,
            "duration_ms":      result.duration_ms,
        })

    assert {r["system"] for r in records} == {"A", "B", "C", "D", "E", "T"}
    for record in records:
        assert "success" in record
        assert "output"  in record
        assert "observed_effects" in record
