"""Tests for Baseline B: the existing flow registry runtime.

These tests verify that `FlowRegistryRuntime`:

  1. Returns the stored response on an exact task match.
  2. Falls through to the stub LLM on an unknown task.
  3. Produces a `WasmRunResult` with the same field shape as
     `WasmtimeRuntime`.
  4. Diverges from SKG on a task that the SKG launcher would route to
     a real WASI node, confirming the two runtimes are not aliased.
"""

from __future__ import annotations

from pathlib import Path

from skg.baselines.flow_registry import FlowRegistryRuntime
from skg.wasmtime_launcher import WasmRunResult, WasmtimeRuntime


def test_flow_registry_hit_replays_stored_response() -> None:
    """An exact task match returns the stored response from the registry."""
    registry = [
        ("draft reviewer ping", "Hi @bob, please review PR #42."),
        ("summarise commits",   "2 commits by alice and bob."),
    ]
    runtime = FlowRegistryRuntime(registry=registry)

    result = runtime.execute(
        wasm_path="/dev/null",
        node_id="reviewer-ping-draft",
        task="draft reviewer ping",
        context={"pr_number": 42},
        granted_effects=["text.generate"],
    )

    assert result.success is True
    assert result.output["source"]   == "registry"
    assert result.output["response"] == "Hi @bob, please review PR #42."
    assert result.observed_effects   == []


def test_flow_registry_miss_falls_through_to_stub_llm() -> None:
    """An unknown task path uses the stub LLM, not a registry entry."""
    runtime = FlowRegistryRuntime(registry=[("a known task", "stored")])

    result = runtime.execute(
        wasm_path="/dev/null",
        node_id="any-node",
        task="something the registry has never seen",
        context={"k": "v"},
        granted_effects=[],
    )

    assert result.success is True
    assert result.output["source"] == "stub_llm"
    assert "stub-llm-response" in result.output["response"]
    assert "something the registry has never seen" in result.output["response"]


def test_flow_registry_result_shape_matches_wasmtime() -> None:
    """`WasmRunResult` fields match the SKG launcher's contract."""
    runtime = FlowRegistryRuntime(registry=[])

    result = runtime.execute(
        wasm_path="/dev/null",
        node_id="shape-check",
        task="any task",
        context={},
        granted_effects=[],
    )

    assert isinstance(result, WasmRunResult)
    expected_fields = {
        "node_id",
        "success",
        "output",
        "error",
        "duration_ms",
        "observed_effects",
    }
    assert expected_fields.issubset(result.to_dict().keys())
    assert result.node_id == "shape-check"
    assert isinstance(result.duration_ms, float)
    assert isinstance(result.observed_effects, list)


def test_flow_registry_differs_from_skg_on_basic_case() -> None:
    """SKG and Baseline B return different output shapes on the same task.

    The SKG launcher returns the WASI node's output dict (or a fail
    when the artifact path is bogus). The flow registry returns a
    static response keyed by task string. The two outputs are not
    equal on the same input, which confirms Baseline B is not
    aliased to the SKG runtime.
    """
    task = "draft reviewer ping"
    registry = [(task, "registered response")]
    flow = FlowRegistryRuntime(registry=registry)
    skg  = WasmtimeRuntime()

    flow_result = flow.execute(
        wasm_path="/dev/null",
        node_id="reviewer-ping-draft",
        task=task,
        context={},
        granted_effects=[],
    )
    skg_result = skg.execute(
        wasm_path=Path("/nonexistent/skg.wasm"),
        node_id="reviewer-ping-draft",
        task=task,
        context={},
        granted_effects=[],
    )

    assert flow_result.success is True
    assert skg_result.success  is False
    assert flow_result.output != skg_result.output
