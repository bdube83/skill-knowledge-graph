"""Tests for Baseline C: the semantic response cache runtime.

These tests verify that `SemanticCacheRuntime`:

  1. Returns the cached response when a near-duplicate task scores
     above the cosine threshold.
  2. Falls through to the stub LLM when no seed is similar enough.
  3. Produces a `WasmRunResult` with the same field shape as
     `WasmtimeRuntime`.
  4. Diverges from SKG on a task that the SKG launcher would route to
     a real WASI node, confirming the two runtimes are not aliased.
"""

from __future__ import annotations

from pathlib import Path

from skg.baselines.semantic_cache import SemanticCacheRuntime
from skg.wasmtime_launcher import WasmRunResult, WasmtimeRuntime


def test_semantic_cache_hit_on_near_duplicate_task() -> None:
    """A task that re-uses every seed token hits the cache.

    The seed and query tokens are identical after lower-casing and
    tokenising; cosine similarity is 1.0, well above the 0.85 default.
    """
    seeds = [("draft reviewer ping for pr 42", "Hi @bob, please review PR #42.")]
    runtime = SemanticCacheRuntime(seed_pairs=seeds)

    result = runtime.execute(
        wasm_path="/dev/null",
        node_id="reviewer-ping-draft",
        task="draft reviewer PING for PR 42",
        context={"pr_number": 42},
        granted_effects=["text.generate"],
    )

    assert result.success is True
    assert result.output["source"]   == "cache"
    assert result.output["response"] == "Hi @bob, please review PR #42."
    assert result.output["similarity"] >= 0.85


def test_semantic_cache_miss_falls_through_to_stub_llm() -> None:
    """A task with no token overlap to any seed misses the cache."""
    seeds = [("summarise commits in repo", "2 commits.")]
    runtime = SemanticCacheRuntime(seed_pairs=seeds, threshold=0.85)

    result = runtime.execute(
        wasm_path="/dev/null",
        node_id="other-node",
        task="reticulate splines on planet xenon",
        context={},
        granted_effects=[],
    )

    assert result.success is True
    assert result.output["source"] == "stub_llm"
    assert "stub-llm-response" in result.output["response"]
    assert result.output["similarity"] < 0.85


def test_semantic_cache_result_shape_matches_wasmtime() -> None:
    """`WasmRunResult` fields match the SKG launcher's contract."""
    runtime = SemanticCacheRuntime(seed_pairs=[])

    result = runtime.execute(
        wasm_path="/dev/null",
        node_id="shape-check",
        task="anything",
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
    assert isinstance(result.duration_ms, float)
    assert isinstance(result.observed_effects, list)


def test_semantic_cache_differs_from_skg_on_basic_case() -> None:
    """SKG and Baseline C return different outputs on the same task.

    With a missing artifact the SKG launcher returns `success=False`.
    The semantic cache returns a stub response with `success=True`.
    The two are not aliased.
    """
    task = "summarise commits"
    cache = SemanticCacheRuntime(seed_pairs=[(task, "cached")], threshold=0.5)
    skg   = WasmtimeRuntime()

    cache_result = cache.execute(
        wasm_path="/dev/null",
        node_id="git-summary",
        task=task,
        context={},
        granted_effects=[],
    )
    skg_result = skg.execute(
        wasm_path=Path("/nonexistent/skg.wasm"),
        node_id="git-summary",
        task=task,
        context={},
        granted_effects=[],
    )

    assert cache_result.success is True
    assert skg_result.success   is False
    assert cache_result.output != skg_result.output


def test_semantic_cache_stores_miss_for_later_hit() -> None:
    """A miss writes the new pair so a re-issue of the same task hits."""
    runtime = SemanticCacheRuntime(seed_pairs=[], threshold=0.85)

    first = runtime.execute(
        wasm_path="/dev/null",
        node_id="any",
        task="brand new task",
        context={"k": 1},
        granted_effects=[],
    )
    assert first.output["source"] == "stub_llm"

    second = runtime.execute(
        wasm_path="/dev/null",
        node_id="any",
        task="brand new task",
        context={"k": 1},
        granted_effects=[],
    )
    assert second.output["source"]   == "cache"
    assert second.output["response"] == first.output["response"]
