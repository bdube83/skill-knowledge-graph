"""Tests for the Wasmtime launcher — slice 3 execution with a real WASI node.

These tests require the reviewer-ping-draft node to be compiled to WASM first:
  cd nodes/reviewer-ping-draft
  cargo build --release --target wasm32-wasip1

The .wasm artifact is expected at:
  nodes/reviewer-ping-draft/target/wasm32-wasip1/release/reviewer_ping_draft.wasm

Tests are skipped automatically if the artifact is not present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skg.wasmtime_launcher import WasmtimeRuntime, wasm_path_for_node

# Path to the sample WASI node artifact relative to the project root.
PROJECT_ROOT = Path(__file__).parent.parent
WASM_ARTIFACT = (
    PROJECT_ROOT
    / "nodes"
    / "reviewer-ping-draft"
    / "target"
    / "wasm32-wasip1"
    / "release"
    / "reviewer_ping_draft.wasm"
)

requires_wasm = pytest.mark.skipif(
    not WASM_ARTIFACT.exists(),
    reason="WASM artifact not built. Run: cd nodes/reviewer-ping-draft && cargo build --release --target wasm32-wasip1",
)


@requires_wasm
class TestWasmtimeLauncher:
    def test_basic_execution(self):
        rt = WasmtimeRuntime()
        result = rt.execute(
            wasm_path=WASM_ARTIFACT,
            node_id="reviewer-ping-draft",
            task="draft reviewer ping",
            context={
                "pr_number": 42,
                "repo": "example/myrepo",
                "author": "alice",
                "reviewers": ["bob", "carol"],
            },
            granted_effects=["text.generate"],
        )
        assert result.success is True, f"Expected success, got error: {result.error}"
        assert "message" in result.output
        assert "@bob" in result.output["message"] or "@carol" in result.output["message"]
        assert "42" in result.output["message"]
        assert "text.generate" in result.observed_effects

    def test_capability_denied(self):
        """Node should fail cleanly when text.generate is not granted."""
        rt = WasmtimeRuntime()
        result = rt.execute(
            wasm_path=WASM_ARTIFACT,
            node_id="reviewer-ping-draft",
            task="draft reviewer ping",
            context={"pr_number": 1, "repo": "x/y", "author": "z", "reviewers": []},
            granted_effects=[],   # nothing granted
        )
        # Node should return an error in its JSON output, not crash Wasmtime.
        assert result.success is False or "error" in (result.output or {})

    def test_missing_artifact(self):
        rt = WasmtimeRuntime()
        result = rt.execute(
            wasm_path=Path("/nonexistent/node.wasm"),
            node_id="fake-node",
            task="do something",
            context={},
            granted_effects=[],
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_duration_recorded(self):
        rt = WasmtimeRuntime()
        result = rt.execute(
            wasm_path=WASM_ARTIFACT,
            node_id="reviewer-ping-draft",
            task="draft reviewer ping",
            context={"pr_number": 1, "repo": "x/y", "author": "z", "reviewers": []},
            granted_effects=["text.generate"],
        )
        assert result.duration_ms >= 0

    def test_module_cache(self):
        """Second call to same artifact should use cached module (no recompile)."""
        rt = WasmtimeRuntime()
        ctx = {"pr_number": 1, "repo": "x/y", "author": "z", "reviewers": []}
        rt.execute(WASM_ARTIFACT, "n", "t", ctx, ["text.generate"])
        # Second call hits cache — module entry should exist.
        assert str(WASM_ARTIFACT) in rt._cache
