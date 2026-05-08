"""Tests for Baseline E: the declared-capability runtime.

These tests verify that `DeclaredCapabilityRuntime`:

  1. Runs the three promoted nodes (doc-update, git-summary,
     reviewer-ping-draft) with the same output shape as the SKG
     launcher.
  2. Instantiates a module that imports a WASI function outside the
     SKG MINIMUM_WASI set (e.g. `path_open`), because Baseline E
     wires the full WASI surface.
  3. Fails to instantiate a module that imports `skg.http_get`,
     because Baseline E has no SKG host-import layer.

The first set of tests is skipped when the .wasm artifacts are not
present on disk.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from skg.baselines.declared import DeclaredCapabilityRuntime
from skg.wasmtime_launcher import WasmtimeRuntime


PROJECT_ROOT = Path(__file__).parent.parent

NODE_ARTIFACTS: dict[str, Path] = {
    "doc-update": (
        PROJECT_ROOT / "nodes" / "doc-update" / "target"
        / "wasm32-wasip1" / "release" / "doc-update.wasm"
    ),
    "git-summary": (
        PROJECT_ROOT / "nodes" / "git-summary" / "target"
        / "wasm32-wasip1" / "release" / "git-summary.wasm"
    ),
    "reviewer-ping-draft": (
        PROJECT_ROOT / "nodes" / "reviewer-ping-draft" / "target"
        / "wasm32-wasip1" / "release" / "reviewer_ping_draft.wasm"
    ),
}


NODE_INPUTS: dict[str, dict] = {
    "doc-update": {
        "task": "update doc section",
        "context": {
            "doc_path":           "README.md",
            "section_title":      "Setup",
            "existing_content":   "Run pip install.",
            "change_description": "Mention the venv.",
        },
        "granted_effects": ["text.generate"],
    },
    "git-summary": {
        "task": "summarise commits",
        "context": {
            "repo":   "example/repo",
            "branch": "main",
            "commits": [
                {"sha": "a1", "author": "alice", "message": "init"},
                {"sha": "b2", "author": "bob",   "message": "fix"},
            ],
        },
        "granted_effects": ["git.read"],
    },
    "reviewer-ping-draft": {
        "task": "draft reviewer ping",
        "context": {
            "pr_number": 42,
            "repo":      "example/myrepo",
            "author":    "alice",
            "reviewers": ["bob", "carol"],
        },
        "granted_effects": ["text.generate"],
    },
}


def _artifact_or_skip(node_id: str) -> Path:
    path = NODE_ARTIFACTS[node_id]
    if not path.exists():
        pytest.skip(f"WASM artifact not built for {node_id}: {path}")
    return path


@pytest.mark.parametrize("node_id", list(NODE_ARTIFACTS.keys()))
def test_baseline_runs_promoted_node(node_id: str) -> None:
    """Baseline E runs each promoted node and produces the same shape."""
    artifact = _artifact_or_skip(node_id)
    inputs   = NODE_INPUTS[node_id]

    baseline = DeclaredCapabilityRuntime()
    skg      = WasmtimeRuntime()

    baseline_result = baseline.execute(
        wasm_path=artifact,
        node_id=node_id,
        task=inputs["task"],
        context=inputs["context"],
        granted_effects=inputs["granted_effects"],
    )
    skg_result = skg.execute(
        wasm_path=artifact,
        node_id=node_id,
        task=inputs["task"],
        context=inputs["context"],
        granted_effects=inputs["granted_effects"],
    )

    assert baseline_result.success is True, (
        f"Baseline E failed for {node_id}: {baseline_result.error}"
    )
    assert skg_result.success is True, (
        f"SKG launcher failed for {node_id}: {skg_result.error}"
    )

    # Both runtimes feed the same JSON payload to the same WASM module
    # over stdin and read output from stdout. Output must match.
    assert baseline_result.output == skg_result.output
    assert baseline_result.observed_effects == skg_result.observed_effects


def _wat_with_import(
    module_name: str,
    func_name:   str,
    params:      str,
    results:     str,
) -> str:
    return textwrap.dedent(f'''
        (module
          (import "{module_name}" "{func_name}"
            (func $imported (param {params}) (result {results})))
          (memory (export "memory") 1)
          (func (export "_start"))
        )
    ''').strip()


def _write_wat(wat_text: str, tmp_path: Path, name: str) -> Path:
    out = tmp_path / f"{name}.wat"
    out.write_text(wat_text)
    return out


def test_baseline_allows_full_wasi_surface(tmp_path: Path) -> None:
    """A module importing `path_open` instantiates under Baseline E.

    Under SKG, the same module fails because `path_open` is outside
    MINIMUM_WASI and is only wired when LOCAL_READ or LOCAL_WRITE is
    granted. Under Baseline E, `define_wasi()` wires the full surface
    so instantiation succeeds.
    """
    wat = _wat_with_import(
        "wasi_snapshot_preview1",
        "path_open",
        "i32 i32 i32 i32 i32 i64 i64 i32 i32",
        "i32",
    )
    wasm = _write_wat(wat, tmp_path, "uses_path_open")

    baseline = DeclaredCapabilityRuntime()
    result = baseline.execute(
        wasm_path=wasm,
        node_id="adversarial-path-open",
        task="attack",
        context={},
        granted_effects=[],
    )
    assert result.success is True, (
        f"Baseline E should instantiate path_open module, got: {result.error}"
    )

    # Confirm the same module fails under SKG with the same grants.
    skg = WasmtimeRuntime()
    skg_result = skg.execute(
        wasm_path=wasm,
        node_id="adversarial-path-open",
        task="attack",
        context={},
        granted_effects=[],
    )
    assert skg_result.success is False
    assert "path_open" in skg_result.error or "import" in skg_result.error.lower()


def test_baseline_rejects_skg_http_get(tmp_path: Path) -> None:
    """A module importing `skg.http_get` fails under Baseline E.

    Baseline E wires no `skg.*` host imports. Wasmtime cannot resolve
    the import, so instantiation fails. This is the realistic
    behaviour of a declared-capability runtime that lacks SKG's
    host-import layer.
    """
    wat = _wat_with_import("skg", "http_get", "i32 i32", "i32")
    wasm = _write_wat(wat, tmp_path, "needs_skg_http_get")

    baseline = DeclaredCapabilityRuntime()
    result = baseline.execute(
        wasm_path=wasm,
        node_id="adversarial-http-get",
        task="attack",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is False
    err = result.error.lower()
    assert "http_get" in result.error or "import" in err or "unknown" in err


def test_baseline_drops_unknown_effects(tmp_path: Path) -> None:
    """Unknown effect strings are dropped without raising.

    The baseline mirrors `_to_effects` in the SKG launcher: strings
    that do not match the `Effect` enum are silently dropped.
    """
    wat = _wat_with_import(
        "wasi_snapshot_preview1",
        "fd_write",
        "i32 i32 i32 i32",
        "i32",
    )
    wasm = _write_wat(wat, tmp_path, "minimal_fd_write")

    baseline = DeclaredCapabilityRuntime()
    result = baseline.execute(
        wasm_path=wasm,
        node_id="effects-validation",
        task="noop",
        context={},
        granted_effects=["not.a.real.effect", "local.read"],
    )
    assert result.success is True, (
        f"Baseline should accept the run with unknown strings dropped: {result.error}"
    )


def test_baseline_missing_artifact() -> None:
    """A missing artifact path returns a clean failed result."""
    baseline = DeclaredCapabilityRuntime()
    result = baseline.execute(
        wasm_path=Path("/nonexistent/baseline.wasm"),
        node_id="missing",
        task="noop",
        context={},
        granted_effects=[],
    )
    assert result.success is False
    assert "not found" in result.error.lower()
