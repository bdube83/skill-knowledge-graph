"""Tests for Baseline D: the flat tool library runtime.

These tests verify that `FlatToolLibraryRuntime`:

  1. Runs a promoted node with the same output shape as the SKG
     launcher when the .wasm artifact is on disk.
  2. Instantiates a module that imports a WASI function outside the
     SKG MINIMUM_WASI set, because Baseline D wires the full surface.
  3. Returns a `WasmRunResult` with the same field shape as
     `WasmtimeRuntime`.
  4. Diverges from SKG on a module that imports `path_open`, because
     SKG refuses imports outside MINIMUM_WASI when the matching grant
     is absent.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from skg.baselines.flat_library import FlatToolLibraryRuntime
from skg.wasmtime_launcher import WasmRunResult, WasmtimeRuntime


PROJECT_ROOT = Path(__file__).parent.parent

DOC_UPDATE_ARTIFACT = (
    PROJECT_ROOT / "nodes" / "doc-update" / "target"
    / "wasm32-wasip1" / "release" / "doc-update.wasm"
)


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


def test_flat_library_runs_promoted_node() -> None:
    """Baseline D runs doc-update with the same output as the SKG launcher."""
    if not DOC_UPDATE_ARTIFACT.exists():
        pytest.skip(f"WASM artifact not built: {DOC_UPDATE_ARTIFACT}")

    inputs = {
        "task": "update doc section",
        "context": {
            "doc_path":           "README.md",
            "section_title":      "Setup",
            "existing_content":   "Run pip install.",
            "change_description": "Mention the venv.",
        },
        "granted_effects": ["text.generate"],
    }

    flat = FlatToolLibraryRuntime()
    skg  = WasmtimeRuntime()

    flat_result = flat.execute(
        wasm_path=DOC_UPDATE_ARTIFACT,
        node_id="doc-update",
        **inputs,
    )
    skg_result = skg.execute(
        wasm_path=DOC_UPDATE_ARTIFACT,
        node_id="doc-update",
        **inputs,
    )

    assert flat_result.success is True, f"Baseline D failed: {flat_result.error}"
    assert skg_result.success  is True, f"SKG failed: {skg_result.error}"
    assert flat_result.output           == skg_result.output
    assert flat_result.observed_effects == skg_result.observed_effects


def test_flat_library_allows_full_wasi_surface(tmp_path: Path) -> None:
    """A module importing `path_open` instantiates under Baseline D.

    SKG refuses the same module when no matching grant is in scope.
    Baseline D wires the full WASI surface so instantiation succeeds.
    """
    wat = _wat_with_import(
        "wasi_snapshot_preview1",
        "path_open",
        "i32 i32 i32 i32 i32 i64 i64 i32 i32",
        "i32",
    )
    wasm = _write_wat(wat, tmp_path, "uses_path_open")

    flat = FlatToolLibraryRuntime()
    result = flat.execute(
        wasm_path=wasm,
        node_id="adversarial-path-open",
        task="attack",
        context={},
        granted_effects=[],
    )
    assert result.success is True, (
        f"Baseline D should instantiate path_open module, got: {result.error}"
    )


def test_flat_library_result_shape_matches_wasmtime(tmp_path: Path) -> None:
    """`WasmRunResult` fields match the SKG launcher's contract."""
    wat = _wat_with_import(
        "wasi_snapshot_preview1",
        "fd_write",
        "i32 i32 i32 i32",
        "i32",
    )
    wasm = _write_wat(wat, tmp_path, "minimal_fd_write")

    flat = FlatToolLibraryRuntime()
    result = flat.execute(
        wasm_path=wasm,
        node_id="shape-check",
        task="noop",
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


def test_flat_library_differs_from_skg_on_path_open(tmp_path: Path) -> None:
    """SKG refuses `path_open` without a grant; Baseline D allows it.

    This is the core differentiator the paper points to: D is a flat
    tool library with no per-grant import gate. Confirm the two
    runtimes diverge on the same input.
    """
    wat = _wat_with_import(
        "wasi_snapshot_preview1",
        "path_open",
        "i32 i32 i32 i32 i32 i64 i64 i32 i32",
        "i32",
    )
    wasm = _write_wat(wat, tmp_path, "diff_path_open")

    flat = FlatToolLibraryRuntime()
    skg  = WasmtimeRuntime()

    flat_result = flat.execute(
        wasm_path=wasm,
        node_id="diff",
        task="x",
        context={},
        granted_effects=[],
    )
    skg_result = skg.execute(
        wasm_path=wasm,
        node_id="diff",
        task="x",
        context={},
        granted_effects=[],
    )

    assert flat_result.success is True
    assert skg_result.success  is False


def test_flat_library_missing_artifact() -> None:
    """A missing artifact path returns a clean failed result."""
    flat = FlatToolLibraryRuntime()
    result = flat.execute(
        wasm_path=Path("/nonexistent/flat.wasm"),
        node_id="missing",
        task="noop",
        context={},
        granted_effects=[],
    )
    assert result.success is False
    assert "not found" in result.error.lower()
