"""Adversarial corpus for the GrantedLinker (Phase 4).

Each test crafts a WAT module that attempts a specific bypass and runs
it under the SKG launcher. The test asserts the expected containment
outcome. The corpus is organized by attack class per the design doc
(designs/proposed/skg-wasm-import-enforcement.md, "Adversarial corpus
design").

This file covers the attack classes that depend only on Phase 1, 2,
and 3 surface:

  - Path escape (class 2)
  - WASI introspection (class 7)
  - Fuel exhaustion (class 6)

Manifest lies, URL escape, replay, and confused deputy live in
companion files once Phase 3b (handle table) and Phase 5 (baseline E)
land. The 7-attack-class containment matrix lives in
`tests/test_containment_matrix.py` (added later).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from skg.wasmtime_launcher import WASI_MODULE, WasmtimeRuntime


def _wat_with_wasi_import(
    func_name: str,
    params:    str,
    results:   str,
) -> str:
    return textwrap.dedent(f'''
        (module
          (import "{WASI_MODULE}" "{func_name}"
            (func $imp (param {params}) (result {results})))
          (memory (export "memory") 1)
          (func (export "_start"))
        )
    ''').strip()


def _wat_no_imports() -> str:
    return textwrap.dedent('''
        (module
          (memory (export "memory") 1)
          (func (export "_start")))
    ''').strip()


def _wat_infinite_loop() -> str:
    """A WAT module whose _start spins forever; depletes fuel."""
    return textwrap.dedent('''
        (module
          (memory (export "memory") 1)
          (func (export "_start")
            (loop $forever
              br $forever)))
    ''').strip()


def _write_wat(tmp_path: Path, name: str, wat: str) -> Path:
    out = tmp_path / f"{name}.wat"
    out.write_text(wat)
    return out


def _run(wasm_path: Path, granted_effects: list[str]) -> "object":
    rt = WasmtimeRuntime()
    return rt.execute(
        wasm_path=wasm_path,
        node_id="adversarial",
        task="attack",
        context={},
        granted_effects=granted_effects,
    )


# ---- Class 2: Path escape ----------------------------------------------------

class TestPathEscape:
    """A node tries WASI filesystem operations beyond its grant."""

    def test_path_open_without_local_read_fails(self, tmp_path: Path) -> None:
        wat  = _wat_with_wasi_import("path_open", "i32 i32 i32 i32 i32 i64 i64 i32 i32", "i32")
        wasm = _write_wat(tmp_path, "path_escape_no_grant", wat)
        result = _run(wasm, granted_effects=[])
        assert result.success is False

    def test_local_read_does_not_grant_write_surface(self, tmp_path: Path) -> None:
        """LOCAL_READ should not bring path_create_directory into the linker."""
        wat  = _wat_with_wasi_import("path_create_directory", "i32 i32 i32", "i32")
        wasm = _write_wat(tmp_path, "path_escape_read_only_grant", wat)
        result = _run(wasm, granted_effects=["local.read"])
        assert result.success is False

    def test_local_write_does_not_grant_filestat_get(self, tmp_path: Path) -> None:
        """LOCAL_WRITE should not bring path_filestat_get (a LOCAL_READ import)."""
        wat  = _wat_with_wasi_import("path_filestat_get", "i32 i32 i32 i32 i32", "i32")
        wasm = _write_wat(tmp_path, "path_escape_write_only_grant", wat)
        result = _run(wasm, granted_effects=["local.write"])
        assert result.success is False


# ---- Class 7: WASI introspection ---------------------------------------------

class TestWasiIntrospection:
    """A node tries WASI imports outside MINIMUM_WASI without a grant."""

    def test_poll_oneoff_absent(self, tmp_path: Path) -> None:
        wat  = _wat_with_wasi_import("poll_oneoff", "i32 i32 i32 i32", "i32")
        wasm = _write_wat(tmp_path, "introspect_poll_oneoff", wat)
        result = _run(wasm, granted_effects=[])
        assert result.success is False

    def test_sched_yield_absent(self, tmp_path: Path) -> None:
        wat  = _wat_with_wasi_import("sched_yield", "", "i32")
        wasm = _write_wat(tmp_path, "introspect_sched_yield", wat)
        result = _run(wasm, granted_effects=[])
        assert result.success is False

    def test_clock_time_get_absent(self, tmp_path: Path) -> None:
        """clock_time_get is not in MINIMUM_WASI per the audit (2026-05-08)."""
        wat  = _wat_with_wasi_import("clock_time_get", "i32 i64 i32", "i32")
        wasm = _write_wat(tmp_path, "introspect_clock_time_get", wat)
        result = _run(wasm, granted_effects=[])
        assert result.success is False


# ---- Class 6: Fuel exhaustion ------------------------------------------------

class TestFuelExhaustion:
    """A node depletes fuel; the launcher reports a failure."""

    def test_infinite_loop_runs_out_of_fuel(self, tmp_path: Path) -> None:
        wat  = _wat_infinite_loop()
        wasm = _write_wat(tmp_path, "fuel_exhaustion", wat)
        result = _run(wasm, granted_effects=[])
        assert result.success is False

    def test_no_op_module_succeeds(self, tmp_path: Path) -> None:
        """Sanity: a module with no work and no imports succeeds under any grant."""
        wat  = _wat_no_imports()
        wasm = _write_wat(tmp_path, "no_op", wat)
        result = _run(wasm, granted_effects=[])
        assert result.success is True, f"baseline no-op failed: {result.error}"
