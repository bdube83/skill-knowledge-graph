"""Security tests for the GrantedLinker pattern.

These tests verify the paper's H3 reduction argument: a node that
imports a host function it was not granted fails at instantiate-time
because the import is absent from the linker.

Each test crafts a tiny WAT module that imports a single host
function, compiles it via Wasmtime's WAT parser, and runs it under
the launcher with a known grant set. The expected outcome is a
specific instantiate-time error.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from skg.wasmtime_launcher import WasmtimeRuntime, WASI_MODULE


def _wat_with_import(module_name: str, func_name: str, params: str, results: str) -> str:
    """Build a minimal WAT that imports one function and returns immediately."""
    return textwrap.dedent(f'''
        (module
          (import "{module_name}" "{func_name}"
            (func $imported (param {params}) (result {results})))
          (memory (export "memory") 1)
          (func (export "_start"))
        )
    ''').strip()


def _compile_to_disk(wat_text: str, tmp_path: Path, name: str) -> Path:
    """Write WAT to a temp .wat file the launcher can load via Module.from_file."""
    out = tmp_path / f"{name}.wat"
    out.write_text(wat_text)
    return out


def _try_run(wasm_path: Path, granted_effects: list[str]) -> "object":
    rt = WasmtimeRuntime()
    return rt.execute(
        wasm_path=wasm_path,
        node_id="adversarial",
        task="attack",
        context={},
        granted_effects=granted_effects,
    )


def test_ungranted_skg_http_get_fails_at_instantiate(tmp_path: Path) -> None:
    wat = _wat_with_import("skg", "http_get", "i32 i32", "i32")
    wasm = _compile_to_disk(wat, tmp_path, "attack_http_get")
    result = _try_run(wasm, granted_effects=[])
    assert result.success is False
    assert "http_get" in result.error or "unknown import" in result.error.lower() \
        or "not defined" in result.error.lower() or "import" in result.error.lower()


def test_ungranted_skg_external_send_fails_at_instantiate(tmp_path: Path) -> None:
    wat = _wat_with_import("skg", "external_send", "i32 i32 i32", "i32")
    wasm = _compile_to_disk(wat, tmp_path, "attack_external_send")
    result = _try_run(wasm, granted_effects=["external.draft"])
    assert result.success is False
    assert "external_send" in result.error or "import" in result.error.lower()


def test_ungranted_wasi_path_open_fails_at_instantiate(tmp_path: Path) -> None:
    wat = _wat_with_import(
        WASI_MODULE,
        "path_open",
        "i32 i32 i32 i32 i32 i64 i64 i32 i32",
        "i32",
    )
    wasm = _compile_to_disk(wat, tmp_path, "attack_path_open")
    result = _try_run(wasm, granted_effects=[])
    assert result.success is False


def test_local_read_grant_no_longer_unsupported(tmp_path: Path) -> None:
    """LOCAL_READ WASI imports are wired now; granting it instantiates."""
    wat = _wat_with_import(WASI_MODULE, "fd_write", "i32 i32 i32 i32", "i32")
    wasm = _compile_to_disk(wat, tmp_path, "needs_local_read")
    result = _try_run(wasm, granted_effects=["local.read"])
    assert result.success is True, f"expected success, got error: {result.error}"
    assert "unsupported wasi imports" not in result.error.lower()


def test_granted_skg_http_get_instantiates(tmp_path: Path) -> None:
    """If NETWORK_READ is granted, skg.http_get is wired and the module loads.

    Phase 3b signature: (handle, payload_ptr, payload_len, response_ptr,
    response_max_len, actual_len_ptr) -> errno.
    """
    wat = _wat_with_import("skg", "http_get", "i32 i32 i32 i32 i32 i32", "i32")
    wasm = _compile_to_disk(wat, tmp_path, "ok_http_get")
    result = _try_run(wasm, granted_effects=["network.read"])
    assert result.success is True, f"expected success, got error: {result.error}"


def test_granted_skg_external_send_instantiates(tmp_path: Path) -> None:
    """If EXTERNAL_SEND is granted, skg.external_send is wired and the module loads.

    Phase 3b approval-gated signature: (handle, payload_ptr, payload_len,
    response_ptr, response_max_len, actual_len_ptr, approval_token) -> errno.
    """
    wat = _wat_with_import(
        "skg",
        "external_send",
        "i32 i32 i32 i32 i32 i32 i32",
        "i32",
    )
    wasm = _compile_to_disk(wat, tmp_path, "ok_external_send")
    result = _try_run(wasm, granted_effects=["external.send"])
    assert result.success is True, f"expected success, got error: {result.error}"


def test_minimum_wasi_always_present(tmp_path: Path) -> None:
    """fd_write is in MINIMUM_WASI; granting nothing still allows it."""
    wat = _wat_with_import(WASI_MODULE, "fd_write", "i32 i32 i32 i32", "i32")
    wasm = _compile_to_disk(wat, tmp_path, "minimum_wasi")
    result = _try_run(wasm, granted_effects=[])
    assert result.success is True, f"expected success, got error: {result.error}"
