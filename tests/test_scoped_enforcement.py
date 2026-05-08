"""End-to-end tests for per-call URL and path scoping (Phase 3c).

The Phase 3b host wrappers validate the per-run handle table on every
call. The launcher's default `_mint_handles` creates wildcard-scoped
handles (any URL, any path). These tests subclass `WasmtimeRuntime` to
mint scoped handles and exercise the wrapper through a calling WAT
module that hands the wrapper a specific URL.

Outcome encoding. Each WAT calls the host import once and forwards the
errno to `proc_exit`. The launcher catches the resulting `WasiExit`
and reports `success=True` for errno 0, `success=False` for errno != 0.
The error string contains "exited with code N".

What this proves. The wrapper extracts the URL from the JSON payload,
calls `HandleTable.validate` with that URL, and rejects calls outside
the handle's scope. End-to-end through real WASM execution.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from skg.effects import Effect
from skg.wasmtime_launcher import WasmtimeRuntime


class _ScopedRuntime(WasmtimeRuntime):
    """Test-only runtime that mints handles with caller-supplied scopes."""

    def __init__(self, scopes: dict[Effect, tuple[str, str]]) -> None:
        super().__init__()
        self._scopes = scopes

    def _mint_handles(self, table, effects: list[Effect]) -> dict[str, int]:
        approval_effects = {
            Effect.EXTERNAL_SEND,
            Effect.GIT_WRITE,
            Effect.PRODUCTION_WRITE,
        }
        handles: dict[str, int] = {}
        for effect in effects:
            url_pattern, path_str = self._scopes.get(effect, ("*", "/"))
            handle_id = table.mint(
                effect,
                url_pattern=url_pattern,
                path_scope=Path(path_str),
                approval_token=1 if effect in approval_effects else 0,
            )
            handles[effect.value] = handle_id
        return handles


def _http_get_wat(payload: str) -> str:
    """WAT that calls skg.http_get once with the given JSON payload.

    The payload is embedded as a data segment at memory offset 0. The
    `_start` function calls `skg.http_get` with handle id 1 (the
    launcher mints handles starting at 1) and forwards the errno
    through `proc_exit`.
    """
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


def _write(tmp_path: Path, name: str, wat: str) -> Path:
    out = tmp_path / f"{name}.wat"
    out.write_text(wat)
    return out


def test_in_scope_url_succeeds(tmp_path: Path) -> None:
    """A request to a URL the handle covers returns errno 0."""
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: ("https://allowed.example/*", "/"),
    })
    wat  = _http_get_wat('{"url":"https://allowed.example/foo"}')
    wasm = _write(tmp_path, "in_scope", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="scope_test",
        task="fetch",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is True, f"expected success, got: {result.error}"


def test_out_of_scope_url_denied(tmp_path: Path) -> None:
    """A request to a URL outside the handle's scope returns errno 13."""
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: ("https://allowed.example/*", "/"),
    })
    wat  = _http_get_wat('{"url":"https://attacker.example/exfiltrate"}')
    wasm = _write(tmp_path, "out_of_scope", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="scope_test",
        task="exfiltrate",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is False
    assert "13" in result.error


def test_url_in_different_subdomain_denied(tmp_path: Path) -> None:
    """fnmatch is strict: subdomain swap is out of scope."""
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: ("https://api.allowed.example/*", "/"),
    })
    wat  = _http_get_wat('{"url":"https://internal.allowed.example/foo"}')
    wasm = _write(tmp_path, "wrong_subdomain", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="scope_test",
        task="cross-subdomain",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is False
    assert "13" in result.error


def test_stale_handle_denied(tmp_path: Path) -> None:
    """A WAT that hardcodes a non-existent handle id is rejected."""
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: ("*", "/"),
    })
    payload_bytes = '{"url":"https://allowed.example/"}'.encode("utf-8")
    payload_len   = len(payload_bytes)
    wat = textwrap.dedent(f'''
        (module
          (import "skg" "http_get"
            (func $http_get (param i32 i32 i32 i32 i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "proc_exit"
            (func $proc_exit (param i32)))
          (memory (export "memory") 1)
          (data (i32.const 0) "{{\\"url\\":\\"https://allowed.example/\\"}}")
          (func (export "_start")
            (call $proc_exit
              (call $http_get
                (i32.const 99)
                (i32.const 0)
                (i32.const {payload_len})
                (i32.const 1024)
                (i32.const 1024)
                (i32.const 2048)))))
    ''').strip()
    wasm = _write(tmp_path, "stale_handle", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="scope_test",
        task="replay",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is False
    assert "13" in result.error
