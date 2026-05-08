"""End-to-end tests for the real HTTP host adapters.

The Phase 3e adapters in `skg/host_adapters.py` make real `urllib`
calls after the Phase 3b wrapper has validated the handle. These
tests spin up a loopback HTTP server, point the WAT-driven node at
it, and verify both the scope check and the real fetch.

What these tests prove together with the scope tests:
  - The wrapper enforces the URL pattern. (test_scoped_enforcement.py)
  - When the URL passes the wrapper, the real adapter performs the
    fetch and returns its bytes back into WASM memory. (this file)
"""

from __future__ import annotations

import http.server
import socketserver
import textwrap
import threading
from pathlib import Path

import pytest

from skg.effects import Effect
from skg.wasmtime_launcher import WasmtimeRuntime


class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
    """Minimal handler that returns 200 OK with body `LOOPBACK`."""

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"LOOPBACK")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body   = self.rfile.read(length) if length else b""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ECHO:" + body)

    def log_message(self, *_args, **_kwargs) -> None:
        return  # silent


@pytest.fixture
def loopback_server():
    """Start an http.server on 127.0.0.1, yield (host, port), shut down."""
    server = socketserver.TCPServer(("127.0.0.1", 0), _LoopbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


class _ScopedRuntime(WasmtimeRuntime):
    """Runtime that mints handles with caller-supplied scopes."""

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


def _http_get_wat(payload: str) -> str:
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


def _http_post_wat(payload: str) -> str:
    payload_bytes = payload.encode("utf-8")
    payload_len   = len(payload_bytes)
    escaped       = payload.replace('\\', '\\\\').replace('"', '\\"')
    return textwrap.dedent(f'''
        (module
          (import "skg" "http_post"
            (func $http_post (param i32 i32 i32 i32 i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "proc_exit"
            (func $proc_exit (param i32)))
          (memory (export "memory") 1)
          (data (i32.const 0) "{escaped}")
          (func (export "_start")
            (call $proc_exit
              (call $http_post
                (i32.const 1)
                (i32.const 0)
                (i32.const {payload_len})
                (i32.const 1024)
                (i32.const 1024)
                (i32.const 2048)))))
    ''').strip()


def _write_wat(tmp_path: Path, name: str, wat: str) -> Path:
    out = tmp_path / f"{name}.wat"
    out.write_text(wat)
    return out


def test_real_http_get_succeeds_in_scope(loopback_server, tmp_path: Path) -> None:
    host, port = loopback_server
    url = f"http://{host}:{port}/hello"
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: (f"http://{host}:{port}/*", "/"),
    })
    wat  = _http_get_wat(f'{{"url":"{url}"}}')
    wasm = _write_wat(tmp_path, "real_http_get_in_scope", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="fetch",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is True, f"expected success, got: {result.error}"


def test_real_http_get_denied_out_of_scope(loopback_server, tmp_path: Path) -> None:
    host, port = loopback_server
    url = f"http://{host}:{port}/hello"
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: ("http://other.example/*", "/"),
    })
    wat  = _http_get_wat(f'{{"url":"{url}"}}')
    wasm = _write_wat(tmp_path, "real_http_get_out_of_scope", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="exfiltrate",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is False
    assert "13" in result.error


def test_real_http_post_succeeds_in_scope(loopback_server, tmp_path: Path) -> None:
    host, port = loopback_server
    url = f"http://{host}:{port}/echo"
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_WRITE: (f"http://{host}:{port}/*", "/"),
    })
    wat  = _http_post_wat(
        f'{{"url":"{url}","body":"hello","content_type":"text/plain"}}'
    )
    wasm = _write_wat(tmp_path, "real_http_post_in_scope", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="post",
        context={},
        granted_effects=["network.write"],
    )
    assert result.success is True, f"expected success, got: {result.error}"


def test_real_http_get_unreachable_url_returns_io_error(tmp_path: Path) -> None:
    """If scope passes but the real fetch fails, the wrapper returns ERRNO_IO."""
    runtime = _ScopedRuntime(scopes={
        Effect.NETWORK_READ: ("http://127.0.0.1:1/*", "/"),
    })
    wat  = _http_get_wat('{"url":"http://127.0.0.1:1/never"}')
    wasm = _write_wat(tmp_path, "real_http_get_unreachable", wat)
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="fetch",
        context={},
        granted_effects=["network.read"],
    )
    assert result.success is False
    assert "29" in result.error
