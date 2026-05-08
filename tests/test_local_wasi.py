"""End-to-end tests for the LOCAL_READ and LOCAL_WRITE WASI host imports.

These tests exercise the path_open / fd_seek / fd_read / fd_write /
fd_close / path_filestat_get / path_create_directory wrappers added in
Phase 3 of the WASM import-level enforcement work
(designs/proposed/skg-wasm-import-enforcement.md).

Each test compiles a small WAT module, drives it through the launcher
under a scoped runtime that mints LOCAL_READ or LOCAL_WRITE handles
with a tmpdir path scope, and inspects either the run result or the
filesystem state afterwards.

Invariants:
  - A read inside the granted scope succeeds and the bytes reach the
    program (then stdout via fd_write).
  - A read outside the granted scope fails with a non-zero errno.
  - A write inside the granted scope creates a file with the expected
    bytes.
  - LOCAL_READ alone does not bring path_create_directory into the
    linker; a module that imports it fails at instantiate-time.
"""

from __future__ import annotations

import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest

from skg.effects import Effect
from skg.wasmtime_launcher import WasmtimeRuntime


class _LocalScopedRuntime(WasmtimeRuntime):
    """Runtime that mints LOCAL_READ / LOCAL_WRITE handles with a fixed scope."""

    def __init__(self, scope: Path) -> None:
        super().__init__()
        self._scope = scope

    def _mint_handles(self, table, effects):
        handles: dict[str, int] = {}
        for effect in effects:
            handle_id = table.mint(
                effect,
                url_pattern="*",
                path_scope=self._scope,
                approval_token=0,
            )
            handles[effect.value] = handle_id
        return handles


def _write_wat(tmp_path: Path, name: str, wat: str) -> Path:
    out = tmp_path / f"{name}.wat"
    out.write_text(wat)
    return out


def _scope_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="skg_local_wasi_"))


def test_path_open_in_scope_reads_file_to_stdout(tmp_path: Path) -> None:
    """A WAT module opens a file inside the granted scope, reads it,
    and writes the contents to stdout. Verify the output reaches the
    captured stdout buffer."""
    scope = _scope_dir()
    try:
        target = scope / "hello.txt"
        target.write_bytes(b"hello-skg-wasi")

        # Memory layout:
        #   0..16  path bytes "hello.txt"
        #   64     fd_out_ptr (i32 receives the new fd)
        #   80..   iovec [buf_ptr=128, buf_len=64]
        #   128..  read buffer
        #   200    nread_ptr (i32 receives bytes read)
        #   216    write iovec [buf_ptr=128, buf_len=<nread>] is built at runtime
        #   240    nwritten_ptr
        path_str  = "hello.txt"
        path_len  = len(path_str)
        wat = textwrap.dedent(f'''
            (module
              (import "wasi_snapshot_preview1" "path_open"
                (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
              (import "wasi_snapshot_preview1" "fd_read"
                (func $fd_read (param i32 i32 i32 i32) (result i32)))
              (import "wasi_snapshot_preview1" "fd_write"
                (func $fd_write (param i32 i32 i32 i32) (result i32)))
              (import "wasi_snapshot_preview1" "fd_close"
                (func $fd_close (param i32) (result i32)))
              (import "wasi_snapshot_preview1" "proc_exit"
                (func $proc_exit (param i32)))
              (memory (export "memory") 1)
              (data (i32.const 0) "{path_str}")
              (func (export "_start")
                (local $fd i32)
                (local $rc i32)
                (local $nread i32)

                ;; path_open(dirfd=3, dirflags=0, path_ptr=0, path_len, oflags=0,
                ;;           fs_rights_base=FD_READ (0x2), fs_rights_inheriting=0,
                ;;           fdflags=0, fd_out_ptr=64)
                (local.set $rc
                  (call $path_open
                    (i32.const 3) (i32.const 0)
                    (i32.const 0) (i32.const {path_len})
                    (i32.const 0)
                    (i64.const 2) (i64.const 0)
                    (i32.const 0) (i32.const 64)))
                (if (i32.ne (local.get $rc) (i32.const 0))
                  (then (call $proc_exit (local.get $rc))))

                (local.set $fd (i32.load (i32.const 64)))

                ;; build read iovec at 80: [buf_ptr=128, buf_len=64]
                (i32.store (i32.const 80) (i32.const 128))
                (i32.store (i32.const 84) (i32.const 64))

                ;; fd_read(fd, iovs_ptr=80, iovs_len=1, nread_ptr=200)
                (local.set $rc
                  (call $fd_read
                    (local.get $fd) (i32.const 80) (i32.const 1) (i32.const 200)))
                (if (i32.ne (local.get $rc) (i32.const 0))
                  (then (call $proc_exit (local.get $rc))))

                (local.set $nread (i32.load (i32.const 200)))

                ;; build write iovec at 216: [buf_ptr=128, buf_len=nread]
                (i32.store (i32.const 216) (i32.const 128))
                (i32.store (i32.const 220) (local.get $nread))

                ;; fd_write(1, 216, 1, 240)
                (local.set $rc
                  (call $fd_write
                    (i32.const 1) (i32.const 216) (i32.const 1) (i32.const 240)))
                (if (i32.ne (local.get $rc) (i32.const 0))
                  (then (call $proc_exit (local.get $rc))))

                (drop (call $fd_close (local.get $fd)))
                (call $proc_exit (i32.const 0))))
        ''').strip()

        wasm = _write_wat(tmp_path, "read_in_scope", wat)
        runtime = _LocalScopedRuntime(scope)
        # Use _try_raw to read the raw stdout buffer; the launcher's
        # JSON-decoding path expects a JSON object on stdout. Run the
        # module manually to inspect raw bytes.
        from skg import wasi_minimal
        from skg.cap_to_imports import wasi_imports_for
        from wasmtime import Linker, Module, Store

        module = Module.from_file(runtime._engine, str(wasm))
        effects = [Effect.LOCAL_READ]
        wasi_set = wasi_imports_for(effects)
        state = wasi_minimal.WasiState()
        runtime._mint_handles(state.handle_table, effects)

        linker = Linker(runtime._engine)
        wasi_minimal.define_into_linker(linker, state, wasi_set)
        store = Store(runtime._engine)
        store.set_fuel(1_000_000_000)
        try:
            instance = linker.instantiate(store, module)
            instance.exports(store)["_start"](store)
        except wasi_minimal.WasiExit as exit_signal:
            assert exit_signal.code == 0, (
                f"expected exit 0, got {exit_signal.code}; "
                f"stdout: {bytes(state.stdout_buffer)!r}"
            )

        assert bytes(state.stdout_buffer) == b"hello-skg-wasi"
    finally:
        shutil.rmtree(scope, ignore_errors=True)


def test_path_open_outside_scope_fails(tmp_path: Path) -> None:
    """A WAT module tries to open a file outside the granted scope and
    receives a non-zero errno (ERRNO_NOENT=44 because the path is
    rejected by the scope check)."""
    scope = _scope_dir()
    outside = _scope_dir()
    try:
        target = outside / "secret.txt"
        target.write_bytes(b"out-of-scope-bytes")
        absolute = str(target)
        path_len = len(absolute.encode("utf-8"))

        wat = textwrap.dedent(f'''
            (module
              (import "wasi_snapshot_preview1" "path_open"
                (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
              (import "wasi_snapshot_preview1" "proc_exit"
                (func $proc_exit (param i32)))
              (memory (export "memory") 1)
              (data (i32.const 0) "{absolute}")
              (func (export "_start")
                (call $proc_exit
                  (call $path_open
                    (i32.const 3) (i32.const 0)
                    (i32.const 0) (i32.const {path_len})
                    (i32.const 0)
                    (i64.const 2) (i64.const 0)
                    (i32.const 0) (i32.const 256)))))
        ''').strip()

        wasm = _write_wat(tmp_path, "read_out_of_scope", wat)
        runtime = _LocalScopedRuntime(scope)
        result = runtime.execute(
            wasm_path=wasm,
            node_id="local_wasi_test",
            task="read out of scope",
            context={},
            granted_effects=["local.read"],
        )
        assert result.success is False
        # ERRNO_NOENT = 44 (path outside scope)
        assert "44" in result.error, f"expected errno 44, got: {result.error}"
    finally:
        shutil.rmtree(scope, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


def test_local_write_creates_file_with_bytes(tmp_path: Path) -> None:
    """A WAT module opens a new file inside the write scope, writes
    bytes, closes it. Verify the file exists with the expected
    contents."""
    scope = _scope_dir()
    try:
        path_str = "out.txt"
        path_len = len(path_str)
        # Bytes to write (kept short and printable).
        payload  = b"skg-write-ok"
        # Memory layout:
        #   0..   path bytes "out.txt"
        #   16..  payload bytes
        #   64    fd_out_ptr
        #   80    write iovec [buf_ptr=16, buf_len=len(payload)]
        #   100   nwritten_ptr
        wat = textwrap.dedent(f'''
            (module
              (import "wasi_snapshot_preview1" "path_open"
                (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
              (import "wasi_snapshot_preview1" "fd_write"
                (func $fd_write (param i32 i32 i32 i32) (result i32)))
              (import "wasi_snapshot_preview1" "fd_close"
                (func $fd_close (param i32) (result i32)))
              (import "wasi_snapshot_preview1" "proc_exit"
                (func $proc_exit (param i32)))
              (memory (export "memory") 1)
              (data (i32.const 0)  "{path_str}")
              (data (i32.const 16) "{payload.decode()}")
              (func (export "_start")
                (local $fd i32)
                (local $rc i32)

                ;; path_open with FD_WRITE right (0x40) and O_CREAT (oflags bit 0).
                (local.set $rc
                  (call $path_open
                    (i32.const 3) (i32.const 0)
                    (i32.const 0) (i32.const {path_len})
                    (i32.const 1)
                    (i64.const 64) (i64.const 0)
                    (i32.const 0) (i32.const 64)))
                (if (i32.ne (local.get $rc) (i32.const 0))
                  (then (call $proc_exit (local.get $rc))))

                (local.set $fd (i32.load (i32.const 64)))

                (i32.store (i32.const 80) (i32.const 16))
                (i32.store (i32.const 84) (i32.const {len(payload)}))

                (local.set $rc
                  (call $fd_write
                    (local.get $fd) (i32.const 80) (i32.const 1) (i32.const 100)))
                (if (i32.ne (local.get $rc) (i32.const 0))
                  (then (call $proc_exit (local.get $rc))))

                (drop (call $fd_close (local.get $fd)))
                (call $proc_exit (i32.const 0))))
        ''').strip()

        wasm = _write_wat(tmp_path, "write_in_scope", wat)
        runtime = _LocalScopedRuntime(scope)
        result = runtime.execute(
            wasm_path=wasm,
            node_id="local_wasi_test",
            task="write in scope",
            context={},
            granted_effects=["local.write"],
        )
        assert result.success is True, f"expected success, got: {result.error}"
        out = scope / "out.txt"
        assert out.exists(), "expected output file to exist"
        assert out.read_bytes() == payload
    finally:
        shutil.rmtree(scope, ignore_errors=True)


def test_local_read_does_not_bring_path_create_directory(tmp_path: Path) -> None:
    """A node granted only LOCAL_READ that imports path_create_directory
    fails at instantiate-time. The import is not in EFFECT_WASI[LOCAL_READ]
    so it never reaches the linker."""
    scope = _scope_dir()
    try:
        wat = textwrap.dedent('''
            (module
              (import "wasi_snapshot_preview1" "path_create_directory"
                (func $mkdir (param i32 i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "_start")))
        ''').strip()
        wasm = _write_wat(tmp_path, "read_only_attempts_mkdir", wat)
        runtime = _LocalScopedRuntime(scope)
        result = runtime.execute(
            wasm_path=wasm,
            node_id="local_wasi_test",
            task="attempt mkdir with read grant",
            context={},
            granted_effects=["local.read"],
        )
        assert result.success is False
        assert "path_create_directory" in result.error \
            or "import" in result.error.lower() \
            or "unknown import" in result.error.lower()
    finally:
        shutil.rmtree(scope, ignore_errors=True)
