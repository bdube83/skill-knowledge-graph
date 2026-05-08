"""Minimal WASI snapshot preview1 host implementation for the GrantedLinker.

Replaces `Linker.define_wasi()` with per-run, per-grant host function
wiring. Phase 2 of the WASM import-level enforcement work
(designs/proposed/skg-wasm-import-enforcement.md).

Why a custom implementation. `Linker.define_wasi()` wires the entire
WASI snapshot preview1 surface, which makes the runtime gate weaker
than the paper's reduction argument requires. With this module, only
the WASI imports the grant set permits are present in the linker;
modules referencing other imports fail at instantiate-time because
Wasmtime cannot resolve the import.

Scope. This module implements only the imports listed in
`cap_to_imports.MINIMUM_WASI` and `cap_to_imports.EFFECT_WASI`.
Anything outside that set is intentionally absent.

State model. Each `execute()` call builds a fresh `WasiState` and a
fresh linker that closes over it. State is per-run; the linker is
not shared across runs with different grants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable

from wasmtime import Caller, FuncType, Linker, Memory, ValType

from .effects import Effect
from .handle_table import HandleTable, _path_in_scope


ERRNO_SUCCESS = 0
ERRNO_BADF    = 8
ERRNO_IO      = 29
ERRNO_INVAL   = 28
ERRNO_NOENT   = 44
ERRNO_PERM    = 63


# WASI fs_rights_base bits we care about. The full bitset is large; only
# FD_READ and FD_WRITE drive open-mode selection here.
_RIGHTS_FD_READ  = 1 << 1
_RIGHTS_FD_WRITE = 1 << 6


@dataclass
class FileDescriptorRow:
    """One open file in the per-run fd table."""

    fd_id:       int
    path:        Path
    mode:        str            # "r" or "w"
    file_handle: BinaryIO
    position:    int = 0


@dataclass
class WasiState:
    """Per-run WASI state shared with host function closures."""

    stdin_bytes:   bytes                          = b""
    stdin_offset:  int                            = 0
    stdout_buffer: bytearray                      = field(default_factory=bytearray)
    exit_code:     int | None                     = None
    handle_table:  HandleTable                    = field(default_factory=HandleTable)
    fd_table:      dict[int, FileDescriptorRow]   = field(default_factory=dict)
    next_fd:       int                            = 3


class WasiExit(Exception):
    """Raised by proc_exit to terminate WASM execution cleanly."""

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"WASI proc_exit({code})")


def _memory(caller: Caller) -> Memory | None:
    """Return the calling instance's exported `memory`, or None."""
    return caller.get("memory")


def _proc_exit(state: WasiState) -> Callable:
    def proc_exit(caller: Caller, code: int) -> None:
        state.exit_code = code
        raise WasiExit(code)
    return proc_exit


def _fd_read(state: WasiState) -> Callable:
    def fd_read(
        caller:    Caller,
        fd:        int,
        iovs_ptr:  int,
        iovs_len:  int,
        nread_ptr: int,
    ) -> int:
        memory = _memory(caller)
        if memory is None:
            return ERRNO_BADF

        # stdin (fd 0) reads from the launcher-provided JSON buffer.
        if fd == 0:
            iovecs = memory.read(caller, iovs_ptr, iovs_ptr + iovs_len * 8)
            total_read = 0
            remaining  = state.stdin_bytes[state.stdin_offset:]
            for i in range(iovs_len):
                off = i * 8
                buf_ptr = int.from_bytes(iovecs[off:off+4],     "little")
                buf_len = int.from_bytes(iovecs[off+4:off+8],   "little")
                if buf_len == 0 or not remaining:
                    continue
                chunk = remaining[:buf_len]
                memory.write(caller, chunk, buf_ptr)
                total_read += len(chunk)
                remaining   = remaining[len(chunk):]

            state.stdin_offset += total_read
            memory.write(caller, total_read.to_bytes(4, "little"), nread_ptr)
            return ERRNO_SUCCESS

        # Non-stdin: must be a row in the fd_table opened for read.
        row = state.fd_table.get(fd)
        if row is None or row.mode != "r":
            return ERRNO_BADF

        iovecs = memory.read(caller, iovs_ptr, iovs_ptr + iovs_len * 8)
        total_read = 0
        try:
            row.file_handle.seek(row.position)
            for i in range(iovs_len):
                off = i * 8
                buf_ptr = int.from_bytes(iovecs[off:off+4],   "little")
                buf_len = int.from_bytes(iovecs[off+4:off+8], "little")
                if buf_len == 0:
                    continue
                chunk = row.file_handle.read(buf_len)
                if not chunk:
                    break
                memory.write(caller, chunk, buf_ptr)
                total_read += len(chunk)
            row.position += total_read
        except OSError:
            return ERRNO_IO

        memory.write(caller, total_read.to_bytes(4, "little"), nread_ptr)
        return ERRNO_SUCCESS
    return fd_read


def _fd_write(state: WasiState) -> Callable:
    def fd_write(
        caller:       Caller,
        fd:           int,
        iovs_ptr:     int,
        iovs_len:     int,
        nwritten_ptr: int,
    ) -> int:
        memory = _memory(caller)
        if memory is None:
            return ERRNO_BADF

        # stdout/stderr behaviour: fd 1 collects into stdout_buffer; fd 2
        # is consumed and discarded. Both report total bytes written.
        if fd in (1, 2):
            iovecs = memory.read(caller, iovs_ptr, iovs_ptr + iovs_len * 8)
            total_written = 0
            for i in range(iovs_len):
                off = i * 8
                buf_ptr = int.from_bytes(iovecs[off:off+4],   "little")
                buf_len = int.from_bytes(iovecs[off+4:off+8], "little")
                if buf_len == 0:
                    continue
                data = memory.read(caller, buf_ptr, buf_ptr + buf_len)
                if fd == 1:
                    state.stdout_buffer.extend(data)
                total_written += buf_len

            memory.write(caller, total_written.to_bytes(4, "little"), nwritten_ptr)
            return ERRNO_SUCCESS

        # Non-stdout/stderr: must be a row in fd_table opened for write.
        row = state.fd_table.get(fd)
        if row is None or row.mode != "w":
            return ERRNO_BADF

        iovecs = memory.read(caller, iovs_ptr, iovs_ptr + iovs_len * 8)
        total_written = 0
        try:
            row.file_handle.seek(row.position)
            for i in range(iovs_len):
                off = i * 8
                buf_ptr = int.from_bytes(iovecs[off:off+4],   "little")
                buf_len = int.from_bytes(iovecs[off+4:off+8], "little")
                if buf_len == 0:
                    continue
                data = memory.read(caller, buf_ptr, buf_ptr + buf_len)
                row.file_handle.write(bytes(data))
                total_written += buf_len
            row.file_handle.flush()
            row.position += total_written
        except OSError:
            return ERRNO_IO

        memory.write(caller, total_written.to_bytes(4, "little"), nwritten_ptr)
        return ERRNO_SUCCESS
    return fd_write


def _read_path(caller: Caller, path_ptr: int, path_len: int) -> str | None:
    """Read a path string from WASM memory. Returns None on memory failure."""
    memory = _memory(caller)
    if memory is None:
        return None
    try:
        raw = memory.read(caller, path_ptr, path_ptr + path_len)
    except Exception:
        return None
    try:
        return bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _scope_for_effect(state: WasiState, effect: Effect) -> Path | None:
    """Return the path scope of the first handle in `state.handle_table`
    granting `effect`.

    Path-effect handles are minted by the launcher's `_mint_handles` with
    one row per granted effect. The host wrapper looks the row up by
    effect rather than by handle id because WASI imports do not carry
    a handle argument. If no matching row exists, returns None which
    callers treat as "no grant" and refuse the call.
    """
    for row in state.handle_table.rows():
        if row.effect == effect:
            return row.path_scope
    return None


def _path_open(state: WasiState) -> Callable:
    def path_open(
        caller:               Caller,
        dirfd:                int,
        dirflags:             int,
        path_ptr:             int,
        path_len:             int,
        oflags:               int,
        fs_rights_base:       int,
        fs_rights_inheriting: int,
        fdflags:              int,
        fd_out_ptr:           int,
    ) -> int:
        memory = _memory(caller)
        if memory is None:
            return ERRNO_BADF

        path_str = _read_path(caller, path_ptr, path_len)
        if path_str is None:
            return ERRNO_INVAL

        # Decide read vs write from fs_rights_base. If FD_WRITE is set we
        # treat this as a write open; otherwise a read open. Some callers
        # set both bits, in which case write wins (open for read+write
        # falls back to write semantics for our tests).
        wants_write = bool(fs_rights_base & _RIGHTS_FD_WRITE)
        effect = Effect.LOCAL_WRITE if wants_write else Effect.LOCAL_READ
        scope = _scope_for_effect(state, effect)
        if scope is None:
            return ERRNO_PERM

        try:
            candidate = Path(path_str)
            if not candidate.is_absolute():
                candidate = scope / candidate
        except (TypeError, ValueError):
            return ERRNO_INVAL

        if not _path_in_scope(candidate, scope):
            return ERRNO_NOENT

        try:
            if wants_write:
                # WASI O_CREAT bit is 1<<0; O_TRUNC bit is 1<<3.
                creat = bool(oflags & 0x1)
                trunc = bool(oflags & 0x8)
                if creat:
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                # "wb" truncates and creates. "r+b" updates without trunc.
                if trunc or creat:
                    fh = open(candidate, "wb")
                else:
                    fh = open(candidate, "r+b")
                mode = "w"
            else:
                fh = open(candidate, "rb")
                mode = "r"
        except FileNotFoundError:
            return ERRNO_NOENT
        except PermissionError:
            return ERRNO_PERM
        except OSError:
            return ERRNO_IO

        fd_id = state.next_fd
        state.next_fd += 1
        state.fd_table[fd_id] = FileDescriptorRow(
            fd_id=fd_id,
            path=candidate,
            mode=mode,
            file_handle=fh,
            position=0,
        )
        try:
            memory.write(caller, fd_id.to_bytes(4, "little"), fd_out_ptr)
        except Exception:
            return ERRNO_INVAL
        return ERRNO_SUCCESS
    return path_open


def _fd_seek(state: WasiState) -> Callable:
    def fd_seek(
        caller:         Caller,
        fd:             int,
        offset:         int,
        whence:         int,
        newoffset_ptr:  int,
    ) -> int:
        memory = _memory(caller)
        if memory is None:
            return ERRNO_BADF
        row = state.fd_table.get(fd)
        if row is None:
            return ERRNO_BADF

        if whence == 0:        # SET
            new_pos = offset
        elif whence == 1:      # CUR
            new_pos = row.position + offset
        elif whence == 2:      # END
            try:
                end = row.path.stat().st_size
            except OSError:
                return ERRNO_IO
            new_pos = end + offset
        else:
            return ERRNO_INVAL

        if new_pos < 0:
            return ERRNO_INVAL

        row.position = new_pos
        try:
            memory.write(caller, new_pos.to_bytes(8, "little", signed=False), newoffset_ptr)
        except Exception:
            return ERRNO_INVAL
        return ERRNO_SUCCESS
    return fd_seek


def _fd_close(state: WasiState) -> Callable:
    def fd_close(_caller: Caller, fd: int) -> int:
        row = state.fd_table.pop(fd, None)
        if row is None:
            return ERRNO_BADF
        try:
            row.file_handle.close()
        except OSError:
            return ERRNO_IO
        return ERRNO_SUCCESS
    return fd_close


def _path_filestat_get(state: WasiState) -> Callable:
    def path_filestat_get(
        caller:    Caller,
        _fd:       int,
        _flags:    int,
        path_ptr:  int,
        path_len:  int,
        buf_ptr:   int,
    ) -> int:
        memory = _memory(caller)
        if memory is None:
            return ERRNO_BADF

        path_str = _read_path(caller, path_ptr, path_len)
        if path_str is None:
            return ERRNO_INVAL

        scope = _scope_for_effect(state, Effect.LOCAL_READ)
        if scope is None:
            return ERRNO_PERM

        try:
            candidate = Path(path_str)
            if not candidate.is_absolute():
                candidate = scope / candidate
        except (TypeError, ValueError):
            return ERRNO_INVAL

        if not _path_in_scope(candidate, scope):
            return ERRNO_NOENT

        if not candidate.exists():
            return ERRNO_NOENT

        # Stub filestat: 64 zero bytes. Real fields (dev, ino, filetype,
        # nlink, size, atim, mtim, ctim) are not surfaced; tests that
        # need them belong to a later phase.
        try:
            memory.write(caller, b"\x00" * 64, buf_ptr)
        except Exception:
            return ERRNO_INVAL
        return ERRNO_SUCCESS
    return path_filestat_get


def _path_create_directory(state: WasiState) -> Callable:
    def path_create_directory(
        caller:    Caller,
        _fd:       int,
        path_ptr:  int,
        path_len:  int,
    ) -> int:
        path_str = _read_path(caller, path_ptr, path_len)
        if path_str is None:
            return ERRNO_INVAL

        scope = _scope_for_effect(state, Effect.LOCAL_WRITE)
        if scope is None:
            return ERRNO_PERM

        try:
            candidate = Path(path_str)
            if not candidate.is_absolute():
                candidate = scope / candidate
        except (TypeError, ValueError):
            return ERRNO_INVAL

        if not _path_in_scope(candidate, scope):
            return ERRNO_NOENT

        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return ERRNO_PERM
        except OSError:
            return ERRNO_IO
        return ERRNO_SUCCESS
    return path_create_directory


def _environ_get(_state: WasiState) -> Callable:
    def environ_get(_caller: Caller, _environ_ptr: int, _environ_buf_ptr: int) -> int:
        return ERRNO_SUCCESS
    return environ_get


def _environ_sizes_get(_state: WasiState) -> Callable:
    def environ_sizes_get(
        caller:        Caller,
        count_ptr:     int,
        buf_size_ptr:  int,
    ) -> int:
        memory = _memory(caller)
        if memory is None:
            return ERRNO_BADF
        memory.write(caller, (0).to_bytes(4, "little"), count_ptr)
        memory.write(caller, (0).to_bytes(4, "little"), buf_size_ptr)
        return ERRNO_SUCCESS
    return environ_sizes_get


def _random_get(_state: WasiState) -> Callable:
    def random_get(caller: Caller, buf_ptr: int, buf_len: int) -> int:
        if buf_len == 0:
            return ERRNO_SUCCESS
        memory = _memory(caller)
        if memory is None:
            return ERRNO_BADF
        memory.write(caller, os.urandom(buf_len), buf_ptr)
        return ERRNO_SUCCESS
    return random_get


# FuncType signatures are described as (param ValTypes, result ValTypes)
# tuples and constructed per-call inside define_into_linker, because
# wasmtime FuncType objects bind to the first Engine they touch and
# reusing them across Engines triggers a "comes_from_same_engine" panic.
_HANDLERS: dict[str, tuple[
    Callable[[], list],
    Callable[[], list],
    Callable[[WasiState], Callable],
]] = {
    "proc_exit":             (lambda: [ValType.i32()],
                              lambda: [],
                              _proc_exit),
    "fd_read":               (lambda: [ValType.i32(), ValType.i32(),
                                       ValType.i32(), ValType.i32()],
                              lambda: [ValType.i32()],
                              _fd_read),
    "fd_write":              (lambda: [ValType.i32(), ValType.i32(),
                                       ValType.i32(), ValType.i32()],
                              lambda: [ValType.i32()],
                              _fd_write),
    "fd_close":              (lambda: [ValType.i32()],
                              lambda: [ValType.i32()],
                              _fd_close),
    "fd_seek":               (lambda: [ValType.i32(), ValType.i64(),
                                       ValType.i32(), ValType.i32()],
                              lambda: [ValType.i32()],
                              _fd_seek),
    "path_open":             (lambda: [ValType.i32(), ValType.i32(),
                                       ValType.i32(), ValType.i32(),
                                       ValType.i32(), ValType.i64(),
                                       ValType.i64(), ValType.i32(),
                                       ValType.i32()],
                              lambda: [ValType.i32()],
                              _path_open),
    "path_filestat_get":     (lambda: [ValType.i32(), ValType.i32(),
                                       ValType.i32(), ValType.i32(),
                                       ValType.i32()],
                              lambda: [ValType.i32()],
                              _path_filestat_get),
    "path_create_directory": (lambda: [ValType.i32(), ValType.i32(),
                                       ValType.i32()],
                              lambda: [ValType.i32()],
                              _path_create_directory),
    "environ_get":           (lambda: [ValType.i32(), ValType.i32()],
                              lambda: [ValType.i32()],
                              _environ_get),
    "environ_sizes_get":     (lambda: [ValType.i32(), ValType.i32()],
                              lambda: [ValType.i32()],
                              _environ_sizes_get),
    "random_get":            (lambda: [ValType.i32(), ValType.i32()],
                              lambda: [ValType.i32()],
                              _random_get),
}


def supported_imports() -> frozenset[str]:
    """Return the set of WASI imports this module can wire."""
    return frozenset(_HANDLERS)


def define_into_linker(
    linker:    Linker,
    state:     WasiState,
    imports:   frozenset[str],
    *,
    module:    str = "wasi_snapshot_preview1",
) -> None:
    """Wire the listed WASI imports into the linker, closing over state.

    Imports outside `_HANDLERS` are silently skipped; the caller is
    expected to validate first via `unsupported_imports()`.
    """
    for name in imports:
        spec = _HANDLERS.get(name)
        if spec is None:
            continue
        params_factory, results_factory, handler_factory = spec
        func_type = FuncType(params_factory(), results_factory())
        linker.define_func(
            module,
            name,
            func_type,
            handler_factory(state),
            access_caller=True,
        )


def unsupported_imports(imports: frozenset[str]) -> frozenset[str]:
    """Return imports the caller asked for that this module cannot wire."""
    return frozenset(name for name in imports if name not in _HANDLERS)
