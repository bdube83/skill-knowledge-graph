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
from typing import Callable

from wasmtime import Caller, FuncType, Linker, Memory, ValType

from .handle_table import HandleTable


ERRNO_SUCCESS = 0
ERRNO_BADF    = 8
ERRNO_INVAL   = 28
ERRNO_IO      = 29


@dataclass
class WasiState:
    """Per-run WASI state shared with host function closures."""

    stdin_bytes:   bytes              = b""
    stdin_offset:  int                = 0
    stdout_buffer: bytearray          = field(default_factory=bytearray)
    exit_code:     int | None         = None
    handle_table:  HandleTable        = field(default_factory=HandleTable)


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
        if fd != 0:
            return ERRNO_BADF
        memory = _memory(caller)
        if memory is None:
            return ERRNO_BADF

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
    return fd_read


def _fd_write(state: WasiState) -> Callable:
    def fd_write(
        caller:       Caller,
        fd:           int,
        iovs_ptr:     int,
        iovs_len:     int,
        nwritten_ptr: int,
    ) -> int:
        if fd not in (1, 2):
            return ERRNO_BADF
        memory = _memory(caller)
        if memory is None:
            return ERRNO_BADF

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
    return fd_write


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
    "proc_exit":         (lambda: [ValType.i32()],
                          lambda: [],
                          _proc_exit),
    "fd_read":           (lambda: [ValType.i32(), ValType.i32(),
                                   ValType.i32(), ValType.i32()],
                          lambda: [ValType.i32()],
                          _fd_read),
    "fd_write":          (lambda: [ValType.i32(), ValType.i32(),
                                   ValType.i32(), ValType.i32()],
                          lambda: [ValType.i32()],
                          _fd_write),
    "environ_get":       (lambda: [ValType.i32(), ValType.i32()],
                          lambda: [ValType.i32()],
                          _environ_get),
    "environ_sizes_get": (lambda: [ValType.i32(), ValType.i32()],
                          lambda: [ValType.i32()],
                          _environ_sizes_get),
    "random_get":        (lambda: [ValType.i32(), ValType.i32()],
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
