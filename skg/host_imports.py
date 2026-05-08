"""Custom host imports for SKG capability grants.

Phase 3b of the WASM import-level enforcement work
(designs/proposed/skg-wasm-import-enforcement.md). These imports live
in the `skg` Wasmtime module namespace and are wired into the linker
only when the corresponding grant is present. A node that imports
`skg.http_get` without `network.read` granted fails at instantiate-time
because the import is absent from the linker.

What Phase 3b adds. Each host import is now a per-call grant-validating
wrapper backed by the per-run `HandleTable` on `WasiState`. The wrapper:

  1. Reads the integer handle id passed by the caller.
  2. Decodes the JSON payload from WASM memory.
  3. Validates the handle against the requested effect, the URL pattern
     (for network/external/browser effects), or the path scope (for
     local effects).
  4. For approval-gated effects, checks the approval_token argument.
  5. Writes a stub response (`{"stub": true, "effect": "..."}`) to the
     response buffer in WASM memory.
  6. Writes the response length to actual_len_ptr.
  7. Returns ERRNO_SUCCESS or ERRNO_DENIED.

What this still does not do. No real HTTP, file I/O, git, browser, or
LLM operation runs. Real adapters are downstream work; the Phase 3b
wrappers exist to prove that grant validation happens on every call.

Common signature for non-approval imports:
    (handle: i32,
     payload_ptr: i32,
     payload_len: i32,
     response_ptr: i32,
     response_max_len: i32,
     actual_len_ptr: i32) -> errno: i32

Common signature for approval-gated imports
(`skg.external_send`, `skg.git_write`, `skg.production_write`):
    (handle: i32,
     payload_ptr: i32,
     payload_len: i32,
     response_ptr: i32,
     response_max_len: i32,
     actual_len_ptr: i32,
     approval_token: i32) -> errno: i32
"""

from __future__ import annotations

import json
from typing import Callable

from wasmtime import Caller, FuncType, Linker, Memory, ValType

from . import host_adapters
from .cap_to_imports import APPROVAL_HOST
from .effects import Effect
from .wasi_minimal import WasiState


SKG_MODULE = "skg"

ERRNO_SUCCESS  = 0
ERRNO_DENIED   = 13
ERRNO_INVAL    = 28
ERRNO_NOT_IMPL = 58


# Map qualified host import names to the Effect they authorise.
_HOST_TO_EFFECT: dict[str, Effect] = {
    "skg.http_get":         Effect.NETWORK_READ,
    "skg.http_post":        Effect.NETWORK_WRITE,
    "skg.external_draft":   Effect.EXTERNAL_DRAFT,
    "skg.external_send":    Effect.EXTERNAL_SEND,
    "skg.browser_read":     Effect.BROWSER_READ,
    "skg.browser_write":    Effect.BROWSER_WRITE,
    "skg.git_read":         Effect.GIT_READ,
    "skg.git_write":        Effect.GIT_WRITE,
    "skg.secret_read":      Effect.SECRET_READ,
    "skg.production_write": Effect.PRODUCTION_WRITE,
    "skg.text_generate":    Effect.TEXT_GENERATE,
}


# Effects that look up a URL inside the payload at the "url" key.
_URL_EFFECTS: frozenset[Effect] = frozenset({
    Effect.NETWORK_READ,
    Effect.NETWORK_WRITE,
    Effect.EXTERNAL_DRAFT,
    Effect.EXTERNAL_SEND,
    Effect.BROWSER_READ,
    Effect.BROWSER_WRITE,
})


# Effects that look up a path inside the payload at the "path" key.
_PATH_EFFECTS: frozenset[Effect] = frozenset()


def _short_name(qualified: str) -> str:
    """Strip the `skg.` prefix from a qualified host import name."""
    return qualified.split(".", 1)[1] if "." in qualified else qualified


def _memory(caller: Caller) -> Memory | None:
    """Return the calling instance's exported `memory`, or None."""
    return caller.get("memory")


def _read_payload(caller: Caller, ptr: int, length: int) -> dict | None:
    """Read `length` bytes at `ptr` and decode as a JSON object.

    Returns None when memory is missing, the read fails, the bytes are
    not valid JSON, or the decoded value is not a dict.
    """
    if length == 0:
        return {}
    memory = _memory(caller)
    if memory is None:
        return None
    try:
        raw = memory.read(caller, ptr, ptr + length)
    except Exception:
        return None
    try:
        decoded = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def _write_response(
    caller:           Caller,
    response_ptr:     int,
    response_max_len: int,
    actual_len_ptr:   int,
    body:             bytes,
) -> int:
    """Write `body` to memory at response_ptr and the length to actual_len_ptr.

    Returns ERRNO_SUCCESS on success, ERRNO_INVAL on memory or capacity
    failures.
    """
    memory = _memory(caller)
    if memory is None:
        return ERRNO_INVAL
    if len(body) > response_max_len:
        body = body[:response_max_len]
    try:
        memory.write(caller, body, response_ptr)
        memory.write(caller, len(body).to_bytes(4, "little"), actual_len_ptr)
    except Exception:
        return ERRNO_INVAL
    return ERRNO_SUCCESS


def _stub_body(effect: Effect) -> bytes:
    """Return the canonical stub response bytes for an effect."""
    return json.dumps({"stub": True, "effect": effect.value}).encode("utf-8")


def _make_no_approval_handler(
    state:          WasiState,
    effect:         Effect,
    qualified_name: str,
) -> Callable:
    def handler(
        caller:           Caller,
        handle:           int,
        payload_ptr:      int,
        payload_len:      int,
        response_ptr:     int,
        response_max_len: int,
        actual_len_ptr:   int,
    ) -> int:
        payload = _read_payload(caller, payload_ptr, payload_len)
        if payload is None:
            return ERRNO_INVAL

        url = payload.get("url") if effect in _URL_EFFECTS else None
        path = payload.get("path") if effect in _PATH_EFFECTS else None

        if not state.handle_table.validate(handle, effect, url=url, path=path):
            return ERRNO_DENIED

        if host_adapters.has_adapter(qualified_name):
            errno, body = host_adapters.ADAPTERS[qualified_name](payload)
            if errno != ERRNO_SUCCESS:
                return errno
            return _write_response(
                caller,
                response_ptr,
                response_max_len,
                actual_len_ptr,
                body,
            )

        return _write_response(
            caller,
            response_ptr,
            response_max_len,
            actual_len_ptr,
            _stub_body(effect),
        )
    return handler


def _make_approval_handler(
    state:          WasiState,
    effect:         Effect,
    qualified_name: str,
) -> Callable:
    def handler(
        caller:           Caller,
        handle:           int,
        payload_ptr:      int,
        payload_len:      int,
        response_ptr:     int,
        response_max_len: int,
        actual_len_ptr:   int,
        approval_token:   int,
    ) -> int:
        if approval_token == 0:
            return ERRNO_DENIED

        payload = _read_payload(caller, payload_ptr, payload_len)
        if payload is None:
            return ERRNO_INVAL

        url = payload.get("url") if effect in _URL_EFFECTS else None
        path = payload.get("path") if effect in _PATH_EFFECTS else None

        ok = state.handle_table.validate(
            handle,
            effect,
            url=url,
            path=path,
            approval_token=approval_token,
        )
        if not ok:
            return ERRNO_DENIED

        if host_adapters.has_adapter(qualified_name):
            errno, body = host_adapters.ADAPTERS[qualified_name](payload)
            if errno != ERRNO_SUCCESS:
                return errno
            return _write_response(
                caller,
                response_ptr,
                response_max_len,
                actual_len_ptr,
                body,
            )

        return _write_response(
            caller,
            response_ptr,
            response_max_len,
            actual_len_ptr,
            _stub_body(effect),
        )
    return handler


def _params_no_approval() -> list:
    return [
        ValType.i32(),  # handle
        ValType.i32(),  # payload_ptr
        ValType.i32(),  # payload_len
        ValType.i32(),  # response_ptr
        ValType.i32(),  # response_max_len
        ValType.i32(),  # actual_len_ptr
    ]


def _params_approval() -> list:
    return _params_no_approval() + [ValType.i32()]  # approval_token


def _results() -> list:
    return [ValType.i32()]


def define_into_linker(
    linker:        Linker,
    host_imports:  frozenset[str],
    state:         WasiState | None = None,
) -> None:
    """Wire the listed custom host imports into the linker.

    Each entry in `host_imports` is a qualified name like `skg.http_get`.
    The function defines the corresponding short name in the `skg`
    module namespace. Each handler validates its handle against the
    per-run `HandleTable` carried by `state`.

    `state` is optional for backwards compatibility with callers that
    pre-date Phase 3b. When omitted, a fresh empty `WasiState` is used,
    which causes every call to fail validation. Production callers
    must always pass the same `WasiState` they wired into the WASI
    layer.
    """
    if state is None:
        state = WasiState()

    for qualified in host_imports:
        short = _short_name(qualified)
        effect = _HOST_TO_EFFECT.get(qualified)
        if effect is None:
            continue
        approval_required = qualified in APPROVAL_HOST
        if approval_required:
            params  = _params_approval()
            handler = _make_approval_handler(state, effect, qualified)
        else:
            params  = _params_no_approval()
            handler = _make_no_approval_handler(state, effect, qualified)
        func_type = FuncType(params, _results())
        linker.define_func(
            SKG_MODULE,
            short,
            func_type,
            handler,
            access_caller=True,
        )
