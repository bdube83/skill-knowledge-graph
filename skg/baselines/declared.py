"""Baseline E: declared-capability runtime.

This baseline matches Section 7.2 of the paper. The manifest is
checked at declaration time; the runtime then trusts the declaration
and exposes the full WASI snapshot preview1 surface via Wasmtime's
built-in `Linker.define_wasi()`. There is no per-grant import gate
and no custom `skg.*` host-import layer.

The class implements the same `execute(...)` signature as
`skg.wasmtime_launcher.WasmtimeRuntime`. It exists so the paper can
run the adversarial corpus through both runtimes and report a
containment matrix.

Differences from `WasmtimeRuntime`:
  1. Wires the entire WASI surface (path_open, fd_seek, sock_*, ...).
  2. Wires no `skg.*` host imports. Nodes that import `skg.http_get`
     fail at instantiate-time, which is the realistic behaviour of a
     declared-capability runtime that lacks SKG's host-import layer.
  3. Validates declared effects against the `Effect` enum and drops
     unknown strings. This mirrors `_to_effects` in the SKG launcher.

Reference:
  designs/in-progress/skill-graph-codex-v10/paper-draft.md  Section 7.2
  designs/proposed/skg-wasm-import-enforcement.md           Phase 5
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from wasmtime import (
    Config,
    Engine,
    Linker,
    Module,
    Store,
    WasiConfig,
)

from ..effects import Effect
from ..wasmtime_launcher import WasmRunResult


__all__ = ["DeclaredCapabilityRuntime", "WasmRunResult"]


DEFAULT_TIMEOUT_MS = 5_000


def _validate_effects(declared: list[str]) -> list[Effect]:
    """Translate declared-effect strings into Effect enum members.

    Strings that do not match any Effect class are dropped silently.
    This matches the SKG launcher's `_to_effects` behaviour and keeps
    the two runtimes comparable on the same adversarial corpus.
    """
    parsed: list[Effect] = []
    for raw in declared:
        try:
            parsed.append(Effect(raw))
        except ValueError:
            continue
    return parsed


class DeclaredCapabilityRuntime:
    """Wasmtime-based execution runtime for the declared-capability baseline.

    Each `execute()` call loads the WASM module, builds a fresh Linker
    with the full WASI surface wired via `define_wasi()`, and runs the
    module in a fresh Store. Stdin and stdout are routed through a
    `WasiConfig` whose `stdin_file` points at a tmp file holding the
    JSON payload and whose `stdout_custom` callback captures bytes.

    Module caching: compiled modules are cached by wasm path and mtime
    to avoid recompilation. Linkers are not cached because the WASI
    state binds to the per-run Store.
    """

    def __init__(self, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        cfg = Config()
        cfg.consume_fuel = True
        self._engine     = Engine(cfg)
        self._timeout_ms = timeout_ms
        self._cache: dict[str, tuple[float, Module]] = {}

    def execute(
        self,
        wasm_path: Path | str,
        node_id: str,
        task: str,
        context: dict[str, Any],
        granted_effects: list[str],
        dry_run: bool = False,
    ) -> WasmRunResult:
        """Execute a WASI node under the declared-capability baseline."""
        path = Path(wasm_path)
        if not path.exists():
            return WasmRunResult(
                node_id=node_id,
                success=False,
                error=f"WASM artifact not found: {path}",
            )

        module = self._load_module(path)
        if module is None:
            return WasmRunResult(
                node_id=node_id,
                success=False,
                error=f"Failed to compile WASM module: {path}",
            )

        # The declaration check is the only effect-level gate in this
        # baseline. The runtime trusts the declaration after this point.
        _validate_effects(granted_effects)

        payload = json.dumps({
            "task":    task,
            "context": context,
            "grants":  granted_effects if not dry_run else [],
        }).encode()

        stdout_buffer = bytearray()

        def _capture_stdout(chunk: bytes) -> int:
            stdout_buffer.extend(chunk)
            return len(chunk)

        with tempfile.NamedTemporaryFile(
            prefix=f"skg-baseline-{node_id}-",
            suffix=".stdin",
            delete=False,
        ) as stdin_tmp:
            stdin_tmp.write(payload)
            stdin_path = Path(stdin_tmp.name)

        try:
            wasi_cfg = WasiConfig()
            wasi_cfg.stdin_file     = stdin_path
            wasi_cfg.stdout_custom  = _capture_stdout

            linker = Linker(self._engine)
            # Full WASI surface, intentionally. Baseline E does not
            # trim imports by grant.
            linker.define_wasi()

            store = Store(self._engine)
            store.set_wasi(wasi_cfg)
            store.set_fuel(1_000_000_000)

            start = time.monotonic()
            try:
                instance = linker.instantiate(store, module)
                start_fn = instance.exports(store)["_start"]
                start_fn(store)
                success = True
                error   = ""
            except Exception as exc:
                # Wasmtime raises a generic error for proc_exit; treat
                # exit code 0 as success when the module wrote output.
                msg = str(exc)
                if "exit" in msg.lower() and ("code 0" in msg or "code: 0" in msg):
                    success = True
                    error   = ""
                else:
                    success = False
                    error   = msg

            duration_ms = round((time.monotonic() - start) * 1000, 2)
        finally:
            try:
                stdin_path.unlink()
            except FileNotFoundError:
                pass

        raw_output = bytes(stdout_buffer)

        output_dict: dict[str, Any] = {}
        observed: list[str]         = []

        if raw_output:
            try:
                parsed = json.loads(raw_output.decode())
                output_dict = parsed.get("output", {})
                observed    = parsed.get("observed_effects", [])
                if "error" in parsed and not output_dict:
                    success = False
                    error   = parsed["error"]
                # When the module wrote a JSON result, treat as success
                # even if proc_exit reached the linker as an exception.
                if output_dict and not error:
                    success = True
            except json.JSONDecodeError as e:
                success = False
                error   = f"Node stdout is not valid JSON: {e}. Raw: {raw_output[:200]!r}"

        return WasmRunResult(
            node_id=node_id,
            success=success,
            output=output_dict,
            error=error,
            duration_ms=duration_ms,
            observed_effects=observed,
        )

    # ---- Module cache -------------------------------------------------------

    def _load_module(self, path: Path) -> Module | None:
        mtime = path.stat().st_mtime
        cached = self._cache.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            module = Module.from_file(self._engine, str(path))
            self._cache[str(path)] = (mtime, module)
            return module
        except Exception:
            return None
