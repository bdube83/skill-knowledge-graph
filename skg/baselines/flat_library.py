"""Baseline D: flat tool library runtime.

Section 7.2 of the paper defines Baseline D as a system that exposes
the same nodes as SKG, with no graph composition and no capability
tokens. The runtime loads the .wasm artifact, wires the full WASI
surface, and trusts the caller's `granted_effects` list without any
manifest validation.

Baseline D differs from Baseline E (`DeclaredCapabilityRuntime`) in one
respect: D performs no manifest check at all, even at declaration
time. The point of D in the paper is to isolate the value of
capability tokens (E vs D) from the value of graph composition (T vs D).

The class implements the same `execute(...)` signature as
`skg.wasmtime_launcher.WasmtimeRuntime`. The eval harness can swap
runtimes without changing call sites.

Reference:
  designs/in-progress/skill-graph-codex-v10/paper-draft.md  Section 7.2
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

from ..wasmtime_launcher import WasmRunResult


__all__ = ["FlatToolLibraryRuntime", "WasmRunResult"]


DEFAULT_TIMEOUT_MS = 5_000


class FlatToolLibraryRuntime:
    """Wasmtime-based execution runtime for the flat-tool-library baseline.

    Each `execute()` call loads the WASM module, builds a fresh Linker
    with the full WASI surface, and runs the module in a fresh Store.
    The `granted_effects` list is passed through to the node payload
    unchanged. The runtime performs no manifest validation and no
    effect filtering.

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
        """Execute a WASI node under the flat-tool-library baseline."""
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

        # No manifest validation. The caller's grant list passes through
        # to the node payload unchanged.
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
            prefix=f"skg-flat-{node_id}-",
            suffix=".stdin",
            delete=False,
        ) as stdin_tmp:
            stdin_tmp.write(payload)
            stdin_path = Path(stdin_tmp.name)

        try:
            wasi_cfg = WasiConfig()
            wasi_cfg.stdin_file    = stdin_path
            wasi_cfg.stdout_custom = _capture_stdout

            linker = Linker(self._engine)
            # Full WASI surface, no per-grant import gate.
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
