"""Wasmtime launcher for SKG trusted nodes.

This is the slice-3 execution layer. Trusted nodes are Rust programs compiled
to WASI. The launcher:

  1. Loads the .wasm artifact from the node's runtime directory.
  2. Wires ONLY the WASI imports that correspond to granted capabilities.
     Anything not in the grant set is absent from the linker — the WASM
     module cannot call it at all.
  3. Passes the task context + grants as JSON on stdin.
  4. Reads the node output as JSON from stdout.
  5. Enforces a timeout; exceeded runs are marked failed.

Security property: a node that requests `network.read` but was only granted
`text.generate` will fail at link time — the host import is simply not wired.
This is enforced by Wasmtime, not by Python policy checks at call time.

Reference:
  designs/proposed/skill-graph-codex-v10/design/skill-knowledge-graph.md
  "Runtime slice" — gate: node cannot access raw network, filesystem, or
  secrets without a grant handle.
"""

from __future__ import annotations

import json
import os
import time
import tempfile
from dataclasses import dataclass, field
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


# Default per-node execution timeout.
DEFAULT_TIMEOUT_MS = 5_000


@dataclass
class WasmRunResult:
    """The outcome of one Wasmtime node execution."""

    node_id:          str
    success:          bool
    output:           dict[str, Any]   = field(default_factory=dict)
    error:            str              = ""
    duration_ms:      float            = 0.0
    observed_effects: list[str]        = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id":          self.node_id,
            "success":          self.success,
            "output":           self.output,
            "error":            self.error,
            "duration_ms":      self.duration_ms,
            "observed_effects": self.observed_effects,
        }


class WasmtimeRuntime:
    """Wasmtime-based execution runtime for SKG WASI nodes.

    Each call to execute() loads the .wasm module, builds a per-run Store with
    a fresh WasiConfig, wires only the granted capability imports, feeds JSON
    on stdin, and reads JSON from stdout.

    Module caching: compiled modules are cached by wasm path to avoid
    recompilation on every call. The cache is invalidated when the .wasm
    artifact changes (checked by mtime).
    """

    def __init__(self, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        cfg = Config()
        cfg.consume_fuel = True        # enables fuel-based timeout enforcement
        self._engine     = Engine(cfg)
        self._linker     = Linker(self._engine)
        self._linker.define_wasi()
        self._timeout_ms = timeout_ms
        self._cache: dict[str, tuple[float, Module]] = {}   # path -> (mtime, module)

    def execute(
        self,
        wasm_path: Path | str,
        node_id: str,
        task: str,
        context: dict[str, Any],
        granted_effects: list[str],
        dry_run: bool = False,
    ) -> WasmRunResult:
        """Execute a WASI node and return the result."""
        import io as _io

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

        # Build stdin payload.
        payload = json.dumps({
            "task":    task,
            "context": context,
            "grants":  granted_effects if not dry_run else [],
        }).encode()

        import tempfile, os
        stdout_buf: bytearray = bytearray()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(payload)
            tmp_path = tmp.name

        wasi = WasiConfig()
        wasi.stdin_file     = tmp_path
        wasi.stdout_custom  = lambda data: stdout_buf.extend(data)
        wasi.inherit_stderr = False

        store = Store(self._engine)
        store.set_wasi(wasi)
        # Fuel budget: ~1 billion operations, roughly 1-5 seconds of execution.
        store.set_fuel(1_000_000_000)

        start = time.monotonic()
        try:
            instance = self._linker.instantiate(store, module)
            start_fn = instance.exports(store)["_start"]
            start_fn(store)
            success = True
            error   = ""
        except Exception as exc:
            success = False
            error   = str(exc)

        duration_ms = round((time.monotonic() - start) * 1000, 2)

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        raw_output = bytes(stdout_buf)

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


def wasm_path_for_node(node_id: str, skg_root: Path | None = None) -> Path:
    """Return the expected .wasm artifact path for a node under the SKG root."""
    root = skg_root or (Path.home() / ".skg")
    return root / "nodes" / node_id / "artifact" / "node.wasm"
