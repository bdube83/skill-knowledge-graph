"""Wasmtime launcher for SKG trusted nodes.

This is the slice-3 execution layer. Trusted nodes are Rust programs compiled
to WASI. The launcher:

  1. Loads the .wasm artifact from the node's runtime directory.
  2. Builds a fresh per-run Linker that wires only the WASI imports the
     grant set permits (via cap_to_imports.wasi_imports_for).
  3. Wires custom host imports for non-WASI grants (skg.http_get, etc.)
     when their grant is present (Phase 3, not yet implemented).
  4. Validates the module's import list against what we wired; any
     unwired import causes Wasmtime to fail at instantiate-time.
  5. Passes the task context + grants as JSON on stdin (the WasiState's
     stdin_bytes).
  6. Reads node output as JSON from the WasiState's stdout_buffer.
  7. Enforces a fuel-based timeout; exceeded runs are marked failed.

Security property: a node that requests `network.read` but was only
granted `external.draft` will fail at instantiate-time. The relevant
host imports are absent from the linker. This is enforced by Wasmtime's
import resolution, not by Python policy checks at call time.

Reference:
  designs/proposed/skg-wasm-import-enforcement.md
  designs/in-progress/skill-graph-codex-v10/design/skill-knowledge-graph.md
  "Runtime slice" gate: node cannot access raw network, filesystem, or
  secrets without a grant handle.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wasmtime import (
    Config,
    Engine,
    Linker,
    Module,
    Store,
)

from . import host_imports as skg_host
from . import wasi_minimal
from .cap_to_imports import host_imports_for, wasi_imports_for
from .effects import Effect


DEFAULT_TIMEOUT_MS = 5_000

WASI_MODULE = "wasi_snapshot_preview1"


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


def _to_effects(granted_effects: list[str]) -> list[Effect]:
    """Translate granted-effect strings into Effect enum members.

    Strings that do not match any Effect class are dropped. The launcher
    treats unknown strings as ungranted; the import check then refuses
    any node that needs WASI or host imports for unknown effects.
    """
    parsed: list[Effect] = []
    for raw in granted_effects:
        try:
            parsed.append(Effect(raw))
        except ValueError:
            continue
    return parsed


class WasmtimeRuntime:
    """Wasmtime-based execution runtime for SKG WASI nodes.

    Each call to execute() loads the .wasm module, builds a fresh
    `Linker` wiring only the imports permitted by the grant set, runs
    the module in a fresh `Store`, and reads JSON output from the
    captured stdout buffer.

    Module caching: compiled modules are cached by wasm path to avoid
    recompilation on every call. Linkers are not cached because they
    close over per-run WasiState.
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
        """Execute a WASI node and return the result."""
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

        effects   = _to_effects(granted_effects)
        wasi_set  = wasi_imports_for(effects)
        host_set  = host_imports_for(effects) if not dry_run else frozenset()

        unsupported = wasi_minimal.unsupported_imports(wasi_set)
        if unsupported:
            return WasmRunResult(
                node_id=node_id,
                success=False,
                error=f"Unsupported WASI imports requested: {sorted(unsupported)}",
            )

        state = wasi_minimal.WasiState()

        # Mint one handle per granted effect. The default scope is
        # wildcard (any URL, any path); the policy engine will replace
        # this with per-grant scopes once Phase 3d wires manifest scope
        # declarations through. Tests that need scoped grants subclass
        # this runtime and override `_mint_handles`.
        handles: dict[str, int] = self._mint_handles(state.handle_table, effects)

        state.stdin_bytes = json.dumps({
            "task":    task,
            "context": context,
            "grants":  granted_effects if not dry_run else [],
            "handles": handles if not dry_run else {},
        }).encode()

        linker = Linker(self._engine)
        wasi_minimal.define_into_linker(linker, state, wasi_set, module=WASI_MODULE)
        skg_host.define_into_linker(linker, host_set, state)

        store = Store(self._engine)
        store.set_fuel(1_000_000_000)

        start = time.monotonic()
        try:
            instance = linker.instantiate(store, module)
            start_fn = instance.exports(store)["_start"]
            start_fn(store)
            success = True
            error   = ""
        except wasi_minimal.WasiExit as exit_signal:
            success = (exit_signal.code == 0)
            error   = "" if success else f"node exited with code {exit_signal.code}"
        except Exception as exc:
            success = False
            error   = str(exc)

        duration_ms = round((time.monotonic() - start) * 1000, 2)

        raw_output = bytes(state.stdout_buffer)

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

    def _mint_handles(self, table, effects: list[Effect]) -> dict[str, int]:
        """Mint one handle per granted effect into `table`.

        Default behaviour: wildcard URL pattern, root path scope,
        approval_token=1 for the three approval-gated effects. Subclass
        and override to inject test-specific scopes.
        """
        approval_effects = {
            Effect.EXTERNAL_SEND,
            Effect.GIT_WRITE,
            Effect.PRODUCTION_WRITE,
        }
        handles: dict[str, int] = {}
        for effect in effects:
            handle_id = table.mint(
                effect,
                url_pattern="*",
                path_scope=Path("/"),
                approval_token=1 if effect in approval_effects else 0,
            )
            handles[effect.value] = handle_id
        return handles

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
