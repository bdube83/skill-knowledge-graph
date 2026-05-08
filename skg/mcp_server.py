"""MCP server exposing SKG router and runtime as tools.

This server lets any MCP-compliant host (Claude Code, Claude Desktop,
Cursor, ChatGPT Desktop, Codex CLI) call SKG without switching CLIs.

Three tools are registered:

    skg_route(task, context=None)
        Route a task through SKG. Returns hit/stage/node_id/header/reason.

    skg_execute(node_id, task, context, granted_effects, dry_run=False)
        Execute a stored WASI node via the Wasmtime runtime. Returns the
        success flag, output dict, error string, duration, and observed
        effects.

    skg_list_nodes()
        List active nodes in the local store.

The server speaks MCP over stdio. Hosts launch it as a subprocess and
exchange JSON-RPC messages on the child's stdin/stdout. Run via
``python -m skg.mcp_server`` or the ``skg-mcp`` console script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


SERVER_NAME = "skg"


def _stage_value(stage: Any) -> str:
    """Return the string form of a RouteStage enum or pass-through string."""
    return getattr(stage, "value", str(stage))


def _route_payload(result: Any) -> dict[str, Any]:
    """Shape a RouteResult into the JSON contract the MCP tool returns."""
    node = getattr(result, "node", None)
    header = node.manifest.header if node is not None else None
    return {
        "hit":     bool(getattr(result, "hit", False)),
        "stage":   _stage_value(getattr(result, "stage", "")),
        "node_id": node.id if node is not None else None,
        "header":  header,
        "reason":  getattr(result, "reason", "") or None,
    }


def _granted_effects(result: Any) -> list[str]:
    """Extract granted effect strings from a RouteResult, or return []."""
    grant = getattr(result, "grant", None)
    if grant is None:
        return []
    return [c.effect for c in getattr(grant, "granted", [])]


def _build_server() -> FastMCP:
    """Create the FastMCP server and register the three SKG tools.

    Imports of skg internals happen at tool-call time so the server
    starts cleanly even when the local SKG store has issues that the
    host can surface as a tool error rather than a startup crash.
    """
    server = FastMCP(SERVER_NAME)

    @server.tool(
        name="skg_route",
        description=(
            "Route a task through the local Skill Knowledge Graph. "
            "Returns whether a stored procedure matches, the stage that "
            "matched, the node id, and the node header."
        ),
    )
    def skg_route(task: str, context: dict | None = None) -> dict:
        from skg.integrations.agent_proxy import route_proposal
        result = route_proposal(task, context=context)
        return _route_payload(result)

    @server.tool(
        name="skg_execute",
        description=(
            "Execute a stored SKG node by node_id via the Wasmtime "
            "runtime. Returns success flag, output dict, error string, "
            "duration in ms, and observed effects."
        ),
    )
    def skg_execute(
        node_id: str,
        task: str,
        context: dict,
        granted_effects: list[str],
        dry_run: bool = False,
    ) -> dict:
        from skg.graph import SKG
        from skg.wasmtime_launcher import WasmtimeRuntime, wasm_path_for_node

        skg = SKG()
        node = skg.get_node(node_id)
        if node is None:
            return {
                "success":          False,
                "output":           {},
                "error":            f"node not found: {node_id}",
                "duration_ms":      0.0,
                "observed_effects": [],
            }

        wasm_path = wasm_path_for_node(node_id)
        if not wasm_path.exists():
            return {
                "success":          False,
                "output":           {},
                "error":            f"no WASI artifact at {wasm_path}",
                "duration_ms":      0.0,
                "observed_effects": [],
            }

        runtime = WasmtimeRuntime()
        run = runtime.execute(
            wasm_path=wasm_path,
            node_id=node_id,
            task=task,
            context=context or {},
            granted_effects=granted_effects or [],
            dry_run=dry_run,
        )
        return {
            "success":          bool(run.success),
            "output":           run.output or {},
            "error":            run.error or "",
            "duration_ms":      float(run.duration_ms),
            "observed_effects": list(run.observed_effects or []),
        }

    @server.tool(
        name="skg_list_nodes",
        description=(
            "List active SKG nodes in the local store. Returns node id, "
            "task type, header, and tags for each."
        ),
    )
    def skg_list_nodes() -> list[dict]:
        from skg.graph import SKG
        skg = SKG()
        nodes = skg._store.list_active()
        return [
            {
                "node_id":   n.id,
                "task_type": n.manifest.task_type,
                "header":    n.manifest.header,
                "tags":      list(n.manifest.tags),
            }
            for n in nodes
        ]

    return server


# Module-level handle that tests inspect to verify registration.
SERVER = _build_server()


def main() -> None:
    """Run the SKG MCP server over stdio.

    Hosts launch this as a subprocess. The server speaks MCP JSON-RPC
    on stdin/stdout until the host disconnects.
    """
    SERVER.run()


if __name__ == "__main__":
    main()
