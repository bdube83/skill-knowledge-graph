"""Tests for skg.mcp_server.

The MCP server registers three tools: skg_route, skg_execute, and
skg_list_nodes. These tests inspect the FastMCP server registration,
then call each tool's underlying handler with monkeypatched SKG
internals so the suite stays offline. No real MCP host or .wasm
artifact is touched.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest


# ---- Helpers --------------------------------------------------------------


@dataclass
class _FakeStage:
    value: str


@dataclass
class _FakeManifest:
    header:    str       = "fake header"
    task_type: str       = "fake_task"
    tags:      list[str] = field(default_factory=lambda: ["alpha", "beta"])


@dataclass
class _FakeNode:
    id:       str
    manifest: _FakeManifest = field(default_factory=_FakeManifest)


@dataclass
class _FakeCap:
    effect: str


@dataclass
class _FakeGrant:
    granted: list[_FakeCap] = field(default_factory=list)


@dataclass
class _FakeRouteResult:
    hit:    bool
    stage:  _FakeStage
    node:   _FakeNode | None     = None
    grant:  _FakeGrant | None    = None
    reason: str                  = ""

    @property
    def miss(self) -> bool:
        return not self.hit


def _call_tool(server, name, arguments):
    """Invoke a FastMCP tool by name and decode its JSON content payload."""
    blocks = asyncio.run(server.call_tool(name, arguments))
    # FastMCP returns a (content_blocks, structured_or_dict) pair when
    # the tool produces structured output, or just content blocks for
    # plain text. We accept both shapes here.
    if isinstance(blocks, tuple) and len(blocks) == 2:
        _, structured = blocks
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured
    if isinstance(blocks, list):
        # Last-resort: parse JSON text from the first text block.
        for block in blocks:
            text = getattr(block, "text", None)
            if text:
                return json.loads(text)
    return blocks


# ---- Tests ----------------------------------------------------------------


def test_server_registers_exactly_three_tools():
    """The server exposes skg_route, skg_execute, and skg_list_nodes."""
    from skg import mcp_server

    server = mcp_server._build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"skg_route", "skg_execute", "skg_list_nodes"}


def test_skg_route_hit_shape(monkeypatch):
    """skg_route returns the documented shape on a hit."""
    from skg import mcp_server

    def _fake_route(task, context=None):  # noqa: ARG001
        return _FakeRouteResult(
            hit=True,
            stage=_FakeStage("exact"),
            node=_FakeNode(id="node-1"),
            grant=_FakeGrant(granted=[_FakeCap(effect="external.draft")]),
        )

    monkeypatch.setattr(
        "skg.integrations.agent_proxy.route_proposal", _fake_route,
    )
    server = mcp_server._build_server()
    payload = _call_tool(server, "skg_route", {"task": "do the thing"})

    assert payload["hit"] is True
    assert payload["stage"] == "exact"
    assert payload["node_id"] == "node-1"
    assert payload["header"] == "fake header"
    # Reason is None or an empty-equivalent on hit.
    assert payload["reason"] in (None, "")


def test_skg_route_miss_shape(monkeypatch):
    """skg_route returns the documented shape on a miss."""
    from skg import mcp_server

    def _fake_route(task, context=None):  # noqa: ARG001
        return _FakeRouteResult(
            hit=False,
            stage=_FakeStage("miss"),
            reason="no match",
        )

    monkeypatch.setattr(
        "skg.integrations.agent_proxy.route_proposal", _fake_route,
    )
    server = mcp_server._build_server()
    payload = _call_tool(server, "skg_route", {"task": "unmatched"})

    assert payload["hit"] is False
    assert payload["stage"] == "miss"
    assert payload["node_id"] is None
    assert payload["header"] is None
    assert payload["reason"] == "no match"


def test_skg_execute_returns_runtime_shape(monkeypatch, tmp_path):
    """skg_execute returns success/output/error/duration/effects."""
    from skg import mcp_server

    fake_node = _FakeNode(id="exec-node")

    class _FakeSKG:
        def __init__(self, *a, **kw):  # noqa: ARG002
            pass

        def get_node(self, node_id):  # noqa: ARG002
            return fake_node

    @dataclass
    class _FakeRun:
        node_id:          str = "exec-node"
        success:          bool = True
        output:           dict = field(default_factory=lambda: {"ok": True})
        error:            str = ""
        duration_ms:      float = 12.5
        observed_effects: list[str] = field(default_factory=lambda: ["external.draft"])

    class _FakeRuntime:
        def __init__(self, *a, **kw):  # noqa: ARG002
            pass

        def execute(self, **kwargs):  # noqa: ARG002
            return _FakeRun()

    wasm_stub = tmp_path / "node.wasm"
    wasm_stub.write_bytes(b"\x00asm")  # any non-empty body satisfies exists()

    monkeypatch.setattr("skg.graph.SKG", _FakeSKG)
    monkeypatch.setattr(
        "skg.wasmtime_launcher.WasmtimeRuntime", _FakeRuntime,
    )
    monkeypatch.setattr(
        "skg.wasmtime_launcher.wasm_path_for_node",
        lambda node_id, skg_root=None: wasm_stub,
    )

    server = mcp_server._build_server()
    payload = _call_tool(
        server,
        "skg_execute",
        {
            "node_id":         "exec-node",
            "task":            "do it",
            "context":         {"k": "v"},
            "granted_effects": ["external.draft"],
            "dry_run":         False,
        },
    )

    assert payload["success"] is True
    assert payload["output"] == {"ok": True}
    assert payload["error"] == ""
    assert payload["duration_ms"] == 12.5
    assert payload["observed_effects"] == ["external.draft"]


def test_skg_execute_missing_artifact(monkeypatch, tmp_path):
    """skg_execute reports the missing-artifact error path."""
    from skg import mcp_server

    class _FakeSKG:
        def __init__(self, *a, **kw):  # noqa: ARG002
            pass

        def get_node(self, node_id):  # noqa: ARG002
            return _FakeNode(id="nope")

    monkeypatch.setattr("skg.graph.SKG", _FakeSKG)
    monkeypatch.setattr(
        "skg.wasmtime_launcher.wasm_path_for_node",
        lambda node_id, skg_root=None: tmp_path / "missing.wasm",
    )

    server = mcp_server._build_server()
    payload = _call_tool(
        server,
        "skg_execute",
        {
            "node_id":         "nope",
            "task":            "x",
            "context":         {},
            "granted_effects": [],
        },
    )
    assert payload["success"] is False
    assert "no WASI artifact" in payload["error"]


def test_skg_list_nodes_returns_list(monkeypatch):
    """skg_list_nodes returns a JSON list with the documented shape."""
    from skg import mcp_server

    nodes = [
        _FakeNode(id="a", manifest=_FakeManifest(header="alpha header", tags=["t1"])),
        _FakeNode(id="b", manifest=_FakeManifest(header="beta header", tags=[])),
    ]

    class _FakeStore:
        def list_active(self):
            return nodes

    class _FakeSKG:
        def __init__(self, *a, **kw):  # noqa: ARG002
            self._store = _FakeStore()

    monkeypatch.setattr("skg.graph.SKG", _FakeSKG)

    server = mcp_server._build_server()
    payload = _call_tool(server, "skg_list_nodes", {})

    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["node_id"] == "a"
    assert payload[0]["task_type"] == "fake_task"
    assert payload[0]["header"] == "alpha header"
    assert payload[0]["tags"] == ["t1"]
    assert payload[1]["node_id"] == "b"
    assert payload[1]["tags"] == []


def test_main_entry_point_exists():
    """The skg-mcp console script target is importable and callable."""
    from skg import mcp_server
    assert callable(mcp_server.main)
    # SERVER is registered at import time; ensure it is a FastMCP instance.
    from mcp.server.fastmcp import FastMCP
    assert isinstance(mcp_server.SERVER, FastMCP)


def test_install_command_prints_claude_code_block(monkeypatch, capsys, tmp_path):
    """`skg install --client claude-code` prints a JSON snippet."""
    from skg import cli

    monkeypatch.setenv("SKG_HOME", str(tmp_path / "skg-home"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "user-home")
    (tmp_path / "user-home").mkdir(parents=True, exist_ok=True)

    rc = cli.main(["install", "--client", "claude-code"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    payload = json.loads(out.split("\n\n")[0]) if "\n\n" in out else json.loads(out)
    assert "mcpServers" in payload
    assert "skg" in payload["mcpServers"]
    assert payload["mcpServers"]["skg"]["command"] == "skg-mcp"


def test_install_write_merges_into_claude_code(monkeypatch, capsys, tmp_path):
    """`--write` merges the snippet into ~/.claude.json."""
    from skg import cli

    home = tmp_path / "user-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: home)

    # Pre-existing config with one unrelated server.
    target = home / ".claude.json"
    target.write_text(
        json.dumps({"mcpServers": {"other": {"command": "x"}}}, indent=2),
        encoding="utf-8",
    )

    rc = cli.main(["install", "--client", "claude-code", "--write"])
    assert rc == cli.EXIT_OK
    capsys.readouterr()

    body = json.loads(target.read_text(encoding="utf-8"))
    assert body["mcpServers"]["other"]["command"] == "x"
    assert body["mcpServers"]["skg"]["command"] == "skg-mcp"


def test_install_chatgpt_desktop_flags_uncertain(capsys):
    """ChatGPT Desktop install path is flagged as uncertain in stderr."""
    from skg import cli

    rc = cli.main(["install", "--client", "chatgpt-desktop"])
    assert rc == cli.EXIT_OK
    captured = capsys.readouterr()
    assert "mcpServers" in captured.out
    assert "not stable" in captured.err or "not known" in captured.err or "note:" in captured.err
