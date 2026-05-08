"""skg shell CLI.

skg wraps any LLM CLI behind a Skill Knowledge Graph cache. On each
``skg run "<task>"`` call the CLI routes the task through the local SKG
first. A hit replays a stored procedure for free; a miss falls through
to the configured vendor (OpenAI, Claude CLI, or Codex CLI) and the
vendor's reply is printed.

Subcommands:

    skg init                          create ~/.skg/ and config.toml
    skg run "<task>"                  route via SKG; on miss call vendor
    skg run --vendor claude "<task>"  override vendor for one call
    skg run --json "<task>"           wrap output in a JSON envelope
    skg run --dry-run "<task>"        report SKG hit/miss; no vendor call
    skg run --execute "<task>"        on hit, run the WASI node
    skg list                          list nodes in ~/.skg/
    skg config get vendor             read the configured vendor
    skg config set vendor openai      write the configured vendor
    skg mcp                           run the SKG MCP server over stdio
    skg install --client <host>       print the MCP config block for a host
    skg --version                     print the package version
    skg --help                        print this help

Exit codes:

    0   success (hit, miss-with-vendor-success, dry-run, list, config, init)
    1   bad arguments or invalid config
    2   SKG miss but the configured vendor is unavailable (no key, no binary)
    3   vendor was reached but returned an error
    4   ``--execute`` requested but the node has no WASI artifact
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Public so tests can import it.
DEFAULT_VENDOR  = "openai"
KNOWN_VENDORS   = ("openai", "claude", "codex")
CONFIG_FILENAME = "config.toml"

EXIT_OK             = 0
EXIT_BAD_ARGS       = 1
EXIT_VENDOR_NOAVAIL = 2
EXIT_VENDOR_FAILED  = 3
EXIT_NO_ARTIFACT    = 4


def _version() -> str:
    """Return the installed package version.

    Prefers ``importlib.metadata`` (works for installed wheels). Falls back
    to reading ``pyproject.toml`` from the repo root so ``python -m skg.cli``
    inside a fresh checkout still prints the real version.
    """
    try:
        from importlib.metadata import version
        return version("skill-knowledge-graph")
    except Exception:
        pass
    try:
        import tomllib
        repo_root = Path(__file__).resolve().parent.parent
        with (repo_root / "pyproject.toml").open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    except Exception:
        return "0.0.0"


# ---- Config IO -------------------------------------------------------------

def _skg_root() -> Path:
    """Return the SKG root directory honouring the SKG_HOME override."""
    override = os.environ.get("SKG_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".skg"


def _config_path() -> Path:
    return _skg_root() / CONFIG_FILENAME


def _default_config() -> dict[str, Any]:
    return {"vendor": DEFAULT_VENDOR, "model": ""}


def _serialize_toml(cfg: dict[str, Any]) -> str:
    """Serialise the small flat config dict to TOML.

    The config is one table of string keys to scalars, so a hand-rolled
    writer is enough; pulling in tomli_w would be overkill.
    """
    lines = ["# skg config. See `skg config --help`.\n"]
    for key in sorted(cfg):
        val = cfg[key]
        if isinstance(val, bool):
            lines.append(f"{key} = {'true' if val else 'false'}\n")
        elif isinstance(val, (int, float)):
            lines.append(f"{key} = {val}\n")
        else:
            escaped = str(val).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"\n')
    return "".join(lines)


def _load_config() -> dict[str, Any]:
    """Read the config from disk. Return defaults if the file is missing."""
    path = _config_path()
    if not path.exists():
        return _default_config()
    try:
        import tomllib
        with path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return _default_config()
    cfg = _default_config()
    cfg.update({k: v for k, v in data.items() if isinstance(k, str)})
    return cfg


def _write_config(cfg: dict[str, Any]) -> None:
    root = _skg_root()
    root.mkdir(parents=True, exist_ok=True)
    _config_path().write_text(_serialize_toml(cfg), encoding="utf-8")


# ---- Hit summary container -------------------------------------------------

@dataclass
class HitSummary:
    """Plain shape returned by ``_summarize_hit`` so JSON output is stable."""

    hit:        bool
    stage:      str
    node_id:    str | None
    output:     str
    tokens:     int | None = None


# ---- Subcommand: init ------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Create the SKG root and write a default config if absent."""
    root = _skg_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "nodes").mkdir(parents=True, exist_ok=True)
    path = _config_path()
    if path.exists():
        print(f"already initialised: {path}")
        return EXIT_OK
    _write_config(_default_config())
    print(f"initialised SKG at {root}")
    print(f"wrote default config to {path}")
    return EXIT_OK


# ---- Subcommand: config ----------------------------------------------------

def cmd_config(args: argparse.Namespace) -> int:
    """Read or write a single config key."""
    if args.config_cmd == "get":
        cfg = _load_config()
        if args.key not in cfg:
            print(f"unknown config key: {args.key}", file=sys.stderr)
            return EXIT_BAD_ARGS
        print(cfg[args.key])
        return EXIT_OK

    if args.config_cmd == "set":
        if args.key == "vendor" and args.value not in KNOWN_VENDORS:
            print(
                f"vendor must be one of {list(KNOWN_VENDORS)}, got {args.value!r}",
                file=sys.stderr,
            )
            return EXIT_BAD_ARGS
        cfg = _load_config()
        cfg[args.key] = args.value
        _write_config(cfg)
        print(f"{args.key} = {args.value}")
        return EXIT_OK

    print("config: expected 'get' or 'set'", file=sys.stderr)
    return EXIT_BAD_ARGS


# ---- Subcommand: list ------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    """List nodes in the active SKG store."""
    try:
        from skg.graph import SKG
        skg = SKG()
        nodes = skg._store.list_active() if not args.list_all else skg._store.list_all()
    except Exception as exc:
        print(f"failed to open SKG store: {exc}", file=sys.stderr)
        return EXIT_BAD_ARGS

    if args.json_out:
        out = [
            {
                "id":      n.id,
                "status":  n.status.value if hasattr(n.status, "value") else str(n.status),
                "header":  n.manifest.header,
                "task_type": n.manifest.task_type,
            }
            for n in nodes
        ]
        print(json.dumps(out, indent=2))
        return EXIT_OK

    if not nodes:
        print("no nodes")
        return EXIT_OK
    for n in nodes:
        status = n.status.value if hasattr(n.status, "value") else str(n.status)
        print(f"  {n.id:<40} {status:<10} {n.manifest.header[:60]}")
    return EXIT_OK


# ---- Subcommand: run -------------------------------------------------------

def _summarize_hit(result: Any, execute: bool) -> HitSummary:
    """Build a HitSummary from a RouteResult.

    On hit, the output is the node header by default. With ``execute=True``
    the WASI node runs and stdout from the run is the output. If the node
    has no .wasm artifact the summary's output is an explanatory string and
    the caller maps that to EXIT_NO_ARTIFACT.
    """
    node = getattr(result, "node", None)
    stage = getattr(getattr(result, "stage", None), "value", str(getattr(result, "stage", "")))
    node_id = node.id if node is not None else None

    if not execute or node is None:
        header = node.manifest.header if node is not None else ""
        return HitSummary(hit=True, stage=stage, node_id=node_id, output=header)

    # Execute path: try to run the WASI artifact.
    try:
        from skg.wasmtime_launcher import WasmtimeRuntime, wasm_path_for_node
    except Exception as exc:
        return HitSummary(
            hit=True, stage=stage, node_id=node_id,
            output=f"runtime unavailable: {exc}",
        )

    wasm_path = wasm_path_for_node(node.id, _skg_root())
    if not wasm_path.exists():
        return HitSummary(
            hit=True, stage=stage, node_id=node_id,
            output=f"no WASI artifact at {wasm_path}",
        )

    grants = []
    grant = getattr(result, "grant", None)
    if grant is not None:
        grants = [c.effect for c in grant.granted]

    runtime = WasmtimeRuntime()
    run = runtime.execute(
        wasm_path=wasm_path,
        node_id=node.id,
        task=node.manifest.header,
        context={},
        granted_effects=grants,
    )
    text = json.dumps(run.output) if run.success else f"node failed: {run.error}"
    return HitSummary(hit=True, stage=stage, node_id=node_id, output=text)


def _emit(args: argparse.Namespace, summary: dict[str, Any], plain: str) -> None:
    """Print either the JSON envelope or the plain text body."""
    if args.json_out:
        print(json.dumps(summary, indent=2))
    else:
        print(plain)


def cmd_run(args: argparse.Namespace) -> int:
    """Route a task through SKG and call the vendor on miss."""
    task = args.task
    cfg = _load_config()
    vendor_name = args.vendor or cfg.get("vendor") or DEFAULT_VENDOR
    if vendor_name not in KNOWN_VENDORS:
        print(
            f"vendor must be one of {list(KNOWN_VENDORS)}, got {vendor_name!r}",
            file=sys.stderr,
        )
        return EXIT_BAD_ARGS

    # Routing.
    try:
        from skg.integrations.agent_proxy import route_proposal
        result = route_proposal(task)
    except Exception as exc:
        print(f"SKG routing failed: {exc}", file=sys.stderr)
        return EXIT_BAD_ARGS

    stage = getattr(getattr(result, "stage", None), "value", str(getattr(result, "stage", "")))
    node = getattr(result, "node", None)
    node_id = node.id if node is not None else None

    if args.dry_run:
        envelope = {
            "hit":         bool(result.hit),
            "stage":       stage,
            "node_id":     node_id,
            "output":      "" if result.hit else "(would call vendor: %s)" % vendor_name,
            "tokens_used": None,
            "vendor":      vendor_name,
            "dry_run":     True,
        }
        plain = (
            f"HIT  [{stage}] {node_id}" if result.hit
            else f"MISS would call vendor: {vendor_name}"
        )
        _emit(args, envelope, plain)
        return EXIT_OK

    if result.hit:
        summary = _summarize_hit(result, execute=args.execute)
        if args.execute and summary.output.startswith("no WASI artifact"):
            envelope = {
                "hit":         True,
                "stage":       stage,
                "node_id":     summary.node_id,
                "output":      summary.output,
                "tokens_used": None,
            }
            _emit(args, envelope, summary.output)
            return EXIT_NO_ARTIFACT
        envelope = {
            "hit":         True,
            "stage":       stage,
            "node_id":     summary.node_id,
            "output":      summary.output,
            "tokens_used": summary.tokens,
        }
        _emit(args, envelope, summary.output)
        return EXIT_OK

    # Miss: call the vendor.
    from skg.cli_vendors import (
        ERRNO_NOAVAIL, get_vendor,
    )
    try:
        vendor_fn = get_vendor(vendor_name)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BAD_ARGS

    response = vendor_fn(task, model=cfg.get("model") or None)
    if response.error:
        envelope = {
            "hit":         False,
            "stage":       stage,
            "node_id":     None,
            "output":      "",
            "tokens_used": None,
            "vendor":      vendor_name,
            "error":       response.error,
        }
        _emit(args, envelope, f"vendor error ({vendor_name}): {response.error}")
        return EXIT_VENDOR_NOAVAIL if response.error.startswith(ERRNO_NOAVAIL) else EXIT_VENDOR_FAILED

    envelope = {
        "hit":         False,
        "stage":       stage,
        "node_id":     None,
        "output":      response.text,
        "tokens_used": response.tokens_used,
        "vendor":      vendor_name,
        "model":       response.model,
    }
    _emit(args, envelope, response.text)
    return EXIT_OK


# ---- Subcommand: mcp -------------------------------------------------------

def cmd_mcp(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Run the SKG MCP server over stdio.

    Hosts launch this as a child process and exchange JSON-RPC frames
    on stdin/stdout. The call blocks until the host disconnects.
    """
    try:
        from skg.mcp_server import main as mcp_main
    except ImportError as exc:
        print(f"MCP support not installed: {exc}", file=sys.stderr)
        print("install with: pip install 'mcp>=1.27.0'", file=sys.stderr)
        return EXIT_BAD_ARGS
    mcp_main()
    return EXIT_OK


# ---- Subcommand: install ---------------------------------------------------

KNOWN_CLIENTS = (
    "claude-code",
    "codex",
    "copilot",
)

# Hosts whose config-file location and key layout we can write directly.
# Hosts marked uncertain print the snippet only and leave installation to
# the user. The set is scoped to the three CLI vendors agent-proxy-kit
# routes through (config.json `vendors`: claude, codex, copilot).
_CONFIG_TARGETS: dict[str, dict[str, Any]] = {
    "claude-code": {
        "path":   "~/.claude.json",
        "key":    "mcpServers",
        "certain": True,
    },
    "codex": {
        "path":   None,
        "key":    "mcpServers",
        "certain": False,
        "note":   (
            "Codex CLI 0.129+ accepts MCP servers via "
            "`codex mcp add <name> -- <command>`. The on-disk path "
            "is `~/.codex/config.toml` under `[mcp.servers.<name>]`, "
            "which is TOML and not directly mergeable from this "
            "JSON-shaped snippet. Use the `codex mcp add` CLI."
        ),
    },
    "copilot": {
        "path":   "~/.copilot/mcp-config.json",
        "key":    "mcpServers",
        "certain": True,
        "stdio_extras": {
            "type":  "stdio",
            "tools": ["*"],
        },
    },
}


def _resolve_skg_mcp() -> str:
    """Return the absolute path to ``skg-mcp``, falling back to bare name.

    Looks first in the running interpreter's bin directory (so a fresh
    `pip install -e .` puts the script in a known place even when the
    venv is not activated), then on PATH. If neither resolves, returns
    the bare ``skg-mcp`` so the snippet still parses; the host will
    fail at connect-time with a clear message rather than silently.
    """
    here = Path(sys.executable).parent / "skg-mcp"
    if here.is_file():
        return str(here)
    found = shutil.which("skg-mcp")
    if found:
        return found
    return "skg-mcp"


def _mcp_snippet(command: str | None = None) -> dict[str, Any]:
    """Build the JSON snippet that hosts paste under mcpServers.

    Defaults to the absolute path of ``skg-mcp`` resolved against the
    running interpreter. This avoids the common foot-gun where a host
    process (Claude Code, Cursor, etc.) cannot find ``skg-mcp`` on its
    own PATH because the binary lives inside a venv.
    """
    cmd = command or _resolve_skg_mcp()
    return {
        "skg": {
            "command": cmd,
            "args":    [],
        }
    }


def _merge_mcp_config(target: Path, key: str, snippet: dict[str, Any]) -> None:
    """Merge the snippet into a host's JSON config under the given key.

    The file is created when missing. An existing file is read, the
    ``key`` table is merged with the snippet (snippet wins on collision),
    and the result is written back with two-space indentation.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    servers = existing.get(key) or {}
    if not isinstance(servers, dict):
        servers = {}
    servers.update(snippet)
    existing[key] = servers
    target.write_text(
        json.dumps(existing, indent=2) + "\n", encoding="utf-8",
    )


def cmd_install(args: argparse.Namespace) -> int:
    """Print (and optionally write) the MCP config block for a host."""
    client = args.client
    if client not in _CONFIG_TARGETS:
        print(
            f"client must be one of {list(KNOWN_CLIENTS)}, got {client!r}",
            file=sys.stderr,
        )
        return EXIT_BAD_ARGS

    target = _CONFIG_TARGETS[client]
    snippet = _mcp_snippet(command=getattr(args, "launch_command", None))
    extras = target.get("stdio_extras")
    if isinstance(extras, dict):
        snippet["skg"].update(extras)
    block = {target["key"]: snippet}
    print(json.dumps(block, indent=2))

    if not target["certain"]:
        note = target.get("note", "")
        if note:
            print(f"\nnote: {note}", file=sys.stderr)
        if args.write:
            print(
                "refusing to --write: install path for this host is not "
                "known. paste the snippet manually.",
                file=sys.stderr,
            )
            return EXIT_BAD_ARGS
        return EXIT_OK

    config_path = Path(target["path"]).expanduser()
    if args.write:
        _merge_mcp_config(config_path, target["key"], snippet)
        print(f"\nmerged into {config_path}", file=sys.stderr)
    else:
        print(
            f"\npaste the block above into {config_path} under "
            f"{target['key']!r}, or rerun with --write to merge it in.",
            file=sys.stderr,
        )
    return EXIT_OK


# ---- Argument parser -------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skg",
        description=(
            "Skill Knowledge Graph CLI. Wraps any LLM CLI behind a local "
            "procedure cache."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"skg {_version()}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create ~/.skg/ and config.toml.")

    p_run = sub.add_parser("run", help="Route a task through SKG.")
    p_run.add_argument("task", help="Natural-language task.")
    p_run.add_argument(
        "--vendor", choices=list(KNOWN_VENDORS), default=None,
        help="Override the configured vendor for this call.",
    )
    p_run.add_argument(
        "--json", action="store_true", dest="json_out",
        help="Emit a JSON envelope.",
    )
    p_run.add_argument(
        "--dry-run", action="store_true",
        help="Report whether SKG would hit; do not call the vendor.",
    )
    p_run.add_argument(
        "--execute", action="store_true",
        help="On hit, execute the WASI node and print stdout.",
    )

    p_list = sub.add_parser("list", help="List SKG nodes.")
    p_list.add_argument(
        "--all", action="store_true", dest="list_all",
        help="Include candidate and stale nodes.",
    )
    p_list.add_argument(
        "--json", action="store_true", dest="json_out",
        help="Emit JSON.",
    )

    sub.add_parser("mcp", help="Run the SKG MCP server over stdio.")

    p_install = sub.add_parser(
        "install",
        help="Print the MCP config block for a host.",
    )
    p_install.add_argument(
        "--client", choices=list(KNOWN_CLIENTS), required=True,
        help="Target host whose MCP config layout to emit.",
    )
    p_install.add_argument(
        "--write", action="store_true",
        help="Merge the snippet into the host's config file (when known).",
    )
    p_install.add_argument(
        "--launch", dest="launch_command", default=None,
        help="Override the launch command (default: skg-mcp).",
    )

    p_cfg = sub.add_parser("config", help="Read or write config.")
    cfg_sub = p_cfg.add_subparsers(dest="config_cmd", required=True)
    p_cfg_get = cfg_sub.add_parser("get", help="Print a config value.")
    p_cfg_get.add_argument("key")
    p_cfg_set = cfg_sub.add_parser("set", help="Write a config value.")
    p_cfg_set.add_argument("key")
    p_cfg_set.add_argument("value")

    return parser


# ---- Dispatcher ------------------------------------------------------------

_DISPATCH = {
    "init":    cmd_init,
    "run":     cmd_run,
    "list":    cmd_list,
    "config":  cmd_config,
    "mcp":     cmd_mcp,
    "install": cmd_install,
}


def main(argv: list[str] | None = None) -> int:
    """Parse argv and dispatch. Return the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help(sys.stderr)
        return EXIT_BAD_ARGS
    rc = handler(args)
    # When called as a script entry point, propagate via sys.exit.
    if argv is None:
        sys.exit(rc)
    return rc


if __name__ == "__main__":
    main()
