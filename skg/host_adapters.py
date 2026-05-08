"""Real adapter implementations for SKG custom host imports.

Phase 3e of the WASM import-level enforcement work. The Phase 3b host
wrappers in `skg/host_imports.py` validate the per-run handle table on
every call. This module replaces stub bodies with real operations:
HTTP via urllib, git via subprocess, secrets via the local filesystem,
and external/browser/production writes via append-only audit
directories.

What is implemented here:
  - skg.http_get, skg.http_post: real HTTP via urllib.request
  - skg.git_read, skg.git_write: subprocess to a whitelisted `git`
    subcommand under a caller-supplied cwd
  - skg.secret_read: reads `~/.skg/secrets/<name>` if present
  - skg.external_draft, skg.external_send: writes JSON to
    `~/.skg/drafts/` and `~/.skg/sent/` for the audit trail
  - skg.browser_read, skg.browser_write: queues the request in
    `~/.skg/browser_requests/` or `~/.skg/browser_writes/` for a
    downstream draining process
  - skg.production_write: queues the request in `~/.skg/production_log/`
    for per-system adapters; this module never performs the write

The wrapper still owns the security checks. Adapters run after scope
validation and after any approval-token check.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


HTTP_TIMEOUT_S    = 5
GIT_TIMEOUT_S     = 30


# Errno values shared with skg.host_imports
ERRNO_SUCCESS = 0
ERRNO_DENIED  = 13
ERRNO_NOENT   = 44
ERRNO_INVAL   = 28
ERRNO_IO      = 29


# Whitelists for git subcommands. The wrapper has already validated
# the GIT_READ or GIT_WRITE grant. The whitelists narrow the surface
# beyond the grant: a node holding GIT_WRITE still cannot run `push`
# from inside this adapter. Network-touching subcommands stay out so
# git is not a side channel for network.write.
_GIT_READ_COMMANDS  = frozenset({
    "log", "status", "diff", "show", "branch", "rev-parse", "ls-files",
})
_GIT_WRITE_COMMANDS = frozenset({
    "commit", "add", "tag", "branch",
})


def _skg_root() -> Path:
    """Return the SKG state root, honouring `Path.home` at call time.

    Tests monkeypatch `Path.home` to a tmp dir, so the path resolves
    inside the test sandbox rather than the user's real `~/.skg`. The
    function builds the path on every call rather than caching, so
    monkeypatched home paths take effect.
    """
    return Path.home() / ".skg"


def _now_iso() -> str:
    """Return a filesystem-safe UTC ISO-8601 timestamp."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return stamp.replace(":", "-").replace("+", "_")


def _atomic_write(path: Path, body: bytes) -> bool:
    """Write `body` to `path`, creating parents. Return False on error."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return True
    except OSError:
        return False


# ---- HTTP --------------------------------------------------------------


def http_get_body(url: str) -> tuple[int, bytes]:
    """Fetch a URL via HTTP GET. Returns (errno, body_bytes)."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return ERRNO_SUCCESS, resp.read()
    except (urllib.error.URLError, ValueError, TimeoutError):
        return ERRNO_IO, b""


def http_post_body(url: str, body: bytes, content_type: str) -> tuple[int, bytes]:
    """Send an HTTP POST. Returns (errno, response_body_bytes)."""
    try:
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return ERRNO_SUCCESS, resp.read()
    except (urllib.error.URLError, ValueError, TimeoutError):
        return ERRNO_IO, b""


def real_http_get(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.http_get. Returns (errno, response body)."""
    url = payload.get("url")
    if not isinstance(url, str) or not url:
        return ERRNO_INVAL, b""
    return http_get_body(url)


def real_http_post(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.http_post. Returns (errno, response body)."""
    url = payload.get("url")
    if not isinstance(url, str) or not url:
        return ERRNO_INVAL, b""
    body_value = payload.get("body", "")
    if isinstance(body_value, str):
        body_bytes = body_value.encode("utf-8")
    elif isinstance(body_value, (dict, list)):
        body_bytes = json.dumps(body_value).encode("utf-8")
    else:
        return ERRNO_INVAL, b""
    content_type = payload.get("content_type", "application/json")
    if not isinstance(content_type, str):
        content_type = "application/json"
    return http_post_body(url, body_bytes, content_type)


# ---- Git ---------------------------------------------------------------


def _run_git(
    command:   str,
    cwd:       str,
    args:      list[str],
    whitelist: frozenset[str],
) -> tuple[int, bytes]:
    """Run a whitelisted git subcommand. Return (errno, stdout_bytes)."""
    if command not in whitelist:
        return ERRNO_INVAL, b""
    if not isinstance(cwd, str) or not cwd:
        return ERRNO_INVAL, b""
    sanitised_args: list[str] = []
    for arg in args:
        if not isinstance(arg, str):
            return ERRNO_INVAL, b""
        sanitised_args.append(arg)
    try:
        result = subprocess.run(
            ["git", command, *sanitised_args],
            cwd=cwd,
            capture_output=True,
            text=False,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ERRNO_IO, b""
    if result.returncode != 0:
        return ERRNO_IO, result.stdout or b""
    return ERRNO_SUCCESS, result.stdout or b""


def real_git_read(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.git_read. Whitelists read-only subcommands."""
    command = payload.get("command")
    cwd     = payload.get("cwd")
    args    = payload.get("args", [])
    if not isinstance(command, str) or not isinstance(args, list):
        return ERRNO_INVAL, b""
    return _run_git(command, cwd, args, _GIT_READ_COMMANDS)


def real_git_write(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.git_write. Whitelists local-only subcommands.

    The wrapper has already enforced the approval token and the
    GIT_WRITE grant. This adapter narrows further: `push`, `fetch`,
    `pull`, and `reset` stay rejected so a compromised node holding
    GIT_WRITE cannot rewrite history or speak to a remote.

    A `dry_run` payload key short-circuits the subprocess call and
    echoes back the planned argv as JSON. The wrapper does not consult
    `dry_run`; the adapter owns that flag.
    """
    command = payload.get("command")
    cwd     = payload.get("cwd")
    args    = payload.get("args", [])
    if not isinstance(command, str) or not isinstance(args, list):
        return ERRNO_INVAL, b""
    if payload.get("dry_run") is True:
        plan = {"plan": ["git", command, *[str(a) for a in args]], "cwd": cwd}
        return ERRNO_SUCCESS, json.dumps(plan).encode("utf-8")
    return _run_git(command, cwd, args, _GIT_WRITE_COMMANDS)


# ---- Secrets -----------------------------------------------------------


def real_secret_read(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.secret_read. Reads `~/.skg/secrets/<name>`.

    The name must be a single path component. Path traversal (`..`,
    leading `/`, embedded separators) returns ERRNO_INVAL. A missing
    file returns ERRNO_NOENT so callers distinguish absent secrets
    from access faults.
    """
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return ERRNO_INVAL, b""
    if "/" in name or "\\" in name or name in {".", ".."}:
        return ERRNO_INVAL, b""
    secret_path = _skg_root() / "secrets" / name
    if not secret_path.is_file():
        return ERRNO_NOENT, b""
    try:
        return ERRNO_SUCCESS, secret_path.read_bytes()
    except OSError:
        return ERRNO_IO, b""


# ---- External draft / send ---------------------------------------------


def _serialise_external(channel: str, body: Any) -> bytes:
    """JSON-encode the external payload plus a server timestamp."""
    record = {
        "channel":   channel,
        "body":      body,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    return json.dumps(record, sort_keys=True).encode("utf-8")


def _write_external(payload: dict, subdir: str, id_key: str) -> tuple[int, bytes]:
    """Shared writer for external_draft, external_send, browser_*, and production_write."""
    channel = payload.get("channel")
    body    = payload.get("body")
    if not isinstance(channel, str) or not channel:
        return ERRNO_INVAL, b""
    if "/" in channel or "\\" in channel:
        return ERRNO_INVAL, b""
    if body is None:
        return ERRNO_INVAL, b""
    filename = f"{_now_iso()}-{channel}.json"
    target   = _skg_root() / subdir / filename
    if not _atomic_write(target, _serialise_external(channel, body)):
        return ERRNO_IO, b""
    return ERRNO_SUCCESS, json.dumps({id_key: filename}).encode("utf-8")


def real_external_draft(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.external_draft. Writes to `~/.skg/drafts/`."""
    return _write_external(payload, "drafts", "draft_id")


def real_external_send(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.external_send. Writes to `~/.skg/sent/`.

    The wrapper has already enforced the approval token. The send
    adapter mirrors the draft path so the audit trail records both
    halves of the draft-then-send protocol with separate filesystem
    roots. A real channel transmitter (Slack, email, GitHub) reads
    from `~/.skg/sent/` and dispatches.
    """
    return _write_external(payload, "sent", "sent_id")


# ---- Browser queue -----------------------------------------------------


def _queue_browser(payload: dict, subdir: str) -> tuple[int, bytes]:
    """Queue a browser request to a per-direction subdirectory.

    Real Chrome MCP integration is downstream. This adapter persists
    the request so a separate process can drain the queue, run the
    browser action, and write the result somewhere the caller can
    poll. Each request gets a UUID so two writes inside the same
    microsecond do not collide.
    """
    action = payload.get("action")
    args   = payload.get("args", {})
    if not isinstance(action, str) or not action:
        return ERRNO_INVAL, b""
    if not isinstance(args, dict):
        return ERRNO_INVAL, b""
    request_id = f"{_now_iso()}-{uuid4().hex}"
    body = json.dumps(
        {
            "action":     action,
            "args":       args,
            "request_id": request_id,
            "timestamp":  datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        },
        sort_keys=True,
    ).encode("utf-8")
    target = _skg_root() / subdir / f"{request_id}.json"
    if not _atomic_write(target, body):
        return ERRNO_IO, b""
    response = {"queued": True, "request_id": request_id}
    return ERRNO_SUCCESS, json.dumps(response).encode("utf-8")


def real_browser_read(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.browser_read. Queues to `~/.skg/browser_requests/`."""
    return _queue_browser(payload, "browser_requests")


def real_browser_write(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.browser_write. Queues to `~/.skg/browser_writes/`."""
    return _queue_browser(payload, "browser_writes")


# ---- Production write --------------------------------------------------


def real_production_write(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.production_write. Queues to `~/.skg/production_log/`.

    The wrapper has already enforced the approval token and the
    PRODUCTION_WRITE grant. This adapter never performs the write.
    Each production system needs its own per-system design that picks
    up the queued request, validates it against system-specific
    invariants, and then dispatches. Persisting the request first
    means every attempt is auditable, even when the downstream
    dispatcher rejects it.
    """
    system = payload.get("system")
    if not isinstance(system, str) or not system:
        return ERRNO_INVAL, b""
    if "/" in system or "\\" in system:
        return ERRNO_INVAL, b""
    audit_id = f"{_now_iso()}-{system}"
    target   = _skg_root() / "production_log" / f"{audit_id}.json"
    record = {
        "system":    system,
        "payload":   payload,
        "audit_id":  audit_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    if not _atomic_write(target, json.dumps(record, sort_keys=True).encode("utf-8")):
        return ERRNO_IO, b""
    response = {"queued": True, "audit_id": audit_id}
    return ERRNO_SUCCESS, json.dumps(response).encode("utf-8")


# ---- Registry ----------------------------------------------------------


# Registry mapping qualified host import name to a function that takes
# the decoded JSON payload and returns (errno, response_body).
ADAPTERS: dict[str, Callable[[dict], tuple[int, bytes]]] = {
    "skg.http_get":         real_http_get,
    "skg.http_post":        real_http_post,
    "skg.git_read":         real_git_read,
    "skg.git_write":        real_git_write,
    "skg.secret_read":      real_secret_read,
    "skg.external_draft":   real_external_draft,
    "skg.external_send":    real_external_send,
    "skg.browser_read":     real_browser_read,
    "skg.browser_write":    real_browser_write,
    "skg.production_write": real_production_write,
}


def has_adapter(qualified_name: str) -> bool:
    """Return True if a real adapter exists for this host import."""
    return qualified_name in ADAPTERS
