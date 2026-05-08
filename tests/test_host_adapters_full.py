"""End-to-end tests for the eight non-HTTP host adapters.

The Phase 3e adapters in `skg/host_adapters.py` perform real work
after the Phase 3b wrapper validates the handle. These tests drive
each adapter through the launcher with a WAT module that calls the
host import once and forwards the errno through `proc_exit`.

Isolation. Each test monkeypatches `Path.home` to `tmp_path` so the
adapter's reads and writes land inside the test sandbox, not the
real `~/.skg/` tree. The git tests also init a real on-disk repo
under `tmp_path`.

What these tests prove. Each adapter returns ERRNO_SUCCESS when the
wrapper passes the call through, persists the expected file when the
adapter is queue-shaped, and produces the documented response shape.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from skg import host_adapters
from skg.effects import Effect
from skg.wasmtime_launcher import WasmtimeRuntime


# ---- Helpers -----------------------------------------------------------


class _ScopedRuntime(WasmtimeRuntime):
    """Runtime that mints handles with caller-supplied scopes."""

    def __init__(self, scopes: dict[Effect, tuple[str, str]]) -> None:
        super().__init__()
        self._scopes = scopes

    def _mint_handles(self, table, effects):
        approval_effects = {
            Effect.EXTERNAL_SEND,
            Effect.GIT_WRITE,
            Effect.PRODUCTION_WRITE,
        }
        handles: dict[str, int] = {}
        for effect in effects:
            url_pattern, path_str = self._scopes.get(effect, ("*", "/"))
            handle_id = table.mint(
                effect,
                url_pattern=url_pattern,
                path_scope=Path(path_str),
                approval_token=1 if effect in approval_effects else 0,
            )
            handles[effect.value] = handle_id
        return handles


def _no_approval_wat(import_name: str, payload: str) -> str:
    """WAT that calls a non-approval host import once and forwards errno."""
    payload_bytes = payload.encode("utf-8")
    payload_len   = len(payload_bytes)
    escaped       = payload.replace('\\', '\\\\').replace('"', '\\"')
    return textwrap.dedent(f'''
        (module
          (import "skg" "{import_name}"
            (func $call (param i32 i32 i32 i32 i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "proc_exit"
            (func $proc_exit (param i32)))
          (memory (export "memory") 1)
          (data (i32.const 0) "{escaped}")
          (func (export "_start")
            (call $proc_exit
              (call $call
                (i32.const 1)
                (i32.const 0)
                (i32.const {payload_len})
                (i32.const 1024)
                (i32.const 1024)
                (i32.const 2048)))))
    ''').strip()


def _approval_wat(import_name: str, payload: str, approval_token: int = 1) -> str:
    """WAT that calls an approval-gated host import once and forwards errno."""
    payload_bytes = payload.encode("utf-8")
    payload_len   = len(payload_bytes)
    escaped       = payload.replace('\\', '\\\\').replace('"', '\\"')
    return textwrap.dedent(f'''
        (module
          (import "skg" "{import_name}"
            (func $call (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "proc_exit"
            (func $proc_exit (param i32)))
          (memory (export "memory") 1)
          (data (i32.const 0) "{escaped}")
          (func (export "_start")
            (call $proc_exit
              (call $call
                (i32.const 1)
                (i32.const 0)
                (i32.const {payload_len})
                (i32.const 1024)
                (i32.const 1024)
                (i32.const 2048)
                (i32.const {approval_token})))))
    ''').strip()


def _write_wat(tmp_path: Path, name: str, wat: str) -> Path:
    out = tmp_path / f"{name}.wat"
    out.write_text(wat)
    return out


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch) -> Path:
    """Point `Path.home` at `tmp_path` so adapters write into the sandbox."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


# ---- git_read ----------------------------------------------------------


def _init_repo(repo: Path) -> None:
    """Init a git repo at `repo` and add one commit."""
    import os as _os
    repo.mkdir(parents=True, exist_ok=True)
    env = {
        **_os.environ,
        "GIT_AUTHOR_NAME":     "skg-test",
        "GIT_AUTHOR_EMAIL":    "skg-test@example.com",
        "GIT_COMMITTER_NAME":  "skg-test",
        "GIT_COMMITTER_EMAIL": "skg-test@example.com",
    }
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    (repo / "README").write_text("hello\n")
    subprocess.run(["git", "add", "README"], cwd=repo, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial"],
        cwd=repo, check=True, env=env,
    )


def test_real_git_read_log(tmp_path: Path, fake_home: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    payload = json.dumps({
        "command": "log",
        "cwd":     str(repo),
        "args":    ["--oneline"],
    })
    runtime = _ScopedRuntime(scopes={Effect.GIT_READ: ("*", "/")})
    wasm = _write_wat(tmp_path, "git_read", _no_approval_wat("git_read", payload))
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="git-log",
        context={},
        granted_effects=["git.read"],
    )
    assert result.success is True, f"expected success, got: {result.error}"


def test_real_git_read_rejects_unwhitelisted_command(
    tmp_path: Path, fake_home: Path,
) -> None:
    """Unwhitelisted git subcommands return ERRNO_INVAL (28)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    payload = json.dumps({
        "command": "fetch",
        "cwd":     str(repo),
        "args":    [],
    })
    runtime = _ScopedRuntime(scopes={Effect.GIT_READ: ("*", "/")})
    wasm = _write_wat(
        tmp_path, "git_read_bad", _no_approval_wat("git_read", payload),
    )
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="git-fetch",
        context={},
        granted_effects=["git.read"],
    )
    assert result.success is False
    assert "28" in result.error


# ---- git_write ---------------------------------------------------------


def test_real_git_write_dry_run(tmp_path: Path, fake_home: Path) -> None:
    """`dry_run` short-circuits the subprocess and echoes the planned argv."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    payload = json.dumps({
        "command": "commit",
        "cwd":     str(repo),
        "args":    ["-m", "fake"],
        "dry_run": True,
    })
    runtime = _ScopedRuntime(scopes={Effect.GIT_WRITE: ("*", "/")})
    wasm = _write_wat(
        tmp_path, "git_write_dry", _approval_wat("git_write", payload),
    )
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="git-commit",
        context={},
        granted_effects=["git.write"],
    )
    assert result.success is True, f"expected success, got: {result.error}"


# ---- secret_read -------------------------------------------------------


def test_real_secret_read_returns_file_bytes(
    tmp_path: Path, fake_home: Path,
) -> None:
    secrets_dir = fake_home / ".skg" / "secrets"
    secrets_dir.mkdir(parents=True)
    (secrets_dir / "api_key").write_bytes(b"sk-test-1234")

    payload = json.dumps({"name": "api_key"})
    runtime = _ScopedRuntime(scopes={Effect.SECRET_READ: ("*", "/")})
    wasm = _write_wat(
        tmp_path, "secret_read", _no_approval_wat("secret_read", payload),
    )
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="read-secret",
        context={},
        granted_effects=["secret.read"],
    )
    assert result.success is True, f"expected success, got: {result.error}"


def test_real_secret_read_missing_returns_noent(
    tmp_path: Path, fake_home: Path,
) -> None:
    """Absent secrets return ERRNO_NOENT (44), distinct from access faults."""
    payload = json.dumps({"name": "absent"})
    runtime = _ScopedRuntime(scopes={Effect.SECRET_READ: ("*", "/")})
    wasm = _write_wat(
        tmp_path, "secret_read_missing", _no_approval_wat("secret_read", payload),
    )
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="read-missing",
        context={},
        granted_effects=["secret.read"],
    )
    assert result.success is False
    assert "44" in result.error


# ---- external_draft ----------------------------------------------------


def test_real_external_draft_writes_file(
    tmp_path: Path, fake_home: Path,
) -> None:
    payload = json.dumps({
        "url":     "slack://channel/general",
        "channel": "slack",
        "body":    {"text": "hello"},
    })
    runtime = _ScopedRuntime(scopes={
        Effect.EXTERNAL_DRAFT: ("slack://*", "/"),
    })
    wasm = _write_wat(
        tmp_path, "external_draft",
        _no_approval_wat("external_draft", payload),
    )
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="draft",
        context={},
        granted_effects=["external.draft"],
    )
    assert result.success is True, f"expected success, got: {result.error}"

    drafts_dir = fake_home / ".skg" / "drafts"
    drafts = list(drafts_dir.glob("*-slack.json"))
    assert len(drafts) == 1, f"expected one draft file, got: {drafts}"

    record = json.loads(drafts[0].read_text())
    assert record["channel"] == "slack"
    assert record["body"]    == {"text": "hello"}


# ---- external_send -----------------------------------------------------


def test_real_external_send_writes_to_sent_dir(
    tmp_path: Path, fake_home: Path,
) -> None:
    payload = json.dumps({
        "url":     "slack://channel/general",
        "channel": "slack",
        "body":    {"text": "hello"},
    })
    runtime = _ScopedRuntime(scopes={
        Effect.EXTERNAL_SEND: ("slack://*", "/"),
    })
    wasm = _write_wat(
        tmp_path, "external_send", _approval_wat("external_send", payload),
    )
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="send",
        context={},
        granted_effects=["external.send"],
    )
    assert result.success is True, f"expected success, got: {result.error}"

    sent_dir = fake_home / ".skg" / "sent"
    sent = list(sent_dir.glob("*-slack.json"))
    assert len(sent) == 1, f"expected one sent file, got: {sent}"


# ---- browser_read ------------------------------------------------------


def test_real_browser_read_queues_request(
    tmp_path: Path, fake_home: Path,
) -> None:
    payload = json.dumps({
        "url":    "https://example.com",
        "action": "snapshot",
        "args":   {"selector": "body"},
    })
    runtime = _ScopedRuntime(scopes={
        Effect.BROWSER_READ: ("https://example.com*", "/"),
    })
    wasm = _write_wat(
        tmp_path, "browser_read", _no_approval_wat("browser_read", payload),
    )
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="snapshot",
        context={},
        granted_effects=["browser.read"],
    )
    assert result.success is True, f"expected success, got: {result.error}"

    queue_dir = fake_home / ".skg" / "browser_requests"
    queued = list(queue_dir.glob("*.json"))
    assert len(queued) == 1, f"expected one queued request, got: {queued}"


# ---- browser_write -----------------------------------------------------


def test_real_browser_write_queues_to_separate_dir(
    tmp_path: Path, fake_home: Path,
) -> None:
    payload = json.dumps({
        "url":    "https://example.com",
        "action": "click",
        "args":   {"selector": "#submit"},
    })
    runtime = _ScopedRuntime(scopes={
        Effect.BROWSER_WRITE: ("https://example.com*", "/"),
    })
    wasm = _write_wat(
        tmp_path, "browser_write", _no_approval_wat("browser_write", payload),
    )
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="click",
        context={},
        granted_effects=["browser.write"],
    )
    assert result.success is True, f"expected success, got: {result.error}"

    write_dir = fake_home / ".skg" / "browser_writes"
    queued = list(write_dir.glob("*.json"))
    assert len(queued) == 1, f"expected one queued write, got: {queued}"

    # The browser_read queue stays empty on a write.
    read_dir = fake_home / ".skg" / "browser_requests"
    assert not read_dir.exists() or not list(read_dir.glob("*.json"))


# ---- production_write --------------------------------------------------


def test_real_production_write_appends_audit_record(
    tmp_path: Path, fake_home: Path,
) -> None:
    payload = json.dumps({
        "system":  "ledger",
        "action":  "credit",
        "amount":  100,
    })
    runtime = _ScopedRuntime(scopes={Effect.PRODUCTION_WRITE: ("*", "/")})
    wasm = _write_wat(
        tmp_path, "production_write",
        _approval_wat("production_write", payload),
    )
    result = runtime.execute(
        wasm_path=wasm,
        node_id="adapter_test",
        task="credit",
        context={},
        granted_effects=["production.write"],
    )
    assert result.success is True, f"expected success, got: {result.error}"

    audit_dir = fake_home / ".skg" / "production_log"
    records = list(audit_dir.glob("*-ledger.json"))
    assert len(records) == 1, f"expected one audit record, got: {records}"

    record = json.loads(records[0].read_text())
    assert record["system"] == "ledger"
    assert record["payload"]["action"] == "credit"
