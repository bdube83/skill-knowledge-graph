"""Tests for the skg shell CLI.

The CLI wraps SKG routing around any vendor LLM. These tests cover the
subcommand dispatch surface without making real network calls. Vendor
adapters and SKG routing are monkeypatched so tests run fully offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from skg import cli
from skg import cli_vendors


# ---- Fixtures --------------------------------------------------------------

@pytest.fixture
def skg_home(tmp_path, monkeypatch):
    """Point SKG_HOME at a temp dir so init/config writes do not touch ~."""
    home = tmp_path / "skg-home"
    monkeypatch.setenv("SKG_HOME", str(home))
    # Also redirect Path.home() in case any code reads ~ directly.
    monkeypatch.setattr(
        "pathlib.Path.home", lambda: tmp_path / "user-home",
    )
    (tmp_path / "user-home").mkdir(parents=True, exist_ok=True)
    return home


# Stand-ins for skg.router.RouteResult so tests do not hit the SQLite store.

@dataclass
class _FakeStage:
    value: str


@dataclass
class _FakeManifest:
    header: str = "fake header line"
    task_type: str = "fake_task"


@dataclass
class _FakeNode:
    id: str = "fake-node-1"
    manifest: _FakeManifest = None  # populated in __post_init__

    def __post_init__(self):
        if self.manifest is None:
            self.manifest = _FakeManifest()


@dataclass
class _FakeResult:
    hit: bool
    stage: _FakeStage
    node: _FakeNode | None = None
    grant: object = None
    reason: str = ""

    @property
    def miss(self) -> bool:
        return not self.hit


def _fake_hit_route(*_a, **_kw):
    return _FakeResult(
        hit=True,
        stage=_FakeStage("exact"),
        node=_FakeNode(),
    )


def _fake_miss_route(*_a, **_kw):
    return _FakeResult(
        hit=False,
        stage=_FakeStage("miss"),
        reason="no match",
    )


# ---- Tests -----------------------------------------------------------------

def test_version_flag_prints_non_empty_string(capsys):
    """--version prints `skg <version>` and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("skg ")
    assert len(out.strip()) > len("skg ")


def test_init_creates_config_file(skg_home, capsys):
    """`skg init` writes ~/.skg/config.toml with defaults."""
    rc = cli.main(["init"])
    assert rc == cli.EXIT_OK
    assert skg_home.exists()
    cfg_path = skg_home / "config.toml"
    assert cfg_path.exists()
    body = cfg_path.read_text()
    assert 'vendor = "openai"' in body
    out = capsys.readouterr().out
    assert "initialised SKG" in out


def test_config_set_then_get_round_trip(skg_home, capsys):
    """`skg config set vendor claude` then `skg config get vendor` returns claude."""
    cli.main(["init"])
    capsys.readouterr()  # clear
    rc = cli.main(["config", "set", "vendor", "claude"])
    assert rc == cli.EXIT_OK
    capsys.readouterr()
    rc = cli.main(["config", "get", "vendor"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out.strip()
    assert out == "claude"


def test_config_set_rejects_unknown_vendor(skg_home, capsys):
    """Setting an unknown vendor exits with EXIT_BAD_ARGS."""
    cli.main(["init"])
    capsys.readouterr()
    rc = cli.main(["config", "set", "vendor", "made-up-vendor"])
    assert rc == cli.EXIT_BAD_ARGS
    err = capsys.readouterr().err
    assert "vendor must be one of" in err


def test_run_dry_run_json_envelope(skg_home, monkeypatch, capsys):
    """`skg run --dry-run --json "..."` emits a JSON object with hit/stage."""
    cli.main(["init"])
    capsys.readouterr()
    monkeypatch.setattr(
        "skg.integrations.agent_proxy.route_proposal", _fake_miss_route,
    )
    rc = cli.main(["run", "--dry-run", "--json", "build a thing"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["hit"] is False
    assert payload["stage"] == "miss"
    assert payload["dry_run"] is True
    assert payload["vendor"] == "openai"


def test_run_hit_prints_node_header(skg_home, monkeypatch, capsys):
    """On a routing hit `skg run` prints the matched node's header."""
    cli.main(["init"])
    capsys.readouterr()
    monkeypatch.setattr(
        "skg.integrations.agent_proxy.route_proposal", _fake_hit_route,
    )
    rc = cli.main(["run", "do the thing"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out.strip()
    assert out == "fake header line"


def test_run_miss_calls_vendor(skg_home, monkeypatch, capsys):
    """On a miss the configured vendor is called and its text is printed."""
    cli.main(["init"])
    capsys.readouterr()
    monkeypatch.setattr(
        "skg.integrations.agent_proxy.route_proposal", _fake_miss_route,
    )
    seen = {}

    def _stub(task, *, model=None):
        seen["task"]  = task
        seen["model"] = model
        return cli_vendors.VendorResponse(
            text="vendor reply", tokens_used=42, model="stub-model",
        )

    monkeypatch.setitem(cli_vendors.VENDORS, "openai", _stub)
    rc = cli.main(["run", "summarise the report"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out.strip()
    assert out == "vendor reply"
    assert seen["task"] == "summarise the report"


def test_run_miss_vendor_unavailable_exits_2(skg_home, monkeypatch, capsys):
    """A vendor that returns ERRNO_NOAVAIL maps to EXIT_VENDOR_NOAVAIL."""
    cli.main(["init"])
    capsys.readouterr()
    monkeypatch.setattr(
        "skg.integrations.agent_proxy.route_proposal", _fake_miss_route,
    )

    def _stub(_task, *, model=None):  # noqa: ARG001
        return cli_vendors.VendorResponse(
            text="", error=cli_vendors.ERRNO_NOAVAIL, model="stub",
        )

    monkeypatch.setitem(cli_vendors.VENDORS, "claude", _stub)
    rc = cli.main(["run", "--vendor", "claude", "task"])
    assert rc == cli.EXIT_VENDOR_NOAVAIL


def test_run_miss_json_envelope_shape(skg_home, monkeypatch, capsys):
    """--json on a miss emits the documented envelope keys."""
    cli.main(["init"])
    capsys.readouterr()
    monkeypatch.setattr(
        "skg.integrations.agent_proxy.route_proposal", _fake_miss_route,
    )

    def _stub(_task, *, model=None):  # noqa: ARG001
        return cli_vendors.VendorResponse(
            text="hello", tokens_used=7, model="m",
        )

    monkeypatch.setitem(cli_vendors.VENDORS, "openai", _stub)
    rc = cli.main(["run", "--json", "task"])
    assert rc == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    for key in ("hit", "stage", "node_id", "output", "tokens_used"):
        assert key in payload
    assert payload["hit"] is False
    assert payload["output"] == "hello"
    assert payload["tokens_used"] == 7


def test_vendor_registry_holds_three_defaults():
    """The shipped registry covers the three documented vendors."""
    assert set(cli_vendors.VENDORS) >= {"openai", "claude", "codex"}
    for name in ("openai", "claude", "codex"):
        assert callable(cli_vendors.VENDORS[name])
