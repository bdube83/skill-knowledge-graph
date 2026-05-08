"""Unit tests for the per-run handle table.

Phase 3b of the WASM import-level enforcement work
(designs/proposed/skg-wasm-import-enforcement.md). The handle table
mediates every custom host import call: the host wrapper validates
the integer handle id before letting the call proceed.
"""

from __future__ import annotations

from pathlib import Path

from skg.effects import Effect
from skg.handle_table import GrantHandle, HandleTable


def test_mint_returns_monotonically_increasing_ids() -> None:
    table = HandleTable()
    a = table.mint(Effect.NETWORK_READ)
    b = table.mint(Effect.NETWORK_WRITE)
    c = table.mint(Effect.GIT_READ)
    assert a == 1
    assert b == 2
    assert c == 3


def test_lookup_returns_minted_grant_handle() -> None:
    table = HandleTable()
    handle_id = table.mint(
        Effect.NETWORK_READ,
        url_pattern="https://api.example.com/*",
    )
    row = table.lookup(handle_id)
    assert isinstance(row, GrantHandle)
    assert row.effect == Effect.NETWORK_READ
    assert row.url_pattern == "https://api.example.com/*"
    assert row.approval_token == 0


def test_lookup_unknown_id_returns_none() -> None:
    table = HandleTable()
    table.mint(Effect.NETWORK_READ)
    assert table.lookup(99) is None


def test_validate_matches_effect() -> None:
    table = HandleTable()
    handle_id = table.mint(Effect.NETWORK_READ)
    assert table.validate(handle_id, Effect.NETWORK_READ) is True


def test_validate_rejects_mismatched_effect() -> None:
    table = HandleTable()
    handle_id = table.mint(Effect.NETWORK_READ)
    assert table.validate(handle_id, Effect.NETWORK_WRITE) is False


def test_validate_rejects_unknown_handle_id() -> None:
    table = HandleTable()
    table.mint(Effect.NETWORK_READ)
    assert table.validate(999, Effect.NETWORK_READ) is False


def test_validate_url_pattern_accepts_match() -> None:
    table = HandleTable()
    handle_id = table.mint(
        Effect.NETWORK_READ,
        url_pattern="https://api.example.com/*",
    )
    assert table.validate(
        handle_id,
        Effect.NETWORK_READ,
        url="https://api.example.com/foo",
    ) is True


def test_validate_url_pattern_rejects_other_origin() -> None:
    table = HandleTable()
    handle_id = table.mint(
        Effect.NETWORK_READ,
        url_pattern="https://api.example.com/*",
    )
    assert table.validate(
        handle_id,
        Effect.NETWORK_READ,
        url="https://attacker.example/",
    ) is False


def test_validate_url_pattern_wildcard_accepts_anything() -> None:
    table = HandleTable()
    handle_id = table.mint(Effect.NETWORK_READ, url_pattern="*")
    assert table.validate(
        handle_id,
        Effect.NETWORK_READ,
        url="https://anywhere.example/foo",
    ) is True


def test_validate_path_inside_scope(tmp_path: Path) -> None:
    scope = tmp_path / "allowed"
    scope.mkdir()
    inside = scope / "file.txt"
    inside.write_text("x")

    table = HandleTable()
    handle_id = table.mint(Effect.LOCAL_READ, path_scope=scope)

    assert table.validate(handle_id, Effect.LOCAL_READ, path=inside) is True


def test_validate_path_outside_scope(tmp_path: Path) -> None:
    scope = tmp_path / "allowed"
    scope.mkdir()
    outside_dir = tmp_path / "other"
    outside_dir.mkdir()
    outside = outside_dir / "file.txt"
    outside.write_text("x")

    table = HandleTable()
    handle_id = table.mint(Effect.LOCAL_READ, path_scope=scope)

    assert table.validate(handle_id, Effect.LOCAL_READ, path=outside) is False


def test_validate_path_root_scope_accepts_anything(tmp_path: Path) -> None:
    target = tmp_path / "anywhere.txt"
    target.write_text("x")

    table = HandleTable()
    handle_id = table.mint(Effect.LOCAL_READ, path_scope=Path("/"))

    assert table.validate(handle_id, Effect.LOCAL_READ, path=target) is True


def test_validate_approval_token_match() -> None:
    table = HandleTable()
    handle_id = table.mint(Effect.EXTERNAL_SEND, approval_token=42)
    assert table.validate(
        handle_id,
        Effect.EXTERNAL_SEND,
        approval_token=42,
    ) is True


def test_validate_approval_token_mismatch() -> None:
    table = HandleTable()
    handle_id = table.mint(Effect.EXTERNAL_SEND, approval_token=42)
    assert table.validate(
        handle_id,
        Effect.EXTERNAL_SEND,
        approval_token=99,
    ) is False
