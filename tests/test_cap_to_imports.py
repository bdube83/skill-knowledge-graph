"""Unit tests for the capability-to-import mapping.

Phase 1 of the WASM import-level enforcement work. These tests pin the
mapping behaviour so Phase 2 (GrantedLinker) can rely on it.
"""

from __future__ import annotations

from skg.cap_to_imports import (
    APPROVAL_HOST,
    EFFECT_HOST,
    EFFECT_WASI,
    MINIMUM_WASI,
    host_imports_for,
    requires_approval_token,
    wasi_imports_for,
)
from skg.effects import APPROVAL_REQUIRED, Effect


def test_minimum_wasi_present_for_empty_grants() -> None:
    imports = wasi_imports_for([])
    assert imports == MINIMUM_WASI


def test_minimum_wasi_subset_of_any_grant_set() -> None:
    for effect in Effect:
        imports = wasi_imports_for([effect])
        assert MINIMUM_WASI <= imports, f"MINIMUM_WASI not preserved for {effect}"


def test_local_read_adds_path_open() -> None:
    imports = wasi_imports_for([Effect.LOCAL_READ])
    assert "path_open" in imports
    assert "path_filestat_get" in imports


def test_local_write_adds_create_directory() -> None:
    imports = wasi_imports_for([Effect.LOCAL_WRITE])
    assert "path_create_directory" in imports


def test_non_local_effects_add_no_wasi() -> None:
    for effect in Effect:
        if effect in EFFECT_WASI:
            continue
        imports = wasi_imports_for([effect])
        assert imports == MINIMUM_WASI, f"{effect} should not add WASI imports"


def test_no_host_imports_for_empty_grants() -> None:
    assert host_imports_for([]) == frozenset()


def test_network_read_wires_http_get() -> None:
    assert host_imports_for([Effect.NETWORK_READ]) == frozenset({"skg.http_get"})


def test_network_write_wires_http_post() -> None:
    assert host_imports_for([Effect.NETWORK_WRITE]) == frozenset({"skg.http_post"})


def test_local_effects_have_no_custom_host_imports() -> None:
    assert host_imports_for([Effect.LOCAL_READ]) == frozenset()
    assert host_imports_for([Effect.LOCAL_WRITE]) == frozenset()


def test_each_non_local_effect_has_a_host_import() -> None:
    non_local = [e for e in Effect if e not in EFFECT_WASI]
    for effect in non_local:
        assert EFFECT_HOST.get(effect), f"{effect} has no host import wired"


def test_approval_required_effects_have_approval_hosts() -> None:
    for effect in APPROVAL_REQUIRED:
        for name in EFFECT_HOST[effect]:
            assert name in APPROVAL_HOST
            assert requires_approval_token(name)


def test_non_approval_hosts_do_not_require_token() -> None:
    non_approval_effects = set(EFFECT_HOST) - set(APPROVAL_REQUIRED)
    for effect in non_approval_effects:
        for name in EFFECT_HOST[effect]:
            assert not requires_approval_token(name)


def test_combined_grants_union_imports() -> None:
    grants = [Effect.LOCAL_READ, Effect.NETWORK_READ, Effect.GIT_READ]
    wasi = wasi_imports_for(grants)
    host = host_imports_for(grants)
    assert "path_open" in wasi
    assert host == frozenset({"skg.http_get", "skg.git_read"})


def test_every_effect_has_either_wasi_or_host_mapping() -> None:
    for effect in Effect:
        in_wasi = effect in EFFECT_WASI
        in_host = effect in EFFECT_HOST
        assert in_wasi or in_host, f"{effect} has no mapping"
