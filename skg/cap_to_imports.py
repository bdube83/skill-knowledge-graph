"""Capability-to-import mapping for the GrantedLinker.

Maps SKG effect classes (skg/effects.py) to the set of WASI imports and
custom host imports that must be wired into the Wasmtime Linker for a
node holding that grant.

Phase 1 of the WASM import-level enforcement work
(designs/proposed/skg-wasm-import-enforcement.md). No runtime change in
this phase. Phase 2 consumes this mapping when constructing the
GrantedLinker.

Design contract: the mapping is the only place that knows how an effect
class translates to host surface. The launcher reads from this module
and wires only what the grant set permits. Adding a new effect class
requires updating MINIMUM_WASI (no), EFFECT_WASI, EFFECT_HOST, and
APPROVAL_HOST.
"""

from __future__ import annotations

from .effects import APPROVAL_REQUIRED, Effect


MINIMUM_WASI: frozenset[str] = frozenset({
    "proc_exit",
    "fd_read",
    "fd_write",
    "fd_close",
    "environ_get",
    "environ_sizes_get",
    "random_get",
})


EFFECT_WASI: dict[Effect, frozenset[str]] = {
    Effect.LOCAL_READ:  frozenset({"path_open", "fd_seek", "path_filestat_get"}),
    Effect.LOCAL_WRITE: frozenset({"path_open", "fd_seek", "path_create_directory"}),
}


EFFECT_HOST: dict[Effect, frozenset[str]] = {
    Effect.NETWORK_READ:     frozenset({"skg.http_get"}),
    Effect.NETWORK_WRITE:    frozenset({"skg.http_post"}),
    Effect.EXTERNAL_DRAFT:   frozenset({"skg.external_draft"}),
    Effect.EXTERNAL_SEND:    frozenset({"skg.external_send"}),
    Effect.BROWSER_READ:     frozenset({"skg.browser_read"}),
    Effect.BROWSER_WRITE:    frozenset({"skg.browser_write"}),
    Effect.GIT_READ:         frozenset({"skg.git_read"}),
    Effect.GIT_WRITE:        frozenset({"skg.git_write"}),
    Effect.SECRET_READ:      frozenset({"skg.secret_read"}),
    Effect.PRODUCTION_WRITE: frozenset({"skg.production_write"}),
    Effect.TEXT_GENERATE:    frozenset({"skg.text_generate"}),
}


APPROVAL_HOST: frozenset[str] = frozenset(
    name for effect in APPROVAL_REQUIRED for name in EFFECT_HOST.get(effect, frozenset())
)


def wasi_imports_for(grants: list[Effect]) -> frozenset[str]:
    """Return the WASI imports that must be wired for this grant set.

    Always includes MINIMUM_WASI. Adds per-effect WASI imports for any
    effect that maps to additional WASI surface (currently LOCAL_READ
    and LOCAL_WRITE; all other effects are mediated by custom host
    imports).
    """
    imports: set[str] = set(MINIMUM_WASI)
    for effect in grants:
        imports |= EFFECT_WASI.get(effect, frozenset())
    return frozenset(imports)


def host_imports_for(grants: list[Effect]) -> frozenset[str]:
    """Return the custom host imports that must be wired for this grant set."""
    imports: set[str] = set()
    for effect in grants:
        imports |= EFFECT_HOST.get(effect, frozenset())
    return frozenset(imports)


def requires_approval_token(host_import_name: str) -> bool:
    """Return True if the host import requires an approval_token argument."""
    return host_import_name in APPROVAL_HOST
