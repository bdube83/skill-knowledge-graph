"""Per-run handle table for SKG capability grants.

Phase 3b of the WASM import-level enforcement work
(designs/proposed/skg-wasm-import-enforcement.md). The handle table
sits between the launcher and the custom host imports defined in
skg/host_imports.py. The launcher mints one handle per granted effect
before instantiating the module. Host import wrappers receive the
integer handle id from the WASM caller and validate it before letting
the call proceed.

Lifecycle. A `HandleTable` is created fresh inside each `execute()`
call. Handle ids are integers starting at 1, monotonically increasing
within one table. Tables are not shared across runs, which closes the
replay attack class from the design doc (a captured handle from one
run does not validate in any other run).

Validation. `validate(handle_id, effect, *, url, path, approval_token)`
checks four things:
  1. The handle id resolves to a `GrantHandle` in this table.
  2. The handle's effect equals the requested effect.
  3. If `url` is given, it matches the handle's `url_pattern`.
  4. If `path` is given, it sits inside the handle's `path_scope`.

Approval tokens are stored on the handle for reference but the policy
engine owns the token lifecycle. Phase 3b does not check the token
against the policy engine; it only refuses tokens of value 0 in the
host wrappers themselves (see skg/host_imports.py).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

from .effects import Effect


@dataclass
class GrantHandle:
    """One row of the handle table.

    `effect` names the effect class this handle authorises. `url_pattern`
    is an fnmatch-style glob applied to URL strings; `path_scope` is a
    filesystem prefix applied to filesystem paths. `approval_token` is
    the integer token minted by the policy engine for approval-gated
    effects; 0 means no token has been issued.
    """

    effect:          Effect
    url_pattern:     str | None  = None
    path_scope:      Path | None = None
    approval_token:  int         = 0


class HandleTable:
    """Per-run integer-keyed table of grant handles.

    Use one `HandleTable` per `execute()` call. The launcher mints one
    handle per granted effect before instantiating the WASM module,
    then passes the resulting ids to the node via stdin.
    """

    def __init__(self) -> None:
        self._next_id: int = 1
        self._rows:    dict[int, GrantHandle] = {}

    def mint(
        self,
        effect:          Effect,
        url_pattern:     str | None  = None,
        path_scope:      Path | None = None,
        approval_token:  int         = 0,
    ) -> int:
        """Add a new handle and return its integer id."""
        handle_id = self._next_id
        self._next_id += 1
        self._rows[handle_id] = GrantHandle(
            effect=effect,
            url_pattern=url_pattern,
            path_scope=path_scope,
            approval_token=approval_token,
        )
        return handle_id

    def lookup(self, handle_id: int) -> GrantHandle | None:
        """Return the handle for this id, or None if no such row."""
        return self._rows.get(handle_id)

    def rows(self) -> list[GrantHandle]:
        """Return all handle rows in mint order. Used by WASI host
        wrappers that look up scope by effect rather than by id (path_*
        WASI imports do not carry a handle argument)."""
        return [self._rows[i] for i in sorted(self._rows)]

    def validate(
        self,
        handle_id:      int,
        effect:         Effect,
        *,
        url:            str | None  = None,
        path:           Path | str | None = None,
        approval_token: int | None  = None,
    ) -> bool:
        """Return True if the handle authorises this call.

        Checks the handle resolves, the effect matches, the URL fits the
        handle's pattern (when a URL is given), and the path sits inside
        the handle's scope (when a path is given). When `approval_token`
        is given, also checks the token equals the handle's stored
        token.
        """
        row = self._rows.get(handle_id)
        if row is None:
            return False
        if row.effect != effect:
            return False
        if url is not None and not _url_in_scope(url, row.url_pattern):
            return False
        if path is not None and not _path_in_scope(path, row.path_scope):
            return False
        if approval_token is not None and approval_token != row.approval_token:
            return False
        return True


def _url_in_scope(url: str, pattern: str | None) -> bool:
    """Return True if `url` matches the fnmatch pattern, or pattern is wildcard."""
    if pattern is None or pattern == "*":
        return True
    return fnmatch.fnmatchcase(url, pattern)


def _path_in_scope(path: Path | str, scope: Path | None) -> bool:
    """Return True if `path` sits inside `scope`, or scope is the root."""
    if scope is None:
        return True
    candidate = Path(path).resolve()
    base = scope.resolve()
    if base == Path("/"):
        return True
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        return False
