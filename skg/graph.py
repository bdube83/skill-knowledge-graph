"""SKG facade: wires store, router, and policy.

This is slices 1 and 2 of the north-star build:
  - Kernel: manifest parsing, effect algebra, policy grants, node store, attestation
  - Router: exact lookup, SQLite FTS, graph expansion

Slice 3 (Wasmtime execution with grant handles) is not implemented here.
Node execution requires Rust-to-WASI compilation and a Wasmtime linker that
enforces granted handles at the import level. That boundary cannot be safely
approximated in Python without losing the core security property.

See: designs/proposed/skill-graph-codex-v10/design/skill-knowledge-graph.md
     "Runtime slice" — the gate is: a node cannot access network, filesystem,
     or secrets without a grant handle. Python cannot enforce this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skg.attestation import AttestationStore
from skg.node import Node, NodeStatus
from skg.policy import PolicyEngine, default_allow_policy
from skg.router import Router, RouteResult
from skg.store import NodeStore


_DEFAULT_SKG_DIR = Path.home() / ".skg"
_DEFAULT_DB_PATH = _DEFAULT_SKG_DIR / "skg.db"
_DEFAULT_ATTEST  = _DEFAULT_SKG_DIR / "attestations.jsonl"
_DEFAULT_POLICY  = Path.home() / ".agent-proxy" / "policy.yaml"


class SKG:
    """Skill Knowledge Graph.

    Covers kernel and router slices only. Execution is deferred to the
    Wasmtime runtime slice; calling execute() on a RouteResult raises
    NotImplementedError with a clear explanation.

    Parameters
    ----------
    store_path:
        Path to the SQLite database. Created if it does not exist.
    policy_path:
        Path to the YAML policy table. Falls back to a permissive default if
        the file does not exist.
    attestation_path:
        Path for the attestation JSONL log.
    max_router_depth:
        Maximum graph expansion depth during routing.
    """

    def __init__(
        self,
        store_path: Path | str | None = None,
        policy_path: Path | str | None = None,
        attestation_path: Path | str | None = None,
        max_router_depth: int = 3,
    ) -> None:
        _DEFAULT_SKG_DIR.mkdir(parents=True, exist_ok=True)

        db_path  = Path(store_path)       if store_path       else _DEFAULT_DB_PATH
        pol_path = Path(policy_path)      if policy_path      else _DEFAULT_POLICY
        att_path = Path(attestation_path) if attestation_path else _DEFAULT_ATTEST

        self._store       = NodeStore(db_path).connect()
        self._policy      = (
            PolicyEngine(pol_path) if pol_path.exists()
            else default_allow_policy()
        )
        self._router      = Router(self._store, self._policy, max_depth=max_router_depth)
        self._attestation = AttestationStore(att_path)

    # ---- Node management -------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add or update a node in the store."""
        self._store.put(node)

    def get_node(self, node_id: str) -> Node | None:
        return self._store.get(node_id)

    def promote(self, node_id: str) -> None:
        """Promote a candidate node to active status.

        The full promotion protocol (six gates including verifier contracts,
        dry-run success, and replay acceptance) is defined in the verifier
        and promotion slice. This method advances node status in the store
        only; gate enforcement is the caller's responsibility until that
        slice ships.
        """
        self._store.promote(node_id)

    def retire(self, node_id: str) -> None:
        """Mark a node stale so the router stops routing to it."""
        node = self._store.get(node_id)
        if node:
            node.status = NodeStatus.STALE
            self._store.put(node)

    # ---- Routing ---------------------------------------------------------

    def route(
        self,
        request: str,
        context: dict[str, Any] | None = None,
    ) -> RouteResult:
        """Route a task request through exact, FTS, and graph stages.

        Returns a RouteResult. If hit is True, the result carries the matched
        node and an issued policy grant. The caller is responsible for
        execution via the Wasmtime runtime (slice 3).
        """
        return self._router.route(request, context)

    # ---- Execution placeholder -------------------------------------------

    def execute(self, result: RouteResult) -> None:  # noqa: ARG002
        """Not implemented. Requires the Wasmtime runtime slice.

        Node execution depends on Rust-to-WASI compilation and a Wasmtime
        linker that enforces granted handles at the import level. See:
        designs/proposed/skill-graph-codex-v10/design/skill-knowledge-graph.md
        "Runtime slice".
        """
        raise NotImplementedError(
            "execute() requires the Wasmtime runtime slice (slice 3). "
            "Nodes must be compiled to WASI targets; grant handles are "
            "enforced by the Wasmtime linker, not by Python. "
            "See designs/proposed/skill-graph-codex-v10/design/"
            "skill-knowledge-graph.md for the build plan."
        )

    # ---- Attestation -------------------------------------------------

    def attestations(self, node_id: str) -> list:
        """Return all attestation records for a node."""
        return self._attestation.get_all(node_id)
