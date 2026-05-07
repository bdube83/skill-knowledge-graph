"""Multi-stage local router for SKG.

The router tries stages in order before reaching the LLM:
    1. Exact lookup by canonical task type
    2. Full-text search (SQLite FTS5)
    3. Vector search (Qdrant, local-hash-v1 embedding)
    4. Graph expansion on FTS/vector candidates

Each stage narrows the candidate set. The router returns a RouteResult
indicating which stage produced the match (or a miss).

TAU (vector similarity threshold) is defined in skg.index and should be
calibrated against the replay corpus before publishing results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from skg.node import Node, NodeStatus, EdgeType
from skg.policy import PolicyEngine, Grant
from skg.store import NodeStore

if TYPE_CHECKING:
    from skg.index import VectorIndex


class RouteStage(str, Enum):
    EXACT  = "exact"
    FTS    = "fts"
    VECTOR = "vector"
    GRAPH  = "graph"
    MISS   = "miss"


@dataclass
class GraphPlan:
    """An ordered list of nodes forming a capability-safe execution plan."""

    root:       Node
    nodes:      list[Node]   = field(default_factory=list)
    depth:      int          = 0


@dataclass
class RouteResult:
    """The outcome of a routing decision."""

    hit:       bool
    stage:     RouteStage
    plan:      GraphPlan | None     = None
    grant:     Grant | None         = None
    node:      Node | None          = None
    context:   dict[str, Any]       = field(default_factory=dict)
    reason:    str                  = ""

    @property
    def miss(self) -> bool:
        return not self.hit


class Router:
    """Multi-stage router. Construct with a NodeStore, PolicyEngine, and optional VectorIndex."""

    def __init__(
        self,
        store: NodeStore,
        policy: PolicyEngine,
        vector_index: "VectorIndex | None" = None,
        max_depth: int = 3,
        fts_limit: int = 20,
        vector_limit: int = 10,
    ) -> None:
        self._store        = store
        self._policy       = policy
        self._vector       = vector_index
        self._max_depth    = max_depth
        self._fts_limit    = fts_limit
        self._vector_limit = vector_limit

    def route(
        self,
        request: str,
        context: dict[str, Any] | None = None,
    ) -> RouteResult:
        """Route a task request through all stages. Return the first hit or a miss."""
        ctx = context or {}
        task_type = normalize_request(request)

        # Stage 1: exact lookup by canonical task type
        candidates = self._store.exact_lookup(task_type)
        candidates = self._filter(candidates, ctx)
        if candidates:
            return self._build_result(candidates[0], ctx, RouteStage.EXACT)

        # Stage 2: full-text search (SQLite FTS5)
        fts_hits  = self._store.fts_search(request, limit=self._fts_limit)
        fts_nodes = self._filter([n for n, _ in fts_hits], ctx)
        if fts_nodes:
            return self._build_result(fts_nodes[0], ctx, RouteStage.FTS)

        # Stage 3: vector search (Qdrant, local-hash-v1 fallback)
        if self._vector:
            vec_hits = self._vector.search(request, limit=self._vector_limit)
            for node_id, _score in vec_hits:
                node = self._store.get(node_id)
                if node and self._filter([node], ctx):
                    return self._build_result(node, ctx, RouteStage.VECTOR)

        # Stage 4: graph expansion on FTS candidates
        all_fts = [n for n, _ in fts_hits]
        for node in all_fts:
            plan = self._expand_graph(node, ctx, depth=0)
            if plan:
                grant = self._policy.grant(node.id, node.manifest, ctx)
                if grant.can_execute_dry_run:
                    return RouteResult(
                        hit=True,
                        stage=RouteStage.GRAPH,
                        plan=plan,
                        grant=grant,
                        node=node,
                        context=ctx,
                    )

        return RouteResult(
            hit=False,
            stage=RouteStage.MISS,
            reason=f"No node matched '{request}' after exact, FTS, vector, and graph stages.",
            context=ctx,
        )

    # ---- Internal --------------------------------------------------------

    def _filter(
        self,
        candidates: list[Node],
        ctx: dict,
    ) -> list[Node]:
        """Apply precondition and policy filters. Remove stale nodes."""
        result = []
        for node in candidates:
            if node.status == NodeStatus.STALE:
                continue
            if not _preconditions_hold(node, ctx):
                continue
            if not self._policy.can_grant(node.manifest, ctx):
                continue
            result.append(node)
        return result

    def _build_result(
        self,
        node: Node,
        ctx: dict,
        stage: RouteStage,
    ) -> RouteResult:
        grant = self._policy.grant(node.id, node.manifest, ctx)
        plan  = self._expand_graph(node, ctx, depth=0) or GraphPlan(root=node, nodes=[node])
        return RouteResult(
            hit=True,
            stage=stage,
            plan=plan,
            grant=grant,
            node=node,
            context=ctx,
        )

    def _expand_graph(
        self,
        node: Node,
        ctx: dict,
        depth: int,
    ) -> GraphPlan | None:
        """Expand a node's call graph up to max_depth. Return None if invalid."""
        if depth > self._max_depth:
            return None

        plan_nodes = [node]
        for edge in node.edges:
            if edge.type != EdgeType.CALLS:
                continue
            child = self._store.get(edge.target_id)
            if not child or child.status == NodeStatus.STALE:
                continue
            if not _preconditions_hold(child, ctx):
                continue
            if not self._policy.can_grant(child.manifest, ctx):
                continue
            child_plan = self._expand_graph(child, ctx, depth + 1)
            if child_plan:
                plan_nodes.extend(child_plan.nodes)

        return GraphPlan(root=node, nodes=plan_nodes, depth=depth)


def normalize_request(request: str) -> str:
    """Convert a natural-language request into a canonical task type string.

    This is a heuristic normalization: lowercase, strip punctuation, compress
    whitespace, replace common synonyms. A real implementation would use a
    small classifier or the LLM.
    """
    text = request.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", "_", text.strip())
    # Trim to a reasonable length for use as a lookup key.
    return text[:120]


def _preconditions_hold(node: Node, context: dict) -> bool:
    """Evaluate all preconditions declared in the node manifest.

    Each precondition is a dict with keys:
        name:   str
        check:  str  (Python expression evaluated against context)
        reason: str

    A missing or empty check is treated as passing. Evaluation errors
    are treated as failures (conservative).
    """
    for pc in node.manifest.preconditions:
        check = pc.get("check", "").strip()
        if not check:
            continue
        try:
            result = eval(check, {"__builtins__": {}}, context)  # noqa: S307
            if not result:
                return False
        except Exception:
            return False
    return True
