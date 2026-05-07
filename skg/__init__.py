"""Skill Knowledge Graph — capability-governed procedure reuse for LLM agents.

Public API
----------
    from skg import SKG, Node, RouteResult

    graph = SKG()
    result = graph.route("draft reviewer ping for PR #42", context={}, policy={})
    if result.hit:
        print(result.output)
"""

from skg.graph import SKG
from skg.node import Node, Manifest, CapabilityRequest, NodeStatus
from skg.router import RouteResult

__all__ = [
    "SKG",
    "Node",
    "Manifest",
    "CapabilityRequest",
    "NodeStatus",
    "RouteResult",
]

__version__ = "0.1.0"
