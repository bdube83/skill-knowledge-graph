"""Node and Manifest dataclasses for SKG.

A node is the fundamental unit of procedure reuse. It stores everything needed
to route, execute, verify, and audit a single reusable agent procedure.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeStatus(str, Enum):
    CANDIDATE = "candidate"   # newly created, awaiting promotion gates
    ACTIVE    = "active"      # promoted, eligible for routing
    STALE     = "stale"       # retired, excluded from all routing paths
    LEARNED   = "learned"     # alias for CANDIDATE (legacy)


class EdgeType(str, Enum):
    CALLS      = "calls"
    REQUIRES   = "requires"
    PRODUCES   = "produces"
    VALIDATES  = "validates"
    SUPERSEDES = "supersedes"
    CONFLICTS  = "conflicts"


@dataclass
class CapabilityRequest:
    effect:       str                    # one of the 12 Effect classes (string value)
    adapter:      str                    # adapter name, e.g. "github", "gmail"
    scope:        dict[str, Any] = field(default_factory=dict)
    url_pattern:  str | None     = None  # fnmatch glob applied to URLs at handle-mint time
    path_scope:   str | None     = None  # filesystem prefix applied to paths at handle-mint time

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"effect": self.effect, "adapter": self.adapter, "scope": self.scope}
        if self.url_pattern is not None:
            d["url_pattern"] = self.url_pattern
        if self.path_scope is not None:
            d["path_scope"] = self.path_scope
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CapabilityRequest":
        return cls(
            effect=d["effect"],
            adapter=d.get("adapter", ""),
            scope=d.get("scope", {}),
            url_pattern=d.get("url_pattern"),
            path_scope=d.get("path_scope"),
        )


@dataclass
class Edge:
    type:      EdgeType
    target_id: str          # node_id of the related node
    weight:    float        = 1.0

    def to_dict(self) -> dict:
        return {"type": self.type.value, "target_id": self.target_id, "weight": self.weight}

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(type=EdgeType(d["type"]), target_id=d["target_id"], weight=d.get("weight", 1.0))


@dataclass
class Manifest:
    """Typed contract for a node.

    The manifest is a declaration, not a grant. The policy engine reads it
    to compute which capabilities to grant for a given run.
    """

    task_type:               str
    header:                  str
    preconditions:           list[dict[str, str]] = field(default_factory=list)
    requested_capabilities:  list[CapabilityRequest] = field(default_factory=list)
    forbidden_capabilities:  list[str] = field(default_factory=list)
    verifiers:               list[dict[str, str]] = field(default_factory=list)
    input_schema:            dict[str, Any] = field(default_factory=dict)
    output_schema:           dict[str, Any] = field(default_factory=dict)
    tags:                    list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_type":              self.task_type,
            "header":                 self.header,
            "preconditions":          self.preconditions,
            "requested_capabilities": [c.to_dict() for c in self.requested_capabilities],
            "forbidden_capabilities": self.forbidden_capabilities,
            "verifiers":              self.verifiers,
            "input_schema":           self.input_schema,
            "output_schema":          self.output_schema,
            "tags":                   self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(
            task_type=d["task_type"],
            header=d["header"],
            preconditions=d.get("preconditions", []),
            requested_capabilities=[
                CapabilityRequest.from_dict(c)
                for c in d.get("requested_capabilities", [])
            ],
            forbidden_capabilities=d.get("forbidden_capabilities", []),
            verifiers=d.get("verifiers", []),
            input_schema=d.get("input_schema", {}),
            output_schema=d.get("output_schema", {}),
            tags=d.get("tags", []),
        )


@dataclass
class Node:
    """A single reusable agent procedure stored in the SKG node store."""

    id:          str
    manifest:    Manifest
    source:      str              # Python source code for the procedure
    edges:       list[Edge]       = field(default_factory=list)
    status:      NodeStatus       = NodeStatus.LEARNED
    created_at:  str              = ""
    promoted_at: str | None       = None

    # Computed at store time; verified at attestation time.
    source_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.source_sha256 and self.source:
            self.source_sha256 = hashlib.sha256(
                self.source.encode("utf-8")
            ).hexdigest()

    @classmethod
    def new(cls, manifest: Manifest, source: str, id: str | None = None, edges: list | None = None) -> "Node":
        """Create a new candidate node. ID is auto-generated if not provided."""
        import datetime
        return cls(
            id=id or str(uuid.uuid4()),
            manifest=manifest,
            source=source,
            edges=edges or [],
            status=NodeStatus.CANDIDATE,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "manifest":       self.manifest.to_dict(),
            "source":         self.source,
            "source_sha256":  self.source_sha256,
            "edges":          [e.to_dict() for e in self.edges],
            "status":         self.status.value,
            "created_at":     self.created_at,
            "promoted_at":    self.promoted_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            id=d["id"],
            manifest=Manifest.from_dict(d["manifest"]),
            source=d.get("source", ""),
            source_sha256=d.get("source_sha256", ""),
            edges=[Edge.from_dict(e) for e in d.get("edges", [])],
            status=NodeStatus(d.get("status", "learned")),
            created_at=d.get("created_at", ""),
            promoted_at=d.get("promoted_at"),
        )
