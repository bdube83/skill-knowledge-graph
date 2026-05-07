"""Tests for skg.node — Node, Manifest, CapabilityRequest, Edge."""

import hashlib
import json
import pytest

from skg.node import (
    CapabilityRequest,
    Edge,
    EdgeType,
    Manifest,
    Node,
    NodeStatus,
)


def make_manifest(**kwargs) -> Manifest:
    defaults = dict(
        task_type="send_reviewer_ping",
        header="Draft a reviewer-ping message for a pull request.",
        tags=["git", "review"],
        requested_capabilities=[CapabilityRequest(effect="text.generate", adapter="local")],
        forbidden_capabilities=[],
        preconditions=[],
        verifiers=[],
    )
    defaults.update(kwargs)
    return Manifest(**defaults)


class TestManifest:
    def test_round_trip(self):
        m = make_manifest()
        d = m.to_dict()
        m2 = Manifest.from_dict(d)
        assert m2.task_type == m.task_type
        assert m2.header    == m.header
        assert len(m2.requested_capabilities) == 1
        assert m2.requested_capabilities[0].effect == "text.generate"

    def test_empty_capabilities(self):
        m = make_manifest(requested_capabilities=[])
        assert m.requested_capabilities == []


class TestNode:
    def test_new_factory(self):
        m = make_manifest()
        n = Node.new(id="ping-001", manifest=m, source="def fn(): pass")
        assert n.id     == "ping-001"
        assert n.status == NodeStatus.CANDIDATE
        assert len(n.source_sha256) == 64

    def test_source_sha256(self):
        source = "fn main() {}"
        expected = hashlib.sha256(source.encode()).hexdigest()
        m = make_manifest()
        n = Node.new(id="x", manifest=m, source=source)
        assert n.source_sha256 == expected

    def test_round_trip(self):
        m = make_manifest()
        n = Node.new(id="ping-001", manifest=m, source="fn main() {}")
        d = n.to_dict()
        n2 = Node.from_dict(d)
        assert n2.id            == n.id
        assert n2.source_sha256 == n.source_sha256
        assert n2.status        == n.status

    def test_edges(self):
        m = make_manifest()
        e = Edge(type=EdgeType.CALLS, target_id="other-node", weight=1.0)
        n = Node.new(id="ping-001", manifest=m, source="", edges=[e])
        d = n.to_dict()
        n2 = Node.from_dict(d)
        assert len(n2.edges)           == 1
        assert n2.edges[0].target_id   == "other-node"
        assert n2.edges[0].type        == EdgeType.CALLS
