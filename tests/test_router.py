"""Tests for skg.router — multi-stage routing with real store and policy."""

import pytest

from skg.node import CapabilityRequest, Manifest, Node, NodeStatus
from skg.policy import default_allow_policy
from skg.router import Router, RouteStage
from skg.store import NodeStore


def make_active_node(node_id: str, task_type: str, header: str) -> Node:
    m = Manifest(
        task_type=task_type,
        header=header,
        tags=["test"],
        requested_capabilities=[CapabilityRequest(effect="text.generate", adapter="local")],
        forbidden_capabilities=[],
        preconditions=[],
        verifiers=[],
    )
    n = Node.new(id=node_id, manifest=m, source="fn main() {}")
    n.status = NodeStatus.ACTIVE
    return n


@pytest.fixture
def router(tmp_path):
    store = NodeStore(tmp_path / "skg.db")
    store.connect()
    policy = default_allow_policy()
    r = Router(store=store, policy=policy)
    yield r, store
    store.close()


class TestRouterExactStage:
    def test_exact_hit(self, router):
        r, store = router
        node = make_active_node("ping-001", "send_reviewer_ping", "Ping reviewers.")
        store.put(node)
        result = r.route("send_reviewer_ping")
        assert result.hit is True
        assert result.stage == RouteStage.EXACT
        assert result.node.id == "ping-001"

    def test_exact_miss_falls_to_fts(self, router):
        r, store = router
        node = make_active_node("ping-001", "send_reviewer_ping", "Draft ping for pull request reviewer")
        store.put(node)
        result = r.route("send ping to reviewer for pull request")
        # Should hit on FTS or better, not exact
        assert result.hit is True
        assert result.stage in (RouteStage.FTS, RouteStage.VECTOR, RouteStage.GRAPH)


class TestRouterFTSStage:
    def test_fts_hit(self, router):
        r, store = router
        node = make_active_node("ping-001", "xunrelated", "Draft a reviewer ping message for pull requests")
        store.put(node)
        result = r.route("reviewer ping pull request")
        assert result.hit is True

    def test_miss_on_unrelated_query(self, router):
        r, store = router
        node = make_active_node("ping-001", "send_ping", "reviewer ping pull request")
        store.put(node)
        result = r.route("completely unrelated xyz123 task")
        assert result.hit is False
        assert result.stage == RouteStage.MISS

    def test_stale_node_not_returned(self, router):
        r, store = router
        node = make_active_node("ping-001", "send_reviewer_ping", "Ping reviewers.")
        node.status = NodeStatus.STALE
        store.put(node)
        result = r.route("send_reviewer_ping")
        assert result.hit is False


class TestRouterGrant:
    def test_hit_includes_grant(self, router):
        r, store = router
        node = make_active_node("ping-001", "send_reviewer_ping", "Ping reviewers.")
        store.put(node)
        result = r.route("send_reviewer_ping")
        assert result.hit is True
        assert result.grant is not None
        granted_effects = [c.effect for c in result.grant.granted]
        assert "text.generate" in granted_effects

    def test_miss_has_no_grant(self, router):
        r, store = router
        result = r.route("nothing here at all xyz")
        assert result.grant is None
