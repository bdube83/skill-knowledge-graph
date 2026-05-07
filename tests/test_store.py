"""Tests for skg.store — NodeStore put/get/list/fts/promote."""

import tempfile
from pathlib import Path

import pytest

from skg.node import CapabilityRequest, Manifest, Node, NodeStatus
from skg.store import NodeStore


def make_node(node_id: str, task_type: str = "send_ping", header: str = "Draft a ping.") -> Node:
    m = Manifest(
        task_type=task_type,
        header=header,
        tags=["test"],
        requested_capabilities=[CapabilityRequest(effect="text.generate", adapter="local")],
        forbidden_capabilities=[],
        preconditions=[],
        verifiers=[],
    )
    return Node.new(id=node_id, manifest=m, source="fn main() {}")


@pytest.fixture
def store(tmp_path):
    s = NodeStore(tmp_path / "skg.db")
    s.connect()
    yield s
    s.close()


class TestNodeStore:
    def test_put_and_get(self, store):
        n = make_node("ping-001")
        store.put(n)
        fetched = store.get("ping-001")
        assert fetched is not None
        assert fetched.id == "ping-001"

    def test_get_missing(self, store):
        assert store.get("does-not-exist") is None

    def test_put_updates(self, store):
        n = make_node("ping-001")
        store.put(n)
        n.status = NodeStatus.ACTIVE
        store.put(n)
        fetched = store.get("ping-001")
        assert fetched.status == NodeStatus.ACTIVE

    def test_exact_lookup(self, store):
        store.put(make_node("ping-001", task_type="send_ping"))
        store.put(make_node("ping-002", task_type="send_ping"))
        store.put(make_node("other-001", task_type="other_task"))
        results = store.exact_lookup("send_ping")
        assert len(results) == 2
        ids = {n.id for n in results}
        assert "ping-001" in ids
        assert "ping-002" in ids

    def test_exact_lookup_excludes_stale(self, store):
        n = make_node("ping-001", task_type="send_ping")
        n.status = NodeStatus.STALE
        store.put(n)
        results = store.exact_lookup("send_ping")
        assert results == []

    def test_fts_search(self, store):
        store.put(make_node("ping-001", header="Draft a reviewer ping for pull request review"))
        store.put(make_node("other-001", header="Send email notification to team"))
        results = store.fts_search("reviewer pull request", limit=10)
        assert any(n.id == "ping-001" for n, _ in results)

    def test_promote(self, store):
        n = make_node("ping-001")
        store.put(n)
        store.promote("ping-001")
        fetched = store.get("ping-001")
        assert fetched.status == NodeStatus.ACTIVE

    def test_list_active(self, store):
        n1 = make_node("ping-001")
        n1.status = NodeStatus.ACTIVE
        n2 = make_node("ping-002")
        n2.status = NodeStatus.STALE
        store.put(n1)
        store.put(n2)
        active = store.list_active()
        assert len(active) == 1
        assert active[0].id == "ping-001"

    def test_list_all(self, store):
        store.put(make_node("a"))
        store.put(make_node("b"))
        assert len(store.list_all()) == 2

    def test_rebuild_fts(self, store):
        store.put(make_node("ping-001", header="reviewer ping message"))
        count = store.rebuild_fts()
        assert count >= 1
