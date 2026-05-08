"""Tests for the vector stage runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _qdrant_memory_or_skip():
    """Return a fresh in-memory QdrantClient or skip the test."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        pytest.skip("qdrant-client not installed")
    try:
        client = QdrantClient(":memory:")
        client.get_collections()
    except Exception as e:
        pytest.skip(f"in-memory Qdrant unavailable: {e}")
    return client


def test_load_promoted_nodes_returns_three():
    """The runner finds three promoted node headers on disk."""
    from eval.vector_stage_runner import load_promoted_nodes

    nodes = load_promoted_nodes()
    assert len(nodes) == 3
    ids = {n["node_id"] for n in nodes}
    assert ids == {"reviewer-ping-draft", "git-summary", "doc-update"}
    for n in nodes:
        assert n["task_type"]
        assert n["header"]


def test_open_qdrant_memory_returns_client():
    """Memory mode opens a usable client."""
    _qdrant_memory_or_skip()  # gate the env

    from eval.vector_stage_runner import open_qdrant

    client, mode, note = open_qdrant(prefer="memory")
    assert client is not None
    assert mode == "memory"
    assert note == ""


def test_index_nodes_creates_collection():
    """Indexing the promoted nodes upserts three points into the collection."""
    _qdrant_memory_or_skip()

    from skg.index import COLLECTION
    from eval.vector_stage_runner import index_nodes, load_promoted_nodes, open_qdrant

    client, mode, _ = open_qdrant(prefer="memory")
    assert mode == "memory"
    nodes = load_promoted_nodes()
    count = index_nodes(client, nodes)
    assert count == 3

    info = client.get_collection(COLLECTION)
    assert info.points_count == 3


def test_run_vector_stage_small_corpus_produces_non_empty_report():
    """Run end-to-end on a small corpus and confirm the report has entries."""
    _qdrant_memory_or_skip()

    from skg.index import EMBED_DIM, TAU
    from eval.vector_stage_runner import load_promoted_nodes, run_vector_stage

    corpus = [
        {"id": "t0001", "task": "Draft a reviewer ping for PR review"},
        {"id": "t0002", "task": "Summarise recent git commits for a repository branch."},
        {"id": "t0003", "task": "Update a documentation section given a change description."},
        {"id": "t0004", "task": "Order a coffee from the espresso bar"},
    ]
    nodes = load_promoted_nodes()

    report = run_vector_stage(corpus, nodes, prefer="memory")

    assert report.qdrant_mode == "memory"
    assert report.tau == TAU
    assert report.embedding_dim == EMBED_DIM
    assert report.task_count == len(corpus)
    assert len(report.per_task) == len(corpus)
    for r in report.per_task:
        assert r.task_id
        assert r.task_text
        assert 0.0 <= r.top_score <= 1.0


def test_exact_header_match_scores_one():
    """A query identical to an indexed header scores 1.0 and counts as a hit."""
    _qdrant_memory_or_skip()

    from eval.vector_stage_runner import (
        load_promoted_nodes,
        run_vector_stage,
    )

    nodes = load_promoted_nodes()
    # Build a corpus where each task is the literal "task_type header" the
    # runner indexes. This is the strongest possible match for the local
    # hash embedding.
    corpus = [
        {"id": f"hit-{i}", "task": f"{n['task_type']} {n['header']}"}
        for i, n in enumerate(nodes)
    ]
    report = run_vector_stage(corpus, nodes, prefer="memory")

    assert report.vector_hit_count == len(corpus)
    assert report.vector_hit_rate == 1.0
    for r in report.per_task:
        assert r.vector_hit
        assert r.top_score >= 0.99
        assert r.top_node_id


def test_write_report_emits_required_fields(tmp_path: Path):
    """The written JSON contains every field the spec requires."""
    _qdrant_memory_or_skip()

    from eval.vector_stage_runner import (
        load_promoted_nodes,
        run_vector_stage,
        write_report,
    )

    corpus = [
        {"id": "t0001", "task": "Draft a reviewer ping for PR review"},
        {"id": "t0002", "task": "Order a coffee"},
    ]
    nodes = load_promoted_nodes()
    report = run_vector_stage(corpus, nodes, prefer="memory")

    out_path = tmp_path / "results" / "vector_stage_report.json"
    write_report(report, out_path)

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    required = {
        "qdrant_mode",
        "tau",
        "task_count",
        "vector_hit_count",
        "vector_hit_rate",
        "top_score_p50",
        "top_score_p95",
        "embedding_dim",
    }
    assert required.issubset(payload.keys())
    assert payload["task_count"] == 2
    assert payload["embedding_dim"] == report.embedding_dim


def test_unavailable_qdrant_writes_clean_failure_report(monkeypatch, tmp_path: Path):
    """If Qdrant cannot start, the runner emits qdrant_mode='unavailable'."""
    from eval import vector_stage_runner as vsr

    def fake_open(prefer="memory"):
        return None, "unavailable", "simulated failure for test"

    monkeypatch.setattr(vsr, "open_qdrant", fake_open)

    corpus = [{"id": "t1", "task": "anything"}]
    nodes = vsr.load_promoted_nodes()
    report = vsr.run_vector_stage(corpus, nodes, prefer="memory")

    assert report.qdrant_mode == "unavailable"
    assert report.vector_hit_count == 0
    assert "simulated failure" in report.note

    out_path = tmp_path / "vector_stage_report.json"
    vsr.write_report(report, out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["qdrant_mode"] == "unavailable"
    assert payload["note"]


def test_local_embed_returns_normalised_vector_of_correct_dim():
    """The deterministic embedding has the right shape and unit norm."""
    from skg.index import EMBED_DIM, local_embed

    vec = local_embed("Draft a reviewer-ping message for a pull request.")
    assert len(vec) == EMBED_DIM
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-5
