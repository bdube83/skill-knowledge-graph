"""Qdrant-backed vector index for SKG router (slice 2: vector stage).

The vector index is the third routing stage, after exact lookup and SQLite FTS.
It embeds task headers using a local hash-based fallback (no external model
required for v0.1) and queries Qdrant for nearest neighbours above a
similarity threshold TAU.

TAU = 0.88 — calibrate this against the replay corpus before publishing.
See: designs/proposed/skill-graph-codex-v10/design/skill-knowledge-graph.md
     "Router and index slice".

Qdrant is run locally. No cloud API key required. The collection is rebuilt
from the node store whenever rebuild_index() is called.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


# Cosine similarity threshold — router does not return a vector match below this.
TAU = 0.88

# Dimension of the local hash embedding.
EMBED_DIM = 128

# Qdrant collection name.
COLLECTION = "skg_nodes"


def local_embed(text: str) -> list[float]:
    """Deterministic local embedding using SHA-256 hash projection.

    This is the local-hash-v1 fallback described in the design. It produces
    a stable 128-dim float vector from any text string without calling an
    external model. Similarity is weak compared to a real encoder but
    sufficient for exact and near-exact header matches.

    Replace with a real sentence encoder (e.g. sentence-transformers,
    text-embedding-3-small) once the router slice benchmarks are complete.
    """
    digest = hashlib.sha256(text.lower().encode()).digest()  # 32 bytes
    # Tile to EMBED_DIM floats by repeating and normalising.
    raw = []
    seed = digest
    while len(raw) < EMBED_DIM:
        seed = hashlib.sha256(seed).digest()
        raw.extend(struct.unpack("16f", seed[:64]))
    vec = raw[:EMBED_DIM]
    norm = (sum(v * v for v in vec) ** 0.5) or 1.0
    return [v / norm for v in vec]


class VectorIndex:
    """Qdrant-backed vector index for node headers.

    Construction is lazy: the Qdrant client connects on first use. If Qdrant
    is not reachable, all searches return empty results (the router falls back
    to graph expansion).
    """

    def __init__(self, qdrant_path: Path | str | None = None) -> None:
        self._path   = Path(qdrant_path) if qdrant_path else Path.home() / ".skg" / "indexes" / "qdrant"
        self._client: Any = None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if not _QDRANT_AVAILABLE:
            return False
        try:
            self._path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self._path))
            return True
        except Exception:
            return False

    def _ensure_collection(self) -> None:
        assert self._client
        existing = [c.name for c in self._client.get_collections().collections]
        if COLLECTION not in existing:
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    def upsert(self, node_id: str, header_text: str, task_type: str) -> None:
        if not self._ensure_client():
            return
        self._ensure_collection()
        vec = local_embed(f"{task_type} {header_text}")
        point_id = abs(hash(node_id)) % (2**53)
        self._client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector=vec,
                payload={"node_id": node_id, "task_type": task_type},
            )],
        )

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Return (node_id, score) pairs above TAU, best-first."""
        if not self._ensure_client():
            return []
        try:
            self._ensure_collection()
            vec = local_embed(query)
            hits = self._client.search(
                collection_name=COLLECTION,
                query_vector=vec,
                limit=limit,
                score_threshold=TAU,
            )
            return [(h.payload["node_id"], h.score) for h in hits]
        except Exception:
            return []

    def delete(self, node_id: str) -> None:
        if not self._ensure_client():
            return
        try:
            point_id = abs(hash(node_id)) % (2**53)
            self._client.delete(
                collection_name=COLLECTION,
                points_selector=[point_id],
            )
        except Exception:
            pass

    def rebuild(self, nodes: list[dict]) -> int:
        """Rebuild the entire collection from a list of node dicts.

        Each dict must have: node_id, task_type, header (str).
        Returns the number of nodes indexed.
        """
        if not self._ensure_client():
            return 0
        self._ensure_collection()
        try:
            self._client.delete_collection(COLLECTION)
        except Exception:
            pass
        self._ensure_collection()
        count = 0
        for node in nodes:
            self.upsert(node["node_id"], node.get("header", ""), node.get("task_type", ""))
            count += 1
        return count
