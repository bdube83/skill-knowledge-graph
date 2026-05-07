"""SQLite-backed node store with FTS5 full-text search.

The store is the source of authority for all nodes and their manifests.
Retrieval indexes (FTS5, vector) are derived from the store and rebuildable.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator

from skg.node import Node, NodeStatus


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id           TEXT PRIMARY KEY,
    task_type    TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'candidate',
    manifest_json TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT '',
    source_sha256 TEXT NOT NULL DEFAULT '',
    edges_json   TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL DEFAULT '',
    promoted_at  TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    node_id,
    header,
    task_type,
    tags
);

CREATE TABLE IF NOT EXISTS fts_sync (
    node_id TEXT PRIMARY KEY,
    indexed_at TEXT NOT NULL
);
"""


class NodeStore:
    """Persistent SQLite node store with FTS5 routing index.

    Thread safety: each connection is used from one thread only.
    Use NodeStore.open() to get a context-managed connection.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> "NodeStore":
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "NodeStore":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.close()

    # ---- Write operations ------------------------------------------------

    def put(self, node: Node) -> None:
        """Insert or replace a node. Updates FTS index."""
        assert self._conn
        manifest = node.manifest
        self._conn.execute(
            """
            INSERT OR REPLACE INTO nodes
                (id, task_type, status, manifest_json, source, source_sha256,
                 edges_json, created_at, promoted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                manifest.task_type,
                node.status.value,
                json.dumps(manifest.to_dict()),
                node.source,
                node.source_sha256,
                json.dumps([e.to_dict() for e in node.edges]),
                node.created_at,
                node.promoted_at,
            ),
        )
        self._index_fts(node)
        self._conn.commit()

    def mark_stale(self, node_id: str) -> None:
        assert self._conn
        self._conn.execute(
            "UPDATE nodes SET status = 'stale' WHERE id = ?", (node_id,)
        )
        self._conn.commit()

    def promote(self, node_id: str, promoted_at: str | None = None) -> None:
        """Mark a node as active (promoted). Caller is responsible for gate checks."""
        import datetime
        assert self._conn
        ts = promoted_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE nodes SET status = ?, promoted_at = ? WHERE id = ?",
            (NodeStatus.ACTIVE.value, ts, node_id),
        )
        self._conn.commit()

    def delete(self, node_id: str) -> None:
        assert self._conn
        self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.execute(
            "DELETE FROM nodes_fts WHERE node_id = ?", (node_id,)
        )
        self._conn.execute("DELETE FROM fts_sync WHERE node_id = ?", (node_id,))
        self._conn.commit()

    # ---- Read operations -------------------------------------------------

    def get(self, node_id: str) -> Node | None:
        assert self._conn
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def list(
        self,
        status: NodeStatus | None = None,
        task_type: str | None = None,
    ) -> list[Node]:
        assert self._conn
        query = "SELECT * FROM nodes WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    def exact_lookup(self, task_type: str) -> list[Node]:
        """Return all non-stale nodes for the given task type."""
        assert self._conn
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE task_type = ? AND status != 'stale'",
            (task_type,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def fts_search(self, query: str, limit: int = 20) -> list[tuple[Node, float]]:
        """Full-text search over headers, task types, and tags.

        Returns (node, bm25_score) pairs sorted by relevance.
        BM25 scores from SQLite FTS5 are negative; more negative means more
        relevant. We negate them so higher is better.
        """
        assert self._conn
        # Sanitize query: remove FTS5 special characters that would cause parse errors.
        safe_q = _sanitize_fts_query(query)
        if not safe_q:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT n.*, -bm25(nodes_fts) AS score
                FROM nodes_fts
                JOIN nodes n ON n.id = nodes_fts.node_id
                WHERE nodes_fts MATCH ?
                  AND n.status != 'stale'
                ORDER BY score DESC
                LIMIT ?
                """,
                (safe_q, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(self._row_to_node(r), r["score"]) for r in rows]

    def iter_all(self) -> Iterator[Node]:
        assert self._conn
        for row in self._conn.execute("SELECT * FROM nodes ORDER BY created_at"):
            yield self._row_to_node(row)

    def list_all(self) -> list[Node]:
        return list(self.iter_all())

    def list_active(self) -> list[Node]:
        return self.list(status=NodeStatus.ACTIVE)
    def rebuild_fts(self) -> int:
        """Rebuild the FTS index from all nodes in the store. Returns count."""
        assert self._conn
        self._conn.execute("DELETE FROM nodes_fts")
        self._conn.execute("DELETE FROM fts_sync")
        count = 0
        for node in self.iter_all():
            self._index_fts(node)
            count += 1
        self._conn.commit()
        return count

    # ---- Internal --------------------------------------------------------

    def _index_fts(self, node: Node) -> None:
        assert self._conn
        import datetime
        manifest = node.manifest
        tags_text = " ".join(manifest.tags)
        self._conn.execute(
            "DELETE FROM nodes_fts WHERE node_id = ?", (node.id,)
        )
        self._conn.execute(
            "INSERT INTO nodes_fts (node_id, header, task_type, tags) VALUES (?, ?, ?, ?)",
            (node.id, manifest.header, manifest.task_type, tags_text),
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO fts_sync (node_id, indexed_at)
            VALUES (?, ?)
            """,
            (node.id, datetime.datetime.now(datetime.timezone.utc).isoformat()),
        )

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> Node:
        manifest_data = json.loads(row["manifest_json"])
        from skg.node import Manifest, Edge, EdgeType
        edges_data = json.loads(row["edges_json"])
        edges = [Edge.from_dict(e) for e in edges_data]
        manifest = Manifest.from_dict(manifest_data)
        return Node(
            id=row["id"],
            manifest=manifest,
            source=row["source"],
            source_sha256=row["source_sha256"],
            edges=edges,
            status=NodeStatus(row["status"]),
            created_at=row["created_at"],
            promoted_at=row["promoted_at"],
        )


def _sanitize_fts_query(query: str) -> str:
    """Convert a plain-text query into a safe FTS5 MATCH expression.

    Strips FTS5 operator characters that would cause parse errors when the
    query comes from raw user input or LLM output.
    """
    import re
    # Remove FTS5 special chars: " ' ( ) * : ^
    clean = re.sub(r'["\'\(\)\*\:\^]', ' ', query)
    # Collapse whitespace
    tokens = clean.split()
    if not tokens:
        return ""
    # Join tokens with OR so partial matches still surface candidates
    return " OR ".join(tokens)
