"""Vector routing stage runner for the SKG corpus.

This script measures the vector stage of the router in isolation. It indexes
the three promoted node headers (reviewer-ping-draft, git-summary, doc-update)
into a local Qdrant collection. It then embeds each task in the corpus and
queries the collection. It records per-task hit/miss against the TAU
threshold and writes an aggregate report.

The runner prefers Qdrant in-memory mode. It falls back to a Docker daemon
on localhost:6333 if requested. It writes a clean failure report when no
backend is reachable.

Usage:
    python -m eval.vector_stage_runner \
        --corpus eval/corpus.jsonl \
        --out eval/results/vector_stage_report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from skg.index import COLLECTION, EMBED_DIM, TAU, local_embed


REPO_ROOT     = Path(__file__).resolve().parent.parent
NODES_DIR     = REPO_ROOT / "nodes"
PROMOTED_NODE_IDS = ["reviewer-ping-draft", "git-summary", "doc-update"]


@dataclass
class TaskRecord:
    """One per-task vector lookup result."""

    task_id:      str
    task_text:    str
    vector_hit:   bool   = False
    top_score:    float  = 0.0
    top_node_id:  str    = ""


@dataclass
class VectorStageReport:
    """Aggregate report across the corpus."""

    qdrant_mode:       str
    tau:               float
    task_count:        int
    vector_hit_count:  int
    vector_hit_rate:   float
    top_score_p50:     float
    top_score_p95:     float
    embedding_dim:     int
    note:              str               = ""
    per_task:          list[TaskRecord]  = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        out = {
            "qdrant_mode":      self.qdrant_mode,
            "tau":              self.tau,
            "task_count":       self.task_count,
            "vector_hit_count": self.vector_hit_count,
            "vector_hit_rate":  round(self.vector_hit_rate, 4),
            "top_score_p50":    round(self.top_score_p50, 4),
            "top_score_p95":    round(self.top_score_p95, 4),
            "embedding_dim":    self.embedding_dim,
        }
        if self.note:
            out["note"] = self.note
        return out


def load_promoted_nodes(nodes_dir: Path = NODES_DIR) -> list[dict]:
    """Load the three promoted node manifests as a list of index dicts.

    Each dict has keys: node_id, task_type, header. The runner indexes
    these into Qdrant.
    """
    out = []
    for node_id in PROMOTED_NODE_IDS:
        manifest_path = nodes_dir / node_id / "manifest.yaml"
        if not manifest_path.exists():
            continue
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        out.append({
            "node_id":   node_id,
            "task_type": manifest.get("task_type", ""),
            "header":    manifest.get("header", ""),
        })
    return out


def load_corpus(path: Path) -> list[dict]:
    """Load a JSONL corpus file. Each line is a task dict."""
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return tasks


def open_qdrant(prefer: str = "memory") -> tuple[Any, str, str]:
    """Open a Qdrant client. Return (client, mode, note).

    Prefer order: memory, then docker on localhost:6333. Returns
    (None, "unavailable", reason) when nothing connects.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        return None, "unavailable", f"qdrant-client not installed: {e}"

    if prefer == "memory":
        try:
            client = QdrantClient(":memory:")
            client.get_collections()
            return client, "memory", ""
        except Exception as e:
            mem_err = str(e)
        try:
            client = QdrantClient(host="localhost", port=6333, timeout=2.0)
            client.get_collections()
            return client, "docker", ""
        except Exception as e:
            return None, "unavailable", (
                f"memory mode failed: {mem_err}; docker mode failed: {e}"
            )

    try:
        client = QdrantClient(host="localhost", port=6333, timeout=2.0)
        client.get_collections()
        return client, "docker", ""
    except Exception as e:
        return None, "unavailable", f"docker mode failed: {e}"


def index_nodes(client: Any, nodes: list[dict]) -> int:
    """Create the collection and upsert promoted-node embeddings.

    Returns the number of points indexed.
    """
    from qdrant_client.models import Distance, PointStruct, VectorParams

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    points = []
    for i, node in enumerate(nodes):
        text = f"{node.get('task_type', '')} {node.get('header', '')}".strip()
        vec = local_embed(text)
        points.append(PointStruct(
            id=i + 1,
            vector=vec,
            payload={
                "node_id":   node["node_id"],
                "task_type": node.get("task_type", ""),
            },
        ))
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def query_task(client: Any, task_text: str, limit: int = 5) -> tuple[float, str]:
    """Run a single vector lookup. Return (top_score, top_node_id).

    Scores are returned without applying TAU. The caller decides hit/miss.
    """
    vec = local_embed(task_text)
    try:
        resp = client.query_points(
            collection_name=COLLECTION,
            query=vec,
            limit=limit,
        )
        points = resp.points
    except Exception:
        return 0.0, ""
    if not points:
        return 0.0, ""
    top = points[0]
    return float(top.score), str(top.payload.get("node_id", ""))


def run_vector_stage(
    corpus: list[dict],
    nodes: list[dict],
    prefer: str = "memory",
) -> VectorStageReport:
    """Run the vector stage over the corpus and aggregate the report."""
    client, mode, note = open_qdrant(prefer=prefer)
    if client is None:
        return VectorStageReport(
            qdrant_mode=mode,
            tau=TAU,
            task_count=len(corpus),
            vector_hit_count=0,
            vector_hit_rate=0.0,
            top_score_p50=0.0,
            top_score_p95=0.0,
            embedding_dim=EMBED_DIM,
            note=note,
        )

    index_nodes(client, nodes)

    per_task: list[TaskRecord] = []
    scores: list[float] = []
    hit_count = 0

    for item in corpus:
        task_id   = item.get("id", "")
        task_text = item.get("task", "")
        score, top_node = query_task(client, task_text)
        is_hit = score >= TAU
        if is_hit:
            hit_count += 1
        per_task.append(TaskRecord(
            task_id=task_id,
            task_text=task_text,
            vector_hit=is_hit,
            top_score=round(score, 6),
            top_node_id=top_node,
        ))
        scores.append(score)

    scores_sorted = sorted(scores)
    p50 = statistics.median(scores_sorted) if scores_sorted else 0.0
    if scores_sorted:
        p95_idx = min(len(scores_sorted) - 1, int(len(scores_sorted) * 0.95))
        p95 = scores_sorted[p95_idx]
    else:
        p95 = 0.0

    return VectorStageReport(
        qdrant_mode=mode,
        tau=TAU,
        task_count=len(corpus),
        vector_hit_count=hit_count,
        vector_hit_rate=hit_count / len(corpus) if corpus else 0.0,
        top_score_p50=p50,
        top_score_p95=p95,
        embedding_dim=EMBED_DIM,
        per_task=per_task,
        note=note,
    )


def write_report(report: VectorStageReport, out_path: Path, write_per_task: bool = False) -> None:
    """Write the report JSON to out_path. Create parent dirs as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = report.to_summary_dict()
    if write_per_task:
        payload["per_task"] = [
            {
                "task_id":     r.task_id,
                "task_text":   r.task_text,
                "vector_hit":  r.vector_hit,
                "top_score":   r.top_score,
                "top_node_id": r.top_node_id,
            }
            for r in report.per_task
        ]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---- CLI entry point --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SKG vector stage runner")
    parser.add_argument(
        "--corpus",
        default=str(REPO_ROOT / "eval" / "corpus.jsonl"),
        help="Path to JSONL corpus.",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "eval" / "results" / "vector_stage_report.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--prefer",
        choices=["memory", "docker"],
        default="memory",
        help="Preferred Qdrant backend.",
    )
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="Include the per-task array in the report file.",
    )
    args = parser.parse_args()

    corpus = load_corpus(Path(args.corpus))
    nodes  = load_promoted_nodes()
    report = run_vector_stage(corpus, nodes, prefer=args.prefer)
    write_report(report, Path(args.out), write_per_task=args.per_task)

    print(f"qdrant_mode:      {report.qdrant_mode}")
    print(f"task_count:       {report.task_count}")
    print(f"vector_hit_count: {report.vector_hit_count}")
    print(f"vector_hit_rate:  {report.vector_hit_rate:.4f}")
    print(f"top_score_p50:    {report.top_score_p50:.4f}")
    print(f"top_score_p95:    {report.top_score_p95:.4f}")
    print(f"embedding_dim:    {report.embedding_dim}")
    if report.note:
        print(f"note:             {report.note}")
    print(f"\nReport written to {args.out}")
