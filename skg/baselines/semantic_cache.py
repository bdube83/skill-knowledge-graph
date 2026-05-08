"""Baseline C: semantic response cache.

Section 7.2 of the paper defines Baseline C as a GPTCache-style cosine
similarity cache over task embeddings. The runtime stores
`(task_embedding, response)` pairs. On a new task it embeds the task,
finds the nearest stored embedding by cosine similarity, and returns
that response when similarity is above a threshold (default 0.85). On
miss it falls through to a stub LLM and stores the new pair.

Embedding choice: this baseline uses a deterministic feature-hashed
embedding rather than a real model. Each token in the lower-cased
task string contributes to one of `EMBED_DIM` slots picked by
`hash(token) % EMBED_DIM`. The vector is then L2-normalised. The
embedding is offline and stable across runs. The point is to give the
paper a measurable C baseline, not a faithful semantic model. Replace
this with a real embedding API for production runs; the runtime
contract does not change.

The class implements the same `execute(...)` signature as
`skg.wasmtime_launcher.WasmtimeRuntime`. The `wasm_path` argument is
accepted for signature parity and ignored.

Reference:
  designs/in-progress/skill-graph-codex-v10/paper-draft.md  Section 7.2
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..wasmtime_launcher import WasmRunResult


__all__ = ["SemanticCacheRuntime", "WasmRunResult"]


DEFAULT_TIMEOUT_MS = 5_000
EMBED_DIM          = 64
DEFAULT_THRESHOLD  = 0.85

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _stub_llm(task: str, context: dict[str, Any]) -> str:
    """Offline stand-in for a planning LLM call on a cache miss."""
    ctx_hash = hash(repr(sorted(context.items()))) & 0xFFFF
    return f"stub-llm-response for {task!r} ctx#{ctx_hash:04x}"


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_to_slot(token: str, dim: int) -> int:
    """Deterministic, process-stable token to slot map.

    Python's built-in `hash()` is randomised across processes when
    `PYTHONHASHSEED` is not set. This function uses MD5 to keep the
    embedding stable across runs; that matters for the paper's
    reproducibility claims.
    """
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dim


def _embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Feature-hashed L2-normalised embedding of `text`.

    Each token raises the count in one slot. The resulting vector is
    L2-normalised so cosine similarity reduces to a dot product. An
    empty token list maps to a zero vector; cosine similarity then
    returns 0.0 for any pair, which is below threshold and forces a
    miss.
    """
    vec = np.zeros(dim, dtype=np.float64)
    for token in _tokenise(text):
        vec[_hash_to_slot(token, dim)] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for L2-normalised vectors.

    Falls back to the general formula when either input is the zero
    vector, in which case the similarity is 0.0.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticCacheRuntime:
    """GPTCache-style cosine cache keyed on task embeddings.

    The constructor accepts `seed_pairs` of `(task, response)` tuples
    and an optional `threshold`. Each seed task is embedded and stored
    alongside its response. On `execute()` the runtime embeds the
    incoming task, finds the highest-similarity stored entry, and
    returns its response when similarity meets or exceeds the threshold.
    A miss invokes the stub LLM, stores the new pair for later hits,
    and returns the stub output.

    The `wasm_path`, `granted_effects`, and `dry_run` arguments are
    accepted for signature parity and ignored.
    """

    def __init__(
        self,
        seed_pairs: list[tuple[str, str]] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        self._threshold  = float(threshold)
        self._timeout_ms = timeout_ms
        self._entries: list[tuple[np.ndarray, str, str]] = []
        for task, response in seed_pairs or []:
            self._entries.append((_embed(task), task, response))

    def execute(
        self,
        wasm_path: Path | str,
        node_id: str,
        task: str,
        context: dict[str, Any],
        granted_effects: list[str],
        dry_run: bool = False,
    ) -> WasmRunResult:
        """Embed the task and return the nearest cached response or stub."""
        start = time.monotonic()
        query = _embed(task)

        best_sim   = -1.0
        best_entry: tuple[np.ndarray, str, str] | None = None
        for entry in self._entries:
            sim = _cosine(query, entry[0])
            if sim > best_sim:
                best_sim   = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self._threshold:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            return WasmRunResult(
                node_id=node_id,
                success=True,
                output={
                    "response":   best_entry[2],
                    "source":     "cache",
                    "similarity": round(best_sim, 4),
                    "matched":    best_entry[1],
                },
                error="",
                duration_ms=duration_ms,
                observed_effects=[],
            )

        response = _stub_llm(task, context)
        self._entries.append((query, task, response))
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        return WasmRunResult(
            node_id=node_id,
            success=True,
            output={
                "response":   response,
                "source":     "stub_llm",
                "similarity": round(max(best_sim, 0.0), 4),
            },
            error="",
            duration_ms=duration_ms,
            observed_effects=[],
        )
