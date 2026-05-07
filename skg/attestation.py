"""Attestation records for SKG node runs.

Every run writes an attestation to an append-only JSONL log. Attestations
are the audit trail for routing decisions, dry-run results, verifier
outcomes, and promotion approvals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Attestation:
    """A single attestation record for one node run."""

    attestation_id:     str
    node_id:            str
    run_type:           str           # "dry_run" | "full_run" | "promotion"
    manifest_sha256:    str           = ""
    source_sha256:      str           = ""
    requested_caps:     list[str]     = field(default_factory=list)
    granted_caps:       list[str]     = field(default_factory=list)
    observed_effects:   list[str]     = field(default_factory=list)
    verifier_results:   list[dict]    = field(default_factory=list)
    run_duration_ms:    float         = 0.0
    ungranted_attempts: int           = 0
    promotion_eligible: bool          = False
    error:              str           = ""
    timestamp:          str           = ""
    reviewer_id:        str           = ""

    @property
    def verifiers_passed(self) -> bool:
        return all(v.get("passed", False) for v in self.verifier_results)

    def to_dict(self) -> dict:
        return {
            "attestation_id":     self.attestation_id,
            "node_id":            self.node_id,
            "run_type":           self.run_type,
            "manifest_sha256":    self.manifest_sha256,
            "source_sha256":      self.source_sha256,
            "requested_caps":     self.requested_caps,
            "granted_caps":       self.granted_caps,
            "observed_effects":   self.observed_effects,
            "verifier_results":   self.verifier_results,
            "run_duration_ms":    self.run_duration_ms,
            "ungranted_attempts": self.ungranted_attempts,
            "promotion_eligible": self.promotion_eligible,
            "error":              self.error,
            "timestamp":          self.timestamp,
            "reviewer_id":        self.reviewer_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Attestation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AttestationStore:
    """Append-only JSONL store for attestation records.

    Each record is one JSON line. The log is never rewritten; only appended.
    """

    def __init__(self, log_path: Path | str) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, attestation: Attestation) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(attestation.to_dict(), ensure_ascii=False) + "\n")

    def get_all(self, node_id: str) -> list[Attestation]:
        if not self._path.exists():
            return []
        results = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                if d.get("node_id") == node_id:
                    results.append(Attestation.from_dict(d))
            except (json.JSONDecodeError, TypeError):
                continue
        return results

    def get_latest(self, node_id: str, run_type: str | None = None) -> Attestation | None:
        records = self.get_all(node_id)
        if run_type:
            records = [r for r in records if r.run_type == run_type]
        return records[-1] if records else None


def make_routing_attestation(
    *,
    node_id: str,
    node,
    grant,
    route_stage: str,
    verifier_results: list[dict] | None = None,
) -> Attestation:
    """Build an Attestation for a routing decision (kernel + router slices).

    Execution attestations are produced by the Wasmtime runtime slice and
    written via AttestationStore.write() directly. This helper covers the
    routing path only: what was requested, what policy granted, which stage
    matched.
    """
    import uuid
    import datetime
    import json as _json

    manifest = node.manifest
    manifest_sha256 = hashlib.sha256(
        _json.dumps(manifest.to_dict(), sort_keys=True).encode()
    ).hexdigest()

    requested = [c.effect for c in manifest.requested_capabilities]
    granted   = [c.effect for c in (grant.granted if grant else [])]

    return Attestation(
        attestation_id=str(uuid.uuid4()),
        node_id=node_id,
        run_type=f"route:{route_stage}",
        manifest_sha256=manifest_sha256,
        source_sha256=node.source_sha256,
        requested_caps=requested,
        granted_caps=granted,
        observed_effects=[],          # populated by Wasmtime runtime slice
        verifier_results=verifier_results or [],
        run_duration_ms=0.0,          # populated by Wasmtime runtime slice
        ungranted_attempts=0,
        promotion_eligible=False,     # promotion gates are in slice 4
        error="",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
