"""Verifier contracts and promotion gates for SKG (slice 4).

Promotion is the act of moving a node from CANDIDATE to ACTIVE status. A
promoted node is eligible for routing. Six gates must all pass before
promotion is allowed. Any gate failure leaves the node as CANDIDATE and
records the failure in the attestation log.

The six promotion gates (from the design doc):
  1. Manifest validity     — schema complete, no forbidden effects self-granted.
  2. Source present        — source file is non-empty, sha256 matches manifest.
  3. Dry-run success       — Wasmtime dry-run completes without error.
  4. No ungranted attempts — dry-run did not attempt effects outside the grant.
  5. Verifier contracts    — all declared verifiers pass against dry-run output.
  6. Reviewer sign-off     — a human reviewer ID is recorded (or auto-approved
                             for nodes with no external effects).

Gate 6 is the only gate with a human in the loop. For nodes whose manifest
requests only local effects (text.generate, git.read, file.read), gate 6 is
auto-approved. For any node touching external.send, network.*, or secret.*
a reviewer_id must be supplied explicitly.

Reference:
  designs/proposed/skill-graph-codex-v10/design/skill-knowledge-graph.md
  "Verifier and promotion slice"
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skg.attestation import Attestation, AttestationStore
from skg.effects import APPROVAL_REQUIRED
from skg.node import Manifest, Node, NodeStatus
from skg.policy import Grant
from skg.store import NodeStore


# Effects that always require a human reviewer_id for promotion.
_REVIEWER_REQUIRED_PREFIXES = ("external.", "network.", "secret.", "production.")


@dataclass
class GateResult:
    gate:    int
    name:    str
    passed:  bool
    reason:  str = ""


@dataclass
class PromotionResult:
    node_id:     str
    promoted:    bool
    gates:       list[GateResult]    = field(default_factory=list)
    reviewer_id: str                 = ""
    timestamp:   str                 = ""

    @property
    def failed_gates(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed]

    def to_dict(self) -> dict:
        return {
            "node_id":     self.node_id,
            "promoted":    self.promoted,
            "gates":       [{"gate": g.gate, "name": g.name, "passed": g.passed, "reason": g.reason} for g in self.gates],
            "reviewer_id": self.reviewer_id,
            "timestamp":   self.timestamp,
        }


class PromotionEngine:
    """Runs all six promotion gates and promotes the node if all pass.

    Parameters
    ----------
    store:
        The node store. Used to write the final ACTIVE status.
    attestation_store:
        Where gate results are recorded.
    wasm_root:
        Root directory for WASI node artifacts. Used for dry-run gate.
        Defaults to ~/.skg/nodes/.
    """

    def __init__(
        self,
        store: NodeStore,
        attestation_store: AttestationStore,
        wasm_root: Path | None = None,
    ) -> None:
        self._store       = store
        self._attestation = attestation_store
        self._wasm_root   = wasm_root or (Path.home() / ".skg" / "nodes")

    def promote(
        self,
        node_id: str,
        reviewer_id: str = "",
        dry_run_context: dict[str, Any] | None = None,
    ) -> PromotionResult:
        """Run all six gates and promote if all pass.

        Parameters
        ----------
        node_id:
            The node to promote.
        reviewer_id:
            Human reviewer sign-off. Required for nodes with external effects.
        dry_run_context:
            Context dict forwarded to the WASI dry-run. Should include
            representative inputs so the dry-run exercises real paths.
        """
        node = self._store.get(node_id)
        if not node:
            return PromotionResult(
                node_id=node_id,
                promoted=False,
                gates=[GateResult(0, "node_exists", False, f"Node '{node_id}' not found in store.")],
                timestamp=_now(),
            )

        gates = [
            self._gate1_manifest(node),
            self._gate2_source(node),
            self._gate3_dry_run(node, dry_run_context or {}),
            self._gate4_no_ungranted(node, dry_run_context or {}),
            self._gate5_verifiers(node, dry_run_context or {}),
            self._gate6_reviewer(node, reviewer_id),
        ]

        promoted = all(g.passed for g in gates)
        if promoted:
            self._store.promote(node_id)

        result = PromotionResult(
            node_id=node_id,
            promoted=promoted,
            gates=gates,
            reviewer_id=reviewer_id,
            timestamp=_now(),
        )

        # Record gate results as a promotion attestation.
        self._write_attestation(node, result)
        return result

    # ---- Gates -----------------------------------------------------------

    def _gate1_manifest(self, node: Node) -> GateResult:
        m = node.manifest
        if not m.task_type:
            return GateResult(1, "manifest_valid", False, "task_type is empty.")
        if not m.header:
            return GateResult(1, "manifest_valid", False, "header is empty.")
        # A node cannot grant itself a forbidden capability.
        for cap in m.requested_capabilities:
            if cap.effect in m.forbidden_capabilities:
                return GateResult(
                    1, "manifest_valid", False,
                    f"Effect '{cap.effect}' is both requested and forbidden."
                )
        return GateResult(1, "manifest_valid", True)

    def _gate2_source(self, node: Node) -> GateResult:
        import hashlib
        if not node.source:
            return GateResult(2, "source_present", False, "source is empty.")
        computed = hashlib.sha256(node.source.encode()).hexdigest()
        if node.source_sha256 and computed != node.source_sha256:
            return GateResult(
                2, "source_present", False,
                f"source_sha256 mismatch. stored={node.source_sha256[:12]} computed={computed[:12]}"
            )
        return GateResult(2, "source_present", True)

    def _gate3_dry_run(self, node: Node, context: dict) -> GateResult:
        wasm = self._wasm_root / node.id / "artifact" / "node.wasm"
        if not wasm.exists():
            return GateResult(
                3, "dry_run_success", False,
                f"WASM artifact not found at {wasm}. Compile with: cargo build --release --target wasm32-wasip1"
            )
        try:
            from skg.wasmtime_launcher import WasmtimeRuntime
            rt = WasmtimeRuntime(timeout_ms=5000)
            granted = [c.effect for c in node.manifest.requested_capabilities]
            # Run with real grants, not dry_run mode (which would blank grants).
            # dry_run here means: verify execution succeeds, not that side-effects are blocked.
            result = rt.execute(
                wasm_path=wasm,
                node_id=node.id,
                task=node.manifest.task_type,
                context=context,
                granted_effects=granted,
                dry_run=False,
            )
            if not result.success:
                return GateResult(3, "dry_run_success", False, f"Dry-run failed: {result.error}")
            return GateResult(3, "dry_run_success", True)
        except Exception as e:
            return GateResult(3, "dry_run_success", False, f"Dry-run raised: {e}")

    def _gate4_no_ungranted(self, node: Node, context: dict) -> GateResult:
        wasm = self._wasm_root / node.id / "artifact" / "node.wasm"
        if not wasm.exists():
            return GateResult(4, "no_ungranted_attempts", False, "WASM artifact missing.")
        try:
            from skg.wasmtime_launcher import WasmtimeRuntime
            rt = WasmtimeRuntime(timeout_ms=5000)
            # Run with empty grant set — node should report error, not succeed.
            result = rt.execute(
                wasm_path=wasm,
                node_id=node.id,
                task=node.manifest.task_type,
                context=context,
                granted_effects=[],
                dry_run=True,
            )
            # If it succeeds with zero grants and the manifest requests capabilities,
            # the node is not checking its grants — that is a gate failure.
            requested = node.manifest.requested_capabilities
            if result.success and requested:
                return GateResult(
                    4, "no_ungranted_attempts", False,
                    "Node succeeded with no granted capabilities. It is not validating grants."
                )
            return GateResult(4, "no_ungranted_attempts", True)
        except Exception as e:
            return GateResult(4, "no_ungranted_attempts", False, str(e))

    def _gate5_verifiers(self, node: Node, context: dict) -> GateResult:
        verifiers = node.manifest.verifiers
        if not verifiers:
            return GateResult(5, "verifiers_pass", True, "No verifiers declared.")
        wasm = self._wasm_root / node.id / "artifact" / "node.wasm"
        if not wasm.exists():
            return GateResult(5, "verifiers_pass", False, "WASM artifact missing.")
        try:
            from skg.wasmtime_launcher import WasmtimeRuntime
            rt = WasmtimeRuntime(timeout_ms=5000)
            granted = [c.effect for c in node.manifest.requested_capabilities]
            run = rt.execute(
                wasm_path=wasm,
                node_id=node.id,
                task=node.manifest.task_type,
                context=context,
                granted_effects=granted,
            )
            if not run.success:
                return GateResult(5, "verifiers_pass", False, f"Node run failed before verifiers: {run.error}")

            for v in verifiers:
                check = v.get("check", "")
                if not check:
                    continue
                try:
                    _safe_builtins = {"bool": bool, "isinstance": isinstance, "str": str,
                                      "int": int, "list": list, "dict": dict, "len": len}
                    passed = eval(check, {"__builtins__": _safe_builtins}, {"output": run.output})  # noqa: S307
                    if not passed:
                        return GateResult(5, "verifiers_pass", False, f"Verifier failed: {v.get('name', check)}")
                except Exception as e:
                    return GateResult(5, "verifiers_pass", False, f"Verifier error: {e}")
            return GateResult(5, "verifiers_pass", True)
        except Exception as e:
            return GateResult(5, "verifiers_pass", False, str(e))

    def _gate6_reviewer(self, node: Node, reviewer_id: str) -> GateResult:
        needs_reviewer = any(
            cap.effect.startswith(prefix)
            for cap in node.manifest.requested_capabilities
            for prefix in _REVIEWER_REQUIRED_PREFIXES
        )
        if needs_reviewer and not reviewer_id:
            effects = [c.effect for c in node.manifest.requested_capabilities
                       if any(c.effect.startswith(p) for p in _REVIEWER_REQUIRED_PREFIXES)]
            return GateResult(
                6, "reviewer_signoff", False,
                f"Node requests external effects {effects}. A reviewer_id is required for promotion."
            )
        return GateResult(6, "reviewer_signoff", True, reviewer_id or "auto-approved (local effects only)")

    # ---- Attestation ----------------------------------------------------

    def _write_attestation(self, node: Node, result: PromotionResult) -> None:
        import uuid, json, hashlib
        manifest_sha = hashlib.sha256(
            json.dumps(node.manifest.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        a = Attestation(
            attestation_id=str(uuid.uuid4()),
            node_id=node.id,
            run_type="promotion",
            manifest_sha256=manifest_sha,
            source_sha256=node.source_sha256,
            requested_caps=[c.effect for c in node.manifest.requested_capabilities],
            granted_caps=[],
            observed_effects=[],
            verifier_results=[{"gate": g.gate, "name": g.name, "passed": g.passed, "reason": g.reason} for g in result.gates],
            promotion_eligible=result.promoted,
            reviewer_id=result.reviewer_id,
            timestamp=result.timestamp,
        )
        self._attestation.write(a)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
