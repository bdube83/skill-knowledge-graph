"""Tests for skg.verifier — six promotion gates."""

from pathlib import Path
import pytest

from skg.node import CapabilityRequest, Manifest, Node, NodeStatus
from skg.store import NodeStore
from skg.attestation import AttestationStore
from skg.verifier import PromotionEngine

PROJECT_ROOT = Path(__file__).parent.parent
WASM_ARTIFACT = (
    PROJECT_ROOT
    / "nodes" / "reviewer-ping-draft"
    / "target" / "wasm32-wasip1" / "release"
    / "reviewer_ping_draft.wasm"
)

requires_wasm = pytest.mark.skipif(
    not WASM_ARTIFACT.exists(),
    reason="WASM artifact not built. Run: cd nodes/reviewer-ping-draft && cargo build --release --target wasm32-wasip1",
)


def make_ping_node(node_id: str = "reviewer-ping-draft") -> Node:
    m = Manifest(
        task_type="reviewer_ping_draft",
        header="Draft a reviewer-ping message for a pull request.",
        tags=["git", "review"],
        requested_capabilities=[CapabilityRequest(effect="text.generate", adapter="local")],
        forbidden_capabilities=[],
        verifiers=[
            {"name": "message_present", "check": "bool(output.get('message', ''))"},
        ],
        preconditions=[],
    )
    n = Node.new(id=node_id, manifest=m, source="fn main() { /* reviewer-ping-draft */ }")
    return n


@pytest.fixture
def engine(tmp_path):
    store = NodeStore(tmp_path / "skg.db")
    store.connect()
    attest = AttestationStore(tmp_path / "attestations.jsonl")
    # Artifacts are at ~/.skg/nodes/<node_id>/artifact/node.wasm
    wasm_root = Path.home() / ".skg" / "nodes"
    eng = PromotionEngine(store=store, attestation_store=attest, wasm_root=wasm_root)
    yield eng, store
    store.close()


class TestGate1ManifestValid:
    def test_passes_valid_manifest(self, engine):
        eng, store = engine
        node = make_ping_node()
        store.put(node)
        r = eng._gate1_manifest(node)
        assert r.passed is True

    def test_fails_empty_task_type(self, engine):
        eng, store = engine
        node = make_ping_node()
        node.manifest.task_type = ""
        r = eng._gate1_manifest(node)
        assert r.passed is False
        assert "task_type" in r.reason

    def test_fails_effect_in_both_requested_and_forbidden(self, engine):
        eng, store = engine
        node = make_ping_node()
        node.manifest.forbidden_capabilities = ["text.generate"]
        r = eng._gate1_manifest(node)
        assert r.passed is False
        assert "forbidden" in r.reason


class TestGate2SourcePresent:
    def test_passes_with_source(self, engine):
        eng, store = engine
        node = make_ping_node()
        r = eng._gate2_source(node)
        assert r.passed is True

    def test_fails_empty_source(self, engine):
        eng, store = engine
        node = make_ping_node()
        node.source = ""
        r = eng._gate2_source(node)
        assert r.passed is False

    def test_fails_sha256_mismatch(self, engine):
        eng, store = engine
        node = make_ping_node()
        node.source_sha256 = "0" * 64  # wrong hash
        r = eng._gate2_source(node)
        assert r.passed is False
        assert "mismatch" in r.reason


class TestGate6Reviewer:
    def test_auto_approved_local_effects(self, engine):
        eng, store = engine
        node = make_ping_node()
        r = eng._gate6_reviewer(node, reviewer_id="")
        assert r.passed is True
        assert "auto-approved" in r.reason

    def test_requires_reviewer_for_external_effects(self, engine):
        eng, store = engine
        node = make_ping_node()
        node.manifest.requested_capabilities.append(
            CapabilityRequest(effect="external.send", adapter="email")
        )
        r = eng._gate6_reviewer(node, reviewer_id="")
        assert r.passed is False

    def test_passes_with_reviewer_id(self, engine):
        eng, store = engine
        node = make_ping_node()
        node.manifest.requested_capabilities.append(
            CapabilityRequest(effect="external.send", adapter="email")
        )
        r = eng._gate6_reviewer(node, reviewer_id="alice@example.com")
        assert r.passed is True


@requires_wasm
class TestFullPromotion:
    def test_promote_passes_all_gates(self, engine):
        eng, store = engine
        node = make_ping_node()
        store.put(node)
        ctx = {"pr_number": 1, "repo": "x/y", "author": "z", "reviewers": ["bob"]}
        result = eng.promote(node.id, dry_run_context=ctx)
        assert result.promoted is True
        assert len(result.failed_gates) == 0
        fetched = store.get(node.id)
        assert fetched.status == NodeStatus.ACTIVE

    def test_promote_fails_invalid_manifest(self, engine):
        eng, store = engine
        node = make_ping_node()
        node.manifest.task_type = ""
        store.put(node)
        result = eng.promote(node.id)
        assert result.promoted is False
        assert any(not g.passed and g.gate == 1 for g in result.gates)

    def test_attestation_written_on_promote(self, engine, tmp_path):
        eng, store = engine
        node = make_ping_node()
        store.put(node)
        attest_store = AttestationStore(tmp_path / "attestations.jsonl")
        eng._attestation = attest_store
        ctx = {"pr_number": 1, "repo": "x/y", "author": "z", "reviewers": []}
        eng.promote(node.id, dry_run_context=ctx)
        records = attest_store.get_all(node.id)
        assert len(records) >= 1
        assert records[-1].run_type == "promotion"
