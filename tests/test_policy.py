"""Tests for skg.policy — PolicyEngine grant and can_grant."""

import pytest

from skg.node import CapabilityRequest, Manifest
from skg.policy import PolicyEngine, default_allow_policy


def make_manifest(effects: list[str], forbidden: list[str] | None = None) -> Manifest:
    return Manifest(
        task_type="test_task",
        header="Test node.",
        tags=[],
        requested_capabilities=[
            CapabilityRequest(effect=e, adapter="local") for e in effects
        ],
        forbidden_capabilities=forbidden or [],
        preconditions=[],
        verifiers=[],
    )


class TestDefaultAllowPolicy:
    def test_allows_text_generate(self):
        policy = default_allow_policy()
        m = make_manifest(["text.generate"])
        assert policy.can_grant(m) is True

    def test_denies_external_send(self):
        policy = default_allow_policy()
        m = make_manifest(["external.send"])
        assert policy.can_grant(m) is False

    def test_denies_production_write(self):
        policy = default_allow_policy()
        m = make_manifest(["production.write"])
        assert policy.can_grant(m) is False

    def test_denies_secret_read(self):
        policy = default_allow_policy()
        m = make_manifest(["secret.read"])
        assert policy.can_grant(m) is False

    def test_allows_git_read(self):
        policy = default_allow_policy()
        m = make_manifest(["git.read"])
        assert policy.can_grant(m) is True


class TestPolicyEngineGrant:
    def test_grant_populates_granted_list(self):
        policy = default_allow_policy()
        m = make_manifest(["text.generate", "git.read"])
        grant = policy.grant("node-001", m)
        granted_effects = [c.effect for c in grant.granted]
        assert "text.generate" in granted_effects
        assert "git.read" in granted_effects

    def test_grant_denied_goes_to_denied_list(self):
        policy = default_allow_policy()
        m = make_manifest(["external.send"])
        grant = policy.grant("node-001", m)
        assert "external.send" in grant.denied
        assert grant.fully_granted is False

    def test_forbidden_in_manifest_is_denied(self):
        policy = default_allow_policy()
        # text.generate is in requested AND forbidden — forbidden wins.
        m = make_manifest(["text.generate"], forbidden=["text.generate"])
        grant = policy.grant("node-001", m)
        assert "text.generate" in grant.denied

    def test_grant_has_node_id(self):
        policy = default_allow_policy()
        m = make_manifest(["text.generate"])
        grant = policy.grant("my-node", m)
        assert grant.node_id == "my-node"

    def test_can_execute_dry_run_when_no_denied(self):
        policy = default_allow_policy()
        m = make_manifest(["text.generate"])
        grant = policy.grant("node-001", m)
        assert grant.can_execute_dry_run is True

    def test_cannot_execute_dry_run_when_denied(self):
        policy = default_allow_policy()
        m = make_manifest(["external.send"])
        grant = policy.grant("node-001", m)
        assert grant.can_execute_dry_run is False
