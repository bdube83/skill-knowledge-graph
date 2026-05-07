"""Policy engine for SKG capability grants.

The policy engine takes a node manifest and a run context and returns
a grant (allowed capabilities) or a denial. It reads a YAML policy table
from disk and never consults the node store or any retrieval index.

Policy table format (~/.skg/policy.yaml):
    rules:
      - effect: network.read
        adapter: github
        scope_pattern: "*"      # fnmatch pattern matched against scope keys
        allow: true
      - effect: external.send
        adapter: "*"
        allow: false            # deny all external.send by default
    default: deny               # deny anything not matched by a rule
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from skg.effects import Effect, APPROVAL_REQUIRED
from skg.node import CapabilityRequest, Manifest


@dataclass
class Grant:
    """A policy-issued capability grant for a single node run."""

    grant_id:   str
    node_id:    str
    granted:    list[CapabilityRequest]   = field(default_factory=list)
    denied:     list[str]                 = field(default_factory=list)
    requires_approval: list[str]          = field(default_factory=list)
    issued_at:  str                       = ""
    expires_at: str                       = ""

    @property
    def fully_granted(self) -> bool:
        return len(self.denied) == 0 and len(self.requires_approval) == 0

    @property
    def can_execute_dry_run(self) -> bool:
        """Dry-run is allowed as long as no forbidden effects are present."""
        return len(self.denied) == 0

    def to_dict(self) -> dict:
        return {
            "grant_id":          self.grant_id,
            "node_id":           self.node_id,
            "granted":           [c.to_dict() for c in self.granted],
            "denied":            self.denied,
            "requires_approval": self.requires_approval,
            "issued_at":         self.issued_at,
            "expires_at":        self.expires_at,
        }


class PolicyEngine:
    """Evaluates capability requests against a policy table.

    The policy table is loaded once at construction time. Call reload() to
    pick up changes from disk without constructing a new engine.
    """

    def __init__(self, policy_path: Path | str | None = None) -> None:
        self._path = Path(policy_path) if policy_path else None
        self._rules: list[dict] = []
        self._default: str = "deny"
        if self._path and self._path.exists():
            self._load()

    def reload(self) -> None:
        if self._path and self._path.exists():
            self._load()

    def can_grant(self, manifest: Manifest, context: dict[str, Any] | None = None) -> bool:
        """Return True if policy can grant ALL requested capabilities.

        Does not issue a Grant object; use grant() for that.
        """
        for req in manifest.requested_capabilities:
            decision = self._evaluate(req, context or {})
            if decision == "deny":
                return False
            if req.effect in manifest.forbidden_capabilities:
                return False
        return True

    def grant(
        self,
        node_id: str,
        manifest: Manifest,
        context: dict[str, Any] | None = None,
    ) -> Grant:
        """Issue a Grant for the given manifest and run context."""
        import uuid
        import datetime

        ctx = context or {}
        granted: list[CapabilityRequest] = []
        denied: list[str] = []
        requires_approval: list[str] = []

        for req in manifest.requested_capabilities:
            # Forbidden capabilities can never be granted.
            if req.effect in manifest.forbidden_capabilities:
                denied.append(req.effect)
                continue

            decision = self._evaluate(req, ctx)
            if decision == "deny":
                denied.append(req.effect)
            elif decision == "require_approval":
                requires_approval.append(req.effect)
            else:
                granted.append(req)

        # Effects in APPROVAL_REQUIRED always move to requires_approval.
        for cap in granted[:]:
            if cap.effect in {e.value for e in APPROVAL_REQUIRED}:
                granted.remove(cap)
                requires_approval.append(cap.effect)

        now = datetime.datetime.now(datetime.timezone.utc)
        return Grant(
            grant_id=str(uuid.uuid4()),
            node_id=node_id,
            granted=granted,
            denied=denied,
            requires_approval=requires_approval,
            issued_at=now.isoformat(),
            expires_at=(now + datetime.timedelta(hours=8)).isoformat(),
        )

    # ---- Internal --------------------------------------------------------

    def _load(self) -> None:
        if not _YAML_AVAILABLE:
            return
        assert self._path
        try:
            data = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
            self._rules = data.get("rules", [])
            self._default = data.get("default", "deny")
        except Exception:
            self._rules = []
            self._default = "deny"

    def _evaluate(self, req: CapabilityRequest, context: dict) -> str:
        """Return 'allow', 'deny', or 'require_approval' for one capability."""
        for rule in self._rules:
            if not _matches_rule(req, rule):
                continue
            if rule.get("require_approval"):
                return "require_approval"
            return "allow" if rule.get("allow") else "deny"
        return self._default


def _matches_rule(req: CapabilityRequest, rule: dict) -> bool:
    """Return True if a policy rule applies to a capability request."""
    effect_pattern  = rule.get("effect", "*")
    adapter_pattern = rule.get("adapter", "*")
    if not fnmatch.fnmatch(req.effect, effect_pattern):
        return False
    if not fnmatch.fnmatch(req.adapter, adapter_pattern):
        return False
    return True


def default_allow_policy() -> PolicyEngine:
    """Return a permissive policy engine that allows everything except send/production."""
    engine = PolicyEngine()
    engine._rules = [
        {"effect": "external.send",    "adapter": "*", "allow": False},
        {"effect": "production.write", "adapter": "*", "allow": False},
        {"effect": "secret.read",      "adapter": "*", "allow": False},
        {"effect": "*",                "adapter": "*", "allow": True},
    ]
    engine._default = "deny"
    return engine
