"""agent-proxy-kit integration for SKG.

Import-safe: if skg is not installed, importing this module raises ImportError
and the caller falls back to LLM generation. No side effects on import.

Usage in work_brief_writer.py:

    try:
        from skg.integrations.agent_proxy import route_proposal, RouteResult
        _SKG_AVAILABLE = True
    except ImportError:
        _SKG_AVAILABLE = False

    # Inside _compose_proposal_session_prompt():
    if _SKG_AVAILABLE:
        result = route_proposal(proposal_text, policy_config=config)
        if result.hit:
            return _build_prompt_from_skg_result(result)
    # Fall through to LLM generation.
"""

from __future__ import annotations

from typing import Any

from skg.graph import SKG
from skg.router import RouteResult

# Module-level SKG instance — initialised once on first use.
_skg: SKG | None = None


def _get_skg() -> SKG:
    global _skg
    if _skg is None:
        _skg = SKG()
    return _skg


def route_proposal(
    proposal_text: str,
    policy_config: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> RouteResult:
    """Route a work-brief proposal through SKG.

    Parameters
    ----------
    proposal_text:
        The proposal or task description from the work-brief writer.
    policy_config:
        Optional dict with keys accepted by SKG policy (adapter scopes, etc.).
    context:
        Optional execution context dict forwarded to the router.

    Returns
    -------
    RouteResult
        result.hit == True  → a capable node was found; result.node and
                              result.grant are populated.
        result.hit == False → SKG has no node for this task; caller should
                              fall back to LLM generation.
    """
    ctx = dict(context or {})
    if policy_config:
        ctx["_policy_config"] = policy_config

    skg = _get_skg()
    return skg.route(proposal_text, ctx)


def build_prompt_prefix(result: RouteResult) -> str:
    """Format a short SKG routing summary to prepend to a proposal chat prompt.

    Returns an empty string if the result is a miss.
    """
    if not result.hit or not result.node:
        return ""

    node  = result.node
    grant = result.grant
    lines = [
        f"[SKG] Matched node: {node.id} (stage: {result.stage})",
        f"      Header: {node.manifest.header[:100]}",
    ]
    if grant:
        granted = [c.effect for c in grant.granted]
        if granted:
            lines.append(f"      Granted capabilities: {', '.join(granted)}")
        if grant.requires_approval:
            lines.append(f"      Requires approval: {', '.join(grant.requires_approval)}")
    return "\n".join(lines) + "\n\n"
