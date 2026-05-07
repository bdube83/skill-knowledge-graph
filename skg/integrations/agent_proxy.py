"""agent-proxy-kit integration for SKG.

Import-safe: if skg is not installed, importing this module raises ImportError
and the caller falls back to LLM generation. No side effects on import.

Usage in work_brief_writer.py:

    try:
        from skg.integrations.agent_proxy import route_proposal, build_prompt_prefix
        _SKG_AVAILABLE = True
    except ImportError:
        _SKG_AVAILABLE = False

    # Inside _compose_proposal_session_prompt():
    if _SKG_AVAILABLE:
        result = route_proposal(proposal_text, policy_config=config)
        if result.hit:
            prefix = build_prompt_prefix(result)
"""

from __future__ import annotations

from typing import Any

from skg.graph import SKG
from skg.router import RouteResult
from skg.synthesis import Synthesizer

# Module-level singletons -- initialised once on first use.
_skg: SKG | None = None
_synthesizer: Synthesizer | None = None


def _get_skg() -> SKG:
    global _skg
    if _skg is None:
        _skg = SKG()
    return _skg


def _make_llm_fn():
    """Return a callable that wraps agent_router.invoke() for synthesis prompts.

    Constructed lazily so agent_router is not imported at module load time;
    SKG stays usable outside agent-proxy-kit.
    """
    try:
        import agent_router  # type: ignore[import]

        def _llm(prompt: str) -> str:
            result = agent_router.invoke("synthesis", prompt)
            return result.stdout or ""

        return _llm
    except ImportError:
        return None


def _get_synthesizer() -> Synthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = Synthesizer(_get_skg(), llm_fn=_make_llm_fn())
    return _synthesizer


def route_proposal(
    proposal_text: str,
    policy_config: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    synthesize_on_miss: bool = False,
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
    synthesize_on_miss:
        When True and the router returns a miss, attempt LLM synthesis of a new
        CANDIDATE node. The miss RouteResult is still returned (the synthesised
        node is CANDIDATE, not yet ACTIVE); the caller is not blocked.

    Returns
    -------
    RouteResult
        result.hit == True  -- a capable node was found; result.node and
                               result.grant are populated.
        result.hit == False -- SKG has no node for this task; caller should
                               fall back to LLM generation.
    """
    ctx = dict(context or {})
    if policy_config:
        ctx["_policy_config"] = policy_config

    skg = _get_skg()
    result = skg.route(proposal_text, ctx)

    if result.miss and synthesize_on_miss:
        try:
            _get_synthesizer().synthesize(proposal_text, context=ctx)
        except Exception:
            pass  # Synthesis failure is never fatal.

    return result


def build_prompt_prefix(result: RouteResult) -> str:
    """Format a short SKG routing summary to prepend to a proposal chat prompt.

    Returns an empty string if the result is a miss.
    """
    if not result.hit or not result.node:
        return ""

    node = result.node
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
