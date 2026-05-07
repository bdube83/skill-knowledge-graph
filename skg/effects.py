"""Effect algebra for the SKG capability kernel.

The kernel understands 12 generic effect classes. Adapters map concrete
product APIs onto these classes. No product-specific code lives here.
"""

from __future__ import annotations

from enum import Enum


class Effect(str, Enum):
    """The 12 generic effect classes understood by the capability kernel."""

    LOCAL_READ       = "local.read"
    LOCAL_WRITE      = "local.write"
    NETWORK_READ     = "network.read"
    NETWORK_WRITE    = "network.write"
    EXTERNAL_DRAFT   = "external.draft"
    EXTERNAL_SEND    = "external.send"
    BROWSER_READ     = "browser.read"
    BROWSER_WRITE    = "browser.write"
    GIT_READ         = "git.read"
    GIT_WRITE        = "git.write"
    SECRET_READ      = "secret.read"
    PRODUCTION_WRITE = "production.write"


# Effects that require human approval before any execution.
APPROVAL_REQUIRED: frozenset[Effect] = frozenset({
    Effect.EXTERNAL_SEND,
    Effect.GIT_WRITE,
    Effect.PRODUCTION_WRITE,
})

# Effects that are never composable in a single leaf node.
INCOMPATIBLE_PAIRS: frozenset[frozenset[Effect]] = frozenset({
    frozenset({Effect.EXTERNAL_DRAFT, Effect.EXTERNAL_SEND}),
    frozenset({Effect.LOCAL_READ,     Effect.PRODUCTION_WRITE}),
})


def effects_are_compatible(effects: list[Effect]) -> bool:
    """Return True if no incompatible pair appears in the given effect list."""
    effect_set = set(effects)
    for pair in INCOMPATIBLE_PAIRS:
        if pair.issubset(effect_set):
            return False
    return True
