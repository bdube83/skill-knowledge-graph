"""Baseline runtimes for the SKG paper's evaluation.

Each baseline implements the same `execute(...)` interface as
`skg.wasmtime_launcher.WasmtimeRuntime`. Baselines exist so the paper
can report differential containment results across enforcement
strategies on the same adversarial corpus.

Exports:
  FlowRegistryRuntime       Baseline B (existing flow registry)
  SemanticCacheRuntime      Baseline C (GPTCache-style semantic cache)
  FlatToolLibraryRuntime    Baseline D (flat tool library, full WASI)
  DeclaredCapabilityRuntime Baseline E (declared-capability, full WASI)
"""

from .declared      import DeclaredCapabilityRuntime
from .flat_library  import FlatToolLibraryRuntime
from .flow_registry import FlowRegistryRuntime
from .semantic_cache import SemanticCacheRuntime


__all__ = [
    "DeclaredCapabilityRuntime",
    "FlatToolLibraryRuntime",
    "FlowRegistryRuntime",
    "SemanticCacheRuntime",
]
