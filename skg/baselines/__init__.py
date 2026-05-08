"""Baseline runtimes for the SKG paper's evaluation.

Each baseline implements the same `execute(...)` interface as
`skg.wasmtime_launcher.WasmtimeRuntime`. Baselines exist so the paper
can report differential containment results across enforcement
strategies on the same adversarial corpus.
"""
