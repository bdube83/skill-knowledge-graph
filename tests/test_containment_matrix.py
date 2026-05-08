"""Differential containment matrix: SKG (T) vs Declared-Capability (E).

Phase 4 + Phase 5 integration. For every attack module in the
adversarial corpus, run it under both runtimes and assert the expected
outcome:

  - SKG (T):                   contained (instantiate fails or call denied)
  - Declared-Capability (E):   not contained (module loads and runs)

The differential is what becomes Table 4 in the paper. The "containment
delta" is the count of attacks contained by T minus the count contained
by E, divided by total attacks. The paper claims T contains all attacks
in the corpus that E does not.

Some attack classes are containable by E too (for example, attacks that
import `skg.*` are not containable by E either because E does not wire
SKG host imports). Those rows show parity, not delta. The matrix
records the truth, not a one-sided claim.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from skg.baselines.declared import DeclaredCapabilityRuntime
from skg.wasmtime_launcher import WASI_MODULE, WasmtimeRuntime


@dataclass(frozen=True)
class AttackCase:
    """One adversarial input.

    `attack_class` mirrors the design doc's 7 classes.
    `expected_contained_by_t`/`_e` say what should happen under each
    runtime. Containment means the attack is rejected (success=False).
    """

    name:                     str
    attack_class:             str
    wat:                      str
    granted_effects:          list[str]
    expected_contained_by_t:  bool
    expected_contained_by_e:  bool


def _wat_with_wasi_import(name: str, params: str, results: str) -> str:
    return textwrap.dedent(f'''
        (module
          (import "{WASI_MODULE}" "{name}"
            (func $imp (param {params}) (result {results})))
          (memory (export "memory") 1)
          (func (export "_start"))
        )
    ''').strip()


def _wat_with_skg_import(name: str, params: str, results: str) -> str:
    return textwrap.dedent(f'''
        (module
          (import "skg" "{name}"
            (func $imp (param {params}) (result {results})))
          (memory (export "memory") 1)
          (func (export "_start"))
        )
    ''').strip()


def _wat_no_imports() -> str:
    return textwrap.dedent('''
        (module
          (memory (export "memory") 1)
          (func (export "_start")))
    ''').strip()


CORPUS: list[AttackCase] = [

    # Class 2: Path escape
    AttackCase(
        name="path_escape_no_grant",
        attack_class="path_escape",
        wat=_wat_with_wasi_import(
            "path_open",
            "i32 i32 i32 i32 i32 i64 i64 i32 i32",
            "i32",
        ),
        granted_effects=[],
        expected_contained_by_t=True,
        expected_contained_by_e=False,
    ),
    AttackCase(
        name="path_escape_read_only_for_create_dir",
        attack_class="path_escape",
        wat=_wat_with_wasi_import("path_create_directory", "i32 i32 i32", "i32"),
        granted_effects=["local.read"],
        expected_contained_by_t=True,
        expected_contained_by_e=False,
    ),

    # Class 7: WASI introspection
    AttackCase(
        name="introspect_poll_oneoff",
        attack_class="wasi_introspection",
        wat=_wat_with_wasi_import("poll_oneoff", "i32 i32 i32 i32", "i32"),
        granted_effects=[],
        expected_contained_by_t=True,
        expected_contained_by_e=False,
    ),
    AttackCase(
        name="introspect_sched_yield",
        attack_class="wasi_introspection",
        wat=_wat_with_wasi_import("sched_yield", "", "i32"),
        granted_effects=[],
        expected_contained_by_t=True,
        expected_contained_by_e=False,
    ),
    AttackCase(
        name="introspect_clock_time_get",
        attack_class="wasi_introspection",
        wat=_wat_with_wasi_import("clock_time_get", "i32 i64 i32", "i32"),
        granted_effects=[],
        expected_contained_by_t=True,
        expected_contained_by_e=False,
    ),

    # Class 1: Manifest lies (import a host function the manifest did not declare)
    AttackCase(
        name="manifest_lies_http_get_with_draft_only",
        attack_class="manifest_lies",
        wat=_wat_with_skg_import("http_get", "i32 i32 i32 i32 i32 i32", "i32"),
        granted_effects=["external.draft"],
        expected_contained_by_t=True,
        expected_contained_by_e=True,    # E does not wire skg.* either
    ),
    AttackCase(
        name="manifest_lies_production_write_with_local_read",
        attack_class="manifest_lies",
        wat=_wat_with_skg_import("production_write", "i32 i32 i32 i32 i32 i32 i32", "i32"),
        granted_effects=["local.read"],
        expected_contained_by_t=True,
        expected_contained_by_e=True,
    ),
    AttackCase(
        name="manifest_lies_no_grants_at_all",
        attack_class="manifest_lies",
        wat=_wat_with_skg_import("git_read", "i32 i32 i32 i32 i32 i32", "i32"),
        granted_effects=[],
        expected_contained_by_t=True,
        expected_contained_by_e=True,
    ),
    AttackCase(
        name="manifest_lies_browser_read_with_no_grant",
        attack_class="manifest_lies",
        wat=_wat_with_skg_import("browser_read", "i32 i32 i32 i32 i32 i32", "i32"),
        granted_effects=[],
        expected_contained_by_t=True,
        expected_contained_by_e=True,
    ),
    AttackCase(
        name="manifest_lies_secret_read_with_browser_read",
        attack_class="manifest_lies",
        wat=_wat_with_skg_import("secret_read", "i32 i32 i32 i32 i32 i32", "i32"),
        granted_effects=["browser.read"],
        expected_contained_by_t=True,
        expected_contained_by_e=True,
    ),

    # Class 2: Path escape (additional variants)
    AttackCase(
        name="path_escape_filestat_set_times_with_local_read",
        attack_class="path_escape",
        wat=_wat_with_wasi_import(
            "path_filestat_set_times",
            "i32 i32 i32 i32 i64 i64 i32",
            "i32",
        ),
        granted_effects=["local.read"],
        expected_contained_by_t=True,
        expected_contained_by_e=False,
    ),

    # Class 7: WASI introspection (additional variants)
    AttackCase(
        name="introspect_fd_advise",
        attack_class="wasi_introspection",
        wat=_wat_with_wasi_import("fd_advise", "i32 i64 i64 i32", "i32"),
        granted_effects=[],
        expected_contained_by_t=True,
        expected_contained_by_e=False,
    ),
    AttackCase(
        name="introspect_fd_pread",
        attack_class="wasi_introspection",
        wat=_wat_with_wasi_import("fd_pread", "i32 i32 i32 i64 i32", "i32"),
        granted_effects=[],
        expected_contained_by_t=True,
        expected_contained_by_e=False,
    ),
]


# Sanity: at least one non-malicious case should pass under both runtimes.
SANITY = AttackCase(
    name="benign_no_imports",
    attack_class="sanity",
    wat=_wat_no_imports(),
    granted_effects=[],
    expected_contained_by_t=False,
    expected_contained_by_e=False,
)


def _write(tmp_path: Path, name: str, wat: str) -> Path:
    out = tmp_path / f"{name}.wat"
    out.write_text(wat)
    return out


def _contained(result) -> bool:
    return not result.success


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_t_runtime_matches_expected(case: AttackCase, tmp_path: Path) -> None:
    rt = WasmtimeRuntime()
    wasm = _write(tmp_path, case.name, case.wat)
    result = rt.execute(
        wasm_path=wasm,
        node_id=case.name,
        task="attack",
        context={},
        granted_effects=case.granted_effects,
    )
    assert _contained(result) is case.expected_contained_by_t, (
        f"{case.name}: T expected contained={case.expected_contained_by_t}, "
        f"got success={result.success}, error={result.error[:120]}"
    )


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_e_runtime_matches_expected(case: AttackCase, tmp_path: Path) -> None:
    rt = DeclaredCapabilityRuntime()
    wasm = _write(tmp_path, case.name, case.wat)
    result = rt.execute(
        wasm_path=wasm,
        node_id=case.name,
        task="attack",
        context={},
        granted_effects=case.granted_effects,
    )
    assert _contained(result) is case.expected_contained_by_e, (
        f"{case.name}: E expected contained={case.expected_contained_by_e}, "
        f"got success={result.success}, error={result.error[:120]}"
    )


def test_sanity_benign_module_passes_both_runtimes(tmp_path: Path) -> None:
    wasm = _write(tmp_path, SANITY.name, SANITY.wat)
    rt_t = WasmtimeRuntime()
    rt_e = DeclaredCapabilityRuntime()
    res_t = rt_t.execute(
        wasm_path=wasm,
        node_id=SANITY.name,
        task="benign",
        context={},
        granted_effects=[],
    )
    res_e = rt_e.execute(
        wasm_path=wasm,
        node_id=SANITY.name,
        task="benign",
        context={},
        granted_effects=[],
    )
    assert res_t.success is True, f"T benign failed: {res_t.error}"
    assert res_e.success is True, f"E benign failed: {res_e.error}"


def test_containment_delta_summary(tmp_path: Path) -> None:
    """Aggregate: T contains everything E contains, plus more.

    This is the differential the paper's Table 4 reports. T-containment
    must be a superset of E-containment across the corpus.
    """
    rt_t = WasmtimeRuntime()
    rt_e = DeclaredCapabilityRuntime()

    t_contained = 0
    e_contained = 0

    for case in CORPUS:
        wasm = _write(tmp_path, case.name, case.wat)
        if _contained(rt_t.execute(
            wasm_path=wasm,
            node_id=case.name,
            task="attack",
            context={},
            granted_effects=case.granted_effects,
        )):
            t_contained += 1
        if _contained(rt_e.execute(
            wasm_path=wasm,
            node_id=case.name,
            task="attack",
            context={},
            granted_effects=case.granted_effects,
        )):
            e_contained += 1

    assert t_contained >= e_contained
    assert t_contained == len(CORPUS), (
        f"T contained {t_contained}/{len(CORPUS)} attacks; "
        f"every adversarial case in the corpus must be contained by T."
    )
