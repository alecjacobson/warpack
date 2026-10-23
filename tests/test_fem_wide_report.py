"""Wide-FEM timing claims require complete spectra and both native stopping gates."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from benchmarks.anvil_inputs import array_identity, matrix_identity
from benchmarks.applications.fem_wide_report import FIELDS, compare, summarize
from benchmarks.block_inputs import identities, initial_block, reference_identity
from benchmarks.source_provenance import source_manifest


def fixture():
    q = dict.fromkeys(FIELDS, 1e-13) | dict(
        accepted=True, recovered_nullity=0, status=0, converged=3
    )
    shared = dict(
        quality=q,
        accepted=True,
        physical_and_outer_accepted=True,
        native_stopping_accepted=True,
        seconds=0.1,
    )
    warp = shared | dict(method="Warp BlockLOBPCG")
    cupy = shared | dict(
        method="CuPy LOBPCG",
        returned_absolute_residuals=[1e-13] * 4,
        full_block_absolute_tolerance_met=True,
        returned_k_absolute_tolerance_met=True,
    )
    raw = dict(
        inputs=identities(
            sp.diags(np.arange(1, 41, dtype=float)),
            None,
            sp.diags(np.arange(1, 41, dtype=float) + 0.01),
            reference_identity(
                SimpleNamespace(
                    expected=np.arange(1, 4, dtype=float),
                    requested_basis=np.eye(40)[:, :3],
                    null_basis=np.empty((40, 0)),
                    nullity=0,
                    k=3,
                    require_positive_guard=False,
                )
            ),
            initial_block(40, 4),
            cupy_width=4,
        ),
        accepted=True,
        timing_eligible=True,
        n=40,
        k=3,
        args=dict(public_api=True, cupy=True, repeats=3, modes=3, width=4, tol=1e-11, out="a"),
        fixture=dict(
            boundary_cluster_complete=True,
            actual_nontrivial_modes=3,
            requested_nontrivial_modes=3,
            retained_rigid_modes=0,
        ),
        graph=dict(node_types_including_conditional_body={"KERNEL": 2}, memcpy_kinds={}),
        cupy_adapter=dict(
            width=4,
            absolute_tolerance=1e-10,
            reference_setup_seconds=0.3,
            reference_setup_scope="excludes supplied factors",
            stopping="different criteria",
        ),
        rows=[copy.deepcopy(r) for r in [warp] * 3 + [cupy] * 3],
        total_setup_seconds=0.2,
        capture_seconds=0.4,
        warmup_seconds=0.1,
        setup_scope="setup only",
        dependency=dict(
            warp_version="test", files={"linear": {"sha256": "same"}}, manifest_status="absent"
        ),
        versions={"warp": "test"},
        source_sha256={"a": "b"},
    )
    return raw, dict(timing_eligible=True, exit_code=0)


def test_scope_native_contracts_and_regression_gate():
    raw, occ = fixture()
    result = summarize(raw, occ) | dict(environment={"gpu": "same"})
    assert result["setup_stages"]["total_setup_seconds"] == 0.2
    assert result["cupy_setup_scope"] == "excludes supplied factors"
    assert result["native_stopping"] == "different criteria"
    assert compare(result, result) == {"warp": 1, "cupy": 1}
    slower = copy.deepcopy(result)
    slower["methods"]["warp"]["median_seconds"] *= 1.3
    with pytest.raises(ValueError, match="regression"):
        compare(slower, result)


@pytest.mark.parametrize(
    "failure",
    [
        "raw",
        "occupied",
        "exit",
        "aggregate",
        "count",
        "status",
        "cluster",
        "nullity",
        "requested",
        "copy",
        "host",
        "physical",
        "native",
        "absolute",
        "length",
        "nan",
        "missing",
        "runtime",
        "setup",
    ],
)
def test_failed_evidence_cannot_be_promoted(failure):
    raw, occ = fixture()
    if failure == "raw":
        raw["timing_eligible"] = False
    elif failure == "occupied":
        occ["timing_eligible"] = False
    elif failure == "exit":
        occ["exit_code"] = 1
    elif failure == "aggregate":
        raw["accepted"] = False
    elif failure == "count":
        raw["rows"][0]["quality"]["converged"] = 2
    elif failure == "status":
        raw["rows"][0]["quality"]["status"] = 1
    elif failure == "cluster":
        raw["fixture"]["boundary_cluster_complete"] = False
    elif failure == "nullity":
        raw["rows"][0]["quality"]["recovered_nullity"] = 1
    elif failure == "requested":
        raw["fixture"]["requested_nontrivial_modes"] = 2
    elif failure == "copy":
        raw["graph"]["memcpy_kinds"]["HtoD"] = 1
    elif failure == "host":
        raw["graph"]["node_types_including_conditional_body"]["HOST"] = 1
    elif failure == "physical":
        raw["rows"][0]["quality"]["requested_subspace_error"] = 0.1
    elif failure == "native":
        raw["rows"][0]["native_stopping_accepted"] = False
    elif failure == "absolute":
        raw["rows"][3]["returned_absolute_residuals"][-1] = 1.001e-10
    elif failure == "length":
        raw["rows"][3]["returned_absolute_residuals"].pop()
    elif failure == "nan":
        raw["rows"][3]["returned_absolute_residuals"][-1] = float("nan")
    elif failure == "missing":
        raw["rows"].pop()
    elif failure == "runtime":
        raw["rows"][0]["seconds"] = float("nan")
    elif failure == "setup":
        raw["capture_seconds"] = -1
    with pytest.raises(ValueError):
        summarize(raw, occ)


@pytest.mark.parametrize(
    "key",
    [
        "source_sha256",
        "environment",
        "versions",
        "fixture",
        "n",
        "k",
        "arguments",
        "dependency",
        "inputs",
    ],
)
def test_incompatible_baselines_rejected(key):
    raw, occ = fixture()
    result = summarize(raw, occ) | dict(environment={"gpu": "same"})
    changed = copy.deepcopy(result)
    if isinstance(changed[key], dict):
        changed[key]["different"] = True
    else:
        changed[key] += 1
    if key == "dependency":
        changed[key]["warp_version"] = "other"
    with pytest.raises(ValueError, match="incompatible"):
        compare(changed, result)


@pytest.mark.parametrize("helper", ["hodge", "adapters", "anvil_inputs"])
def test_imported_fixture_adapter_or_validation_edit_rejects_baseline(helper, tmp_path):
    (tmp_path / "worker.py").write_text("import hodge, adapters, anvil_inputs\n")
    for name in ("hodge", "adapters", "anvil_inputs"):
        (tmp_path / (name + ".py")).write_text("# original\n")
    raw, occ = fixture()
    raw["source_sha256"] = source_manifest("worker.py", root=tmp_path)
    baseline = summarize(raw, occ) | dict(environment={"gpu": "same"})
    (tmp_path / (helper + ".py")).write_text("# changed helper, unchanged worker\n")
    raw["source_sha256"] = source_manifest("worker.py", root=tmp_path)
    current = summarize(raw, occ) | dict(environment={"gpu": "same"})
    with pytest.raises(ValueError, match="source_sha256"):
        compare(current, baseline)


@pytest.mark.parametrize(
    "part",
    [
        "operator",
        "mass",
        "shifted",
        "values",
        "requested_basis",
        "null_basis",
        "guard",
        "warp",
        "cupy",
        "pool",
    ],
)
def test_changed_input_bytes_rejected_despite_unchanged_convergence(part):
    raw, occ = fixture()
    baseline = summarize(raw, occ) | dict(environment={"gpu": "same"})
    changed = copy.deepcopy(baseline)
    inputs = changed["inputs"]
    if part == "mass":
        inputs["operators"]["mass"] = matrix_identity(2 * sp.eye(40))
    elif part in ("operator", "shifted"):
        inputs["operators"][part]["values"]["sha256"] = "0" * 64
    elif part == "guard":
        inputs["reference"]["guard"] = array_identity(np.array([5.0]))
    elif part in ("values", "requested_basis", "null_basis"):
        inputs["reference"][part]["sha256"] = "0" * 64
    else:
        inputs["initial"][part]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="incompatible input"):
        compare(changed, baseline)


def test_historical_missing_inputs_are_not_backfilled():
    raw, occ = fixture()
    current = summarize(raw, occ) | dict(environment={"gpu": "same"})
    raw["inputs"] = None
    with pytest.raises(ValueError, match="Missing verified procedural input"):
        summarize(raw, occ)
    old_summary = copy.deepcopy(current)
    del old_summary["inputs"]
    with pytest.raises(ValueError, match="Missing verified procedural input"):
        compare(current, old_summary)
