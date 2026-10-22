"""Dominant-spectrum timing claims require complete native and physical checks."""

import copy

import pytest

from benchmarks.applications.spectral_surface_report import compare, summarize


def fixture(backend="cg"):
    q = dict(
        accepted=True,
        status=0,
        converged=1,
        residual=1e-12,
        orthogonality=1e-14,
        spectrum_error=1e-12,
        requested_subspace_error=1e-10,
    )
    geometry = dict(
        valid_zero_set=True,
        crossings=100,
        sample_distance_rms=0.01,
        zero_set_relative_radial_rms=0.02,
        zero_set_relative_radial_max=0.04,
        normal_alignment_mean=0.98,
    )
    row = dict(quality=q, values=[5.0], seconds=0.2, geometry=geometry)
    if backend == "cg":
        row.update(
            cg_stats=[100, 10, 0, 10],
            fresh_cg_stats=[100, 10, 2, 10],
            fresh_residual_maxima=[1e-11, 5e-13],
        )
    if backend == "block":
        row["preconditioner_stats"] = [100, 25, 2, 10]
    graph = dict(node_types_including_conditional_body={"KERNEL": 10}, memcpy_kinds={})
    device_only = True
    if backend == "cudss":
        graph["memcpy_kinds"]["HtoD"] = 8
        device_only = False
    elif backend == "arpack":
        graph, device_only = None, None
    raw = dict(
        arguments=dict(backend=backend, repeats=3, modes=1),
        accepted=True,
        warmup=copy.deepcopy(row),
        rows=[copy.deepcopy(row) for _ in range(3)],
        graph=graph,
        graph_device_only=device_only,
        assembly_seconds=0.2,
        reference_and_copy_assembly_seconds=0.3,
        setup_seconds=0.4,
        capture_seconds=0.1 if backend != "arpack" else 0,
        setup_capture_device_free_memory_decrease_bytes=1024 if backend != "arpack" else None,
        matrix_sha256="matrix",
        source_sha256={},
        environment={},
        versions={},
        n=100,
        dependency=dict(manifest_status="verified", warp_version="test", files={}),
        inner_role="truncated preconditioner" if backend == "block" else "metric inverse",
    )
    return raw, dict(timing_eligible=True, exit_code=0)


@pytest.mark.parametrize("backend", ["cg", "cudss", "arpack", "block"])
def test_accounting_and_visible_vendor_graph_copies(backend):
    raw, occupancy = fixture(backend)
    summary = summarize(raw, occupancy)
    assert summary["setup_capture_solve_seconds"] == pytest.approx(
        0.6 if backend == "arpack" else 0.7
    )
    assert summary["graph_device_only"] == raw["graph_device_only"]
    assert compare(summary, summary) == 1
    if backend == "cg":
        assert summary["inner_diagnostics"]["max_fresh_misses"] == 2
    if backend == "block":
        assert summary["inner_diagnostics"]["max_recursive_tolerance_misses"] == 2
    slower = copy.deepcopy(summary)
    slower["solve_seconds"] *= 1.3
    with pytest.raises(ValueError, match="regression"):
        compare(slower, summary)
    changed = copy.deepcopy(summary)
    changed["signature"]["matrix_sha256"] = "different"
    with pytest.raises(ValueError, match="incompatible"):
        compare(changed, summary)


@pytest.mark.parametrize(
    "failure",
    [
        "native",
        "count",
        "wrong-spectrum",
        "nan",
        "metric",
        "partial",
        "contended",
        "host",
        "upload",
        "readback",
        "false-graph-flag",
        "missing-geometry",
        "capture",
        "dependency",
    ],
)
@pytest.mark.parametrize("backend", ["cg", "block"])
def test_bad_run_rejected(failure, backend):
    raw, occupancy = fixture(backend)
    row = raw["rows"][1]
    if failure == "native":
        row["quality"]["status"] = 1
    elif failure == "count":
        row["values"] = []
    elif failure == "wrong-spectrum":
        row["quality"]["spectrum_error"] = 0.142
    elif failure == "nan":
        row["seconds"] = float("nan")
    elif failure == "metric":
        row["quality"]["orthogonality"] = 1e-4
    elif failure == "partial":
        raw["rows"].pop()
    elif failure == "contended":
        occupancy["timing_eligible"] = False
    elif failure == "host":
        raw["graph"]["node_types_including_conditional_body"]["HOST"] = 1
    elif failure in ("upload", "readback"):
        raw["graph"]["memcpy_kinds"]["HtoD" if failure == "upload" else "DtoH"] = 1
    elif failure == "false-graph-flag":
        raw["graph_device_only"] = False
    elif failure == "missing-geometry":
        row["geometry"] = None
    elif failure == "capture":
        raw["capture_seconds"] = 0
    else:
        raw["dependency"]["manifest_status"] = "mismatch"
    with pytest.raises(ValueError):
        summarize(raw, occupancy)


@pytest.mark.parametrize("failure", ["role", "negative", "impossible-count"])
def test_truncated_preconditioner_contract(failure):
    raw, occupancy = fixture("block")
    if failure == "role":
        raw["inner_role"] = "metric inverse"
    elif failure == "negative":
        raw["rows"][0]["preconditioner_stats"][0] = -1
    else:
        raw["rows"][0]["preconditioner_stats"][2] = 11
    with pytest.raises(ValueError):
        summarize(raw, occupancy)
