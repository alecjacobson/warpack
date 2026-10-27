"""A correct warm replay must not hide a failed first application solve (#207)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.acceptance import assess_comparison, require_accepted
from benchmarks.applications import compare
from benchmarks.applications.problems import make_problem


@pytest.mark.parametrize("failure", ["physical", "count", "status", "inner", None])
def test_first_warp_output_is_independently_gated(monkeypatch, tmp_path, failure):
    """Inject first-only failures at the device-output boundary of the real reporter."""
    p = make_problem("A", 31)
    values, vectors = la.eigh(p.a.toarray(), p.b.toarray())
    state = dict(launches=0, resets=0)

    def array(first, later):
        return SimpleNamespace(numpy=lambda: np.asarray(first if state["launches"] == 1 else later))

    solver = SimpleNamespace(
        capture=lambda steps: object(),
        converged=array([p.k - (failure == "count")], [p.k]),
        status=array([3 if failure == "status" else 0], [0]),
        iterations=array([2], [2]),
        residuals=array([0.0], [0.0]),
    )

    def launch(graph):
        state["launches"] += 1

    def reset():
        state["resets"] += 1

    inverse = SimpleNamespace(
        stats=array([31, 31, int(failure == "inner"), 1], [31, 31, 0, 1]),
        reset_stats=reset,
    )

    def output(problem, solver):
        # Wrong requested spectrum with small physical residual and exact orthogonality.
        selected = (
            slice(-p.k, None) if failure == "physical" and state["launches"] == 1 else slice(p.k)
        )
        return values[selected], vectors[:, selected], None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compare, "timed", lambda fn: (1.0, fn()))
    monkeypatch.setattr(compare, "other_gpu_processes", lambda: [])
    monkeypatch.setattr(compare, "warp_setup", lambda *a, **k: (solver, inverse))
    monkeypatch.setattr(compare, "warp_output", output)
    monkeypatch.setattr(compare.wp, "capture_launch", launch)
    monkeypatch.setattr(
        compare.wp,
        "capture_debug_dot_print",
        lambda graph, path: Path(path).write_text("digraph {}"),
    )
    row = compare.run("A", p.n, "warp-cg", problem=p, repeats=2)
    first = row["first_solve_quality"]
    assert first["accepted"] == (failure != "physical")
    assert first["solver_full_count"] == (failure != "count")
    assert first["solver_status"] == (3 if failure == "status" else 0)
    assert first["inner_inverse_converged"] == (failure != "inner")
    assert row["first_solve_accepted"] == (failure is None)
    assert all(check["accepted"] and check["solver_full_count"] for check in row["validation"])
    assert all(
        check["solver_status"] == 0 and check["inner_inverse_converged"]
        for check in row["validation"]
    )
    assert row["samples_seconds"] == [1.0, 1.0]
    assert state == dict(launches=3, resets=3)
    report = {}
    assert assess_comparison(report, [row]) == (failure is None)
    if failure is not None:
        assert row["outcome"] == "failed"
        with pytest.raises(SystemExit):
            require_accepted(report)


def test_first_reference_output_is_gated(monkeypatch, tmp_path):
    p = make_problem("A", 31)
    values, vectors = la.eigh(p.a.toarray(), p.b.toarray())
    outputs = iter(
        [(values[-p.k :], vectors[:, -p.k :], None)]
        + [(values[: p.k], vectors[:, : p.k], None)] * 2
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compare, "timed", lambda fn: (1.0, fn()))
    monkeypatch.setattr(compare, "other_gpu_processes", lambda: [])
    monkeypatch.setattr(compare, "scipy_setup", lambda *a, **k: lambda: next(outputs))
    row = compare.run("A", p.n, "scipy", problem=p, repeats=2)
    assert not row["first_solve_accepted"]
    assert all(check["accepted"] for check in row["validation"])
    assert row["samples_seconds"] == [1.0, 1.0]
    assert row["outcome"] == "failed"


@pytest.mark.parametrize("api_class", list("ABCDEGHI"))
def test_actual_application_first_output_and_replay(monkeypatch, tmp_path, api_class):
    monkeypatch.chdir(tmp_path)
    row = compare.run(api_class, 80, "warp", repeats=1, steps=100 if api_class in "DG" else 40)
    (tmp_path / "report.json").write_text(json.dumps(row, indent=2) + "\n")
    assert row["accepted"], json.dumps(row, indent=2)
    assert row["first_solve_accepted"]
    assert row["first_solve_quality"]["solver_full_count"]
    assert len(row["validation"]) == len(row["samples_seconds"]) == 1


def test_conformal_report_preserves_native_stopping_floor(monkeypatch, tmp_path):
    """#209: physical success alone cannot certify this ill-conditioned pencil."""
    monkeypatch.chdir(tmp_path)
    row = compare.run("F", 80, "warp", repeats=1, steps=40)
    (tmp_path / "report.json").write_text(json.dumps(row, indent=2) + "\n")
    checks = [row["first_solve_quality"], *row["validation"]]
    assert all(check["accepted"] for check in checks), json.dumps(row, indent=2)
    # Native counts fluctuate near 1e-11. This tests honest reporting, not
    # strict numerical qualification of the unresolved #209 solver case.
    native = [
        check["solver_status"] == 0 and check["reported_converged"] == row["k"] for check in checks
    ]
    assert row["first_solve_accepted"] == native[0]
    assert row["accepted"] == all(native)
    assert assess_comparison({}, [row]) == all(native)
    assert len(row["samples_seconds"]) == 1


@pytest.mark.parametrize("method", ["warp", "warp-cg"])
def test_changed_input_first_output_after_setup_protocol(monkeypatch, tmp_path, method):
    """Establish graph upload during setup, then change the start before first measured use."""
    setup = compare.warp_setup
    seen = []

    def with_updated_start(p, *args, **kwargs):
        solver, inverse = setup(p, *args, **kwargs)
        start = wp.array(np.ones(p.n), dtype=wp.float64, device=solver.device)
        solver.initial_vector = start
        capture = solver.capture

        def captured(steps):
            graph = capture(steps)
            wp.capture_launch(graph)
            wp.synchronize_device(solver.device)
            replacement = wp.array(
                np.random.default_rng(817).normal(size=p.n), dtype=wp.float64, device=solver.device
            )
            wp.copy(start, replacement)
            seen.append((start, replacement))
            return graph

        solver.capture = captured
        return solver, inverse

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compare, "warp_setup", with_updated_start)
    row = compare.run("A", 80, method, repeats=1, steps=40, cg_tol=1e-14, cg_atol=0.0)
    (tmp_path / "report.json").write_text(json.dumps(row, indent=2) + "\n")
    assert row["accepted"], json.dumps(row, indent=2)
    assert row["first_solve_accepted"] and len(seen) == 1
    np.testing.assert_array_equal(seen[0][0].numpy(), np.random.default_rng(817).normal(size=80))
    if method == "warp-cg":
        assert row["first_solve_quality"]["inner_inverse_converged"]
