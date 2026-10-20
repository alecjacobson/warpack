"""Original-pencil spectrum/mass checks and existing regular-API adapter coverage."""

import numpy as np
import pytest
import scipy.linalg as la

from benchmarks.upstream import run_spectra


def problem():
    rng = np.random.default_rng(313)
    n = 18
    root = np.diag(np.linspace(1, 2, n)) + np.diag(np.full(n - 1, 0.15), 1)
    q = la.qr(rng.normal(size=(n, n)))[0]
    values = np.r_[[-12, -9, -6], np.linspace(-1, 1.5, n - 6), [4, 7, 10]]
    a, b = root.T @ q @ np.diag(values) @ q.T @ root, root.T @ root
    return a, b, values, la.solve(root, q)


@pytest.mark.parametrize("which", ["LM", "LA", "SA", "SM"])
def test_generalized_reference_uses_metric_and_requested_spectrum(which):
    a, b, values, vectors = problem()
    indices = run_spectra.select_symmetric_reference(values, 3, which)
    good = run_spectra.generalized_reference_errors(
        a, b, values[indices], vectors[:, indices], 3, which, 1e-9
    )
    assert good["mandatory_spectrum_covered"]
    assert (
        max(good[key] for key in ("spectrum_error", "orthogonality_error", "subspace_error"))
        < 1e-12
    )
    # Valid exact eigenpairs from the wrong spectrum must not pass.
    wrong = np.setdiff1d(np.arange(len(a)), indices)[:3]
    bad = run_spectra.generalized_reference_errors(
        a, b, values[wrong], vectors[:, wrong], 3, which, 1e-9
    )
    assert bad["spectrum_error"] > 1e-9


def test_generalized_reference_allows_boundary_tie_but_requires_strict_interior():
    values = np.array([-10.0, -9, -8, 8, 9, 10])
    b = np.diag(np.arange(1.0, 7.0))
    a, basis = b @ np.diag(values), np.diag(1 / np.sqrt(np.diag(b)))
    for indices, accepted in (([0, 5, 1], True), ([0, 5, 4], True), ([0, 1, 4], False)):
        check = run_spectra.generalized_reference_errors(
            a, b, values[indices], basis[:, indices], 3, "LM", 1e-9
        )
        assert check["mandatory_spectrum_covered"] == accepted
        assert (check["spectrum_error"] < 1e-9) == accepted


@pytest.mark.parametrize("family", ["SymGEigsCholesky", "SymGEigsRegInv"])
@pytest.mark.parametrize("selection", [0, 3, 7])
def test_regular_generalized_spectra_adapter(family, selection, tmp_path, monkeypatch):
    a, b, _, _ = problem()
    for suffix, matrix in (("a", a), ("b", b)):
        with (tmp_path / f"regular.{suffix}").open("wb") as stream:
            np.asarray(matrix.shape, dtype=np.int64).tofile(stream)
            np.asarray(matrix, dtype=np.complex128).tofile(stream)
    monkeypatch.setattr(run_spectra, "FIX", tmp_path)
    row = run_spectra.run(
        dict(
            id="regular",
            family=family,
            name="indefinite A and consistent SPD B",
            k=3,
            ncv=len(a),
            selection=selection,
            sigma_real=0,
            sigma_imag=0,
        ),
        accuracy_only=True,
        iteration_budget=4,
    )
    assert row["outcome"] == "passed", row
    assert row["solver_type"] == "GeneralizedKrylovSchur"
    assert row["requested_spectrum_checked"] and row["mandatory_spectrum_covered"]
    assert row["reported_converged"] == 3 and row["solver_status"] == 0
    assert row["effective_ncv"] == len(a)
