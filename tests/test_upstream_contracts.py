"""Reporting must distinguish physical accuracy, convergence, and missing evidence."""

import pytest

from benchmarks.upstream.compare_reports import compare, contracts


def example(**changes):
    return dict(
        dict(
            id="Example1_0",
            family="Example1",
            outcome="failed",
            k=3,
            upstream_absolute_tolerance=1e-9,
            max_absolute_residual=2.42e-13,
            spectrum_error=0.142,
            reported_converged=3,
            upstream_allows_nonconvergence=False,
        ),
        **changes,
    )


def test_wrong_spectrum_is_not_numerical_conformance():
    c = contracts(example())
    assert c["reported_count_pass"]
    assert not c["checked_numerical_bounds_pass"]
    assert not c["strict_contract_pass"]
    r = compare({}, {"case": example(outcome="passed")})
    assert r["contradictory_successes"] == ["case"]
    assert not r["regression_free"]


@pytest.mark.parametrize("allowed", [False, True])
def test_numerical_pass_is_distinct_from_count_and_allowed_nonconvergence(allowed):
    c = contracts(
        example(spectrum_error=1e-14, reported_converged=2, upstream_allows_nonconvergence=allowed)
    )
    assert c["checked_numerical_bounds_pass"]
    assert c["numerical_pass_count_short"]
    assert not c["reported_count_pass"] and not c["strict_contract_pass"]
    assert c["upstream_allows_nonconvergence"] == allowed


def test_residual_only_is_not_spectrum_verification():
    row = example(family="SymEigs")
    del row["spectrum_error"]
    assert contracts(row)["checked_numerical_bounds_pass"]
    assert not contracts(row)["requested_spectrum_checked"]


@pytest.mark.parametrize("field", ["orthogonality_error", "subspace_error", "vector_norm_error"])
def test_bad_eigenspace_cannot_pass_with_small_residual(field):
    row = example(family="HermEigs", spectrum_error=1e-14, **{field: 1e-3})
    assert not contracts(row)["checked_numerical_bounds_pass"]
    assert field in contracts(row)["checked_fields"]


def test_svd_checks_vectors_too():
    c = contracts(example(family="SVD", values=1e-15, left_vectors=1e-5, right_vectors=1e-15))
    assert not c["checked_numerical_bounds_pass"]


@pytest.mark.parametrize("error", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_invalid_error_never_passes(error):
    assert contracts(example(spectrum_error=error))["checked_numerical_bounds_pass"] is False


@pytest.mark.parametrize("outcome", ["timeout", "unsupported", "error"])
def test_unevaluated_is_not_pass_even_with_stale_metrics(outcome):
    c = contracts(example(outcome=outcome, spectrum_error=0))
    assert c["checked_numerical_bounds_pass"] is None
    assert not c["strict_contract_pass"]


def test_missing_assertions_are_unknown():
    row = example()
    del row["spectrum_error"]
    assert contracts(row)["checked_numerical_bounds_pass"] is None


def test_hidden_numerical_regression_with_unchanged_failed_label():
    old = example(spectrum_error=1e-14, reported_converged=1)
    new = example(spectrum_error=1e-14, reported_converged=1, max_absolute_residual=1e-4)
    r = compare({"case": old}, {"case": new})
    assert not r["outcome_changes"]
    assert r["numerical_regressions"] == ["case"]
    assert not r["regression_free"]


def test_missing_case_blocks_regression_success():
    r = compare({"case": example()}, {})
    assert r["missing_cases"] == ["case"]
    assert not r["regression_free"]


def test_svd_opposite_left_right_signs_fail_paired_equations():
    import numpy as np

    from benchmarks.upstream.run_spectra import svd_errors

    a = np.diag([4.0, 2.0, 1.0])
    u, values, vt = np.linalg.svd(a)
    errors = svd_errors(a, values[:2], u[:, :2], -vt[:2].T)
    # The upstream absolute-vector assertions cannot detect mismatched signs.
    assert errors["values"] == errors["left_vectors"] == errors["right_vectors"] == 0
    assert errors["left_equation_residual"] == errors["right_equation_residual"] == 8
    row = example(family="SVD", k=2, reported_converged=2, **errors)
    assert not contracts(row)["checked_numerical_bounds_pass"]


@pytest.mark.parametrize("left,right", [(4, 5), (5, 4), (5, 5)])
def test_svd_success_requires_both_physical_counts(left, right):
    row = example(
        family="SVD",
        k=5,
        spectrum_error=0,
        values=0,
        left_vectors=0,
        right_vectors=0,
        reported_converged=5,
        left_converged=left,
        right_converged=right,
    )
    result = contracts(row)
    assert result["reported_count_pass"]
    assert result["physical_count_pass"] == (left == right == 5)
    assert result["strict_contract_pass"] == (left == right == 5)
    assert result["numerical_pass_count_short"] == (min(left, right) < 5)


@pytest.mark.parametrize("tol", [0, -1, float("nan"), float("inf")])
def test_spectra_tolerance_override_rejects_invalid_values(tol):
    from benchmarks.upstream.run_spectra import run

    with pytest.raises(ValueError, match="tolerance_override"):
        run({}, tolerance_override=tol)


def test_spectra_basis_override_rejects_nonpositive_size():
    from benchmarks.upstream.run_spectra import run

    with pytest.raises(ValueError, match="ncv_override"):
        run({}, ncv_override=0)
