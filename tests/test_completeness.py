"""Inertia plus independent eigenpair checks must reject incomplete spectra."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from tests.test_solver import sparse
from warpack import CuDSSInverse
from warpack import LowestModeCertificate as Certificate


def flags(c):
    return int(c.flags.numpy()[0])


def diagonal(spectrum, indices, upper, **kwargs):
    spectrum = np.asarray(spectrum, dtype=float)
    return Certificate(
        sparse(np.diag(spectrum)),
        wp.array(np.eye(len(spectrum))[indices], dtype=wp.float64),
        wp.array(spectrum[indices], dtype=wp.float64),
        upper_bound=upper,
        **kwargs,
    )


@pytest.mark.parametrize(
    "spectrum,indices,upper,expected",
    [
        ([1, 2, 3, 4], [0, 1], 2.5, 0),
        ([-2, 0, 2, 4], [0, 1], 1.0, 0),
        ([0, 1, 1, 3], [0, 1, 2], 2.0, 0),
        # Exact residuals do not imply the requested spectrum is complete.
        ([0, 1, 1, 3], [0, 1], 2.0, Certificate.COUNT_MISMATCH),
        ([0, 1, 2, 3], [1, 2], 2.5, Certificate.COUNT_MISMATCH),
        ([0, 1, 2, 3], [0, 0], 1.5, Certificate.INVALID_BASIS),
        ([0, 1, 2, 3], [0, 2], 1.5, Certificate.GAP_UNRESOLVED),
        # A boundary through a cluster cannot certify it.
        ([0, 1, 1, 3], [0, 1], 1.0, Certificate.COUNT_INCONCLUSIVE),
    ],
)
def test_complete_and_adversarial_exact_pairs(spectrum, indices, upper, expected):
    c = diagonal(spectrum, indices, upper)
    c.run()
    assert np.max(c.residuals.numpy()) == 0
    assert bool(c.certified.numpy()[0]) == (expected == 0)
    assert flags(c) == 0 if expected == 0 else flags(c) & expected


def test_inertia_metadata_and_unavailable_factor_types():
    for mtype, expected in [
        ("symmetric", "available"),
        ("general", "unavailable"),
        ("spd", "unavailable"),
    ]:
        inverse = CuDSSInverse(sparse(np.diag([1.0, 2.0, 3.0])), 1, mtype=mtype)
        try:
            result = inverse.inertia()
            assert result.status == expected
            if expected == "available":
                assert (result.positive, result.negative, result.info, result.pivots) == (
                    3,
                    0,
                    0,
                    0,
                )
            with wp.ScopedCapture():
                with pytest.raises(RuntimeError, match="capture"):
                    inverse.inertia()
        finally:
            inverse.close()
        with pytest.raises(RuntimeError, match="closed"):
            inverse.inertia()


def test_perturbed_pivots_are_inconclusive():
    inverse = CuDSSInverse(sparse(np.diag([-3.0, 0.0, 2.0, 4.0])), 1, mtype="symmetric")
    try:
        result = inverse.inertia()
        assert result.info == 0  # cuDSS accepts this singular input after perturbation.
        assert result.pivots > 0
        assert result.status == "inconclusive"
    finally:
        inverse.close()


def test_without_inertia_never_certifies():
    c = diagonal([0, 1, 2, 3], [0, 1], 1.5, backend=None)
    c.run()
    assert flags(c) == Certificate.COUNT_UNAVAILABLE
    assert c.certified.numpy()[0] == 0


def test_mass_nullspace_and_replay_after_matrix_updates(tmp_path):
    rng = np.random.default_rng(133)
    t = rng.normal(size=(8, 8))
    mass = t @ t.T + np.eye(8)
    q = la.solve_triangular(la.cholesky(mass, lower=True).T, np.eye(8), lower=False)
    spectrum = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    stiffness = mass @ q @ np.diag(spectrum) @ q.T @ mass
    a, m = sparse(stiffness), sparse(mass)
    v = wp.array(q[:, 2:4].T.copy(), dtype=wp.float64)
    nullspace = wp.array(q[:, :2].T.copy(), dtype=wp.float64)
    values = wp.array(spectrum[2:4], dtype=wp.float64)
    c = Certificate(a, v, values, mass=m, nullspace=nullspace, upper_bound=2.5)
    c.run()
    assert flags(c) == 0
    with wp.ScopedCapture() as cap:
        c.run()
    path = tmp_path / "certificate.dot"
    wp.capture_debug_dot_print(cap.graph, str(path))
    dot = path.read_text()
    assert not any(x in dot for x in ("HOST", "HtoD", "DtoH"))
    for scale in (2.0, 0.5, 1.0):
        wp.copy(a.values, sparse(scale * stiffness).values)
        wp.copy(m.values, sparse(scale * mass).values)
        wp.copy(v, wp.array(q[:, 2:4].T.copy() / np.sqrt(scale), dtype=wp.float64))
        wp.copy(nullspace, wp.array(q[:, :2].T.copy() / np.sqrt(scale), dtype=wp.float64))
        assert c.refresh_count().negative == np.count_nonzero(
            la.eigvalsh(scale * stiffness, scale * mass) < 2.5
        )
        assert c.certified.numpy()[0] == 0  # no stale success before checks replay
        wp.capture_launch(cap.graph)
        assert flags(c) == 0
        assert c.certified.numpy()[0] == 1
    # A purported nullspace must really be independent and null, including M-orthogonality with V.
    wp.copy(nullspace, wp.array(q[:, 2:4].T.copy(), dtype=wp.float64))
    wp.capture_launch(cap.graph)
    assert flags(c) & Certificate.INVALID_BASIS
    assert flags(c) & Certificate.UNCONVERGED
    wp.copy(nullspace, wp.array(q[:, :2].T.copy(), dtype=wp.float64))
    wp.capture_launch(cap.graph)
    assert flags(c) == 0
    for bad in (np.zeros((2, 8)), np.full((2, 8), np.nan)):
        wp.copy(v, wp.array(bad, dtype=wp.float64))
        wp.capture_launch(cap.graph)
        assert flags(c) & (Certificate.INVALID_BASIS | Certificate.INVALID_DATA)
        assert c.certified.numpy()[0] == 0
    wp.copy(v, wp.array(q[:, 2:4].T.copy(), dtype=wp.float64))
    values.assign([1.0, 1.5])
    wp.capture_launch(cap.graph)
    assert flags(c) & Certificate.UNCONVERGED
    values.assign([1.0, 2.0])
    wp.capture_launch(cap.graph)
    assert flags(c) == 0
    with wp.ScopedCapture():
        with pytest.raises(RuntimeError, match="capture"):
            c.refresh_count()
    # A failed refresh clears an earlier successful certification.
    c.backend = "cudss"
    from unittest.mock import patch

    with patch("warpack.completeness.CuDSSInverse", side_effect=RuntimeError("factor failed")):
        with pytest.raises(RuntimeError, match="factor failed"):
            c.refresh_count()
    assert c.certified.numpy()[0] == 0
    wp.capture_launch(cap.graph)
    assert flags(c) & Certificate.COUNT_UNAVAILABLE


def test_nondiagonal_sparse_congruence_count():
    # Tridiagonal SPD M and pentadiagonal symmetric K with known inertia.
    n = 40
    t = np.eye(n) + np.diag(np.full(n - 1, 0.1), 1)
    spectrum = np.arange(n, dtype=float)
    mass, stiffness = t.T @ t, t.T @ np.diag(spectrum) @ t
    q = la.solve_triangular(t, np.eye(n))
    c = Certificate(
        sparse(stiffness),
        wp.array(q[:, :7].T.copy(), dtype=wp.float64),
        wp.array(spectrum[:7], dtype=wp.float64),
        mass=sparse(mass),
        upper_bound=6.5,
    )
    c.run()
    assert c.inertia.negative == 7
    assert flags(c) == 0


@pytest.mark.parametrize(
    "metadata,status",
    [
        ({0: None, 2: (0,), 3: (2, 2)}, "unavailable"),
        ({0: (1,), 2: (0,), 3: (2, 2)}, "inconclusive"),
        ({0: (0,), 2: (0,), 3: (2, 1)}, "inconclusive"),
        ({0: (0,), 2: (0,), 3: (-1, 5)}, "inconclusive"),
    ],
)
def test_suspect_count_metadata_never_certifies(monkeypatch, metadata, status):
    monkeypatch.setattr(
        "warpack.cudss_diagnostics._integers", lambda solver, key, count: metadata[key]
    )
    c = diagonal([0, 1, 2, 3], [0, 1], 1.5)
    c.run()
    assert c.inertia.status == status
    assert c.certified.numpy()[0] == 0
    assert flags(c) & (Certificate.COUNT_UNAVAILABLE | Certificate.COUNT_INCONCLUSIVE)


def test_whole_subspace_gap_not_just_individual_rayleigh_values():
    # Two Rayleigh quotients below 1 do not put the entire span below 1.
    # The count is also two, so diagonal-only checking would falsely accept.
    spectrum = np.array([0.0, 0.2, 1.9, 3.0])
    v = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, -1.0, 0.0]]) / np.sqrt(2)
    c = Certificate(
        sparse(np.diag(spectrum)),
        wp.array(v, dtype=wp.float64),
        wp.array([0.95, 0.95], dtype=wp.float64),
        upper_bound=1.0,
        residual_tolerance=1.0,
    )
    c.run()
    assert c.inertia.negative == 2
    assert flags(c) == Certificate.GAP_UNRESOLVED
    assert c.certified.numpy()[0] == 0


def test_refreshed_count_changes_on_existing_graph():
    c = diagonal([0, 1, 2, 3], [0, 1], 1.5)
    c.run()
    with wp.ScopedCapture() as cap:
        c.run()
    wp.copy(c.stiffness.values, sparse(np.diag([0.0, 1.0, 1.25, 3.0])).values)
    assert c.refresh_count().negative == 3
    wp.capture_launch(cap.graph)
    assert flags(c) == Certificate.COUNT_MISMATCH
    assert c.certified.numpy()[0] == 0
    wp.copy(c.stiffness.values, sparse(np.diag([0.0, 1.0, 2.0, 3.0])).values)
    assert c.refresh_count().negative == 2
    wp.capture_launch(cap.graph)
    assert flags(c) == 0
    with wp.ScopedCapture():
        with pytest.raises(RuntimeError, match="capture"):
            Certificate(c.stiffness, c.vectors, c.values, upper_bound=1.5)
