"""Original-pencil subspace gates must use a positive metric for buckling."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from benchmarks.applications.problems import Problem
from benchmarks.applications.runner import accuracy


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("negative", [False, True])
def test_buckling_space_uses_stiffness_metric(coupled, negative):
    transform = np.eye(4)
    if coupled:
        transform += np.diag([0.2, -0.3, 0.4], 1)
    stiffness = transform.T @ np.diag([1.0, 3.0, 2.0, 5.0]) @ transform
    geometric = transform.T @ np.diag([-1.0, -2.0, 3.0, 4.0]) @ transform
    mu, basis = la.eigh(geometric, stiffness)
    selected = 0 if negative else 3
    value = np.array([1 / mu[selected]])
    vector = basis[:, selected : selected + 1]
    problem = Problem(
        "buckling_metric",
        "G",
        sp.csr_matrix(stiffness),
        sp.csr_matrix(geometric),
        1,
        0.2,
        mode="buckling",
        expected=value,
        requested_basis=vector,
    )
    correct = accuracy(problem, value, vector)
    assert correct["accepted"]
    assert correct["requested_subspace_error"] < 1e-14

    # A deliberately inconsistent reference isolates the space gate: the
    # returned pair still has an exact equation, value, and K-normalization.
    other = (selected + 1) % 4
    problem.requested_basis = basis[:, other : other + 1]
    wrong = accuracy(problem, value, vector)
    assert wrong["residual"] < 1e-14
    assert wrong["spectrum_error"] == 0
    assert wrong["orthogonality"] < 1e-14
    assert wrong["requested_subspace_error"] == pytest.approx(1.0)
    assert not wrong["accepted"]


def test_negative_geometric_norm_does_not_hide_wrong_space():
    problem = Problem(
        "false_acceptance",
        "G",
        sp.eye(3, format="csr"),
        sp.diags([-1.0, -2.0, 3.0], format="csr"),
        1,
        0.2,
        mode="buckling",
        expected=np.array([-1.0]),
        requested_basis=np.eye(3)[:, 1:2],
    )
    result = accuracy(problem, np.array([-1.0]), np.eye(3)[:, :1])
    assert result["requested_subspace_error"] == 1.0
    assert not result["accepted"]


@pytest.mark.parametrize("metric", ["identity", "diagonal", "coupled"])
def test_ordinary_requested_space_retains_mass_metric(metric):
    mass = np.eye(3) if metric == "identity" else np.diag([2.0, 3.0, 4.0])
    if metric == "coupled":
        mass += np.diag([0.2, 0.3], 1) + np.diag([0.2, 0.3], -1)
    stiffness = np.diag([1.0, 5.0, 9.0])
    values, basis = la.eigh(stiffness, mass)
    problem = Problem(
        "ordinary_metric",
        "A",
        sp.csr_matrix(stiffness),
        None if metric == "identity" else sp.csr_matrix(mass),
        1,
        0.0,
        expected=values[:1],
        requested_basis=basis[:, :1],
    )
    assert accuracy(problem, values[:1], basis[:, :1])["accepted"]
    problem.requested_basis = basis[:, 1:2]
    wrong = accuracy(problem, values[:1], basis[:, :1])
    assert wrong["requested_subspace_error"] == pytest.approx(1.0)
    assert not wrong["accepted"]
