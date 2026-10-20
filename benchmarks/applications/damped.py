"""Coupled axial FEM branches with nonproportional damping and gyroscopic terms.

Host-side fixture/reference construction only. The two polarization matrices
need not commute, but spatial sine modes reduce the independent reference to
4x4 systems. The sparse solver still receives the assembled first-order matrix.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment


@dataclass
class DampedCase:
    name: str
    a: sp.csr_matrix
    stiffness: sp.csr_matrix
    mass: sp.csr_matrix
    damping: sp.csr_matrix
    gyroscopic: sp.csr_matrix
    target: complex
    values: np.ndarray
    right: np.ndarray
    left: np.ndarray
    full_spectrum: np.ndarray
    nodes: int

    @property
    def k(self):
        return len(self.values)


def coupled_chain(nodes=32, k=8, kind="nonproportional", target=-0.15 + 6.3j):
    """Lumped spatial P1 mass, two coupled components, fixed endpoints.

    The consistent option refers to a non-diagonal *polarization* mass; spatial
    mass remains lumped. This is not a general consistent spatial mass inverse.
    Standardization uses a known2x2 block inverse only during fixture assembly.
    """
    if nodes < 4 or not 1 <= k < 4 * nodes:
        raise ValueError("require nodes>=4 and 1<=k<4*nodes")
    if kind not in ("proportional", "nonproportional", "gyroscopic", "coupled-mass"):
        raise ValueError("unknown damped fixture kind")
    spatial = (
        sp.diags(
            [-np.ones(nodes - 1), 2 * np.ones(nodes), -np.ones(nodes - 1)], [-1, 0, 1], format="csr"
        )
        * (nodes + 1) ** 2
    )
    elastic = np.diag([1.0, 1.7])
    foundation = np.array([[2.0, 0.3], [0.3, 3.0]])
    inertia = np.array([[1.2, 0.15], [0.15, 0.9]]) if kind == "coupled-mass" else np.eye(2)
    damping = 0.2 * inertia if kind == "proportional" else np.array([[0.2, 0.055], [0.055, 0.4]])
    gyroscopic = np.array([[0.0, 0.7], [-0.7, 0.0]]) if kind == "gyroscopic" else np.zeros((2, 2))
    identity = sp.eye(nodes, format="csr")
    stiffness = sp.kron(spatial, elastic, format="csr") + sp.kron(
        identity, foundation, format="csr"
    )
    mass = sp.kron(identity, inertia, format="csr")
    c = sp.kron(identity, damping, format="csr")
    g = sp.kron(identity, gyroscopic, format="csr")
    inverse_mass = sp.kron(identity, la.inv(inertia), format="csr")
    a = sp.bmat(
        [[None, sp.eye(2 * nodes)], [-inverse_mass @ stiffness, -inverse_mass @ (c + g)]],
        format="csr",
    )
    # Independent spatial eigenvalues; no eigensolve of assembled A or K.
    theta = np.arange(1, nodes + 1) * np.pi / (nodes + 1)
    spatial_values = 4 * (nodes + 1) ** 2 * np.sin(theta / 2) ** 2
    local = np.zeros((nodes, 4, 4))
    local[:, :2, 2:] = np.eye(2)
    local[:, 2:, :2] = -np.linalg.solve(
        inertia, spatial_values[:, None, None] * elastic + foundation
    )
    local[:, 2:, 2:] = -la.solve(inertia, damping + gyroscopic)
    all_values = np.linalg.eigvals(local).ravel()
    selected = np.argsort(abs(all_values - target))[:k]
    right, left, values = [], [], []
    for flat in selected:
        mode = int(flat // 4)
        w, vl, vr = la.eig(local[mode], left=True, right=True)
        j = int(np.argmin(abs(w - all_values[flat])))
        sine = np.sqrt(2.0 / (nodes + 1)) * np.sin(theta[mode] * np.arange(1, nodes + 1))
        r = np.r_[np.outer(sine, vr[:2, j]).ravel(), np.outer(sine, vr[2:, j]).ravel()]
        left_vector = np.r_[np.outer(sine, vl[:2, j]).ravel(), np.outer(sine, vl[2:, j]).ravel()]
        r /= np.linalg.norm(r)
        # Reference left vectors are dual to normalized right vectors.
        left_vector /= np.conj(np.vdot(left_vector, r))
        values.append(w[j])
        right.append(r)
        left.append(left_vector)
    return DampedCase(
        kind,
        a,
        stiffness,
        mass,
        c,
        g,
        target,
        np.array(values),
        np.array(right).T,
        np.array(left).T,
        all_values,
        nodes,
    )


def quality(case, values, vectors):
    """Original first-order and quadratic equations, full-set and rank gates."""
    if values.shape != (case.k,) or vectors.shape != (case.a.shape[0], case.k):
        return {"accepted": False, "reason": "wrong output shape"}
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        return {"accepted": False, "reason": "nonfinite output"}
    cost = abs(values[:, None] - case.values) / np.maximum(1, abs(case.values))[None, :]
    rows, cols = linear_sum_assignment(cost)
    av = case.a @ vectors
    residual = np.linalg.norm(av - vectors * values, axis=0) / np.maximum(
        1, np.linalg.norm(av, axis=0) + abs(values) * np.linalg.norm(vectors, axis=0)
    )
    q, velocity = vectors[: 2 * case.nodes], vectors[2 * case.nodes :]
    kq, dq, mq = case.stiffness @ q, (case.damping + case.gyroscopic) @ q, case.mass @ q
    quadratic = np.linalg.norm(kq + dq * values + mq * values**2, axis=0) / np.maximum(
        1,
        np.linalg.norm(kq, axis=0)
        + abs(values) * np.linalg.norm(dq, axis=0)
        + abs(values) ** 2 * np.linalg.norm(mq, axis=0),
    )
    singular = la.svdvals(vectors)
    rank_threshold = max(vectors.shape) * np.finfo(float).eps * singular[0]
    basis = la.orth(case.right)
    # Normalize each column so the subspace test cannot pass small vectors.
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=0), np.finfo(float).tiny)
    result = dict(
        residual=float(residual.max()),
        quadratic_residual=float(quadratic.max()),
        velocity_relation=float(np.linalg.norm(velocity - q * values, axis=0).max()),
        spectrum_error=float(cost[rows, cols].max()),
        norm_error=float(abs(np.linalg.norm(vectors, axis=0) - 1).max()),
        subspace_error=float(np.linalg.norm(normalized - basis @ (basis.conj().T @ normalized))),
        minimum_vector_singular_value=float(singular[-1]),
        independent_vectors=bool(singular[-1] > rank_threshold),
    )
    result["accepted"] = bool(
        result["independent_vectors"]
        and all(
            result[key] < 1e-8
            for key in (
                "residual",
                "quadratic_residual",
                "velocity_relation",
                "spectrum_error",
                "norm_error",
                "subspace_error",
            )
        )
    )
    return result
