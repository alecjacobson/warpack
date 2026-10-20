"""Consistent spatial P1 mass pencil and independent small modal references."""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .damped import DampedCase


def consistent_chain(nodes=32, k=8, kind="nonproportional", target=-0.15 + 6.3j):
    """Assemble [0,I;-K,-D-G] x = lambda diag(I,M) x without inverting M.

    K and M are two-polarization P1 chain matrices with fixed endpoints.
    Their spatial sine modes reduce the independent reference to 4x4 pencils;
    all dense solves below are reference construction, outside the solver.
    """
    if nodes < 4 or not 1 <= k < 4 * nodes:
        raise ValueError("require nodes>=4 and 1<=k<4*nodes")
    if kind not in ("nonproportional", "gyroscopic"):
        raise ValueError("unknown consistent-mass fixture kind")
    h = 1 / (nodes + 1)
    spatial_k = (
        sp.diags(
            [-np.ones(nodes - 1), 2 * np.ones(nodes), -np.ones(nodes - 1)], [-1, 0, 1], format="csr"
        )
        / h
    )
    spatial_m = sp.diags(
        [np.ones(nodes - 1), 4 * np.ones(nodes), np.ones(nodes - 1)], [-1, 0, 1], format="csr"
    ) * (h / 6)
    elastic = np.diag([1.0, 1.7])
    foundation = np.array([[2.0, 0.3], [0.3, 3.0]])
    inertia = np.array([[1.2, 0.15], [0.15, 0.9]])
    damping = np.array([[0.2, 0.055], [0.055, 0.4]])
    gyro = np.array([[0.0, 0.7], [-0.7, 0.0]]) if kind == "gyroscopic" else np.zeros((2, 2))
    stiffness = sp.kron(spatial_k, elastic, format="csr") + sp.kron(
        spatial_m, foundation, format="csr"
    )
    mass = sp.kron(spatial_m, inertia, format="csr")
    c, g = (sp.kron(spatial_m, block, format="csr") for block in (damping, gyro))
    identity = sp.eye(2 * nodes, format="csr")
    a = sp.bmat([[None, identity], [-stiffness, -(c + g)]], format="csr")
    b = sp.block_diag([identity, mass], format="csr")
    theta = np.arange(1, nodes + 1) * np.pi / (nodes + 1)
    kv = 4 / h * np.sin(theta / 2) ** 2
    mv = h * (2 / 3 + np.cos(theta) / 3)
    local_a, local_b = np.zeros((nodes, 4, 4)), np.zeros((nodes, 4, 4))
    local_a[:, :2, 2:] = np.eye(2)
    local_a[:, 2:, :2] = -(kv[:, None, None] * elastic + mv[:, None, None] * foundation)
    local_a[:, 2:, 2:] = -mv[:, None, None] * (damping + gyro)
    local_b[:, :2, :2] = np.eye(2)
    local_b[:, 2:, 2:] = mv[:, None, None] * inertia
    all_values = np.linalg.eigvals(np.linalg.solve(local_b, local_a)).ravel()
    selected = np.argsort(abs(all_values - target))[:k]
    right, left, values = [], [], []
    for flat in selected:
        mode = int(flat // 4)
        w, vl, vr = la.eig(local_a[mode], local_b[mode], left=True, right=True)
        j = int(np.argmin(abs(w - all_values[flat])))
        sine = np.sqrt(2 / (nodes + 1)) * np.sin(theta[mode] * np.arange(1, nodes + 1))

        def lift(vector):
            return np.r_[np.outer(sine, vector[:2]).ravel(), np.outer(sine, vector[2:]).ravel()]

        r, left_vector = lift(vr[:, j]), lift(vl[:, j])
        r /= la.norm(r)
        left_vector /= np.conj(np.vdot(left_vector, b @ r))
        values.append(w[j])
        right.append(r)
        left.append(left_vector)
    return DampedCase(
        "consistent-" + kind,
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
        b,
    )
