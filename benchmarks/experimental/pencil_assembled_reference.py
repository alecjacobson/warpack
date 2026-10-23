"""Reference the stored coefficients of the procedural constant-block chain.

This CPU diagnostic is separate from the library and the original benchmark
reference. It does not replace the benchmark's acceptance gate.
"""

from dataclasses import replace

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


def modal_blocks(matrix, nodes):
    diagonal = matrix[:2, :2].toarray()
    off_diagonal = matrix[:2, 2:4].toarray()
    adjacency = sp.diags([np.ones(nodes - 1), np.ones(nodes - 1)], [-1, 1], format="csr")
    reconstructed = sp.kron(sp.eye(nodes), diagonal) + sp.kron(adjacency, off_diagonal)
    if (matrix - reconstructed).nnz:
        raise ValueError("reference requires identical 2x2 diagonal/off-diagonal chain blocks")
    diagonal = diagonal.astype(np.longdouble)
    off_diagonal = off_diagonal.astype(np.longdouble)
    theta = (
        np.arange(1, nodes + 1, dtype=np.longdouble) * np.arccos(np.longdouble(-1)) / (nodes + 1)
    )
    # D + 2 cos(theta) E = (D + 2 E) - 4 sin(theta/2)^2 E.
    # Form D+2E from the actual stored coefficients in wider precision before
    # adding the small modal term. The unfused physical assembly may have lost
    # some small foundation entries; the original analytic reference does not.
    blocks = diagonal + 2 * off_diagonal - 4 * np.sin(theta[:, None, None] / 2) ** 2 * off_diagonal
    return blocks, theta


def assembled_reference(case):
    nodes = case.nodes
    identity = sp.eye(2 * nodes, format="csr")
    expected_a = sp.bmat(
        [[None, identity], [-case.stiffness, -(case.damping + case.gyroscopic)]], format="csr"
    )
    expected_b = sp.block_diag([identity, case.mass], format="csr")
    if case.b is None or (case.a - expected_a).nnz or (case.b - expected_b).nnz:
        raise ValueError("reference requires the original consistent-chain linearization")
    stiffness, theta = modal_blocks(case.stiffness, nodes)
    mass, _ = modal_blocks(case.mass, nodes)
    damping, _ = modal_blocks(case.damping + case.gyroscopic, nodes)
    a, b = np.zeros((nodes, 4, 4)), np.zeros((nodes, 4, 4))
    a[:, :2, 2:] = np.eye(2)
    b[:, :2, :2] = np.eye(2)
    # Row equilibration of each small reference pencil; no change of eigenpairs.
    a[:, 2:, :2] = -(nodes + 1) * stiffness
    a[:, 2:, 2:] = -(nodes + 1) * damping
    b[:, 2:, 2:] = (nodes + 1) * mass
    all_values = np.linalg.eigvals(np.linalg.solve(b, a)).ravel()
    selected = np.argsort(abs(all_values - case.target))[: case.k]
    values, right, left = [], [], []
    for flat in selected:
        mode = int(flat // 4)
        w, vl, vr = la.eig(a[mode], b[mode], left=True, right=True)
        j = int(np.argmin(abs(w - all_values[flat])))
        sine = (
            np.sqrt(np.longdouble(2) / (nodes + 1))
            * np.sin(theta[mode] * np.arange(1, nodes + 1, dtype=np.longdouble))
        ).astype(float)

        def lift(vector):
            return np.r_[np.outer(sine, vector[:2]).ravel(), np.outer(sine, vector[2:]).ravel()]

        r = lift(vr[:, j])
        r /= la.norm(r)
        # Undo the modal pencil's left row scaling before lifting.
        local_left = vl[:, j].copy()
        local_left[2:] *= nodes + 1
        left_vector = lift(local_left)
        left_vector /= np.conj(np.vdot(left_vector, case.b @ r))
        values.append(w[j])
        right.append(r)
        left.append(left_vector)
    return replace(
        case,
        name=case.name + "-assembled-coefficients",
        values=np.array(values),
        right=np.array(right).T,
        left=np.array(left).T,
        full_spectrum=all_values,
    )
