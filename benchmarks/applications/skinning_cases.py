"""Independent tetrahedral assembly and Appendix B.1 rig constraints (reference only)."""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from .fem_cases import beam_mesh, elastic_matrices


@dataclass
class SkinningCase:
    h: object
    m: object
    c: object
    vertices: np.ndarray
    tets: np.ndarray
    elastic_hessian: object
    effective_jacobian: np.ndarray
    raw_constraints: np.ndarray
    independent_rows: np.ndarray
    shift: float = -0.1

    @property
    def n(self):
        return self.h.shape[0]

    def kkt(self):
        return sp.bmat([[self.h - self.shift * self.m, self.c.T], [self.c, None]], format="csc")


def make_skinning_case(side=3, heterogeneous=True):
    vertices, tets = beam_mesh(side)
    elastic, vector_mass = elastic_matrices(
        vertices, tets, consistent=False, heterogeneous=heterogeneous
    )
    h = sum(elastic[i::3, i::3] for i in range(3)).tocsr()
    m = vector_mass[::3, ::3].tocsr()
    # Two localized angular rig parameters, with explicit mass and leakage.
    center = vertices - np.array([1.5, 0.5, 0.5])
    blend = 1 / (1 + np.exp(3 * center[:, 0]))
    rig = np.stack(
        [
            blend[:, None] * np.cross([0, 0, 1], center),
            (1 - blend[:, None]) * np.cross([0, 1, 0], center),
        ]
    )
    leakage = 0.2 + 0.8 * np.sin(0.6 + vertices[:, 0]) ** 2
    effective = rig * (m.diagonal() * leakage)[None, :, None]
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    raw = np.einsum("pna,nc->pacn", effective, homogeneous).reshape(-1, len(vertices))
    # Reference/input preprocessing only. The solver never constructs a dense Z.
    _, triangular, pivots = la.qr(raw.T, mode="economic", pivoting=True)
    rank = np.count_nonzero(abs(triangular.diagonal()) > abs(triangular[0, 0]) * 1e-11)
    selected = pivots[:rank]
    c = raw[selected]
    c = sp.csr_matrix(c / la.norm(c, axis=1)[:, None])
    return SkinningCase(h, m, c, vertices, tets, elastic, effective, raw, selected)


def reference(case, k):
    """Dense elimination for small cases; independent enlarged ARPACK for large ones."""
    if case.n <= 400:
        z = la.null_space(case.c.toarray())
        values, v = la.eigh(z.T @ (case.h @ z), z.T @ (case.m @ z))
        selected = np.argsort(abs(values - case.shift))[:k]
        return values[selected], z @ v[:, selected], "dense nullspace elimination + LAPACK eigh"
    factor = sla.splu(case.kkt())
    n, r = case.n, case.c.shape[0]
    scale = np.sqrt(case.m.diagonal())

    def apply(x):
        return scale * factor.solve(np.r_[scale * x, np.zeros(r)])[:n]

    transformed = sla.LinearOperator((n, n), matvec=apply, dtype=np.float64)
    w, v = sla.eigsh(
        transformed,
        k=k,
        which="LM",
        tol=2e-13,
        ncv=min(n, max(4 * k + 32, 96)),
        v0=np.random.default_rng(913).normal(size=n),
    )
    return (
        case.shift + 1 / w,
        v / scale[:, None],
        "independent enlarged ARPACK KKT transform (seed 913)",
    )


def validate(case, values, vectors, reference_values, reference_vectors, multipliers=None):
    h, m, c = case.h, case.m, case.c
    if multipliers is None:
        multipliers = la.solve(
            (c @ c.T).toarray(), c @ ((m @ vectors) * values - h @ vectors), assume_a="pos"
        )
    force = c.T @ multipliers
    hv, mv = h @ vectors, m @ vectors
    residual = la.norm(hv + force - mv * values, axis=0) / np.maximum(
        1, la.norm(hv, axis=0) + la.norm(force, axis=0) + abs(values) * la.norm(mv, axis=0)
    )
    constraint = abs(c @ vectors) / (
        np.sqrt(c.multiply(c).sum(axis=1)).A * la.norm(vectors, axis=0)
    )
    raw_constraint = abs(case.raw_constraints @ vectors)
    error = np.max(
        abs(np.sort(values) - np.sort(reference_values))
        / np.maximum(1, abs(np.sort(reference_values)))
    )
    ortho = la.norm(vectors.T @ mv - np.eye(len(values)), ord=2)
    subspace = np.max(abs(la.svdvals(reference_vectors.T @ mv) - 1))
    result = dict(
        equilibrium_residual=float(residual.max()),
        constraint_residual=float(constraint.max()),
        raw_rig_constraint_max=float(raw_constraint.max()),
        eigenvalue_error=float(error),
        mass_orthogonality=float(ortho),
        subspace_error=float(subspace),
    )
    result["accepted"] = bool(
        all(np.isfinite(v) for v in result.values())
        and (
            max(
                result["equilibrium_residual"],
                result["constraint_residual"],
                error,
                ortho,
                subspace,
            )
            < 1e-7
        )
    )
    return result
