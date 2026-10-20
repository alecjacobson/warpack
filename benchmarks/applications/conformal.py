"""Unregularized P1 conformal pencils with a singular centered boundary mass.

Host fixture/reference assembly only. The one-vertex gauge removes translation,
not shape degrees of freedom: both original forms annihilate translations.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


@dataclass
class ConformalCase:
    xyz: np.ndarray
    faces: np.ndarray
    boundary: np.ndarray
    free: np.ndarray
    stiffness: object
    mass: object
    k: int
    shift: float
    height: float

    @property
    def n(self):
        return len(self.free)

    @property
    def reduced(self):
        idx = self.free
        return self.stiffness[idx][:, idx], self.mass[idx][:, idx]

    def lift(self, vectors):
        full = np.zeros((self.stiffness.shape[0], vectors.shape[1]))
        full[self.free] = vectors
        n = len(self.xyz)
        for offset in (0, n):
            full[offset : offset + n] -= full[self.boundary + offset].mean(axis=0)
        return full


def make_conformal_case(side=8, k=6, height=0.0, pin=0, shift=None):
    if side < 3 or k < 2 or k % 2 or k >= 8 * (side - 1) - 2:
        raise ValueError("side>=3 and an even k below the finite spectrum size are required")
    if not 0 <= pin < side**2 or not np.isfinite(height):
        raise ValueError("invalid translation gauge or height")
    if shift is None:
        shift = -1.0 / (side - 1) ** 3
    if not np.isfinite(shift) or shift >= 0:
        raise ValueError("shift must be finite and strictly negative")
    xy = np.indices((side, side)).reshape(2, -1).T / (side - 1)
    z = height * np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])
    xyz = np.column_stack((xy, z))
    a = (np.arange(side - 1)[:, None] * side + np.arange(side - 1)).ravel()
    faces = np.concatenate(
        (np.array([a, a + side, a + side + 1]).T, np.array([a, a + side + 1, a + 1]).T)
    )
    triangles = xyz[faces]
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    length = la.norm(e1, axis=1)
    area2 = la.norm(np.cross(e1, e2), axis=1)
    tx, ty = np.sum(e1 * e2, axis=1) / length, area2 / length
    gx = np.column_stack((-1 / length, 1 / length, np.zeros_like(length)))
    gy = np.column_stack(((tx / length - 1) / ty, -tx / (length * ty), 1 / ty))
    cr = np.stack((np.concatenate((gx, -gy), axis=1), np.concatenate((gy, gx), axis=1)), axis=1)
    local = np.einsum("tqi,tqj,t->tij", cr, cr, area2 / 2)
    n = len(xyz)
    ids = np.concatenate((faces, faces + n), axis=1)
    stiffness = sp.coo_matrix(
        (local.ravel(), (np.repeat(ids, 6, axis=1).ravel(), np.tile(ids, (1, 6)).ravel())),
        shape=(2 * n, 2 * n),
    ).tocsr()
    stiffness.eliminate_zeros()
    boundary = np.flatnonzero(np.any((xy == 0) | (xy == 1), axis=1))
    nb = len(boundary)
    centered = sp.coo_matrix(
        (np.full(nb * nb, -1 / nb), (np.repeat(boundary, nb), np.tile(boundary, nb))), shape=(n, n)
    )
    centered = centered.tocsr() + sp.diags(np.isin(np.arange(n), boundary).astype(float))
    mass = sp.block_diag((centered, centered), format="csr")
    free = np.delete(np.arange(2 * n), (pin, pin + n))
    return ConformalCase(xyz, faces, boundary, free, stiffness, mass, k, shift, height)


def independent_energy(case, vectors):
    """3D gradients and oriented surface Jacobians, independent of CR assembly."""
    n = len(case.xyz)
    triangles = case.xyz[case.faces]
    edges = triangles[:, 1:] - triangles[:, :1]
    gram = edges @ np.swapaxes(edges, 1, 2)
    gradients = np.einsum("tij,tjk->tik", np.linalg.inv(gram), edges)
    u, v = vectors[:n][case.faces], vectors[n:][case.faces]
    gu = np.einsum("tij,tik->tjk", gradients, u[:, 1:] - u[:, :1])
    gv = np.einsum("tij,tik->tjk", gradients, v[:, 1:] - v[:, :1])
    normals = np.cross(edges[:, 0], edges[:, 1])
    area2 = la.norm(normals, axis=1)
    jacobian = np.einsum(
        "tki,ti->tk", np.cross(gu.swapaxes(1, 2), gv.swapaxes(1, 2)), normals / area2[:, None]
    )
    energy = np.sum(area2[:, None] / 2 * (np.sum(gu * gu + gv * gv, axis=1) - 2 * jacobian), axis=0)
    boundary = np.concatenate((vectors[:n][case.boundary], vectors[n:][case.boundary]), axis=0)
    boundary = boundary.reshape(2, len(case.boundary), -1)
    norm = np.sum((boundary - boundary.mean(axis=1, keepdims=True)) ** 2, axis=(0, 1))
    return energy, norm


def dense_reference(case):
    """Homogeneous QZ distinguishes infinite modes; no SPD-mass assumption."""
    stiffness, mass = (m.toarray() for m in case.reduced)
    ab, vectors = la.eig(stiffness, mass, homogeneous_eigvals=True)
    finite = abs(ab[1]) > 1e-10 * np.maximum(1, abs(ab[0]))
    values = ab[0, finite] / ab[1, finite]
    if np.max(abs(values.imag)) > 1e-9:
        raise ValueError("nonreal finite reference spectrum")
    order = np.argsort(values.real)
    basis = vectors[:, finite][:, order[: case.k]].real
    # QZ need not return an orthogonal basis for repeated eigenvalues.
    w, rotation = la.eigh(basis.T @ stiffness @ basis, basis.T @ mass @ basis)
    full = case.lift(basis @ rotation)
    return w, full, int(np.count_nonzero(~finite))


def boundary_reference(case):
    """Independent static condensation, then a dense centered-boundary solve.

    Only the interior Dirichlet stiffness is factored. No reciprocal shifted
    solve or full-volume dense matrix is used. This also verifies the harmonic
    extension, which a boundary seminorm alone cannot measure.
    """
    n, nb = len(case.xyz), len(case.boundary)
    boundary = np.r_[case.boundary, case.boundary + n]
    interior = np.setdiff1d(np.arange(2 * n), boundary)
    energy = case.stiffness
    factor = sla.splu(energy[interior][:, interior].tocsc())
    extension = -factor.solve(energy[interior][:, boundary].toarray())
    schur = energy[boundary][:, boundary].toarray() + energy[boundary][:, interior] @ extension
    q = la.null_space(np.ones((1, nb)))
    q = la.block_diag(q, q)
    reduced = q.T @ schur @ q
    w, v = la.eigh(
        (reduced + reduced.T) / 2, subset_by_index=(0, min(case.k + 3, len(reduced) - 1))
    )
    count = max(case.k, np.count_nonzero(w <= w[case.k - 1] + 1e-10))
    basis = q @ v[:, :count]
    full = np.zeros((2 * n, count))
    full[boundary], full[interior] = basis, extension @ basis
    return w[: case.k], full


def metrics(case, values, vectors, reference):
    """Original full equations, centered gauge and boundary-metric subspaces."""
    w, basis = reference
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    kv, mv = case.stiffness @ vectors, case.mass @ vectors
    residual = la.norm(kv - mv * values, axis=0) / np.maximum(
        1, la.norm(kv, axis=0) + abs(values) * la.norm(mv, axis=0)
    )
    gram = vectors.T @ mv
    remainder = vectors - basis @ (basis.T @ mv)
    # Interior differences matter too: a semidefinite mass is not a norm there.
    subspace = la.norm(remainder) / max(1, la.norm(vectors))
    n = len(case.xyz)
    centered = max(
        abs(vectors[case.boundary].mean(axis=0)).max(),
        abs(vectors[case.boundary + n].mean(axis=0)).max(),
    )
    result = dict(
        residual=float(residual.max()),
        orthogonality=float(abs(gram - np.eye(case.k)).max()),
        spectrum_error=float(np.max(abs(values - w) / np.maximum(1, abs(w)))),
        subspace_error=float(subspace),
        centered_boundary_error=float(centered),
    )
    result["accepted"] = all(np.isfinite(x) and x < 1e-8 for x in result.values())
    return result
