"""Host reference fixtures for low elastic modes; not production assembly."""

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from .problems import Problem


@dataclass
class FEMCase:
    problem: Problem
    spectrum: np.ndarray
    vectors: np.ndarray
    vertices: object
    tets: object
    dofs: object
    reference: str
    nontrivial_modes: int


def beam_mesh(side):
    """Conforming six-tet cube split of a 3:1:1 rectangular beam."""
    shape = (3 * side + 1, side + 1, side + 1)
    vertices = np.array(list(np.ndindex(shape)), dtype=float) / side
    cells = np.array(list(np.ndindex((3 * side, side, side))))
    tets = []
    for order in permutations(range(3)):
        offsets = np.vstack(
            [np.zeros(3, dtype=int), np.cumsum(np.eye(3, dtype=int)[list(order)], axis=0)]
        )
        indices = cells[:, None, :] + offsets[None, :, :]
        tets.append(np.ravel_multi_index(indices.transpose(2, 0, 1), shape))
    return vertices, np.concatenate(tets)


def elastic_matrices(vertices, tets, consistent=True, heterogeneous=False):
    """Independent P1 strain-displacement assembly with engineering shear strains."""
    x = vertices[tets]
    augmented = np.concatenate([np.ones((*x.shape[:2], 1)), x], axis=2)
    gradients = np.linalg.inv(augmented)[:, 1:, :].transpose(0, 2, 1)
    volume = abs(np.linalg.det(x[:, 1:] - x[:, :1])) / 6
    young = np.full(len(tets), 10.0)
    density = np.ones(len(tets))
    if heterogeneous:
        center = x.mean(axis=1)
        young *= np.where(center[:, 0] > 1.5, 20.0, 1.0)
        density *= np.where(center[:, 1] > 0.5, 3.0, 1.0)
    poisson = 0.3
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    strain = np.zeros((len(tets), 6, 12))
    for i in range(4):
        gx, gy, gz = gradients[:, i, :].T
        strain[:, 0, 3 * i] = gx
        strain[:, 1, 3 * i + 1] = gy
        strain[:, 2, 3 * i + 2] = gz
        strain[:, 3, 3 * i : 3 * i + 2] = np.stack([gy, gx], axis=1)
        strain[:, 4, 3 * i + 1 : 3 * i + 3] = np.stack([gz, gy], axis=1)
        strain[:, 5, 3 * i] = gz
        strain[:, 5, 3 * i + 2] = gx
    material = np.zeros((len(tets), 6, 6))
    material[:, :3, :3] = lam[:, None, None]
    for j in range(6):
        material[:, j, j] += (2 if j < 3 else 1) * mu
    ke = strain.transpose(0, 2, 1) @ material @ strain * volume[:, None, None]
    nodal_mass = (np.ones((4, 4)) + np.eye(4)) / 20 if consistent else np.eye(4) / 4
    me = np.kron(nodal_mass, np.eye(3))[None, :, :] * (density * volume)[:, None, None]
    ids = (3 * tets[:, :, None] + np.arange(3)).reshape(-1, 12)
    rows = np.broadcast_to(ids[:, :, None], ke.shape).ravel()
    cols = np.broadcast_to(ids[:, None, :], ke.shape).ravel()
    shape = (3 * len(vertices),) * 2
    a = sp.csr_matrix((ke.ravel(), (rows, cols)), shape=shape)
    b = sp.csr_matrix((me.ravel(), (rows, cols)), shape=shape)
    a.eliminate_zeros()
    b.eliminate_zeros()
    return a, b


def rigid_modes(vertices, mass):
    centered = vertices - vertices.mean(axis=0)
    q = np.column_stack(
        [(np.broadcast_to(e, vertices.shape)).ravel() for e in np.eye(3)]
        + [np.cross(e, centered).ravel() for e in np.eye(3)]
    )
    lower = la.cholesky(q.T @ (mass @ q), lower=True)
    return la.solve_triangular(lower, q.T, lower=True).T


def bending_matrices(elements):
    """Two transverse polarizations of a clamped Euler-Bernoulli beam, EI=rho*A=L=1."""
    h = 1.0 / elements
    ke = (
        np.array(
            [
                [12, 6 * h, -12, 6 * h],
                [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                [-12, -6 * h, 12, -6 * h],
                [6 * h, 2 * h * h, -6 * h, 4 * h * h],
            ]
        )
        / h**3
    )
    me = (
        np.array(
            [
                [156, 22 * h, 54, -13 * h],
                [22 * h, 4 * h * h, 13 * h, -3 * h * h],
                [54, 13 * h, 156, -22 * h],
                [-13 * h, -3 * h * h, -22 * h, 4 * h * h],
            ]
        )
        * h
        / 420
    )
    ids = 2 * np.arange(elements)[:, None] + np.arange(4)
    rows = np.broadcast_to(ids[:, :, None], (elements, 4, 4)).ravel()
    cols = np.broadcast_to(ids[:, None, :], (elements, 4, 4)).ravel()
    shape = (2 * (elements + 1),) * 2
    a = sp.csr_matrix((np.tile(ke.ravel(), elements), (rows, cols)), shape=shape)[2:, 2:]
    b = sp.csr_matrix((np.tile(me.ravel(), elements), (rows, cols)), shape=shape)[2:, 2:]
    return a, b


def reference(a, b, count):
    if a.shape[0] <= 768:
        return (*la.eigh(a.toarray(), b.toarray()), "Dense generalized LAPACK")
    count = min(count, a.shape[0] - 2)
    w, v = sla.eigsh(
        a,
        M=b,
        k=count,
        sigma=-0.01,
        which="LM",
        tol=1e-12,
        ncv=min(a.shape[0], max(4 * count + 16, 80)),
        v0=np.random.default_rng(913).normal(size=a.shape[0]),
    )
    order = np.argsort(w)
    return (
        w[order],
        v[:, order],
        "Independent ARPACK shift-invert reference, seed 913, tol 1e-12, expanded basis",
    )


def make_fem_case(name, side=2, modes=6):
    if modes < 1:
        raise ValueError("Request at least one nontrivial mode")
    if name == "repeated_bending":
        single_a, single_b = bending_matrices(max(8, 4 * side))
        a, b = (
            sp.block_diag([single_a] * 2, format="csr"),
            sp.block_diag([single_b] * 2, format="csr"),
        )
        single_w, single_v, ref = reference(single_a, single_b, modes + 8)
        spectrum = np.repeat(single_w, 2)
        vectors = np.zeros((a.shape[0], 2 * single_v.shape[1]))
        vectors[: single_a.shape[0], 0::2] = single_v
        vectors[single_a.shape[0] :, 1::2] = single_v
        vertices = tets = dofs = None
        q = np.zeros((a.shape[0], 0))
        nullity = 0
    else:
        if name not in (
            "clamped_lumped",
            "clamped_consistent",
            "free",
            "disconnected",
            "heterogeneous",
        ):
            raise ValueError(name)
        vertices, tets = beam_mesh(side)
        a, b = elastic_matrices(
            vertices,
            tets,
            consistent=name != "clamped_lumped",
            heterogeneous=name == "heterogeneous",
        )
        free = name in ("free", "disconnected")
        dofs = np.arange(a.shape[0]) if free else np.flatnonzero(np.repeat(vertices[:, 0] > 0, 3))
        a, b = a[dofs, :][:, dofs], b[dofs, :][:, dofs]
        q = rigid_modes(vertices, b) if free else np.zeros((a.shape[0], 0))
        nullity = 6 if free else 0
        spectrum, vectors, ref = reference(a, b, nullity + modes + 8)
        if name == "disconnected":
            a, b = sp.block_diag([a] * 2, format="csr"), sp.block_diag([b] * 2, format="csr")
            q = la.block_diag(q, q)
            basis = la.block_diag(vectors, vectors)
            order = np.argsort(np.tile(spectrum, 2), kind="stable")
            spectrum, vectors = np.tile(spectrum, 2)[order], basis[:, order]
            nullity *= 2
    k = nullity + modes
    if k >= len(spectrum) - 1:
        raise ValueError("Reference capacity must include a positive guard above requested modes")
    # Include the entire cluster at the requested boundary, never silently split repeated modes.
    while abs(spectrum[k] - spectrum[k - 1]) < 1e-8 * max(1, abs(spectrum[k])):
        k += 1
        if k >= len(spectrum) - 1:
            raise ValueError("Insufficient reference capacity for complete boundary cluster")
    p = Problem(
        "fem_" + name,
        "A",
        a,
        b,
        k,
        -0.01,
        expected=spectrum[:k],
        nullity=nullity,
        null_basis=q,
        requested_basis=vectors[:, :k],
        notes=f"{ref}; retain {nullity} known rigid modes and {k - nullity} nontrivial modes; ROM caller discards the known rigid block; full boundary cluster included",
    )
    return FEMCase(p, spectrum, vectors, vertices, tets, dofs, ref, k - nullity)
