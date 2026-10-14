"""Deterministic chain-complex fixtures and independent topology references."""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .problems import Problem, torus_hodge


@dataclass
class HodgeCase:
    problem: Problem
    boundary1: object
    boundary2: object
    rank1: int
    rank2: int
    spectrum: np.ndarray
    vectors: object
    reference: str


def incidence(faces):
    vertices = sorted({v for f in faces for v in f})
    vertex = {v: i for i, v in enumerate(vertices)}
    faces = [tuple(vertex[v] for v in f) for f in faces]
    edges = sorted({tuple(sorted(e)) for f in faces for e in zip(f, f[1:] + f[:1])})
    lookup = {e: i for i, e in enumerate(edges)}
    rows, cols, values = [], [], []
    for j, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            rows.append(lookup[tuple(sorted((a, b)))])
            cols.append(j)
            values.append(1.0 if a < b else -1.0)
    b2 = sp.csr_matrix((values, (rows, cols)), shape=(len(edges), len(faces)))
    b1 = sp.csr_matrix(
        (
            np.tile([-1.0, 1.0], len(edges)),
            (np.array(edges).ravel(), np.repeat(np.arange(len(edges)), 2)),
        ),
        shape=(len(vertices), len(edges)),
    )
    return b1, b2


def torus_spectrum(side):
    """L1 nonzeros are the union of primal L0 and dual L2 nonzeros."""
    theta, phi = np.meshgrid(2 * np.pi * np.arange(side) / side, 2 * np.pi * np.arange(side) / side)
    l0 = (6 - 2 * np.cos(theta) - 2 * np.cos(phi) - 2 * np.cos(theta + phi)).ravel()
    dual = abs(1 + np.exp(1j * theta) + np.exp(1j * phi)).ravel()
    return np.sort(np.r_[0.0, 0.0, l0[1:], (3 - dual)[1:], 3 + dual])


def torus_harmonics(b1, side):
    """Constant closed/co-closed coordinate cochains, in the stored edge orientation."""
    edges = b1.tocsc().indices.reshape(-1, 2)
    delta = np.column_stack(
        (edges[:, 1] // side - edges[:, 0] // side, edges[:, 1] % side - edges[:, 0] % side)
    ).astype(float)
    delta[delta > side / 2] -= side
    delta[delta < -side / 2] += side
    return delta


def make_hodge_case(name, side=4, weak_scale=1e-4):
    if side < 3:
        raise ValueError("A periodic triangulation needs side >= 3")

    def torus():
        return torus_hodge(side)[1:]

    if name == "sphere":
        parts = [incidence([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)])]
    elif name == "disk":
        parts = [incidence([(0, 1, 2), (0, 2, 3)])]
    elif name == "annulus":
        parts = [
            incidence(
                [
                    f
                    for i in range(4)
                    for f in [(i, (i + 1) % 4, (i + 1) % 4 + 4), (i, (i + 1) % 4 + 4, i + 4)]
                ]
            )
        ]
    elif name in ("torus", "disconnected", "large_disconnected"):
        parts = [torus()] * (1 if name == "torus" else 2)
    elif name == "weak_positive":
        ring = incidence(
            [
                f
                for i in range(4)
                for f in [(i, (i + 1) % 4, (i + 1) % 4 + 4), (i, (i + 1) % 4 + 4, i + 4)]
            ]
        )
        parts = [torus(), ring]
    else:
        raise ValueError(name)
    weights = [1.0] * len(parts)
    if name == "weak_positive":
        weights[-1] = weak_scale
    stiffness, mass = [], []
    for (b1, b2), weight in zip(parts, weights):
        n = b1.shape[1]
        stiffness.append(weight * (b1.T @ b1 + b2 @ b2.T))
        t = sp.eye(n, format="csr")
        if name in ("disconnected", "weak_positive"):
            t += sp.diags(np.full(n - 1, 0.05), 1, shape=(n, n), format="csr")
        mass.append(t.T @ t)
    a, b = sp.block_diag(stiffness, format="csr"), sp.block_diag(mass, format="csr")
    b1 = sp.block_diag([x[0] for x in parts], format="csr")
    b2 = sp.block_diag([x[1] for x in parts], format="csr")
    if name == "large_disconnected":
        rank1, rank2 = 2 * (side * side - 1), 2 * (2 * side * side - 1)
        harmonics = la.block_diag(*[torus_harmonics(x[0], side) for x in parts])
        values = np.sort(np.tile(torus_spectrum(side), 2))
        vectors = None
        reference = "Two oriented closed tori: incidence ranks V-2 and F-2; independent Fourier spectra and coordinate harmonic cochains"
    else:
        rank1, rank2 = np.linalg.matrix_rank(b1.toarray()), np.linalg.matrix_rank(b2.toarray())
        harmonics = la.null_space(sp.vstack([b1, b2.T]).toarray())
        values, vectors = la.eigh(a.toarray(), b.toarray())
        reference = "Independent incidence SVD ranks/nullspace; dense generalized spectrum"
    beta = a.shape[0] - rank1 - rank2
    if beta:
        gram = harmonics.T @ (b @ harmonics)
        harmonics = la.solve_triangular(la.cholesky(gram, lower=True), harmonics.T, lower=True).T
    shift = -min(1e-3, weak_scale * 0.1) if name == "weak_positive" else -1e-3
    p = Problem(
        "hodge_" + name,
        "D",
        a,
        b,
        int(beta) + 2,
        shift,
        expected=values[: int(beta) + 2],
        nullity=int(beta),
        null_basis=harmonics,
        notes=f"{reference}; beta1={beta}; retain all zeros and two positive guard modes; fixed k is not a certified Betti counter",
    )
    return HodgeCase(p, b1, b2, int(rank1), int(rank2), values, vectors, reference)
