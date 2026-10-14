"""Host-side benchmark fixtures. These are references, not production assembly."""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


@dataclass
class Problem:
    name: str
    api_class: str
    a: object
    b: object
    k: int
    target: complex
    mode: str = "shift-invert"
    expected: object = None
    nullity: int = 0
    notes: str = ""

    @property
    def n(self):
        return self.a.shape[0]


def bar(n, consistent=True, free=False):
    """Uniform P1 axial FEM on a unit bar, with unit E*A and density*A."""
    elements = n - 1 if free else n + 1
    h = 1.0 / elements
    diag = np.full(elements + 1, 2.0)
    diag[[0, -1]] = 1.0
    stiffness = sp.diags([-np.ones(elements), diag, -np.ones(elements)], [-1, 0, 1]) / h
    if consistent:
        mass = sp.diags([np.ones(elements), 2 * diag, np.ones(elements)], [-1, 0, 1]) * (h / 6)
    else:
        mass = sp.diags(diag * h / 2)
    idx = slice(None) if free else slice(1, -1)
    a, b = stiffness.tocsr()[idx, idx], mass.tocsr()[idx, idx]
    j = np.arange(n) if free else np.arange(1, n + 1)
    theta = j * np.pi / elements
    w = 4 * np.sin(theta / 2) ** 2 / h**2
    if consistent:
        w /= (2 + np.cos(theta)) / 3
    return a, b, w


def torus_hodge(side):
    """Oriented triangular periodic grid, L1=B1.T B1+B2 B2.T, beta1=2."""

    def idx(i, j):
        return (i % side) * side + j % side

    faces = []
    for i in range(side):
        for j in range(side):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)
            faces.extend([(a, b, d), (a, d, c)])
    edges = sorted({tuple(sorted(e)) for f in faces for e in zip(f, f[1:] + f[:1])})
    lookup = {e: i for i, e in enumerate(edges)}
    rows, cols, vals = [], [], []
    for j, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            rows.append(lookup[tuple(sorted((a, b)))])
            cols.append(j)
            vals.append(1 if a < b else -1)
    b2 = sp.csr_matrix((vals, (rows, cols)), shape=(len(edges), len(faces)))
    b1 = sp.csr_matrix(
        (
            np.tile([-1.0, 1.0], len(edges)),
            (np.array(edges).ravel(), np.repeat(np.arange(len(edges)), 2)),
        ),
        shape=(side**2, len(edges)),
    )
    return (b1.T @ b1 + b2 @ b2.T).tocsr(), b1, b2


def conformal(side):
    """P1 Cauchy–Riemann energy on a planar grid and centered boundary norm."""
    xy = np.array([(i, j) for i in range(side) for j in range(side)], float) / (side - 1)
    n = len(xy)
    rows, cols, vals = [], [], []
    for i in range(side - 1):
        for j in range(side - 1):
            a = i * side + j
            for tri in [[a, a + side, a + side + 1], [a, a + side + 1, a + 1]]:
                x = xy[tri]
                twice_area = np.linalg.det(np.column_stack((x[1] - x[0], x[2] - x[0])))
                gx = (
                    np.array([x[1, 1] - x[2, 1], x[2, 1] - x[0, 1], x[0, 1] - x[1, 1]]) / twice_area
                )
                gy = (
                    np.array([x[2, 0] - x[1, 0], x[0, 0] - x[2, 0], x[1, 0] - x[0, 0]]) / twice_area
                )
                cr = np.array([np.r_[gx, -gy], np.r_[gy, gx]])
                local = abs(twice_area) / 2 * cr.T @ cr
                ids = np.r_[tri, np.array(tri) + n]
                rows.extend(np.repeat(ids, 6))
                cols.extend(np.tile(ids, 6))
                vals.extend(local.ravel())
    energy = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n))
    boundary = np.flatnonzero(np.any((xy == 0) | (xy == 1), axis=1))
    nb = len(boundary)
    rr, cc = np.repeat(boundary, nb), np.tile(boundary, nb)
    center = sp.csr_matrix((np.full(nb * nb, -1 / nb), (rr, cc)), shape=(n, n))
    center += sp.diags(np.isin(np.arange(n), boundary).astype(float))
    c = sp.block_diag([center, center], format="csr")
    return c, (energy + 1e-4 * sp.eye(2 * n)).tocsr()


def make_problem(api_class, size=80):
    k = 4
    if api_class in "ABC":
        a, b, w = bar(size)
        target = -1.0
        notes = "P1 clamped axial FEM; non-diagonal consistent mass"
        if api_class == "B":
            target = float((w[9] + w[10]) * 0.5 + 0.01)
        elif api_class == "C":
            a = a - 250 * b
            w = w - 250
            target = -251.0
            notes += "; negative tangent stiffness; target below spectrum selects most negative"
        expected = w[np.argsort(abs(w - target))[:k]]
        return Problem(
            f"bar_{api_class}", api_class, a, b, k, target, expected=expected, notes=notes
        )
    if api_class == "D":
        side = max(4, int(np.sqrt(size / 3)))
        a, _, _ = torus_hodge(side)
        return Problem(
            "torus_harmonics",
            "D",
            a,
            sp.eye(a.shape[0], format="csr"),
            2,
            -0.05,
            expected=np.zeros(2),
            nullity=2,
            notes="Triangulated torus 1-form combinatorial Hodge Laplacian; retain both harmonics",
        )
    if api_class == "E":
        n = size
        phase = 2 * np.pi * 0.23 / n
        row = np.arange(n)
        col = (row + 1) % n
        a = sp.csr_matrix((np.full(n, -np.exp(1j * phase)), (row, col)), shape=(n, n))
        a = a + a.conj().T + 2 * sp.eye(n)
        w = 4 * np.sin((2 * np.pi * np.arange(n) / n + phase) / 2) ** 2
        return Problem(
            "connection_ring",
            "E",
            a.tocsr(),
            None,
            k,
            -1 / n**2,
            expected=np.sort(w)[:k],
            notes="Complex Hermitian U(1) connection with nonintegral holonomy; standard mass",
        )
    if api_class == "F":
        a, b = conformal(max(4, int(np.sqrt(size / 2))))
        return Problem(
            "spectral_conformal",
            "F",
            a,
            b,
            2,
            0.0,
            mode="regular",
            notes="SCP reciprocal pencil C u=mu(Lc+1e-4 I)u; singular C, SPD metric; complete rotation pair",
        )
    if api_class == "G":
        n = size
        t = sp.diags([np.ones(n), np.full(n - 1, 0.2)], [0, 1], format="csr")
        d = np.linspace(1, 40, n)
        signs = np.where(np.arange(n) % 2, -1.0, 1.0)
        a = (t.T @ sp.diags(d) @ t).tocsr()
        b = (t.T @ sp.diags(signs) @ t).tocsr()
        w = d / signs
        target = 5.13
        score = abs(w / (w - target))
        return Problem(
            "coupled_buckling",
            "G",
            a,
            b,
            k,
            target,
            mode="buckling",
            expected=w[np.argsort(-score)[:k]],
            notes="Congruence-coupled SPD/indefinite pencil with independent analytic load factors",
        )
    if api_class == "H":
        n = max(16, size // 2)
        a, _, w = bar(n, consistent=False)
        # Diagonal interior lumped mass: stiffness divided by h gives the physical K.
        stiffness = a * (n + 1)
        damp = 0.2
        linearized = sp.bmat([[None, sp.eye(n)], [-stiffness, -damp * sp.eye(n)]], format="csr")
        freq = np.sqrt(w - damp**2 / 4)
        spectrum = np.r_[-damp / 2 + 1j * freq, -damp / 2 - 1j * freq]
        target = -0.12 + 1j * float(freq[1] + 0.1)
        return Problem(
            "damped_chain",
            "H",
            linearized.astype(complex),
            None,
            k,
            target,
            expected=spectrum[np.argsort(abs(spectrum - target))[:k]],
            notes="Standard first-order linearization of uniformly damped coupled oscillators; right modes",
        )
    if api_class == "I":
        n = size
        m = min(48, n // 2)
        rng = np.random.default_rng(2026)
        u = la.qr(rng.normal(size=(n, m)), mode="economic")[0]
        v = la.qr(rng.normal(size=(m, m)), mode="economic")[0]
        w = np.geomspace(20, 0.05, m)
        w[-2:] = 0
        a = sp.csr_matrix((u * w) @ v.T)
        return Problem(
            "snapshot_pod",
            "I",
            a,
            None,
            k,
            0.0,
            mode="svd",
            expected=w[:k],
            notes="Dense correlated tall snapshots with known singular spectrum and rank deficiency",
        )
    raise ValueError(api_class)
