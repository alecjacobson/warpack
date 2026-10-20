"""Prescribed initial-stress pencils for clamped P1 tetrahedral beams.

Host fixture/reference assembly only. K is the linear elastic stiffness. G is
minus the initial-stress tangent for a prescribed uniform stress -S, so a
load multiplier lambda gives K-lambda*G. This exercises signed buckling load
factors; it is not a nonlinear equilibrium or prestress-generation solver.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from .fem_cases import beam_mesh, elastic_matrices
from .problems import Problem


@dataclass
class BucklingCase:
    problem: Problem
    vertices: np.ndarray
    tets: np.ndarray
    dofs: np.ndarray
    stress: np.ndarray
    reference: str
    reference_residual: float
    reference_orthogonality: float
    selection_margin: float
    shifted_spd: bool


def geometric_stiffness(vertices, tets, stress):
    """Assemble G_ia,jb=delta_ab sum_t V_t grad(N_i)^T S grad(N_j).

    Its quadratic energy is 1/2 sum_t V_t tr(grad(u) S grad(u)^T).
    S is symmetric and may be indefinite; no SPD assumption is made for G.
    """
    stress = np.asarray(stress, dtype=float)
    if (
        stress.shape != (3, 3)
        or not np.all(np.isfinite(stress))
        or not np.allclose(stress, stress.T)
    ):
        raise ValueError("stress must be finite symmetric 3x3")
    x = vertices[tets]
    augmented = np.concatenate([np.ones((*x.shape[:2], 1)), x], axis=2)
    gradients = np.linalg.inv(augmented)[:, 1:, :].transpose(0, 2, 1)
    volume = abs(np.linalg.det(x[:, 1:] - x[:, :1])) / 6
    scalar = (gradients @ stress @ gradients.transpose(0, 2, 1)) * volume[:, None, None]
    local = np.einsum("tij,ab->tiajb", scalar, np.eye(3)).reshape(-1, 12, 12)
    ids = (3 * tets[:, :, None] + np.arange(3)).reshape(-1, 12)
    rows = np.broadcast_to(ids[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(ids[:, None, :], local.shape).ravel()
    g = sp.csr_matrix((local.ravel(), (rows, cols)), shape=(3 * len(vertices),) * 2)
    g.eliminate_zeros()
    return g


def make_buckling_case(side=2, modes=6, sign=1, shift=0.01):
    """Target the lowest signed load factors through ARPACK buckling ordering.

    A small signed shift lies below the first factor on that side. Reference
    eigenpairs come from G x=mu K x, using the SPD K metric; lambda=1/mu.
    Native benchmark buckling solves use a different transformation and seed.
    """
    if side < 1 or modes < 1 or sign not in (-1, 1) or not np.isfinite(shift) or shift <= 0:
        raise ValueError("require side,modes>=1, sign=+/-1, finite shift>0")
    vertices, tets = beam_mesh(side)
    stiffness, _ = elastic_matrices(vertices, tets)
    stress = np.array([[1.0, 0.13, 0.07], [0.13, -0.6, 0.09], [0.07, 0.09, 0.2]])
    geometric = geometric_stiffness(vertices, tets, stress)
    dofs = np.flatnonzero(np.repeat(vertices[:, 0] > 0, 3))
    stiffness = stiffness[dofs, :][:, dofs]
    geometric = geometric[dofs, :][:, dofs]
    n = stiffness.shape[0]
    count = min(modes + 8, n - 2)
    if modes >= count:
        raise ValueError("need unrequested reference guard modes")
    if n <= 768:
        mu, basis = la.eigh(geometric.toarray(), stiffness.toarray())
        label = "Independent dense symmetric reciprocal pencil Gx=mu Kx"
    else:
        mu, basis = sla.eigsh(
            geometric,
            M=stiffness,
            k=count,
            which="LA" if sign > 0 else "SA",
            tol=1e-12,
            ncv=min(n, max(4 * count + 16, 96)),
            v0=np.random.default_rng(913).normal(size=n),
        )
        label = "Independent ARPACK regular reciprocal pencil Gx=mu Kx, seed913, tol1e-12, expanded basis"
    order = np.argsort(-sign * mu)
    mu, basis = mu[order], basis[:, order]
    k = modes
    while k < len(mu) - 1 and abs(mu[k] - mu[k - 1]) < 1e-10 * max(1, abs(mu[k])):
        k += 1
    if k >= len(mu) - 1 or np.any(sign * mu[: k + 1] <= 0):
        raise ValueError("requested side needs a complete same-sign boundary guard")
    sigma = sign * shift
    # For the selected extreme side, all eigenvalues of K^-1 G are on the
    # safe side of the pole iff 1-sigma*mu_extreme>0. Dense tests verify this
    # condition independently; the sparse reference supplies the extreme.
    shifted_spd = bool(1 - sigma * mu[0] > 0)
    if not shifted_spd:
        raise ValueError("shift crosses the first signed load factor")
    values, vectors = 1 / mu[:k], basis[:, :k]
    kv, gv = stiffness @ vectors, geometric @ vectors
    residual = np.linalg.norm(kv - gv * values, axis=0) / np.maximum(
        1, np.linalg.norm(kv, axis=0) + abs(values) * np.linalg.norm(gv, axis=0)
    )
    orth = float(abs(vectors.T @ kv - np.eye(k)).max())
    if residual.max() > 1e-9 or orth > 1e-9:
        raise RuntimeError("independent reciprocal reference failed physical gates")
    scores = abs(1 / (1 - sigma * mu))
    problem = Problem(
        name=f"prestress_buckling_{'positive' if sign > 0 else 'negative'}",
        api_class="G",
        a=stiffness,
        b=geometric,
        k=k,
        target=sigma,
        mode="buckling",
        expected=values,
        requested_basis=vectors,
        block_size=3,
        notes="Clamped tetrahedral linear elasticity, prescribed uniform indefinite initial-stress tangent; K SPD, G indefinite; K-orthonormal modes; transformed magnitude ordering; "
        + label,
    )
    return BucklingCase(
        problem,
        vertices,
        tets,
        dofs,
        stress,
        label,
        float(residual.max()),
        orth,
        float(scores[k - 1] - scores[k]),
        shifted_spd,
    )
