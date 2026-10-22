"""Procedural spectral reconstruction pencil for #136; host benchmark assembly.

Inspired by the objective in Braune et al. (2026), Section 4.2. This is an
independent P1 discretization, not a reproduction of the authors' implementation.
The regularization metric is SPD but is not a physical FEM mass matrix.
"""

from dataclasses import dataclass
from itertools import combinations, permutations
from operator import index

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


@dataclass
class SurfacePencil:
    a: object
    b: object
    laplacian: object
    lumped_mass: np.ndarray
    samples: object
    gradients: object
    normals: np.ndarray
    points: np.ndarray
    vertices: np.ndarray
    tets: np.ndarray
    radii: np.ndarray
    parameters: dict


def cube_tets(side, extent=1.2):
    side = index(side)
    if side < 2 or not np.isfinite(extent) or extent <= 0:
        raise ValueError("side>=2 and positive finite extent required")
    shape = (side + 1,) * 3
    vertices = np.array(list(np.ndindex(shape)), dtype=float) * (2 * extent / side) - extent
    cells = np.array(list(np.ndindex((side,) * 3)))
    tets = []
    for order in permutations(range(3)):
        offsets = np.vstack([np.zeros(3, int), np.eye(3, dtype=int)[list(order)].cumsum(axis=0)])
        indices = cells[:, None, :] + offsets[None, :, :]
        tets.append(np.ravel_multi_index(indices.transpose(2, 0, 1), shape))
    return vertices, np.concatenate(tets)


def sample_operators(points, side, extent=1.2):
    """P1 evaluation and gradients in the containing Freudenthal tetrahedron."""
    side = index(side)
    if side < 2 or not np.isfinite(extent) or extent <= 0:
        raise ValueError("side>=2 and positive finite extent required")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("finite 3D sample points required")
    if np.any(np.abs(points) >= extent):
        raise ValueError("samples must lie strictly inside the background cube")
    h = 2 * extent / side
    grid = (points + extent) / h
    cell = np.floor(grid).astype(int)
    local = grid - cell
    order = np.argsort(-local, axis=1, kind="stable")
    offsets = np.concatenate(
        [np.zeros((len(points), 1, 3), int), np.eye(3, dtype=int)[order].cumsum(axis=1)], axis=1
    )
    corners = cell[:, None, :] + offsets
    ids = np.ravel_multi_index(corners.transpose(2, 0, 1), (side + 1,) * 3)
    xyz = corners * h - extent
    affine = np.concatenate([np.ones((*xyz.shape[:2], 1)), xyz], axis=2)
    inverse = np.linalg.inv(affine)
    bary = np.einsum("si,sij->sj", np.column_stack([np.ones(len(points)), points]), inverse)
    samples = sp.csr_matrix(
        (bary.ravel(), (np.repeat(np.arange(len(points)), 4), ids.ravel())),
        shape=(len(points), (side + 1) ** 3),
    )
    # Rows are sample-major xyz gradients.
    gradients = sp.csr_matrix(
        (
            inverse[:, 1:, :].ravel(),
            (
                np.repeat(np.arange(3 * len(points)), 4),
                np.repeat(ids[:, None, :], 3, axis=1).ravel(),
            ),
        ),
        shape=(3 * len(points), (side + 1) ** 3),
    )
    return samples, gradients


def surface_pencil(
    side=6,
    *,
    sample_count=None,
    radii=(0.8, 0.65, 0.5),
    alpha=100.0,
    beta=0.01,
    gamma=0.001,
    anisotropy=10.0,
):
    """A=<G.T (I+eta*n*n.T) G>, B=alpha<S.T*S>+beta*L+gamma*L.T*M^-1*L.

    M is explicitly lumped here, making the biharmonic term sparse. On this
    connected grid beta>0 and at least one point anchor with alpha>0 remove
    the constant nullspace. Singular, unanchored metrics are rejected.
    """
    side = index(side)
    sample_count = max(64, 6 * side**2) if sample_count is None else index(sample_count)
    radii = np.asarray(radii, dtype=float)
    if radii.shape != (3,) or not np.isfinite(radii).all() or np.any((radii <= 0) | (radii >= 1.2)):
        raise ValueError("three positive radii strictly inside the cube required")
    params = dict(alpha=alpha, beta=beta, gamma=gamma, anisotropy=anisotropy)
    if (
        not all(np.isfinite(v) for v in params.values())
        or alpha <= 0
        or beta <= 0
        or gamma < 0
        or anisotropy < 0
        or sample_count < 4
    ):
        raise ValueError(
            "positive anchoring/gradient weights and nonnegative regularization required"
        )
    vertices, tets = cube_tets(side)
    x = vertices[tets]
    affine = np.concatenate([np.ones((*x.shape[:2], 1)), x], axis=2)
    gradients = np.linalg.inv(affine)[:, 1:, :]
    volumes = np.abs(np.linalg.det(x[:, 1:] - x[:, :1])) / 6
    local = gradients.transpose(0, 2, 1) @ gradients * volumes[:, None, None]
    rows = np.broadcast_to(tets[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(tets[:, None, :], local.shape).ravel()
    laplacian = sp.csr_matrix((local.ravel(), (rows, cols)), shape=(len(vertices),) * 2)
    mass = np.bincount(tets.ravel(), weights=np.repeat(volumes / 4, 4), minlength=len(vertices))
    i = np.arange(sample_count)
    z = 1 - 2 * (i + 0.5) / sample_count
    theta = np.pi * (3 - np.sqrt(5)) * i
    unit = np.column_stack(
        [np.sqrt(1 - z * z) * np.cos(theta), np.sqrt(1 - z * z) * np.sin(theta), z]
    )
    points = unit * radii
    normals = points / radii**2
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    samples, sampled_gradients = sample_operators(points, side)
    c = np.eye(3)[None, :, :] + anisotropy * normals[:, :, None] * normals[:, None, :]
    c = sp.block_diag(c, format="csr")
    a = (sampled_gradients.T @ c @ sampled_gradients) / sample_count
    b = alpha * (samples.T @ samples) / sample_count + beta * laplacian
    b += gamma * (laplacian.T @ sp.diags(1 / mass) @ laplacian)
    a, b = ((m + m.T).tocsr() * 0.5 for m in (a, b))
    for matrix in (a, b, laplacian):
        matrix.eliminate_zeros()
        matrix.sort_indices()
    return SurfacePencil(
        a,
        b,
        laplacian,
        mass,
        samples,
        sampled_gradients,
        normals,
        points,
        vertices,
        tets,
        radii,
        params | dict(side=side, sample_count=sample_count),
    )


def dense_reference(pencil, k=1):
    k = index(k)
    if not 0 < k < pencil.a.shape[0]:
        raise ValueError("0<k<n required")
    values, vectors = la.eigh(pencil.a.toarray(), pencil.b.toarray())
    return values[-k:][::-1].copy(), vectors[:, -k:][:, ::-1].copy()


def reconstruction_quality(pencil, field):
    """Geometry errors are separate from eigensolver residuals and sign-invariant."""
    field = np.asarray(field)
    grad = (pencil.gradients @ field).reshape(-1, 3)
    norm = np.linalg.norm(grad, axis=1)
    if not np.isfinite(field).all() or np.min(norm) <= 0:
        return dict(valid_zero_set=False, reason="invalid field or zero sampled gradient")
    distance = np.abs(pencil.samples @ field) / norm
    pairs = np.array(list(combinations(range(4), 2)))
    edges = np.sort(pencil.tets[:, pairs].reshape(-1, 2), axis=1)
    edges = np.unique(edges, axis=0)
    values = field[edges]
    crossing = (values[:, 0] < 0) != (values[:, 1] < 0)
    edges, values = edges[crossing], values[crossing]
    if len(edges) == 0:
        return dict(valid_zero_set=False, reason="no zero crossings")
    t = values[:, 0] / (values[:, 0] - values[:, 1])
    zeros = (1 - t[:, None]) * pencil.vertices[edges[:, 0]] + t[:, None] * pencil.vertices[
        edges[:, 1]
    ]
    radial = np.abs(np.linalg.norm(zeros / pencil.radii, axis=1) - 1)
    alignment = np.abs(np.sum(grad * pencil.normals, axis=1)) / norm
    return dict(
        valid_zero_set=True,
        sample_distance_rms=float(np.sqrt(np.mean(distance**2))),
        zero_set_relative_radial_rms=float(np.sqrt(np.mean(radial**2))),
        zero_set_relative_radial_max=float(radial.max()),
        normal_alignment_mean=float(alignment.mean()),
        crossings=len(edges),
    )
