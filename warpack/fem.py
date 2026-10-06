"""Linear tetrahedra: stable Neo-Hookean tangent at rest and lumped mass.

Energy density: mu/2 (tr(F^T F)-3) + (lambda+mu)/2 (J-alpha)^2,
alpha = 1 + mu/(lambda+mu). Its stress-free rest tangent has the usual
Lame constants lambda, mu. Geometry, element Hessians, sparse assembly, and
mass normalization are computed in Warp. Mesh decoding is host I/O.
"""

from pathlib import Path

import numpy as np
import warp as wp
import warp.sparse as ws

from . import kernels as K


@wp.kernel
def element_triplets(
    x: wp.array[wp.vec3d],
    t: wp.array[wp.vec4i],
    mu: wp.float64,
    lam: wp.float64,
    density: wp.float64,
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    vals: wp.array[wp.mat33d],
    mass: wp.array[wp.float64],
    invalid: wp.array[wp.int32],
):
    e = wp.tid()
    ids = t[e]
    a, b, c = x[ids[1]] - x[ids[0]], x[ids[2]] - x[ids[0]], x[ids[3]] - x[ids[0]]
    dm = wp.matrix_from_cols(a, b, c)
    det = wp.determinant(dm)
    vol = wp.abs(det) / wp.float64(6.0)
    if vol <= wp.float64(1.0e-30):
        wp.atomic_add(invalid, 0, 1)
    inv = wp.inverse(dm)
    g = wp.matrix(shape=(4, 3), dtype=wp.float64)
    for d in range(3):
        g[0, d] = -inv[0, d] - inv[1, d] - inv[2, d]
        g[1, d] = inv[0, d]
        g[2, d] = inv[1, d]
        g[3, d] = inv[2, d]
    for i in range(4):
        wp.atomic_add(mass, ids[i], density * vol / wp.float64(4.0))
        for j in range(4):
            gi = wp.vec3d(g[i, 0], g[i, 1], g[i, 2])
            gj = wp.vec3d(g[j, 0], g[j, 1], g[j, 2])
            h = vol * (
                mu * wp.dot(gi, gj) * wp.identity(n=3, dtype=wp.float64)
                + lam * wp.outer(gi, gj)
                + mu * wp.outer(gj, gi)
            )
            idx = e * 16 + i * 4 + j
            rows[idx] = ids[i]
            cols[idx] = ids[j]
            vals[idx] = h


@wp.kernel
def normalize_blocks(
    offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    h: wp.array[wp.mat33d],
    mass: wp.array[wp.float64],
    out: wp.array[wp.mat33d],
    shift: wp.float64,
):
    i = wp.tid()
    for e in range(offsets[i], offsets[i + 1]):
        j = cols[e]
        v = h[e] / wp.sqrt(mass[i] * mass[j])
        if i == j:
            v += shift * wp.identity(n=3, dtype=wp.float64)
        out[e] = v


@wp.kernel
def mass_triplets(mass: wp.array[wp.float64], rows: wp.array[wp.int32], vals: wp.array[wp.mat33d]):
    i = wp.tid()
    rows[i] = i
    vals[i] = mass[i] * wp.identity(n=3, dtype=wp.float64)


def read_mesh(path):
    """Read the ASCII Medit vertex/tetrahedron sections (one-based indices)."""
    with Path(path).open() as f:
        vertices = tets = None
        for line in f:
            if line.strip() == "Vertices":
                n = int(next(f))
                vertices = np.loadtxt(f, max_rows=n)[:, :3]
            elif line.strip() == "Tetrahedra":
                n = int(next(f))
                tets = np.loadtxt(f, max_rows=n, dtype=np.int32)[:, :4] - 1
    if vertices is None or tets is None:
        raise ValueError("expected Vertices and Tetrahedra sections")
    if tets.min() < 0 or tets.max() >= len(vertices):
        raise ValueError("tetrahedron vertex index outside mesh")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(tets)


def assemble_rest(vertices, tets, young=1.0e5, poisson=0.3, density=1000.0, device="cuda:0"):
    if young <= 0 or density <= 0 or not -1 < poisson < 0.5:
        raise ValueError("require E>0, density>0, -1<nu<0.5")
    x = wp.array(vertices, dtype=wp.vec3d, device=device)
    t = wp.array(tets, dtype=wp.vec4i, device=device)
    n, nt = len(vertices), len(tets)
    rows = wp.empty(nt * 16, dtype=wp.int32, device=device)
    cols = wp.empty_like(rows)
    vals = wp.empty(nt * 16, dtype=wp.mat33d, device=device)
    mass = wp.zeros(n, dtype=wp.float64, device=device)
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    wp.launch(
        element_triplets,
        nt,
        [x, t, mu, lam, density, rows, cols, vals, mass, invalid],
        device=device,
    )
    if invalid.numpy()[0]:
        raise ValueError("degenerate tetrahedron")
    if np.any(mass.numpy() <= 0):
        raise ValueError("mesh contains vertices without positive mass")
    h = ws.bsr_from_triplets(n, n, rows, cols, vals)
    mr = wp.empty(n, dtype=wp.int32, device=device)
    mv = wp.empty(n, dtype=wp.mat33d, device=device)
    wp.launch(mass_triplets, n, [mass, mr, mv], device=device)
    m = ws.bsr_from_triplets(n, n, mr, mr, mv)
    return h, m, mass


@wp.kernel
def _rigid_body_rows(
    positions: wp.array[wp.vec3d], mass: wp.array[wp.float64], basis: wp.array2d[wp.float64]
):
    mode, vertex = wp.tid()
    axis = wp.vec3d(0.0)
    axis[mode % 3] = wp.float64(1.0)
    value = axis
    if mode >= 3:
        value = wp.cross(axis, positions[vertex])
    value *= wp.sqrt(mass[vertex])
    for d in range(3):
        basis[mode, 3 * vertex + d] = value[d]


def rigid_body_basis(positions, mass):
    """Return six orthonormal rows in lumped-mass-normalized coordinates.

    Inputs are device arrays: FP64 ``vec3d`` positions and positive FP64 nodal
    masses. Positions must span at least a plane. Translations and infinitesimal
    rotations are built and orthogonalized entirely in Warp; no eigensolve or
    host numerical computation is used. This is the nullspace of a connected,
    unconstrained, stress-free elastic body.
    """
    if positions.dtype != wp.vec3d or mass.dtype != wp.float64:
        raise ValueError("positions and mass must use vec3d and FP64")
    if positions.size != mass.size or positions.device != mass.device:
        raise ValueError("positions and mass must have matching lengths and devices")
    device = mass.device
    n = 3 * mass.size
    basis = wp.empty((6, n), dtype=wp.float64, device=device)
    gram = wp.zeros((6, 6), dtype=wp.float64, device=device)
    factor = wp.empty_like(gram)
    status = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(_rigid_body_rows, (6, mass.size), [positions, mass, basis], device=device)
    for _ in range(2):
        gram.zero_()
        wp.launch_tiled(
            K.gram_partial,
            dim=(6, 6, (n + 255) // 256),
            inputs=[basis, basis, gram],
            block_dim=128,
            device=device,
        )
        wp.launch(K.cholesky, 1, [gram, factor, status], device=device)
        wp.launch(K.triangular, n, [basis, factor], device=device)
    return basis


def mass_normalized(h, mass, shift=0.0):
    out = ws.bsr_copy(h)
    wp.launch(
        normalize_blocks,
        h.nrow,
        [h.offsets, h.columns, h.values, mass, out.values, shift],
        device=h.device,
    )
    return out


@wp.kernel
def assemble_into_bsr(
    x: wp.array[wp.vec3d],
    t: wp.array[wp.vec4i],
    material: wp.array[wp.float64],
    offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    vals: wp.array[wp.mat33d],
    mass: wp.array[wp.float64],
    invalid: wp.array[wp.int32],
):
    e = wp.tid()
    young = material[0]
    poisson = material[1]
    density = material[2]
    mu = young / (wp.float64(2.0) * (wp.float64(1.0) + poisson))
    lam = (
        young
        * poisson
        / ((wp.float64(1.0) + poisson) * (wp.float64(1.0) - wp.float64(2.0) * poisson))
    )
    ids = t[e]
    a, b, c = x[ids[1]] - x[ids[0]], x[ids[2]] - x[ids[0]], x[ids[3]] - x[ids[0]]
    dm = wp.matrix_from_cols(a, b, c)
    det = wp.determinant(dm)
    vol = wp.abs(det) / wp.float64(6.0)
    if vol <= wp.float64(1.0e-30):
        wp.atomic_add(invalid, 0, 1)
    inv = wp.inverse(dm)
    g = wp.matrix(shape=(4, 3), dtype=wp.float64)
    for d in range(3):
        g[0, d] = -inv[0, d] - inv[1, d] - inv[2, d]
        g[1, d] = inv[0, d]
        g[2, d] = inv[1, d]
        g[3, d] = inv[2, d]
    for i in range(4):
        wp.atomic_add(mass, ids[i], density * vol / wp.float64(4.0))
        for j in range(4):
            gi = wp.vec3d(g[i, 0], g[i, 1], g[i, 2])
            gj = wp.vec3d(g[j, 0], g[j, 1], g[j, 2])
            h = vol * (
                mu * wp.dot(gi, gj) * wp.identity(n=3, dtype=wp.float64)
                + lam * wp.outer(gi, gj)
                + mu * wp.outer(gj, gi)
            )
            lo = offsets[ids[i]]
            hi = offsets[ids[i] + 1]
            while lo < hi:
                mid = (lo + hi) // 2
                if cols[mid] < ids[j]:
                    lo = mid + 1
                else:
                    hi = mid
            wp.atomic_add(vals, lo, h)


class RestElasticity:
    """Reusable graph-capturable rest-tangent and lumped-mass assembly plan.

    Topology is prepared once. Updating reference positions or material values
    then replaying `update` recomputes H, M, and mass-normalized H on device.
    `material` stores (Young modulus, Poisson ratio, density). `invalid` counts
    degenerate elements; read it explicitly outside capture if desired.
    """

    def __init__(self, vertices, tets, young=1.0e5, poisson=0.3, density=1000.0, device="cuda:0"):
        self.hessian, self.mass_matrix, self.mass = assemble_rest(
            vertices, tets, young, poisson, density, device
        )
        self.device = self.hessian.device
        self.positions = wp.array(vertices, dtype=wp.vec3d, device=device)
        self.tets = wp.array(tets, dtype=wp.vec4i, device=device)
        self.material = wp.array([young, poisson, density], dtype=wp.float64, device=device)
        self.invalid = wp.zeros(1, dtype=wp.int32, device=device)
        self.mass_rows = wp.empty(len(vertices), dtype=wp.int32, device=device)
        self.normalized = mass_normalized(self.hessian, self.mass)

    def update(self):
        h = self.hessian
        self.mass.zero_()
        self.invalid.zero_()
        h.values.zero_()
        wp.launch(
            assemble_into_bsr,
            len(self.tets),
            [
                self.positions,
                self.tets,
                self.material,
                h.offsets,
                h.columns,
                h.values,
                self.mass,
                self.invalid,
            ],
            device=self.device,
        )
        wp.launch(
            mass_triplets,
            len(self.positions),
            [self.mass, self.mass_rows, self.mass_matrix.values],
            device=self.device,
        )
        wp.launch(
            normalize_blocks,
            h.nrow,
            [h.offsets, h.columns, h.values, self.mass, self.normalized.values, 0.0],
            device=self.device,
        )
