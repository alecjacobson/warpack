"""Independent Warp StVK rest assembly for auditing the production tangent.

Uses face-cross-product shape gradients and Green-strain directional derivatives,
not the production inverse-Dm/block formula. Numerical work stays in Warp.
"""

import warp as wp
import warp.sparse as ws


@wp.kernel
def stvk_triplets(
    x: wp.array[wp.vec3d],
    t: wp.array[wp.vec4i],
    mu: wp.float64,
    lam: wp.float64,
    density: wp.float64,
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.mat33d],
    mass: wp.array[wp.float64],
):
    e, i, j = wp.tid()
    ids = t[e]
    a = x[ids[1]] - x[ids[0]]
    b = x[ids[2]] - x[ids[0]]
    c = x[ids[3]] - x[ids[0]]
    determinant = wp.dot(a, wp.cross(b, c))
    volume = wp.abs(determinant) / wp.float64(6.0)
    g = wp.matrix(shape=(4, 3), dtype=wp.float64)
    g[1] = wp.cross(b, c) / determinant
    g[2] = wp.cross(c, a) / determinant
    g[3] = wp.cross(a, b) / determinant
    g[0] = -g[1] - g[2] - g[3]
    block = wp.mat33d()
    for p in range(3):
        for q in range(3):
            ep = wp.vec3d()
            eq = wp.vec3d()
            ep[p] = wp.float64(1.0)
            eq[q] = wp.float64(1.0)
            dfi = wp.outer(ep, g[i])
            dfj = wp.outer(eq, g[j])
            # At F=I, dE = sym(dF), and the second Piola stress is zero.
            dei = (wp.transpose(dfi) + dfi) * wp.float64(0.5)
            dej = (wp.transpose(dfj) + dfj) * wp.float64(0.5)
            block[p, q] = volume * (
                wp.float64(2.0) * mu * wp.ddot(dei, dej) + lam * wp.trace(dei) * wp.trace(dej)
            )
    index = e * 16 + i * 4 + j
    rows[index] = ids[i]
    cols[index] = ids[j]
    values[index] = block
    if i == j:
        wp.atomic_add(mass, ids[i], density * volume / wp.float64(4.0))


def assemble_stvk_rest(vertices, tets, young=1e5, poisson=0.3, density=1000.0, device="cuda:0"):
    x = wp.array(vertices, dtype=wp.vec3d, device=device)
    t = wp.array(tets, dtype=wp.vec4i, device=device)
    rows = wp.empty(len(tets) * 16, dtype=wp.int32, device=device)
    cols = wp.empty_like(rows)
    values = wp.empty(len(tets) * 16, dtype=wp.mat33d, device=device)
    mass = wp.zeros(len(vertices), dtype=wp.float64, device=device)
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    wp.launch(
        stvk_triplets,
        (len(tets), 4, 4),
        [x, t, mu, lam, density, rows, cols, values, mass],
        device=device,
    )
    return ws.bsr_from_triplets(len(vertices), len(vertices), rows, cols, values), mass


@wp.kernel
def consistent_mass_triplets(
    x: wp.array[wp.vec3d],
    t: wp.array[wp.vec4i],
    density: wp.float64,
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.mat33d],
):
    e, i, j = wp.tid()
    ids = t[e]
    volume = wp.abs(
        wp.dot(x[ids[1]] - x[ids[0]], wp.cross(x[ids[2]] - x[ids[0]], x[ids[3]] - x[ids[0]]))
    ) / wp.float64(6.0)
    factor = wp.float64(1.0)
    if i == j:
        factor = wp.float64(2.0)
    index = e * 16 + i * 4 + j
    rows[index] = ids[i]
    cols[index] = ids[j]
    values[index] = (
        density * volume * factor / wp.float64(20.0) * wp.identity(n=3, dtype=wp.float64)
    )


def assemble_consistent_mass(vertices, tets, density=1000.0, device="cuda:0"):
    x = wp.array(vertices, dtype=wp.vec3d, device=device)
    t = wp.array(tets, dtype=wp.vec4i, device=device)
    rows = wp.empty(len(tets) * 16, dtype=wp.int32, device=device)
    cols = wp.empty_like(rows)
    values = wp.empty(len(tets) * 16, dtype=wp.mat33d, device=device)
    wp.launch(
        consistent_mass_triplets,
        (len(tets), 4, 4),
        [x, t, density, rows, cols, values],
        device=device,
    )
    return ws.bsr_from_triplets(len(vertices), len(vertices), rows, cols, values)
