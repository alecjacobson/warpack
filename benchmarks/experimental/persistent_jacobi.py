"""Experimental one-block global-workspace Jacobi; production is unchanged.

Keep the global solver's round schedule and right-then-left arithmetic. The
tile collective barriers order global writes between rounds, as in the global
Cholesky experiment. All workspaces are supplied before capture.
"""

import warp as wp


@wp.func
def pairs(
    h: wp.array2d[wp.float64],
    cs: wp.array[wp.float64],
    ss: wp.array[wp.float64],
    partners: wp.array[wp.int32],
    lane: int,
    round: int,
):
    n = h.shape[0]
    if lane < n // 2:
        p = (round + lane) % (n - 1)
        q = (round - lane + n - 1) % (n - 1)
        if lane == 0:
            q = n - 1
        apq = h[p, q]
        c, s = wp.float64(1.0), wp.float64(0.0)
        if wp.abs(apq) > wp.float64(1.0e-16) * (wp.abs(h[p, p]) + wp.abs(h[q, q])):
            tau = (h[q, q] - h[p, p]) / (wp.float64(2.0) * apq)
            t = wp.float64(1.0) / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
            if tau < wp.float64(0.0):
                t = -t
            c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
            s = t * c
        cs[p] = c
        cs[q] = c
        ss[p] = s
        ss[q] = -s
        partners[p] = q
        partners[q] = p


@wp.func
def update(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    out_h: wp.array2d[wp.float64],
    out_v: wp.array2d[wp.float64],
    c: wp.array[wp.float64],
    s: wp.array[wp.float64],
    p: wp.array[wp.int32],
    i: int,
    j: int,
):
    right_i = c[j] * h[i, j] - s[j] * h[i, p[j]]
    right_p = c[j] * h[p[i], j] - s[j] * h[p[i], p[j]]
    out_h[i, j] = c[i] * right_i - s[i] * right_p
    out_v[i, j] = c[j] * v[i, j] - s[j] * v[i, p[j]]


@wp.kernel(enable_backward=False)
def persistent_jacobi(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    temp: wp.array2d[wp.float64],
    vtemp: wp.array2d[wp.float64],
    cs: wp.array[wp.float64],
    ss: wp.array[wp.float64],
    partners: wp.array[wp.int32],
    sweeps: wp.array[wp.int32],
):
    block, lane = wp.tid()
    n = h.shape[0]
    barrier = wp.tile_zeros(shape=1, dtype=wp.int32, storage="shared")
    for entry in range(lane, n * n, 256):
        i, j = entry // n, entry % n
        v[i, j] = wp.float64(0.0)
        if i == j:
            v[i, j] = wp.float64(1.0)
        if i < j:
            value = (h[i, j] + h[j, i]) * wp.float64(0.5)
            h[i, j] = value
            h[j, i] = value
    wp.tile_scatter_masked(barrier, 0, 0, lane == 0)
    for sweep in range(30):
        for round in range(n - 1):
            if round % 2 == 0:
                pairs(h, cs, ss, partners, lane, round)
            else:
                pairs(temp, cs, ss, partners, lane, round)
            wp.tile_scatter_masked(barrier, 0, round, lane == 0)
            for entry in range(lane, n * n, 256):
                i, j = entry // n, entry % n
                if round % 2 == 0:
                    update(h, v, temp, vtemp, cs, ss, partners, i, j)
                else:
                    update(temp, vtemp, h, v, cs, ss, partners, i, j)
            wp.tile_scatter_masked(barrier, 0, round, lane == 0)
        # Even n has an odd number of rounds; the final result is in temp.
        off, diagonal = wp.float64(0.0), wp.float64(0.0)
        for entry in range(lane, n * n, 256):
            i, j = entry // n, entry % n
            h[i, j] = temp[i, j]
            v[i, j] = vtemp[i, j]
            value = wp.abs(temp[i, j])
            if not wp.isfinite(value):
                off = wp.float64(wp.inf)
            elif i == j:
                diagonal = wp.max(diagonal, value)
            else:
                off = wp.max(off, value)
        wp.tile_scatter_masked(barrier, 0, sweep, lane == 0)
        max_off = wp.tile_extract(wp.tile_max(wp.tile(off)), 0)
        max_diagonal = wp.tile_extract(wp.tile_max(wp.tile(diagonal)), 0)
        if lane == 0:
            sweeps[0] = sweep + 1
        if wp.isfinite(max_off) and max_off <= wp.float64(2.0e-15) * max_diagonal:
            break


class PersistentJacobi:
    """Preallocated experimental equivalent of GlobalJacobi for even n <=512."""

    def __init__(self, h, v, cs, ss, partners, temp, vtemp):
        n = h.shape[0]
        if not h.device.is_cuda or n < 2 or n > 512 or n % 2:
            raise ValueError("persistent Jacobi requires CUDA and an even dimension in 2..512")
        self.h, self.v, self.cs, self.ss = h, v, cs, ss
        self.partners, self.temp, self.vtemp = partners, temp, vtemp
        self.device = h.device
        self.sweeps = wp.zeros(1, dtype=wp.int32, device=self.device)

    def apply(self):
        wp.launch_tiled(
            persistent_jacobi,
            dim=1,
            inputs=[
                self.h,
                self.v,
                self.temp,
                self.vtemp,
                self.cs,
                self.ss,
                self.partners,
                self.sweeps,
            ],
            block_dim=256,
            device=self.device,
        )
