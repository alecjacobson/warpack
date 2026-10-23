"""Benchmark-only FP64 implicit QL for symmetric tridiagonal matrices (#161).

The mathematical decomposition is described by LAPACK DSTEQR. This Warp
implementation replicates scalar QL state per group of eigenvector rows;
rows accumulate rotations concurrently. No vendor routine is called.
"""

import warp as wp


@wp.func
def safe_hypot(a: wp.float64, b: wp.float64):
    scale = wp.max(wp.abs(a), wp.abs(b))
    value = wp.float64(0.0)
    if scale > wp.float64(0.0):
        x, y = a / scale, b / scale
        value = scale * wp.sqrt(x * x + y * y)
    return value


@wp.kernel(enable_backward=False)
def implicit_ql(
    diagonal: wp.array[wp.float64],
    off_diagonal: wp.array[wp.float64],
    values: wp.array[wp.float64],
    vt: wp.array2d[wp.float64],
    d: wp.array2d[wp.float64],
    e: wp.array2d[wp.float64],
    cosines: wp.array2d[wp.float64],
    sines: wp.array2d[wp.float64],
    state: wp.array2d[wp.int32],
    maxiter: int,
):
    block, lane = wp.tid()
    n = diagonal.shape[0]
    row = block * 64 + lane
    barrier = wp.tile_zeros(shape=1, dtype=wp.int32, storage="shared")
    scale = wp.float64(0.0)
    invalid = int(0)
    for i in range(lane, n, 64):
        a = diagonal[i]
        b = wp.float64(0.0)
        if i + 1 < n:
            b = off_diagonal[i]
        if not wp.isfinite(a) or not wp.isfinite(b):
            invalid = 1
        scale = wp.max(scale, wp.max(wp.abs(a), wp.abs(b)))
    scale = wp.tile_extract(wp.tile_max(wp.tile(scale)), 0)
    invalid = wp.tile_extract(wp.tile_max(wp.tile(invalid)), 0)
    if scale == wp.float64(0.0):
        scale = wp.float64(1.0)
    if lane == 0:
        state[block, 1] = invalid
        state[block, 2] = 0
    if invalid != 0:
        return
    for i in range(lane, n, 64):
        d[block, i] = diagonal[i] / scale
        e[block, i] = wp.float64(0.0)
        if i + 1 < n:
            e[block, i] = off_diagonal[i] / scale
    if row < n:
        for i in range(n):
            vt[i, row] = wp.float64(0.0)
            if i == row:
                vt[i, row] = wp.float64(1.0)
    wp.tile_scatter_masked(barrier, 0, 0, lane == 0)
    for left in range(n):
        converged = int(0)
        for iteration in range(maxiter + 1):
            if lane == 0:
                end = int(left)
                while end + 1 < n:
                    threshold = wp.float64(2.0e-16) * (
                        wp.abs(d[block, end]) + wp.abs(d[block, end + 1])
                    )
                    if wp.abs(e[block, end]) <= threshold:
                        e[block, end] = wp.float64(0.0)
                        break
                    end += 1
                state[block, 0] = end
                if end > left and iteration < maxiter:
                    state[block, 2] += 1
                    g = (d[block, left + 1] - d[block, left]) / (wp.float64(2.0) * e[block, left])
                    r = safe_hypot(g, wp.float64(1.0))
                    if g < wp.float64(0.0):
                        r = -r
                    g = d[block, end] - d[block, left] + e[block, left] / (g + r)
                    c, s, shift = wp.float64(1.0), wp.float64(1.0), wp.float64(0.0)
                    for index in range(end - left):
                        i = end - 1 - index
                        f, b = s * e[block, i], c * e[block, i]
                        r = safe_hypot(f, g)
                        e[block, i + 1] = r
                        c, s = wp.float64(1.0), wp.float64(0.0)
                        if r > wp.float64(0.0):
                            c, s = g / r, f / r
                        g = d[block, i + 1] - shift
                        r = (d[block, i] - g) * s + wp.float64(2.0) * c * b
                        shift = s * r
                        d[block, i + 1] = g + shift
                        g = c * r - b
                        cosines[block, i] = c
                        sines[block, i] = s
                    d[block, left] -= shift
                    e[block, left] = g
                    e[block, end] = wp.float64(0.0)
            wp.tile_scatter_masked(barrier, 0, iteration, lane == 0)
            end = state[block, 0]
            if end == left:
                converged = 1
                break
            if iteration == maxiter:
                break
            if row < n:
                for index in range(end - left):
                    i = end - 1 - index
                    c = wp.float64(cosines[block, i])
                    s = wp.float64(sines[block, i])
                    a = wp.float64(vt[i, row])
                    b = wp.float64(vt[i + 1, row])
                    vt[i, row] = c * a - s * b
                    vt[i + 1, row] = s * a + c * b
            wp.tile_scatter_masked(barrier, 0, iteration, lane == 0)
        if converged == 0:
            if lane == 0:
                state[block, 1] = 2
            break
    if block == 0:
        for i in range(lane, n, 64):
            values[i] = d[block, i] * scale


class TridiagonalQL:
    """Bounded device solve with per-block status: 0 success, 1 input, 2 budget."""

    def __init__(self, diagonal, off_diagonal, *, maxiter=30):
        n, device = diagonal.shape[0], diagonal.device
        if not device.is_cuda or not 1 <= n <= 512:
            raise ValueError("prototype requires CUDA and dimension in 1..512")
        if diagonal.dtype != wp.float64 or off_diagonal.dtype != wp.float64:
            raise TypeError("FP64 arrays required")
        if off_diagonal.shape != (n - 1,) or off_diagonal.device != device or maxiter < 0:
            raise ValueError("incompatible off-diagonal or iteration budget")
        self.diagonal, self.off_diagonal, self.maxiter = diagonal, off_diagonal, maxiter
        self.device, self.blocks = device, (n + 63) // 64
        self.values = wp.empty(n, dtype=wp.float64, device=device)
        self.vt = wp.empty((n, n), dtype=wp.float64, device=device)
        self.vectors = self.vt.transpose()
        self.d, self.e, self.cosines, self.sines = (
            wp.empty((self.blocks, n), dtype=wp.float64, device=device) for _ in range(4)
        )
        self.state = wp.zeros((self.blocks, 3), dtype=wp.int32, device=device)

    def apply(self):
        wp.launch_tiled(
            implicit_ql,
            dim=self.blocks,
            inputs=[
                self.diagonal,
                self.off_diagonal,
                self.values,
                self.vt,
                self.d,
                self.e,
                self.cosines,
                self.sines,
                self.state,
                self.maxiter,
            ],
            block_dim=64,
            device=self.device,
        )
