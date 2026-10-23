"""Independent row-wise implicit QL with compiler-local vector workspaces.

Benchmark control for the global scalar-state bottleneck in #161. Repeats
scalar recurrences per eigenvector row so all lanes execute useful work.
"""

from functools import lru_cache

import warp as wp

from benchmarks.experimental.tridiagonal_ql import TridiagonalQL, safe_hypot


@lru_cache(None)
def row_ql_kernel(size):
    n = wp.constant(size)
    vector = wp.types.vector(length=size, dtype=wp.float64)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        diagonal: wp.array[wp.float64],
        off_diagonal: wp.array[wp.float64],
        values: wp.array[wp.float64],
        vt: wp.array2d[wp.float64],
        state: wp.array2d[wp.int32],
        maxiter: int,
    ):
        row = wp.tid()
        d, e, v = vector(), vector(), vector()
        scale = wp.float64(0.0)
        status, iterations = int(0), int(0)
        for i in range(n):
            a = diagonal[i]
            b = wp.float64(0.0)
            if i + 1 < n:
                b = off_diagonal[i]
            if not wp.isfinite(a) or not wp.isfinite(b):
                status = 1
            scale = wp.max(scale, wp.max(wp.abs(a), wp.abs(b)))
        if scale == wp.float64(0.0):
            scale = wp.float64(1.0)
        if status != 0:
            state[row, 1] = status
            state[row, 2] = 0
            return
        for i in range(n):
            d[i] = diagonal[i] / scale
            if i + 1 < n:
                e[i] = off_diagonal[i] / scale
        v[row] = wp.float64(1.0)
        for left in range(n):
            converged = int(0)
            for iteration in range(maxiter + 1):
                end = int(left)
                while end + 1 < n:
                    threshold = wp.float64(2.0e-16) * (wp.abs(d[end]) + wp.abs(d[end + 1]))
                    if wp.abs(e[end]) <= threshold:
                        e[end] = wp.float64(0.0)
                        break
                    end += 1
                if end == left:
                    converged = 1
                    break
                if iteration == maxiter:
                    break
                iterations += 1
                g = (d[left + 1] - d[left]) / (wp.float64(2.0) * e[left])
                r = safe_hypot(g, wp.float64(1.0))
                if g < wp.float64(0.0):
                    r = -r
                g = d[end] - d[left] + e[left] / (g + r)
                c, s, shift = wp.float64(1.0), wp.float64(1.0), wp.float64(0.0)
                for index in range(end - left):
                    i = end - 1 - index
                    f, b = s * e[i], c * e[i]
                    r = safe_hypot(f, g)
                    e[i + 1] = r
                    c, s = wp.float64(1.0), wp.float64(0.0)
                    if r > wp.float64(0.0):
                        c, s = g / r, f / r
                    g = d[i + 1] - shift
                    r = (d[i] - g) * s + wp.float64(2.0) * c * b
                    shift = s * r
                    d[i + 1] = g + shift
                    g = c * r - b
                    a = v[i]
                    b = v[i + 1]
                    v[i] = c * a - s * b
                    v[i + 1] = s * a + c * b
                d[left] -= shift
                e[left] = g
                e[end] = wp.float64(0.0)
            if converged == 0:
                status = 2
                break
        state[row, 1] = status
        state[row, 2] = iterations
        for i in range(n):
            vt[i, row] = v[i]
            if row == 0:
                values[i] = d[i] * scale

    return compute


class RowTridiagonalQL(TridiagonalQL):
    def __init__(self, diagonal, off_diagonal, *, maxiter=30):
        super().__init__(diagonal, off_diagonal, maxiter=maxiter)
        self.state = wp.zeros((diagonal.shape[0], 3), dtype=wp.int32, device=self.device)

    def apply(self):
        wp.launch(
            row_ql_kernel(self.diagonal.shape[0]),
            dim=self.diagonal.shape[0],
            inputs=[
                self.diagonal,
                self.off_diagonal,
                self.values,
                self.vt,
                self.state,
                self.maxiter,
            ],
            block_dim=64,
            device=self.device,
        )
