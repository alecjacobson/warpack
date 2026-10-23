"""Pure-Warp Jacobi round with one writer per 2x2 output block (#172)."""

import warp as wp

from warpack import kernels as K
from warpack.projected import GlobalJacobi, pair_left, pair_right


@wp.kernel(enable_backward=False)
def paired_update(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    out_h: wp.array2d[wp.float64],
    out_v: wp.array2d[wp.float64],
    c: wp.array[wp.float64],
    s: wp.array[wp.float64],
    round: int,
):
    i, j = wp.tid()
    n = h.shape[0]
    p, q = pair_left(i, round, n), pair_right(i, round, n)
    u, w = pair_left(j, round, n), pair_right(j, round, n)
    a00 = h[p, u]
    a01 = h[p, w]
    a10 = h[q, u]
    a11 = h[q, w]
    # Preserve each original output's expression/order, including signs from
    # the angle kernel. Reuse the four right-rotation intermediate entries.
    b00 = c[u] * a00 - s[u] * a01
    b01 = c[w] * a01 - s[w] * a00
    b10 = c[u] * a10 - s[u] * a11
    b11 = c[w] * a11 - s[w] * a10
    out_h[p, u] = c[p] * b00 - s[p] * b10
    out_h[p, w] = c[p] * b01 - s[p] * b11
    out_h[q, u] = c[q] * b10 - s[q] * b00
    out_h[q, w] = c[q] * b11 - s[q] * b01
    v00 = v[p, u]
    v01 = v[p, w]
    v10 = v[q, u]
    v11 = v[q, w]
    out_v[p, u] = c[u] * v00 - s[u] * v01
    out_v[p, w] = c[w] * v01 - s[w] * v00
    out_v[q, u] = c[u] * v10 - s[u] * v11
    out_v[q, w] = c[w] * v11 - s[w] * v10


class PairedOutputJacobi(GlobalJacobi):
    def __init__(self, *args, block_dim=128):
        super().__init__(*args)
        if self.n < 2 or self.n % 2:
            raise ValueError("even dimension >=2 required")
        if block_dim not in (64, 128, 256):
            raise ValueError("supported block sizes are64,128,256")
        self.block_dim = block_dim

    def _sweep(self):
        current_h, current_v = self.h, self.v
        next_h, next_v = self.temp, self.vtemp
        for r in range(self.n - 1):
            wp.launch(
                K.jacobi_pairs,
                self.n // 2,
                [current_h, self.cs, self.ss, self.partners, r],
                device=self.device,
            )
            wp.launch(
                paired_update,
                (self.n // 2, self.n // 2),
                [current_h, current_v, next_h, next_v, self.cs, self.ss, r],
                block_dim=self.block_dim,
                device=self.device,
            )
            current_h, next_h = next_h, current_h
            current_v, next_v = next_v, current_v
        if current_h is not self.h:
            wp.copy(self.h, current_h)
            wp.copy(self.v, current_v)
        self._finish_sweep()
