"""Experimental FP64 block Jacobi for projected symmetric eigensolves (#169).

Each disjoint block pair is diagonalized by a CUDA block using Warp shared
memory. Sparse block rotations are applied to the full matrix and basis.
No vendor dense solver; no production dispatch change.
"""

import functools

import warp as wp

from warpack.projected import GlobalJacobi, global_jacobi_update, pair_left, pair_right


@functools.lru_cache(None)
def local_jacobi_kernel(size, local_sweeps=30, block_dim=64):
    threads = wp.constant(block_dim)
    n = wp.constant(size)
    half = wp.constant(size // 2)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(a: wp.array3d[wp.float64], vectors: wp.array3d[wp.float64]):
        block, lane = wp.tid()
        h = wp.tile_reshape(
            wp.tile_load(a, shape=(1, n, n), offset=(block, 0, 0), storage="shared"), shape=(n, n)
        )
        v = wp.tile_zeros(shape=(n, n), dtype=wp.float64, storage="shared")
        cc = wp.tile_ones(shape=n, dtype=wp.float64, storage="shared")
        ss = wp.tile_zeros(shape=n, dtype=wp.float64, storage="shared")
        wp.tile_scatter_masked(
            v, wp.min(lane, n - 1), wp.min(lane, n - 1), wp.float64(1.0), lane < n
        )
        for sweep in range(wp.static(local_sweeps)):
            for r in range(n - 1):
                ip = wp.min(lane, half - 1)
                p = pair_left(ip, r, n)
                q = pair_right(ip, r, n)
                apq = wp.tile_extract(h, p, q)
                app = wp.tile_extract(h, p, p)
                aqq = wp.tile_extract(h, q, q)
                c = wp.float64(1.0)
                s = wp.float64(0.0)
                if lane < half and wp.abs(apq) > wp.float64(1.0e-16) * (wp.abs(app) + wp.abs(aqq)):
                    tau = (aqq - app) / (wp.float64(2.0) * apq)
                    t = wp.float64(1.0) / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                    if tau < wp.float64(0.0):
                        t = -t
                    c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
                    s = t * c
                wp.tile_scatter_masked(cc, ip, c, lane < half)
                wp.tile_scatter_masked(ss, ip, s, lane < half)
                for batch in range((half * half + threads - 1) // threads):
                    ix = batch * threads + lane
                    safe = wp.min(ix, half * half - 1)
                    i = safe // half
                    j = safe % half
                    p = pair_left(i, r, n)
                    q = pair_right(i, r, n)
                    u = pair_left(j, r, n)
                    w = pair_right(j, r, n)
                    c = wp.tile_extract(cc, i)
                    s = wp.tile_extract(ss, i)
                    d = wp.tile_extract(cc, j)
                    t = wp.tile_extract(ss, j)
                    a00 = wp.tile_extract(h, p, u)
                    a01 = wp.tile_extract(h, p, w)
                    a10 = wp.tile_extract(h, q, u)
                    a11 = wp.tile_extract(h, q, w)
                    b00 = c * a00 - s * a10
                    b01 = c * a01 - s * a11
                    b10 = s * a00 + c * a10
                    b11 = s * a01 + c * a11
                    wp.tile_scatter_masked(h, p, u, d * b00 - t * b01, ix < half * half)
                    wp.tile_scatter_masked(h, p, w, t * b00 + d * b01, ix < half * half)
                    wp.tile_scatter_masked(h, q, u, d * b10 - t * b11, ix < half * half)
                    wp.tile_scatter_masked(h, q, w, t * b10 + d * b11, ix < half * half)
                for batch in range((n * half + threads - 1) // threads):
                    ix = batch * threads + lane
                    safe = wp.min(ix, n * half - 1)
                    i = safe // half
                    j = safe % half
                    p = pair_left(j, r, n)
                    q = pair_right(j, r, n)
                    c = wp.tile_extract(cc, j)
                    s = wp.tile_extract(ss, j)
                    a0 = wp.tile_extract(v, i, p)
                    a1 = wp.tile_extract(v, i, q)
                    wp.tile_scatter_masked(v, i, p, c * a0 - s * a1, ix < n * half)
                    wp.tile_scatter_masked(v, i, q, s * a0 + c * a1, ix < n * half)
            off = wp.float64(0.0)
            diagonal = wp.float64(0.0)
            for batch in range((n * n + threads - 1) // threads):
                ix = batch * threads + lane
                i, j = wp.min(ix // n, n - 1), ix % n
                value = wp.abs(wp.tile_extract(h, i, j))
                if ix < n * n:
                    if i == j:
                        diagonal = wp.max(diagonal, value)
                    else:
                        off = wp.max(off, value)
            max_off = wp.tile_max(wp.tile(off))
            max_diagonal = wp.tile_max(wp.tile(diagonal))
            if wp.tile_extract(max_off, 0) <= wp.float64(2.0e-15) * wp.tile_extract(
                max_diagonal, 0
            ):
                break
        wp.tile_store(a, wp.tile_reshape(h, shape=(1, n, n)), offset=(block, 0, 0))
        wp.tile_store(vectors, wp.tile_reshape(v, shape=(1, n, n)), offset=(block, 0, 0))

    return compute


@wp.func
def block_index(pair: int, local: int, size: int, round: int, blocks: int):
    block = pair_left(pair, round, blocks)
    if local >= size:
        block = pair_right(pair, round, blocks)
    return block * size + local % size


@wp.kernel
def gather_pairs(h: wp.array2d[wp.float64], panels: wp.array3d[wp.float64], size: int, round: int):
    pair, i, j = wp.tid()
    blocks = h.shape[0] // size
    a = block_index(pair, i, size, round, blocks)
    b = block_index(pair, j, size, round, blocks)
    panels[pair, i, j] = h[a, b]


@wp.kernel
def right_rotate(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    rotation: wp.array3d[wp.float64],
    temp: wp.array2d[wp.float64],
    vtemp: wp.array2d[wp.float64],
    size: int,
    round: int,
):
    i, pair, j = wp.tid()
    blocks = h.shape[0] // size
    column = block_index(pair, j, size, round, blocks)
    a, b = wp.float64(0.0), wp.float64(0.0)
    for t in range(2 * size):
        source = block_index(pair, t, size, round, blocks)
        q = rotation[pair, t, j]
        a += h[i, source] * q
        b += v[i, source] * q
    temp[i, column] = a
    vtemp[i, column] = b


@wp.kernel
def left_rotate(
    temp: wp.array2d[wp.float64],
    rotation: wp.array3d[wp.float64],
    h: wp.array2d[wp.float64],
    size: int,
    round: int,
):
    pair, i, j = wp.tid()
    blocks = h.shape[0] // size
    row = block_index(pair, i, size, round, blocks)
    a = wp.float64(0.0)
    for t in range(2 * size):
        source = block_index(pair, t, size, round, blocks)
        a += rotation[pair, t, i] * temp[source, j]
    h[row, j] = a


@wp.kernel
def load_padded(
    source: wp.array2d[wp.float64],
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
):
    i, j = wp.tid()
    a = wp.float64(0.0)
    if i < source.shape[0] and j < source.shape[1]:
        a = (source[i, j] + source[j, i]) * wp.float64(0.5)
    h[i, j] = a
    v[i, j] = wp.float64(i == j)


@wp.kernel
def store_result(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    output: wp.array2d[wp.float64],
    vectors: wp.array2d[wp.float64],
):
    i, j = wp.tid()
    output[i, j] = h[i, j]
    vectors[i, j] = v[i, j]


class BlockJacobi(GlobalJacobi):
    def __init__(self, h, v, cs, ss, partners, temp, vtemp, block_size=8, local_sweeps=30):
        n = h.shape[0]
        if not h.device.is_cuda or n < 2 or n > 1024:
            raise ValueError("block Jacobi requires CUDA and dimension 2..1024")
        if block_size not in (4, 8, 16):
            raise ValueError("block size must be 4, 8 or 16")
        if local_sweeps not in (1, 2, 4, 30):
            raise ValueError("local sweeps must be 1, 2, 4 or 30")
        super().__init__(h, v, cs, ss, partners, temp, vtemp)
        self.size = block_size
        self.padded = ((n + 2 * block_size - 1) // (2 * block_size)) * (2 * block_size)
        self.blocks = self.padded // block_size
        self.work, self.basis, self.right, self.basis_temp = (
            wp.empty((self.padded, self.padded), dtype=wp.float64, device=self.device)
            for _ in range(4)
        )
        self.panels, self.rotations = (
            wp.empty(
                (self.blocks // 2, 2 * block_size, 2 * block_size),
                dtype=wp.float64,
                device=self.device,
            )
            for _ in range(2)
        )
        self.local = local_jacobi_kernel(2 * block_size, local_sweeps)

    def _sweep(self):
        size, pairs = self.size, self.blocks // 2
        for round in range(self.blocks - 1):
            wp.launch(
                gather_pairs,
                self.panels.shape,
                [self.work, self.panels, size, round],
                device=self.device,
            )
            wp.launch_tiled(
                self.local,
                dim=pairs,
                inputs=[self.panels, self.rotations],
                block_dim=64,
                device=self.device,
            )
            wp.launch(
                right_rotate,
                (self.padded, pairs, 2 * size),
                [self.work, self.basis, self.rotations, self.right, self.basis_temp, size, round],
                device=self.device,
            )
            wp.launch(
                left_rotate,
                (pairs, 2 * size, self.padded),
                [self.right, self.rotations, self.work, size, round],
                device=self.device,
            )
            self.basis, self.basis_temp = self.basis_temp, self.basis
        # Restore object identities independently of parity: graph pointers must
        # match on the next captured sweep and every replay.
        if (self.blocks - 1) % 2:
            wp.copy(self.basis_temp, self.basis)
            self.basis, self.basis_temp = self.basis_temp, self.basis
        wp.launch_tiled(
            global_jacobi_update,
            dim=1,
            inputs=[self.work, self.sweeps, self.running],
            block_dim=128,
            device=self.device,
        )

    def _capture(self):
        wp.launch(load_padded, self.work.shape, [self.h, self.work, self.basis], device=self.device)
        self.sweeps.zero_()
        self.running.fill_(1)
        wp.capture_while(self.running, self._sweep)
        wp.launch(
            store_result, self.h.shape, [self.work, self.basis, self.h, self.v], device=self.device
        )


class BlockJacobiPolish(BlockJacobi):
    """A few block-ordered FP64 sweeps, then canonical scalar Jacobi recovery.

    Avoid spending all thirty sweeps in an ordering that can converge slowly
    on clustered/rank-deficient inputs. The bounded refinement uses the same
    stopping test as the existing solver. `sweeps` counts refinement sweeps;
    initializer_sweeps is fixed at construction and charged separately.
    """

    def __init__(self, h, v, cs, ss, partners, temp, vtemp, block_size=8, initializer_sweeps=2):
        if initializer_sweeps not in (1, 2, 4):
            raise ValueError("initializer sweeps must be 1, 2 or 4")
        super().__init__(h, v, cs, ss, partners, temp, vtemp, block_size=block_size, local_sweeps=1)
        self.initializer_sweeps = initializer_sweeps
        self.polish = GlobalJacobi(
            self.work,
            wp.empty_like(self.work),
            wp.empty(self.padded, dtype=wp.float64, device=self.device),
            wp.empty(self.padded, dtype=wp.float64, device=self.device),
            wp.empty(self.padded, dtype=wp.int32, device=self.device),
            self.right,
            wp.empty_like(self.work),
        )
        self.sweeps = self.polish.sweeps
        self.polish_transpose = self.polish.v.transpose()

    def _capture(self):
        from warpack.block import gram

        wp.launch(load_padded, self.work.shape, [self.h, self.work, self.basis], device=self.device)
        self.sweeps.zero_()
        for _ in range(self.initializer_sweeps):
            self._sweep()
        self.polish.apply()
        gram(self.basis, self.polish_transpose, self.basis_temp)
        wp.launch(
            store_result,
            self.h.shape,
            [self.work, self.basis_temp, self.h, self.v],
            device=self.device,
        )
