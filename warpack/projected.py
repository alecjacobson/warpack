"""One-block cyclic Jacobi in shared memory, expressed entirely in Warp tiles."""

import functools

import warp as wp

from ._capture import capture_scope


@wp.func
def pair_left(i: int, r: int, n: int):
    return (r + i) % (n - 1)


@wp.func
def pair_right(i: int, r: int, n: int):
    q = (r - i + n - 1) % (n - 1)
    if i == 0:
        q = n - 1
    return q


@functools.lru_cache(None)
def jacobi_kernel(size, block_dim=256):
    threads = wp.constant(block_dim)
    n = wp.constant(size)
    half = wp.constant(size // 2)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(a: wp.array2d[wp.float64], vectors: wp.array2d[wp.float64]):
        block, lane = wp.tid()
        h = wp.tile_load(a, shape=(n, n), storage="shared")
        v = wp.tile_zeros(shape=(n, n), dtype=wp.float64, storage="shared")
        cc = wp.tile_ones(shape=n, dtype=wp.float64, storage="shared")
        ss = wp.tile_zeros(shape=n, dtype=wp.float64, storage="shared")
        wp.tile_scatter_masked(
            v, wp.min(lane, n - 1), wp.min(lane, n - 1), wp.float64(1.0), lane < n
        )
        for sweep in range(30):
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
        wp.tile_store(a, h)
        wp.tile_store(vectors, v)

    return compute


@wp.kernel
def restore_symmetry(h: wp.array2d[wp.float64]):
    # Jacobi assumes symmetry throughout its iteration. Roundoff skew can
    # prevent convergence on repeated/null spectra even after many sweeps.
    i, j = wp.tid()
    if i < j:
        value = (h[i, j] + h[j, i]) * wp.float64(0.5)
        h[i, j] = value
        h[j, i] = value


@wp.kernel(enable_backward=False)
def normalize_eigenvectors(v: wp.array2d[wp.float64]):
    column, lane = wp.tid()
    square = wp.float64(0.0)
    for row in range(lane, v.shape[0], 128):
        value = v[row, column]
        square += value * value
    square = wp.tile_extract(wp.tile_sum(wp.tile(square)), 0)
    # Preserve invalid outputs for existing failure detection. Never turn a
    # nonfinite/zero vector into a plausible new eigenvector.
    if wp.isfinite(square) and square > wp.float64(0.0):
        inverse = wp.float64(1.0) / wp.sqrt(square)
        for row in range(lane, v.shape[0], 128):
            v[row, column] *= inverse


@wp.kernel
def global_jacobi_update(
    h: wp.array2d[wp.float64],
    sweeps: wp.array[wp.int32],
    running: wp.array[wp.int32],
):
    block, lane = wp.tid()
    n = h.shape[0]
    off = wp.float64(0.0)
    diagonal = wp.float64(0.0)
    for entry in range(lane, n * n, 128):
        i, j = entry // n, entry % n
        value = wp.abs(h[i, j])
        if not wp.isfinite(value):
            off = wp.float64(wp.inf)
        elif i == j:
            diagonal = wp.max(diagonal, value)
        else:
            off = wp.max(off, value)
    max_off = wp.tile_extract(wp.tile_max(wp.tile(off)), 0)
    max_diagonal = wp.tile_extract(wp.tile_max(wp.tile(diagonal)), 0)
    if lane == 0:
        sweeps[0] += 1
        running[0] = 0
        if sweeps[0] < 30 and (
            not wp.isfinite(max_off) or max_off > wp.float64(2.0e-15) * max_diagonal
        ):
            running[0] = 1


class GlobalJacobi:
    """Bounded device convergence for the global-memory projected solver.

    Arrays and loop state are allocated at solver setup. Captured callers embed
    the loop; eager calls reuse a lazily prepared graph. Its first preparation
    may synchronize/load modules, as for outer capture; replay never does.
    """

    _restore_symmetry = True
    _normalize_vectors = True
    _paired_outputs = True

    def __init__(self, h, v, cs, ss, partners, temp, vtemp):
        self.h, self.v = h, v
        self.cs, self.ss, self.partners = cs, ss, partners
        self.temp, self.vtemp = temp, vtemp
        self.n, self.device = h.shape[0], h.device
        self.sweeps = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.running = wp.zeros_like(self.sweeps)
        self.graph = None
        from . import kernels as K

        # Qualified on Ada: smaller projections lose from reduced parallelism.
        # Static setup dispatch only; no device readback or replay host branch.
        self._paired_update = (
            self._paired_outputs
            and self.device.is_cuda
            and self.device.arch == 89
            and 256 <= self.n <= 1024
            and self.n % 2 == 0
        )
        self._update_kernel = (
            K.jacobi_two_sided_paired if self._paired_update else K.jacobi_two_sided
        )
        self._update_shape = (self.n // 2, self.n // 2) if self._paired_update else self.h.shape
        self._update_block_dim = 64 if self._paired_update else 256
        if self._paired_update:
            wp.load_module(module=self._update_kernel.module, device=self.device, block_dim=64)

    def _update(self, h, v, out_h, out_v):
        wp.launch(
            self._update_kernel,
            self._update_shape,
            [h, v, out_h, out_v, self.cs, self.ss, self.partners],
            block_dim=self._update_block_dim,
            device=self.device,
        )

    def _sweep(self):
        from . import kernels as K

        current_h, current_v = self.h, self.v
        next_h, next_v = self.temp, self.vtemp
        for r in range(self.n - 1):
            wp.launch(
                K.jacobi_pairs,
                self.n // 2,
                [current_h, self.cs, self.ss, self.partners, r],
                device=self.device,
            )
            self._update(current_h, current_v, next_h, next_v)
            current_h, next_h = next_h, current_h
            current_v, next_v = next_v, current_v
        # Even projected dimensions have an odd number of rounds. Restore the
        # public buffers once per sweep. Copy locations are fixed at setup;
        # both copies execute on device inside the bounded convergence loop.
        if current_h is not self.h:
            wp.copy(self.h, current_h)
            wp.copy(self.v, current_v)
        self._finish_sweep()

    def _finish_sweep(self):
        if self._restore_symmetry:
            wp.launch(restore_symmetry, self.h.shape, [self.h], device=self.device)
        wp.launch_tiled(
            global_jacobi_update,
            dim=1,
            inputs=[self.h, self.sweeps, self.running],
            block_dim=128,
            device=self.device,
        )

    def _capture(self):
        from . import kernels as K

        wp.launch(K.jacobi_init, self.h.shape, [self.h, self.v], device=self.device)
        self.sweeps.zero_()
        self.running.fill_(1)
        wp.capture_while(self.running, self._sweep)
        if self._normalize_vectors and self.n >= 256:
            # Long rotation sequences accumulate column-length error. Scale
            # only after convergence so H/eigenvalues/stopping are unchanged.
            wp.launch_tiled(
                normalize_eigenvectors,
                dim=self.n,
                inputs=[self.v],
                block_dim=128,
                device=self.device,
            )

    def apply(self):
        if wp.get_stream(self.device).is_capturing:
            self._capture()
            return
        if self.graph is None:
            from . import kernels as K

            wp.load_module(module=K.jacobi_init.module, device=self.device)
            wp.load_module(module=global_jacobi_update.module, device=self.device, block_dim=128)
            # Keep setup graph upload behind pending initialization on drivers
            # affected by #61. No synchronization is performed on later calls.
            with capture_scope(self.device) as capture:
                self._capture()
            self.graph = capture.graph
        wp.capture_launch(self.graph)
