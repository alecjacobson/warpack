"""Reusable, graph-capturable symmetric block eigensolver."""

import math

import warp as wp

from . import kernels as K
from . import krylov_kernels as KR
from . import linalg_tiles as KT


class SparseOperator:
    def __init__(self, matrix, shift=0.0, sign=1.0):
        if matrix.shape[0] < 1 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("operator must be square")
        if not (
            wp.types.types_equal(matrix.values.dtype, wp.float64)
            or wp.types.types_equal(matrix.values.dtype, wp.mat33d)
        ):
            raise TypeError("expected scalar FP64 or 3x3 FP64 BSR")
        if matrix.row_counts is not None:
            raise ValueError("Call warp.sparse.bsr_compress before constructing an operator")
        self.matrix = matrix
        self.n = matrix.shape[0]
        self.device = matrix.device
        self.shift, self.sign = shift, sign

    def apply(self, x, y):
        a = self.matrix
        if a.values.dtype == wp.float64 and a.nnz / a.nrow > 64:
            wp.launch_tiled(
                K.csr_mv_cooperative,
                dim=x.shape,
                inputs=[a.offsets, a.columns, a.values, x, y, self.shift, self.sign],
                block_dim=32,
                device=self.device,
            )
            return
        kernel = K.csr_mm if a.values.dtype == wp.float64 else K.bsr_mm
        wp.launch(
            kernel,
            x.shape,
            [a.offsets, a.columns, a.values, x, y, self.shift, self.sign],
            device=self.device,
        )


class CuDSSInverse:
    """Factorization is setup work; apply() is graph capturable.

    The matrix must already include any shift/mass normalization. General and
    symmetric-indefinite factorizations use one Warp residual-correction step
    by default; SPD factorizations use none. ``matching=True`` enables cuDSS's
    column matching/scaling for general factors and defaults to zero correction
    steps. Matching is setup work, not a replay convergence decision.
    ``refinement_steps`` overrides this fixed budget. ``reordering`` selects a
    cuDSS factorization strategy; global
    pivoting with ``colamd``/``btf_colamd`` should use ``mtype="general"``.
    Correction products and updates run in Warp, with cuDSS only applying the
    existing factors. Check original residuals for the chosen configuration.
    ``cooperative_refinement=True`` evaluates correction products with a warp
    per CSR row; useful for KKT matrices with a few very long constraint rows.
    The default retains the thread-per-row product.
    """

    requires_static_outer_capture = True

    def __init__(
        self,
        matrix,
        width,
        mtype="spd",
        refinement_steps=None,
        reordering="default",
        matching=False,
        cooperative_refinement=False,
    ):
        if width < 1 or matrix.scalar_type != wp.float64:
            raise ValueError("cuDSS inverse needs a positive RHS width and FP64 matrix")
        if matching and (mtype != "general" or reordering in ("colamd", "btf_colamd")):
            raise ValueError("matching requires general factors without COLAMD reordering")
        if refinement_steps is None:
            refinement_steps = 0 if mtype in ("spd", "hpd") or matching else 1
        if not isinstance(refinement_steps, int) or refinement_steps < 0:
            raise ValueError("refinement_steps must be a nonnegative integer")
        self.refinement_steps = refinement_steps
        self.reordering = reordering
        self.matching = bool(matching)
        self.cooperative_refinement = bool(cooperative_refinement)
        self.mtype = mtype
        self.width = width
        import os
        from importlib.metadata import distribution

        if "CUDSS_LIBRARY_PATH" not in os.environ:
            try:
                dist = distribution("nvidia-cudss-cu12")
                for path in dist.files:
                    if str(path).endswith("/libcudss.so.0"):
                        os.environ["CUDSS_LIBRARY_PATH"] = str(dist.locate_file(path))
                        break
            except ModuleNotFoundError:
                pass
        from warp_cudss import CudssSolver

        self.n, self.device = matrix.shape[0], matrix.device
        self.x = wp.empty((width, self.n), dtype=wp.float64, device=self.device)
        self.b = wp.empty_like(self.x)
        if refinement_steps:
            self._rhs = wp.empty_like(self.x)
            self._residual = wp.empty_like(self.x)
            self._correction = wp.empty_like(self.x)
        self.solver = CudssSolver(mtype=mtype, reordering=reordering, device=self.device)
        from .cudss_memory import Workspace

        self.workspace = Workspace(self.solver)
        # Warp nnz is a capacity bound until synchronized after triplet assembly.
        # The optional wrapper expands exactly nnz blocks, so finalize it at setup.
        matrix.nnz_sync()
        try:
            if matching:
                import ctypes

                from warp_cudss._bindings import CUDSS_CONFIG_MATCHING_ALG, _check

                # cuDSS 0.8: MC64 maximum diagonal product, including scaling.
                value = ctypes.c_int(5)
                _check(
                    self.solver._lib.cudssConfigSet(
                        self.solver._config,
                        CUDSS_CONFIG_MATCHING_ALG,
                        ctypes.byref(value),
                        ctypes.sizeof(value),
                    ),
                    "cudssConfigSet(MATCHING_ALG)",
                )
            self.solver.setup(matrix, self.x, self.b, nrhs=width)
        except Exception:
            if self.workspace.error:
                raise self.workspace.error
            raise

    def apply(self, x, y):
        if x.shape != (self.width, self.n) or y.shape != x.shape:
            raise ValueError("cuDSS RHS shape must match the width used at setup")
        if not self.refinement_steps:
            self.solver.solve(x=y, b=x)
            return
        # Preserve b, including when the caller requests an in-place solve.
        wp.copy(self._rhs, x)
        self.solver.solve(x=y, b=self._rhs)
        # Use the wrapper's scalar expansion, including arbitrary BSR blocks.
        # These are the same original matrix values supplied for factorization.
        csr = self.solver._csr
        for _ in range(self.refinement_steps):
            args = [csr.row_offsets, csr.columns, csr.values, y, self._residual, 0.0, 1.0]
            if self.cooperative_refinement:
                wp.launch_tiled(
                    K.csr_mv_cooperative, dim=y.shape, inputs=args, block_dim=32, device=self.device
                )
            else:
                wp.launch(K.csr_mm, y.shape, args, device=self.device)
            wp.launch(K.linear_residual, y.shape, [self._rhs, self._residual], device=self.device)
            self.solver.solve(x=self._correction, b=self._residual)
            wp.launch(K.add_correction, y.shape, [y, self._correction], device=self.device)

    def inertia(self):
        """Explicit, synchronizing host factor metadata; never called by solve/capture.

        Only qualified symmetric-indefinite factors can return available
        inertia. See InertiaResult and LowestModeCertificate for limitations.
        """
        from .cudss_diagnostics import query_inertia

        return query_inertia(self)

    def close(self):
        self.solver.release()


class SymmetricEigensolver:
    """FP64 orthogonal iteration with CholeskyQR2 and Rayleigh–Ritz.

    `operator` is the original symmetric operator; `iteration_operator` can be
    its inverse or a shifted/scaled operator that amplifies the wanted spectrum.
    All work arrays are allocated at construction. `initialize`, `step`, and
    `finalize` can be captured. Results use row-major (k,n) eigenvector storage.
    Fixed iteration budgets make replay deterministic; inspect `converged` and
    `residuals` on device or explicitly transfer them after completion.

    ``initial_vectors`` optionally supplies 1..ncv independent FP64 row vectors
    on the operator device. Remaining rows are seeded randomly. Initialization
    reads its current contents without modifying it, so a captured graph can
    reuse an updated starting subspace. The combined block must have full row
    rank; inspect ``status`` after replay. A starting subspace does not certify
    that all requested eigenvalues have been found.
    """

    def __init__(
        self,
        operator,
        k,
        ncv=None,
        which="SA",
        iteration_operator=None,
        tol=1.0e-9,
        target=None,
        initial_vectors=None,
    ):
        self.op = operator
        self.iter_op = iteration_operator or operator
        self.n, self.device = operator.n, operator.device
        self.k = k
        self.m = min(self.n, max(2 * k, k + 8) if ncv is None else ncv)
        if not 1 <= k <= self.m <= self.n:
            raise ValueError("require 1 <= k <= ncv <= n")
        if which not in ("SA", "LA", "LM", "SM", "BE"):
            raise ValueError("which must be SA, LA, LM, SM, or BE")
        self.which = ("SA", "LA", "LM", "SM", "BE").index(which)
        self.target = target
        if target is not None:
            self.which = 5
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        if iteration_operator is None and self.m < self.n and which != "LM":
            if type(self) is SymmetricEigensolver:
                raise ValueError(
                    "Block iteration needs an explicit iteration_operator for this selection; use KrylovSchur otherwise"
                )
        self.tol = tol
        self.initial_vectors = initial_vectors
        if initial_vectors is not None:
            if (
                initial_vectors.dtype != wp.float64
                or initial_vectors.ndim != 2
                or initial_vectors.shape[1] != self.n
                or not 1 <= initial_vectors.shape[0] <= self.m
            ):
                raise ValueError("initial_vectors must have shape (1..ncv, n) and dtype FP64")
            if initial_vectors.device != self.device:
                raise ValueError("initial_vectors must be on the operator device")

        def empty(shape, dtype=wp.float64):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.q, self.z = empty((self.m, self.n)), empty((self.m, self.n))
        self._initial_rows = (
            self.q[: initial_vectors.shape[0]] if initial_vectors is not None else None
        )
        self.g, self.l, self.v = [empty((self.m, self.m)) for _ in range(3)]
        self.coeff = empty((self.m, self.m))
        self.jtmp, self.vtmp = empty((self.m, self.m)), empty((self.m, self.m))
        self.cs, self.ss = empty(self.m), empty(self.m)
        self.partners = empty(self.m, wp.int32)
        self.ritz = empty(self.m)
        self.order = empty(self.m, wp.int32)
        self.sort_work = empty(self.m, wp.int32)
        self.status = empty(1, wp.int32)
        self.eigenvalues, self.residuals = empty(k), empty(k)
        self.eigenvectors, self.ax = empty((k, self.n)), empty((k, self.n))
        self.norms = empty((k, 1))
        self.converged = empty(1, wp.int32)
        self.iterations = empty(1, wp.int32)
        self.running = empty(1, wp.int32)
        # Measured Ada crossover: the fused global rounds outperform the
        # shared kernel from 48 columns. Preserve other architectures' policy.
        self._prefer_global_projected = (
            self.m >= 48 and self.device.is_cuda and self.device.arch == 89
        )
        self._global_jacobi = None
        if (
            (self.m > 64 or self._prefer_global_projected)
            and self.m % 2 == 0
            and self.device.is_cuda
            and wp.is_conditional_graph_supported()
        ):
            from .projected import GlobalJacobi

            self._global_jacobi = GlobalJacobi(
                self.g, self.v, self.cs, self.ss, self.partners, self.jtmp, self.vtmp
            )

    def rotate(self, q, out):
        wp.launch(
            K.rotation_coefficients,
            (out.shape[0], self.m),
            [self.v, self.order, self.coeff],
            device=self.device,
        )
        wp.launch_tiled(
            KT.rotate_tiled,
            dim=((out.shape[0] + 3) // 4, (self.n + 31) // 32),
            inputs=[q, self.coeff, out],
            block_dim=128,
            device=self.device,
        )

    def diagonalize(self):
        self._diagonalize_projected()
        if self.which >= 4:
            wp.launch(
                K.sort_extra,
                1,
                [
                    self.ritz,
                    self.sort_work,
                    self.order,
                    self.which,
                    float(self.target or 0.0),
                ],
                device=self.device,
            )

    def _diagonalize_projected(self):
        if self.m % 2:
            wp.launch(
                K.jacobi,
                1,
                [self.g, self.v, self.ritz, self.order, self.which],
                device=self.device,
            )
            return
        if self._global_jacobi is not None:
            self._global_jacobi.apply()
            wp.launch(
                K.sort_diagonal,
                self.m,
                [self.g, self.ritz, self.order, self.which],
                device=self.device,
            )
            return
        if self.m <= 64:
            from .projected import jacobi_kernel

            wp.launch_tiled(
                jacobi_kernel(self.m),
                dim=1,
                inputs=[self.g, self.v],
                block_dim=256,
                device=self.device,
            )
            wp.launch(
                K.sort_diagonal,
                self.m,
                [self.g, self.ritz, self.order, self.which],
                device=self.device,
            )
            return
        wp.launch(K.jacobi_init, (self.m, self.m), [self.g, self.v], device=self.device)
        for _ in range(30):
            for r in range(self.m - 1):
                wp.launch(
                    K.jacobi_pairs,
                    self.m // 2,
                    [self.g, self.cs, self.ss, self.partners, r],
                    device=self.device,
                )
                wp.launch(
                    K.jacobi_right,
                    (self.m, self.m),
                    [
                        self.g,
                        self.v,
                        self.jtmp,
                        self.vtmp,
                        self.cs,
                        self.ss,
                        self.partners,
                    ],
                    device=self.device,
                )
                wp.launch(
                    K.jacobi_left,
                    (self.m, self.m),
                    [
                        self.g,
                        self.v,
                        self.jtmp,
                        self.vtmp,
                        self.cs,
                        self.ss,
                        self.partners,
                    ],
                    device=self.device,
                )
        wp.launch(
            K.sort_diagonal,
            self.m,
            [self.g, self.ritz, self.order, self.which],
            device=self.device,
        )

    def gram(self, x, y):
        self.g.zero_()
        wp.launch_tiled(
            KT.gram_tiled,
            dim=((self.m + 7) // 8, (self.m + 7) // 8, (self.n + 255) // 256),
            inputs=[x, y, self.g],
            block_dim=128,
            device=self.device,
        )

    def orthogonalize(self):
        for _ in range(2):
            self.gram(self.q, self.q)
            wp.launch(K.cholesky, 1, [self.g, self.l, self.status], device=self.device)
            wp.launch(K.triangular, self.n, [self.q, self.l], device=self.device)

    def initialize(self, seed=42):
        self.status.zero_()
        if self.initial_vectors is None or self.initial_vectors.shape[0] < self.m:
            wp.launch(K.random_block, self.q.shape, [self.q, seed], device=self.device)
        if self.initial_vectors is not None:
            wp.copy(self._initial_rows, self.initial_vectors)
        self.orthogonalize()

    def step(self):
        self.iter_op.apply(self.q, self.z)
        wp.copy(self.q, self.z)
        self.orthogonalize()

    def finalize(self):
        self.op.apply(self.q, self.z)
        self.gram(self.q, self.z)
        self.diagonalize()
        self.rotate(self.q, self.eigenvectors)
        self.op.apply(self.eigenvectors, self.ax)
        self.norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            K.residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.eigenvectors, self.ax, self.ritz, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            K.output_values,
            self.k,
            [
                self.ritz,
                self.order,
                self.eigenvalues,
                self.norms,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )

    def solve(self, iterations=50, seed=42):
        if iterations < 0:
            raise ValueError("iterations must be nonnegative")
        self.initialize(seed)
        for _ in range(iterations):
            self.step()
        self.finalize()
        return self

    def _update_loop(self, max_iterations):
        wp.launch(
            K.loop_update,
            1,
            [self.converged, self.status, self.iterations, self.running, self.k, max_iterations],
            device=self.device,
        )

    def _prepare_capture(self, finalizer):
        self.solve(1)
        if finalizer is not None:
            finalizer()

    def capture(self, max_iterations=100, adaptive=None, finalizer=None):
        """Capture a GPU-controlled convergence loop (CUDA 12.4+).

        Setup/JIT/warmup happen here; replay never transfers convergence values
        to the host. `iterations`, `converged`, and `status` remain on device.
        Default outer capture respects restrictions declared by both the
        iteration and physical operators, plus a subclass's mass operator.
        """
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        static_only = any(
            getattr(op, "requires_static_outer_capture", False)
            for op in (self.iter_op, self.op, getattr(self, "mass", None))
        )
        if adaptive is None:
            adaptive = not static_only
        if adaptive and static_only:
            raise ValueError("This operator backend requires adaptive=False for the outer graph")
        self._prepare_capture(finalizer)
        wp.synchronize_device(self.device)

        def body():
            self.step()
            self.finalize()
            self._update_loop(max_iterations)

        with wp.ScopedCapture(device=self.device) as capture:
            self.initialize()
            self.iterations.zero_()
            self.running.fill_(1)
            if adaptive:
                wp.capture_while(self.running, body)
            else:
                for _ in range(max_iterations):
                    self.step()
                self.finalize()
                self.iterations.fill_(max_iterations)
            if finalizer is not None:
                finalizer()
        self.graph = capture.graph
        return self.graph


class KrylovSchur(SymmetricEigensolver):
    """Thick-restarted symmetric Krylov / Rayleigh–Ritz iteration.

    Expands the retained Ritz space by operator applications, with two-pass
    classical Gram–Schmidt. Unlike orthogonal iteration, this targets algebraic
    spectrum ends without requiring a spectral shift bound.

    An optional nonzero FP64 ``initial_vector`` supplies the starting direction.
    Initialization normalizes its current device contents, so captured replays
    can use updated starts without recapturing. Otherwise ``seed`` controls the
    random start.

    Adaptive replay also waits for independent-chain exploration (unless the
    basis spans the full problem). Thus it can continue after all ``k`` returned
    residuals pass. ``converged`` counts residual-passing pairs; neither that
    count nor a fixed iteration budget proves global spectrum completeness.

    Unless sparse products are explicitly reapplied after restart, residuals
    use rotated cached operator images (including inverse and generic callback
    operators). At tight tolerances these are estimates, not fresh applications of
    the operator. ``EigenpairEvaluation(..., eigenvalues=solver.eigenvalues)``
    checks the returned pairs with fresh products; use ``rhs_width=1`` for a
    one-RHS inverse. This optional final check does not alter iteration stopping.
    """

    def __init__(self, operator, k, ncv=None, which="LM", tol=1.0e-9, initial_vector=None):
        super().__init__(operator, k, ncv=ncv, which=which, tol=tol)
        self.initial_vector = initial_vector
        if initial_vector is not None:
            if initial_vector.dtype != wp.float64 or initial_vector.size != self.n:
                raise ValueError("initial_vector must contain n FP64 entries")
            if initial_vector.device != self.device:
                raise ValueError("initial_vector must be on the operator device")
            self.initial_vector = initial_vector.reshape((1, self.n))
        # Extremal targets need expansion room to resolve clusters. Interior SM
        # Ritz selection instead needs the larger retained space: reducing it
        # loses accuracy on the transformed-SM SymEigsShift_17 regression.
        retain_limit = self.m - 1 if which == "SM" else max(k, self.m // 2)
        self.keep = min(k + 4, retain_limit)
        # Sparse products can be cheaper and more accurate than a dense basis
        # rotation. Keep cached products for dense operators and inverses.
        self._reapply_after_restart = (
            isinstance(operator, SparseOperator)
            and operator.matrix.nnz / operator.matrix.nrow * operator.matrix.block_shape[1]
            <= self.m // 2
        )
        if self.m <= k:
            raise ValueError("Krylov restart requires ncv > k")
        self.dots = wp.zeros((self.m, 1), dtype=wp.float64, device=self.device)
        self.norm = wp.zeros((1, 1), dtype=wp.float64, device=self.device)
        self.source_norm = wp.zeros_like(self.norm)
        self.breakdown = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.recoveries = wp.zeros_like(self.breakdown)
        self.dot_parts = wp.empty(
            (self.m, (self.n + 1023) // 1024), dtype=wp.float64, device=self.device
        )
        self.norm_parts = wp.empty((self.n + 1023) // 1024, dtype=wp.float64, device=self.device)
        self.retained = wp.empty((self.keep, self.n), dtype=wp.float64, device=self.device)
        self.retained_ax = wp.empty_like(self.retained)
        self.restart_scores = wp.zeros((self.k, 2), dtype=wp.float64, device=self.device)
        self.restart_seed = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.views = [self.q[j : j + 1] for j in range(self.m)]
        self.zviews = [self.z[j : j + 1] for j in range(self.m)]

    def solve(self, iterations=50, seed=42):
        if iterations < 1:
            raise ValueError("Krylov iteration needs at least one restart cycle")
        return super().solve(iterations, seed)

    def _prepare_capture(self, finalizer):
        # CUDA >=12.3 can load Warp kernels during capture. The built-in cuDSS
        # operator only needs one solve to prepare its vendor state, not an
        # entire Krylov cycle (ncv inverse applications plus orthogonalization).
        # Preserve full warmup for custom solvers/operators and callbacks that
        # may inspect initialized eigenpairs while preparing their own kernels.
        driver = wp.get_cuda_driver_version() if self.device.is_cuda else None
        if (
            type(self) is KrylovSchur
            and type(self.op) is CuDSSInverse
            and self.iter_op is self.op
            and finalizer is None
            and driver is not None
            and driver >= (12, 3)
        ):
            # Use eigensolver scratch: callers may supply the inverse's own
            # x/b arrays as starting vectors, which capture must not overwrite.
            self.views[0].zero_()
            self.op.apply(self.views[0], self.zviews[0])
        else:
            super()._prepare_capture(finalizer)

    def _update_loop(self, max_iterations):
        wp.launch(
            KR.loop_update,
            1,
            [
                self.converged,
                self.status,
                self.iterations,
                self.running,
                self.recoveries,
                self.k,
                max_iterations,
                self.m == self.n,
            ],
            device=self.device,
        )

    def _orthogonalize_column(self, j, check):
        for pass_index in range(2):
            rows = j + int(pass_index == 0)
            wp.launch_tiled(
                KR.dot_partials,
                dim=(rows, self.dot_parts.shape[1]),
                inputs=[self.q, self.dot_parts, j],
                block_dim=128,
                device=self.device,
            )
            wp.launch_tiled(
                KR.finish_dots,
                dim=rows,
                inputs=[
                    self.dot_parts,
                    self.dots,
                    self.source_norm,
                    self.g,
                    j,
                    check and pass_index == 0,
                ],
                block_dim=128,
                device=self.device,
            )
            wp.launch_tiled(
                KR.subtract,
                dim=self.norm_parts.size,
                inputs=[self.q, self.dots, self.norm_parts, j, pass_index == 1],
                block_dim=128,
                device=self.device,
            )
        wp.launch_tiled(
            KR.finish_norm,
            dim=1,
            inputs=[
                self.norm_parts,
                self.norm,
                self.source_norm,
                self.breakdown,
                self.recoveries,
                check,
            ],
            block_dim=128,
            device=self.device,
        )

    def expand(self, j):
        self.op.apply(self.views[j - 1], self.zviews[j - 1])
        wp.copy(self.views[j], self.zviews[j - 1])
        self._orthogonalize_column(j, True)

        def recover():
            wp.launch(
                K.fallback_seed,
                self.n,
                [self.q, self.breakdown, self.recoveries, j],
                device=self.device,
            )
            self._orthogonalize_column(j, False)

        if wp.get_stream(self.device).is_capturing:
            wp.capture_if(self.breakdown, recover)
        else:
            # Eager execution also keeps the decision on device.
            self._recover_eager(j)
        wp.launch(K.basis_scale, self.n, [self.q, self.norm, j], device=self.device)

    def _recover_eager(self, j):
        # Gated kernels are only used outside captured replay.
        wp.launch(
            K.fallback_seed,
            self.n,
            [self.q, self.breakdown, self.recoveries, j],
            device=self.device,
        )
        for _ in range(2):
            self.dots.zero_()
            wp.launch_tiled(
                K.fallback_dots,
                dim=(j, (self.n + 255) // 256),
                inputs=[self.q, self.dots, self.breakdown, j],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                K.fallback_subtract,
                self.n,
                [self.q, self.dots, self.breakdown, j],
                device=self.device,
            )
        self.norm.zero_()
        wp.launch_tiled(
            K.basis_norm,
            dim=(self.n + 255) // 256,
            inputs=[self.q, self.norm, j],
            block_dim=128,
            device=self.device,
        )

    def initialize(self, seed=42):
        self.recoveries.zero_()
        self.g.zero_()
        self.status.zero_()
        if self.initial_vector is None:
            wp.launch(K.random_block, (1, self.n), [self.q, seed], device=self.device)
        else:
            wp.copy(self.views[0], self.initial_vector)
        self.norm.zero_()
        wp.launch_tiled(
            K.basis_norm,
            dim=(self.n + 255) // 256,
            inputs=[self.q, self.norm, 0],
            block_dim=128,
            device=self.device,
        )
        wp.launch(K.basis_scale, self.n, [self.q, self.norm, 0], device=self.device)
        for j in range(1, self.keep):
            self.expand(j)

    def step(self):
        for j in range(self.keep, self.m):
            self.expand(j)
        self.op.apply(self.views[-1], self.zviews[-1])
        # Orthogonalization supplied every new projected column except the last.
        self.complete_projection()
        self.diagonalize()
        self.rotate(self.q, self.retained)
        self.update_retained_products()
        wp.copy(self.q, self.retained, count=self.keep * self.n)
        wp.copy(self.z, self.retained_ax, count=self.keep * self.n)
        self.reset_projection()

    def update_retained_products(self):
        if self._reapply_after_restart:
            self.op.apply(self.retained, self.retained_ax)
        else:
            self.rotate(self.z, self.retained_ax)

    def reset_projection(self):
        # After recovery, retained Ritz residuals need not be collinear. Expand
        # the least converged requested pair instead of an arbitrary last row.
        # Only q/z are permuted: retained outputs stay in requested spectrum order.
        self.restart_scores.zero_()
        wp.launch_tiled(
            KR.restart_residuals,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.retained, self.retained_ax, self.ritz, self.order, self.restart_scores],
            block_dim=128,
            device=self.device,
        )
        wp.launch(KR.restart_seed, 1, [self.restart_scores, self.restart_seed], device=self.device)
        wp.launch(
            KR.restart_swap,
            self.n,
            [self.q, self.z, self.restart_seed, self.keep],
            device=self.device,
        )
        # The retained Ritz block is diagonal; future columns start at zero.
        wp.launch(
            KR.restart_projection,
            self.g.shape,
            [self.g, self.ritz, self.order, self.keep, self.restart_seed],
            device=self.device,
        )

    def complete_projection(self):
        wp.launch_tiled(
            KR.last_projection,
            dim=(self.m, self.dot_parts.shape[1]),
            inputs=[self.q, self.z, self.dot_parts],
            block_dim=128,
            device=self.device,
        )
        wp.launch_tiled(
            KR.finish_projection,
            dim=self.m,
            inputs=[self.dot_parts, self.g],
            block_dim=128,
            device=self.device,
        )

    def finalize(self):
        # The retained rows already are Ritz vectors. Remaining old basis rows
        # must not be included in a second projection after the restart.
        wp.copy(self.eigenvectors, self.retained, count=self.k * self.n)
        wp.copy(self.ax, self.retained_ax, count=self.k * self.n)
        self.norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            K.residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.eigenvectors, self.ax, self.ritz, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            K.output_values,
            self.k,
            [
                self.ritz,
                self.order,
                self.eigenvalues,
                self.norms,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )


class EigenpairEvaluation:
    """Evaluate fresh real standard-problem residuals for unit-norm vectors.

    By default, recompute Rayleigh values. Supplying ``eigenvalues`` instead
    checks the current provided values without replacing them by Rayleigh
    values. ``residuals`` is ``||A*x-lambda*x|| / max(1, abs(lambda))`` and
    ``converged`` counts passing pairs; neither certifies spectrum selection.

    ``rhs_width`` optionally splits operator applications into fixed-size row
    batches, e.g. 1 for a one-RHS inverse. It must divide the vector count.
    All work remains on device and can be recorded as a capture finalizer.
    This independent check does not change the eigensolver's stopping rule or
    cached residual estimates. Input array identities/topology must stay fixed.
    """

    def __init__(self, operator, vectors, tol=1.0e-9, *, eigenvalues=None, rhs_width=None):
        self.op, self.vectors, self.tol = operator, vectors, tol
        self.k, self.n = vectors.shape
        self.device = operator.device
        if eigenvalues is not None and (
            eigenvalues.shape != (self.k,)
            or eigenvalues.dtype != wp.float64
            or eigenvalues.device != self.device
        ):
            raise ValueError("eigenvalues must be FP64 (k,) on the operator device")
        if rhs_width is not None:
            from operator import index

            rhs_width = index(rhs_width)
            if rhs_width < 1 or self.k % rhs_width:
                raise ValueError("rhs_width must be positive and divide the vector count")
        self.input_eigenvalues, self.rhs_width = eigenvalues, rhs_width
        self.ax = wp.empty_like(vectors)
        self.quotients = wp.zeros((self.k, 1), dtype=wp.float64, device=self.device)
        self.flat = self.quotients.flatten()
        self.order = wp.empty(self.k, dtype=wp.int32, device=self.device)
        self.values = wp.empty(self.k, dtype=wp.float64, device=self.device)
        self.residuals = wp.empty_like(self.values)
        self.norms = wp.zeros((self.k, 1), dtype=wp.float64, device=self.device)
        self.converged = wp.zeros(1, dtype=wp.int32, device=self.device)

    def run(self):
        if self.rhs_width is None:
            self.op.apply(self.vectors, self.ax)
        else:
            for start in range(0, self.k, self.rhs_width):
                rows = slice(start, start + self.rhs_width)
                self.op.apply(self.vectors[rows], self.ax[rows])
        self.norms.zero_()
        self.converged.zero_()
        wp.launch(K.identity_order, self.k, [self.order], device=self.device)
        if self.input_eigenvalues is None:
            self.quotients.zero_()
            wp.launch_tiled(
                K.rayleigh,
                dim=(self.k, (self.n + 255) // 256),
                inputs=[self.vectors, self.ax, self.quotients],
                block_dim=128,
                device=self.device,
            )
        else:
            wp.copy(self.flat, self.input_eigenvalues)
        wp.launch_tiled(
            K.residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.vectors, self.ax, self.flat, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            K.output_values,
            self.k,
            [
                self.flat,
                self.order,
                self.values,
                self.norms,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )
