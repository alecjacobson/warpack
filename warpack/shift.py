"""Real or complex shift-invert for general complex CSR, on device."""

import warp as wp
import warp.sparse as ws
from warp.optim import linear

from .arnoldi import (
    ComplexCSR,
    GeneralEigensolver,
    complex_outputs,
    complex_residual,
)
from .hermitian import HermitianEigensolver
from .kernels import identity_order
from .schur import cdiv
from .solver import CuDSSInverse


@wp.kernel
def realify_triplets(
    offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    a: wp.array[wp.vec2d],
    n: int,
    nnz: int,
    sigma: wp.vec2d,
    rows_out: wp.array[wp.int32],
    cols_out: wp.array[wp.int32],
    values: wp.array[wp.float64],
):
    i = wp.tid()
    for e in range(offsets[i], offsets[i + 1]):
        j = cols[e]
        v = a[e]
        rows_out[4 * e] = i
        cols_out[4 * e] = j
        values[4 * e] = v[0]
        rows_out[4 * e + 1] = i
        cols_out[4 * e + 1] = j + n
        values[4 * e + 1] = -v[1]
        rows_out[4 * e + 2] = i + n
        cols_out[4 * e + 2] = j
        values[4 * e + 2] = v[1]
        rows_out[4 * e + 3] = i + n
        cols_out[4 * e + 3] = j + n
        values[4 * e + 3] = v[0]
    e = 4 * nnz + 4 * i
    rows_out[e] = i
    cols_out[e] = i
    values[e] = -sigma[0]
    rows_out[e + 1] = i
    cols_out[e + 1] = i + n
    values[e + 1] = sigma[1]
    rows_out[e + 2] = i + n
    cols_out[e + 2] = i
    values[e + 2] = -sigma[1]
    rows_out[e + 3] = i + n
    cols_out[e + 3] = i + n
    values[e + 3] = -sigma[0]


@wp.kernel
def realified_value_map(
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    source_offsets: wp.array[wp.int32],
    n: int,
    capacity: int,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    mapping: wp.array[wp.int32],
):
    t = wp.tid()
    mapping[t] = -1
    # Source CSR may have unused trailing value/column capacity. Those triplets
    # are initialized to zero at setup and must never read padding on replay.
    if t < 4 * source_offsets[n] or t >= 4 * capacity:
        row, col = rows[t], cols[t]
        lo = int(offsets[row])
        hi = int(offsets[row + 1])
        while lo < hi:
            mid = (lo + hi) // 2
            if columns[mid] < col:
                lo = mid + 1
            else:
                hi = mid
        if lo < offsets[row + 1] and columns[lo] == col:
            mapping[t] = lo


@wp.kernel
def refresh_realified_values(
    source: wp.array[wp.vec2d],
    mapping: wp.array[wp.int32],
    capacity: int,
    sigma: wp.vec2d,
    values: wp.array[wp.float64],
):
    t = wp.tid()
    dest = mapping[t]
    if dest >= 0:
        entry, component = t // 4, t % 4
        v = -sigma
        if entry < capacity:
            v = source[entry]
        value = v[0]
        if component == 1:
            value = -v[1]
        elif component == 2:
            value = v[1]
        # Includes duplicate source coordinates and the diagonal shift term.
        wp.atomic_add(values, dest, value)


@wp.kernel
def unpack_complex(x: wp.array2d[wp.vec2d], out: wp.array2d[wp.float64], j: int, n: int):
    i = wp.tid()
    out[0, i] = x[j, i][0]
    out[0, i + n] = x[j, i][1]


@wp.kernel
def pack_complex(x: wp.array2d[wp.float64], out: wp.array2d[wp.vec2d], j: int, n: int):
    i = wp.tid()
    out[j, i] = wp.vec2d(x[0, i], x[0, i + n])


@wp.kernel
def unshift(values: wp.array[wp.vec2d], out: wp.array[wp.vec2d], sigma: wp.vec2d):
    i = wp.tid()
    out[i] = sigma + cdiv(wp.vec2d(1.0, 0.0), values[i])


@wp.kernel
def unshift_selected(
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    out: wp.array[wp.vec2d],
    sigma: wp.vec2d,
):
    i = wp.tid()
    out[i] = sigma + cdiv(wp.vec2d(1.0, 0.0), values[order[i]])


@wp.kernel
def unshift_hermitian(values: wp.array[wp.float64], out: wp.array[wp.vec2d], sigma: wp.vec2d):
    i = wp.tid()
    out[i] = wp.vec2d(sigma[0] + wp.float64(1.0) / values[i], 0.0)


class ComplexShiftInverse:
    """Apply (A-sigma I)^-1 using cuDSS or pure-Warp GMRES.

    The complex matrix is represented as a real 2n system. Initial sparse
    topology/factor setup stays outside capture. Applies support multiple RHS.
    GMRES refreshes realified coefficients from the current source values on
    every apply, including captured replay with fixed topology/addresses.
    cuDSS factors remain tied to the values at construction; recreate that
    inverse and its graphs when source values change.
    """

    requires_static_outer_capture = True

    def __init__(self, operator, sigma=0j, backend="cudss", inner_tol=1.0e-12, inner_maxiter=1000):
        if backend not in ("cudss", "gmres"):
            raise ValueError("backend must be cudss or gmres")
        self.original = operator
        self.complex_n = operator.complex_n
        self.device = operator.device
        n = self.complex_n
        self.n = 2 * n
        nnz = len(operator.values)
        size = 4 * (nnz + n)
        rows = wp.zeros(size, dtype=wp.int32, device=self.device)
        cols = wp.zeros_like(rows)
        vals = wp.zeros(size, dtype=wp.float64, device=self.device)
        self.sigma = wp.vec2d(float(complex(sigma).real), float(complex(sigma).imag))
        wp.launch(
            realify_triplets,
            n,
            [
                operator.offsets,
                operator.columns,
                operator.values,
                n,
                nnz,
                self.sigma,
                rows,
                cols,
                vals,
            ],
            device=self.device,
        )
        self.matrix = ws.bsr_from_triplets(
            2 * n, 2 * n, rows, cols, vals, prune_numerical_zeros=backend != "gmres"
        )
        self._value_map = None
        if backend == "gmres":
            # Preserve every structural component, including zeros that may
            # become nonzero later. Build only the mapping outside capture.
            self._value_map = wp.empty(size, dtype=wp.int32, device=self.device)
            wp.launch(
                realified_value_map,
                size,
                [
                    rows,
                    cols,
                    operator.offsets,
                    n,
                    nnz,
                    self.matrix.offsets,
                    self.matrix.columns,
                    self._value_map,
                ],
                device=self.device,
            )
        self.rhs = wp.zeros((1, 2 * n), dtype=wp.float64, device=self.device)
        self.solution = wp.empty_like(self.rhs)
        self.backend = backend
        if backend == "cudss":
            # Realification can have many zero diagonal entries. Unmatched
            # general factors can then leave a large error even after one
            # correction. Match/scale during setup and retain FP64 correction.
            self.inverse = CuDSSInverse(
                self.matrix, 1, mtype="general", matching=True, refinement_steps=1
            )
        else:
            self.state = linear.gmres(
                self.matrix,
                self.rhs[0],
                self.solution[0],
                tol=inner_tol,
                maxiter=inner_maxiter,
                restart=min(2 * n, 64),
                check_every=0,
                run=False,
            )

    def _solve_real(self):
        if self.backend == "cudss":
            self.inverse.apply(self.rhs, self.solution)
        else:
            self.solution.zero_()
            self.state(b=self.rhs[0], x=self.solution[0])

    def _refresh(self):
        if self._value_map is not None:
            self.matrix.values.zero_()
            wp.launch(
                refresh_realified_values,
                self._value_map.size,
                [
                    self.original.values,
                    self._value_map,
                    len(self.original.values),
                    self.sigma,
                    self.matrix.values,
                ],
                device=self.device,
            )

    def apply(self, x, y):
        """Apply to realified row vectors, as used by Hermitian Krylov."""
        self._refresh()
        for j in range(x.shape[0]):
            wp.copy(self.rhs, x[j : j + 1])
            self._solve_real()
            wp.copy(y[j : j + 1], self.solution)

    def apply_complex(self, x, y):
        self._refresh()
        for j in range(x.shape[0]):
            wp.launch(
                unpack_complex,
                self.complex_n,
                [x, self.rhs, j, self.complex_n],
                device=self.device,
            )
            self._solve_real()
            wp.launch(
                pack_complex,
                self.complex_n,
                [self.solution, y, j, self.complex_n],
                device=self.device,
            )

    def close(self):
        if self.backend == "cudss":
            self.inverse.close()


class ShiftInvertEigensolver:
    """Eigenpairs nearest a real or complex sigma; original-problem residuals.

    ``hermitian=True`` asserts that the complex CSR operator is Hermitian and
    requires a real shift. It uses realified symmetric Krylov instead of general
    Arnoldi, with optional replayable complex ``initial_vector``. ``ncv`` counts
    complex dimensions and is doubled internally on this path. The output
    eigenvalues retain the general API's ``wp.vec2d`` storage, with zero imaginary
    parts; ``converged`` and ``residuals`` check the original complex equation.
    Both paths accept a replayable complex ``initial_vector``. The general
    path preserves complete converged sets during fixed budgets by default;
    candidates must pass the unshifted original equation, and final residuals
    are recomputed after restoration. ``preserve_converged=False`` disables it.
    This setting does not affect the Hermitian path.
    As with the direct Hermitian API, zero/nonfinite starts set ``status=1``;
    outputs must not be used unless ``status`` is zero.
    """

    def __init__(
        self,
        operator,
        k,
        sigma=0j,
        ncv=None,
        backend="cudss",
        tol=1.0e-9,
        inner_tol=1.0e-12,
        inner_maxiter=1000,
        hermitian=False,
        initial_vector=None,
        preserve_converged=True,
    ):
        if hermitian and complex(sigma).imag != 0:
            raise ValueError("Hermitian shift-invert requires a real sigma")
        self.hermitian = bool(hermitian)
        if self.hermitian and not hasattr(operator, "apply_complex"):
            operator = ComplexCSR(
                operator.complex_n, operator.offsets, operator.columns, operator.values
            )
        self.inverse = ComplexShiftInverse(operator, sigma, backend, inner_tol, inner_maxiter)
        try:
            if self.hermitian:
                self.core = HermitianEigensolver(
                    self.inverse,
                    k,
                    ncv=None if ncv is None else 2 * ncv,
                    which="LM",
                    tol=tol * 0.01,
                    initial_vector=initial_vector,
                )
            else:
                self.core = GeneralEigensolver(
                    self.inverse,
                    k,
                    ncv=ncv,
                    which="LM",
                    tol=tol * 0.01,
                    initial_vector=initial_vector,
                    preserve_converged=preserve_converged,
                )
        except Exception:
            self.inverse.close()
            raise
        self.op = operator
        self.device = operator.device
        self.k = k
        self.tol = tol
        self.status = self.core.real.status if self.hermitian else self.core.status
        self.eigenvectors = self.core.eigenvectors
        self.eigenvalues = wp.empty(k, dtype=wp.vec2d, device=self.device)
        self.residuals = wp.empty(k, dtype=wp.float64, device=self.device)
        self.norms = wp.zeros((k, 1), dtype=wp.float64, device=self.device)
        self.ax = wp.empty_like(self.eigenvectors)
        self.order = wp.empty(k, dtype=wp.int32, device=self.device)
        self.converged = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.value_buffer = wp.empty_like(self.eigenvalues)
        if not self.hermitian:
            self.core._checkpoint_validator = self._checkpoint_candidate

    def _checkpoint_candidate(self):
        # Inverse Ritz residuals do not certify the original equation. Check
        # the unshifted candidate before saving, without modifying core state.
        source = self.inverse.original
        vectors = self.core.eigenvectors
        wp.launch(
            unshift_selected,
            self.k,
            [self.core.ritz, self.core.order, self.value_buffer, self.inverse.sigma],
            device=self.device,
        )
        wp.launch(identity_order, self.k, [self.order], device=self.device)
        source.apply_complex(vectors, self.ax)
        self.norms.zero_()
        wp.launch_tiled(
            complex_residual,
            dim=(self.k, (source.complex_n + 255) // 256),
            inputs=[vectors, self.ax, self.value_buffer, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        return self.norms, self.value_buffer, self.order, self.tol

    def finalize(self):
        if self.hermitian:
            self.core.finalize()
        wp.launch(
            unshift_hermitian if self.hermitian else unshift,
            self.k,
            [self.core.eigenvalues, self.value_buffer, self.inverse.sigma],
            device=self.device,
        )
        wp.launch(identity_order, self.k, [self.order], device=self.device)
        self.op.apply_complex(self.eigenvectors, self.ax)
        self.norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            complex_residual,
            dim=(self.k, (self.op.complex_n + 255) // 256),
            inputs=[
                self.eigenvectors,
                self.ax,
                self.value_buffer,
                self.order,
                self.norms,
            ],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            complex_outputs,
            self.k,
            [
                self.value_buffer,
                self.order,
                self.norms,
                self.eigenvalues,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )

        if not self.hermitian:
            self.core._check_start(self.converged)

    def solve(self, iterations=10):
        (self.core.real if self.hermitian else self.core).solve(iterations)
        self.finalize()
        return self

    def capture(self, iterations=10):
        core = self.core.real if self.hermitian else self.core
        self.graph = core.capture(iterations, adaptive=False, finalizer=self.finalize)
        return self.graph

    def close(self):
        self.graph = self.core.graph = None
        if self.hermitian:
            self.core.real.graph = None
        self.inverse.close()
