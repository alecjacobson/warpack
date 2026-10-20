"""Finite right modes of nonsymmetric generalized pencils, entirely on device."""

import math
from functools import partial
from operator import index

import warp as wp

from .arnoldi import GeneralEigensolver
from .hermitian import cmul, norm2
from .kernels import identity_order
from .schur import cabs
from .shift import unshift_selected


@wp.kernel
def pencil_norms(
    ax: wp.array2d[wp.vec2d],
    bx: wp.array2d[wp.vec2d],
    values: wp.array[wp.vec2d],
    norms: wp.array2d[wp.float64],
):
    j, chunk = wp.tid()
    a = wp.tile_load(ax[j], shape=256, offset=chunk * 256)
    b = wp.tile_load(bx[j], shape=256, offset=chunk * 256)
    lam = wp.tile_full(shape=256, value=values[j], dtype=wp.vec2d)
    residual = a - wp.tile_map(cmul, b, lam)
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(norm2, residual)), offset=0)
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(norm2, a)), offset=1)
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(norm2, b)), offset=2)


@wp.kernel
def pencil_errors(
    values: wp.array[wp.vec2d],
    norms: wp.array2d[wp.float64],
    errors_squared: wp.array2d[wp.float64],
):
    j = wp.tid()
    value = values[j]
    denominator = wp.sqrt(norms[j, 1]) + wp.max(wp.float64(1.0), cabs(value)) * wp.sqrt(norms[j, 2])
    error = wp.float64(wp.inf)
    if (
        denominator > wp.float64(0.0)
        and wp.isfinite(denominator)
        and wp.isfinite(norms[j, 0])
        and wp.isfinite(value[0])
        and wp.isfinite(value[1])
    ):
        error = wp.sqrt(norms[j, 0]) / denominator
    errors_squared[j, 0] = error * error


@wp.kernel
def pencil_outputs(
    errors_squared: wp.array2d[wp.float64],
    residuals: wp.array[wp.float64],
    status: wp.array[wp.int32],
    converged: wp.array[wp.int32],
    tol: wp.float64,
):
    j = wp.tid()
    error = wp.sqrt(errors_squared[j, 0])
    residuals[j] = error
    if status[0] == 0 and wp.isfinite(error) and error <= tol:
        wp.atomic_add(converged, 0, 1)


class _PencilOperator:
    def __init__(self, mass, inverse):
        self.mass, self.inverse = mass, inverse
        self.complex_n, self.device = mass.complex_n, mass.device
        self.requires_static_outer_capture = getattr(inverse, "requires_static_outer_capture", True)
        self.rhs = wp.empty((1, self.complex_n), dtype=wp.vec2d, device=self.device)

    def apply_complex(self, x, y):
        # Single-row storage is reused for arbitrary block widths. No dynamic
        # device allocation or callback is needed during capture/replay.
        for j in range(x.shape[0]):
            self.mass.apply_complex(x[j : j + 1], self.rhs)
            self.inverse.apply_complex(self.rhs, y[j : j + 1])


class _PencilEvaluation:
    """Callback state owns operators/workspace, never its owning solver."""

    def __init__(self, operator, mass, k, sigma, tol):
        self.operator, self.mass = operator, mass
        self.sigma = wp.vec2d(sigma.real, sigma.imag)
        self.tol = tol
        self.values = wp.empty(k, dtype=wp.vec2d, device=operator.device)
        self.ax = wp.empty((k, operator.complex_n), dtype=wp.vec2d, device=operator.device)
        self.bx = wp.empty_like(self.ax)
        self.norms = wp.zeros((k, 3), dtype=wp.float64, device=operator.device)
        self.errors_squared = wp.zeros((k, 1), dtype=wp.float64, device=operator.device)
        self.order = wp.empty(k, dtype=wp.int32, device=operator.device)
        # GeneralCheckpoint divides sqrt(norms) by max(1,abs(value)). These
        # zeros keep that denominator one for our already normalized errors.
        self.unit_scaling = wp.zeros(k, dtype=wp.vec2d, device=operator.device)
        wp.launch(identity_order, k, [self.order], device=operator.device)

    def evaluate(self, vectors, values, order):
        k, device = vectors.shape[0], self.operator.device
        wp.launch(unshift_selected, k, [values, order, self.values, self.sigma], device=device)
        self.operator.apply_complex(vectors, self.ax)
        self.mass.apply_complex(vectors, self.bx)
        self.norms.zero_()
        wp.launch_tiled(
            pencil_norms,
            dim=(k, (self.operator.complex_n + 255) // 256),
            inputs=[self.ax, self.bx, self.values, self.norms],
            block_dim=128,
            device=device,
        )
        wp.launch(pencil_errors, k, [self.values, self.norms, self.errors_squared], device=device)
        return self.errors_squared, self.unit_scaling, self.order, self.tol


class GeneralizedShiftInvertEigensolver:
    """Finite right eigenpairs of ``A x = lambda B x`` nearest ``sigma``.

    Supply ``operator=A``, ``mass=B`` and an inverse of ``A-sigma*B``. Each
    provides ``complex_n``, ``device`` and ``apply_complex(x,y)`` on row-major
    ``wp.vec2d`` arrays. Real matrices use zero imaginary components. The
    inverse is caller-owned and must match the current pencil and shift.
    ``ComplexShiftInverse(shifted, sigma=0, backend=...)`` is one implementation;
    construct ``shifted=A-sigma*B`` before supplying that inverse.

    Arnoldi acts on ``(A-sigma*B)^-1 B`` without forming ``B^-1 A``. B need not
    be positive definite or nonsingular, but the pencil must be regular and
    the shifted operator invertible. Only finite nearest modes are supported;
    infinite eigenvalues, singular pencils and left modes are not supported.
    Vectors have unit Euclidean norm, not B-orthogonality.

    Device ``residuals`` are fresh original-equation errors
    ``||Ax-lambda Bx|| / (||Ax|| + max(1,|lambda|)*||Bx||)``. This includes a
    nonzero scale at a zero eigenvalue and is invariant to common rescaling of
    A and B. Zero/nonfinite denominators and infinite values are rejected.
    Require zero ``status`` and ``converged==k``; these do not certify that an
    undiscovered requested mode is absent. Independently validate coverage.
    Fixed-budget checkpoints also require this physical residual, rather than
    trusting the transformed residual of an inaccurate supplied inverse.

    Captured starts are device ``wp.vec2d`` vectors of shape ``(n,)``. Changed
    values/starts at fixed topology are supported when the supplied inverse
    uses current values; rebuild cuDSS factors and graphs after value changes.
    ``close()`` releases this solver's graphs but never closes the borrowed
    inverse. No host numerical decisions are made by solve or replay.
    """

    def __init__(
        self,
        operator,
        mass,
        inverse,
        k,
        sigma=0j,
        ncv=None,
        tol=1.0e-9,
        initial_vector=None,
        preserve_converged=True,
    ):
        sigma = complex(sigma)
        if not math.isfinite(sigma.real) or not math.isfinite(sigma.imag):
            raise ValueError("sigma must be finite")
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        if isinstance(k, bool) or index(k) < 1:
            raise ValueError("k must be a positive integer")
        k = index(k)
        if ncv is not None:
            if isinstance(ncv, bool):
                raise ValueError("ncv must be an integer greater than k")
            ncv = index(ncv)
        for name, op in (("operator", operator), ("mass", mass), ("inverse", inverse)):
            if not callable(getattr(op, "apply_complex", None)) or not hasattr(op, "complex_n"):
                raise TypeError(f"{name} must provide complex_n and apply_complex")
            if not hasattr(op, "device"):
                raise TypeError(f"{name} must provide its device")
            if op.complex_n != operator.complex_n or op.device != operator.device:
                raise ValueError("operators and inverse must have matching dimensions/device")
        if k >= operator.complex_n or (ncv is not None and not k < ncv <= operator.complex_n):
            raise ValueError("require 1 <= k < ncv <= n")
        self.op, self.mass, self.inverse = operator, mass, inverse
        self.device, self.k, self.tol = operator.device, k, tol
        self.transform = _PencilOperator(mass, inverse)
        self.core = GeneralEigensolver(
            self.transform,
            k,
            ncv=ncv,
            which="LM",
            tol=tol * 0.01,
            initial_vector=initial_vector,
            preserve_converged=preserve_converged,
        )
        self.evaluation = _PencilEvaluation(operator, mass, k, sigma, tol)
        self.core._checkpoint_validator = partial(
            self.evaluation.evaluate, self.core.eigenvectors, self.core.ritz, self.core.order
        )
        self.eigenvectors, self.status = self.core.eigenvectors, self.core.status
        self.eigenvalues = self.evaluation.values
        self.residuals = wp.empty(k, dtype=wp.float64, device=self.device)
        self.converged = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.iterations = self.core.iterations
        self.graph = None

    def finalize(self):
        self.evaluation.evaluate(self.eigenvectors, self.core.eigenvalues, self.evaluation.order)
        self.converged.zero_()
        wp.launch(
            pencil_outputs,
            self.k,
            [self.evaluation.errors_squared, self.residuals, self.status, self.converged, self.tol],
            device=self.device,
        )
        self.core._check_start(self.converged)

    def solve(self, iterations=20):
        self.core.solve(iterations)
        self.finalize()
        return self

    def capture(self, iterations=20):
        self.graph = self.core.capture(iterations, adaptive=False, finalizer=self.finalize)
        return self.graph

    def close(self):
        self.graph = self.core.graph = None
