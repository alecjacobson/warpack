# Solver guide

[Installation](installation.md) · [Examples](../README.md#examples) · [Validation](development.md)

## Data and operators

`SparseOperator` accepts compact square FP64 `warp.sparse.BsrMatrix` storage
with scalar blocks (CSR) or 3×3 blocks. Compress mutable sparse rows before
constructing the operator. `BatchedSparseOperator` has the same contract and
shares matrix reads across RHS rows. Operator inputs and outputs are distinct
FP64 arrays of shape `(width, n)` on the same device.

Complex paths use `HermitianCSR` or `ComplexCSR`, storing scalar values as
`wp.vec2d(real, imaginary)`; `SparseOperator` itself is real. Solver instances
own their scratch and retain references to operators and inputs. Application
matrix assembly belongs in caller code; [`examples/problems/`](../examples/problems/README.md)
contains the checkout-only FEM, mesh and skinning helpers.

## Generalized low modes

`BlockLOBPCG` targets the smallest algebraic eigenvalues of `Kx = λMx`, with
symmetric K and SPD M. Omit `mass` for identity M. Use an SPD preconditioner
appropriate to the problem; the bar example uses a diagonal approximation,
while the fast Anvil example uses truncated upstream CG with FSAI.

```python
import warp as wp
from warp.optim import linear
from warpack import BlockLOBPCG, SparseOperator

# K and M are assembled, compact device matrices; n >= 72.
block = BlockLOBPCG(
    SparseOperator(K),
    k=20,
    width=24,
    mass=SparseOperator(M),
    preconditioner=linear.preconditioner(K, ptype="diag"),
    tol=1e-8,
)
graph = block.capture(200)
wp.capture_launch(graph)
```

`width >= k` and `3*width <= n` are required. Outputs are device arrays:
`eigenvalues` has shape `(k,)`, and `eigenvectors` has shape `(k,n)` with vectors
stored as rows. `residuals`, `converged`, `iterations` and `status` provide
acceptance information. Eigenvectors are M-orthonormal. The residual is
`||Kx−λMx||₂ / max(1, ||Kx||₂ + |λ| ||Mx||₂)`.

Supply `initial_vectors` with 1–`width` independent FP64 rows to reuse a computed
basis. Missing rows get random initialization. `initial_perturbation=1e-3`, for
example, adds seeded exploration relative to each supplied row's norm; the
default is zero. Keep separate start storage if copying the previous result
into the next solve. [`examples/block_reuse.py`](../examples/block_reuse.py)
shows this after changing stiffness.

An exact previous invariant space can omit newly lower modes after crossings,
even with zero residuals. Perturbation helps exploration but is not a proof of
requested-spectrum completeness. Validate complete clusters and subspaces
when repeated modes matter. The optional
[`LowestModeCertificate`](completeness.md) checks coverage below a supplied gap
using an extra cuDSS factorization and capturable Warp verification.

`constraints` excludes a **known invariant space**, such as the six rigid
motions of a connected free body. Supply its basis as device rows. Optional
`normalize_constraints=True` normalizes a private copy in the current mass
metric. Independent rigid spaces are needed for disconnected bodies. General
linear equality constraints instead use `ConstrainedKrylovSchur`; see the
[skinning formulation](../benchmarks/applications/SKINNING.md).

For fast block solves, the public options `reuse_operator_images`,
`search_bucket_size`, `keep_history` and `preconditioner_bucket_size` control
cached products and preallocated device-selected workspaces. Their defaults
preserve the base algorithm. `BatchedCGPreconditioner.width` must match the
outer block width. It calls upstream CG with independent RHS coefficients and
stopping tests. FP32 working matrix values refresh from the FP64 source on
each application; factor updates remain caller-managed. Its `stats` counts
scheduled vector iterations, maximum batch iterations, unconverged RHSs and
RHS calls. Unfinished truncated preconditioner solves are distinct from failed
outer eigensolves. The [Anvil recipe](releases/v0.2.0.md#install-and-reproduce-the-fast-anvil-solve)
provides a tested configuration and the required runtime.

Upstream FSAI setup can consume much more device memory than its stored factors.
See the [row-size memory/time comparison](../results/fsai_memory/README.md) before
using a large factor-row cap on a smaller GPU.

## Standard symmetric eigenpairs

```python
from warpack import KrylovSchur, SparseOperator

solver = KrylovSchur(SparseOperator(A), k=20, ncv=48, which="LM", tol=1e-10)
graph = solver.capture(100)
wp.capture_launch(graph)
```

Selection codes are `LM`/`SM` for largest/smallest magnitude, `LA`/`SA` for
largest/smallest algebraic, and `BE` for both ends. Small-magnitude targets are
generally better approached through shift-invert than untransformed Krylov.
For unit-norm vectors, the standard symmetric residual is
`||Ax−λx||₂ / max(1, |λ|)`. This differs from the generalized normalization.

Krylov residuals formed from cached operator images are estimates. Near the
precision floor, independently recompute the original equation using
`EigenpairEvaluation(op, solver.eigenvectors, eigenvalues=solver.eigenvalues)`.
Use `rhs_width=1` for operators accepting one RHS at a time. Omit `eigenvalues`
to recompute Rayleigh values. This diagnostic has its own count and does not
change Krylov stopping. It can be recorded as a capture finalizer.

## Shift-invert and sparse inverse backends

For an interior target `sigma`, assemble `Kshift = K−sigma*M` and supply its
inverse. `GeneralizedKrylovSchur` selects the nearest original eigenvalues with
its default shift-invert `which="LM"` and the matching `target`:

```python
from warpack import CuDSSInverse, GeneralizedKrylovSchur, SparseOperator

sigma = 7.3
inverse = CuDSSInverse(
    Kshift,
    width=1,
    mtype="general",
    matching=True,
    refinement_steps=1,
)
solver = GeneralizedKrylovSchur(
    SparseOperator(K),
    SparseOperator(M),
    inverse,
    k=20,
    ncv=48,
    target=sigma,
    tol=1e-10,
)
graph = solver.capture(40, adaptive=False)
wp.capture_launch(graph)
```

cuDSS matching/scaling and factorization run during setup. Residual corrections
compute `b−Ax` and the update in Warp, using cuDSS only for the sparse solve.
Choose the factor type and correction budget for the shifted system; inspect
fresh original-problem residuals rather than relying solely on transformed
convergence. See the [factorization guidance](../benchmarks/upstream/MATCHING.md).

For a pure-Warp inverse, use `CGInverse` on SPD systems or `GMRESInverse` on
indefinite/nonsymmetric systems. Both call upstream Warp iterative solvers.
GMRES takes a fixed linear right preconditioner; CG needs SPD preconditioning.
Do not apply CG to an indefinite interior shifted matrix. These inverse
applications generally need tighter accuracy than the outer eigensolver.

`CGInverse(..., verify_residuals=True)` exposes fresh residual failures in
`fresh_stats` and maxima in `residual_maxima`, separately from upstream recursive
`stats`. Optional `refinement_steps` and `refinement_tol` add fresh-residual
corrections. `GMRESInverse.stats` includes fresh inner failures. Check inner
and outer diagnostics together for inverse-based solves. This contract differs
from deliberately truncated CG inside preconditioned block iteration.

## Other problem families

- `HermitianEigensolver` handles complex Hermitian standard problems.
  `ShiftInvertEigensolver(..., hermitian=True)` handles Hermitian shift-invert
  with a real shift. General complex shift-invert can use cuDSS or Warp GMRES.
- `GeneralizedShiftInvertEigensolver` applies `(A−sigma*B)^−1 B` for a regular
  pencil, including nonsymmetric or singular B. It returns finite nearest
  **right** modes with fresh original-equation counts. It does not form `B^-1 A`.
  See the [generalized-pencil guide](../benchmarks/applications/GENERALIZED_PENCILS.md).
- `BucklingEigensolver` and `CayleyEigensolver` use transform-specific selection
  rules. Ordinary nearest-shift selection is not interchangeable with buckling
  load-factor ordering. See [transforms](../benchmarks/upstream/TRANSFORMS.md).
- `PartialSVD` uses an augmented symmetric formulation for real matrices.
  `NormalSVD` targets resolved largest nonzero singular triplets through normal
  products. Normal equations square conditioning; unresolved requested values
  are rejected. See [POD examples](../benchmarks/applications/POD.md) and the
  [normal-product API](../benchmarks/upstream/NORMAL_SVD.md).
- `NumericalNullspaceDiagnostics` distinguishes returned near-zero modes,
  ambiguous weak positives, invalid data and capacity limits. It does not
  certify an unknown nullspace dimension or Betti number.

## Graph replay

Capture is a setup operation: it warms kernels, allocates workspaces and may
synchronize. Ordinary compatible paths use device-controlled conditional loops.
cuDSS uses fixed outer capture. `CGInverse` and `GMRESInverse` also default to
fixed outer capture for compatibility; opt-in `adaptive_capture=True` requires
upstream preallocated loop storage, as in the pinned fast environment.
`capture(..., adaptive=False)` explicitly requests a fixed budget.

On CUDA drivers 12.3+, built-in `KrylovSchur` directly using `CuDSSInverse`
prepares capture with one inverse application instead of a full eigensolve.
Warp kernels load while the graph is recorded. Custom solvers/operators,
finalizers and older drivers retain full warmup. Results are valid after
launching the returned graph; capture itself is not a completed solve.

Every replay initializes a new solve from current input values. Keep topology,
array identities and resource lifetimes fixed. Matrix values, constraints and
supplied starts can change; update the associated inverse factors when needed.
The caller owns preconditioner validity. Stale preconditioning may still be
useful for nearby SPD problems, but it is not the updated exact inverse.
For FEM material edits, reserve the full structural sparsity: entries that are
zero through cancellation can become nonzero even on an unchanged mesh. If
they were omitted from storage, updating values alone is insufficient.

Keep the solver, input buffers and factors alive until graph execution finishes.
Instances and shared-width CG variants own mutable scratch and cannot be used
concurrently. Destroy graphs before closing their inverse backend. Never replay
against released factors.

`capture(..., finalizer=callback)` records device postprocessing during setup;
the callback is not a host callback during replay. Preallocate its outputs.
Explicit `.numpy()` reads and caller synchronization belong outside the graph.
The pure-Warp replay has no host numerical decisions. cuDSS currently includes
small vendor HtoD metadata copies; [the audit](../benchmarks/applications/CUDSS_GRAPH.md)
distinguishes these from application synchronization.

Accept a result only when status, requested counts and relevant original
residual/orthogonality checks pass. `status == 0` alone does not mean convergence.
A successful residual count is not a global requested-spectrum certificate.

The [FEM reuse example and measured comparison](../results/fem_public_starts/README.md)
show computed starts for related material configurations, including the setup
costs and the fixed-address replay contract. The
[shape-change qualification](../results/fem_shapes/README.md) additionally updates
K and M together, exercises upstream FSAI refits at fixed storage addresses, and
checks mode crossings and complete repeated clusters. For an opt-in response to
a rejected warm start, see the [bounded recovery example](completeness.md#opt-in-recovery-after-a-rejected-warm-start).

## Limitations

The library has representative application and upstream-test coverage, not full
SciPy/CuPy/Spectra API parity. General eigensolver computation is FP64/complex128;
FP32 factor/inner-CG support does not imply FP32/complex64 outer eigensolvers.
Complex SVD, public left modes and infinite eigenmodes are unsupported.
Arbitrary singular mass is not accepted by the SPD generalized solvers; the
regular-pencil and constrained formulations have separate contracts.

Compact nonsymmetric searches can return a full residual count while missing
the requested extremal set ([#104](https://github.com/alecjacobson/warpack/issues/104)).
Some ill-scaled generalized pencils miss strict original-equation tolerances
([#123](https://github.com/alecjacobson/warpack/issues/123)). Wider validated
profiles and mechanics-specific recovery examples are documented, but are not
generic guarantees. Partial SVD nullspace recovery is also not established for
arbitrary requested zero spaces.

The graph-upload issue [#61](https://github.com/alecjacobson/warpack/issues/61)
is contained by synchronizing warmup before capture; upstream resolution remains
pending. This does not add per-replay synchronization. See the
[issue tracker](https://github.com/alecjacobson/warpack/issues) for remaining work.
