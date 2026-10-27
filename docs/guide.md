# Solver guide

[Documentation](index.md) · [API reference](api/README.md) · [Installation](installation.md)

## Data and operators

Import frontend classes and functions from `warpack`. Real vector blocks are
`wp.float64` arrays with **one vector per row**, on the operator's device.
`SparseOperator` and `BatchedSparseOperator` wrap compact square scalar CSR or
3×3 BSR matrices from `warp.sparse`. They accept strided vector rows but require
nonoverlapping input/output buffers. `SparseOperator(A, shift=s, sign=t)` applies
`t*A*x + s*x`; its additive shift is distinct from a shift-invert target.

Complex scalar CSR uses `wp.vec2d(real, imaginary)` values and `wp.int32`
indices. `HermitianCSR` exposes a realified operator; `ComplexCSR` also exposes
`apply_complex`. Supply the full sparse matrix, not one triangle. See the
[operator reference](api/operators.md) for storage and inverse contracts.

For a custom real operator, provide `n`, `device`, and `apply(x, y)` operating
on FP64 rows `(width, n)`. Preallocate its workspace, overwrite every output
entry, and make the application graph capturable. Complex operators use
`complex_n`, `device`, and `apply_complex(x, y)` on `wp.vec2d` rows. A supplied
operator's symmetry, definiteness and mathematical validity are caller
preconditions; metadata validation is not a numerical proof of those properties.

Application assembly belongs in caller code. Checkout-only FEM, mesh and
skinning helpers live in [examples/problems](../examples/problems/README.md),
outside the installed `warpack` package.

## Generalized low modes

For symmetric `K` and SPD `M`, use
[`BlockLOBPCG`](api/symmetric.md#blocklobpcg) to obtain the smallest algebraic
modes of `K*x = lambda*M*x`. Omit `mass` for identity. An SPD preconditioning
action is optional. With assembled compact device matrices and `n >= 72`:

```python
import warp as wp
from warp.optim import linear
from warpack import BlockLOBPCG, SparseOperator

solver = BlockLOBPCG(
    SparseOperator(K),
    k=20,
    width=24,
    mass=SparseOperator(M),
    preconditioner=linear.preconditioner(K, ptype="diag"),
    tol=1e-8,
)
graph = solver.capture(200)
wp.capture_launch(graph)
# Device outputs; vectors are rows of shape (20, n).
values, vectors = solver.eigenvalues, solver.eigenvectors
```

`width` defaults to `k`; require `1 <= k <= width` and `3*width <= n`.
`solve(steps=...)` runs a fixed budget and returns the solver instance.
`capture(max_iterations=...)` can use device-controlled stopping and returns a
graph. Neither operation implicitly returns host arrays.

Supply `initial_vectors` with 1–`width` independent FP64 rows to reuse a computed
basis. Missing rows are randomized. Optional `initial_perturbation` adds seeded
exploration relative to each supplied row's norm. Keep separate start storage
when copying prior outputs. The [reuse example](../examples/block_reuse.py)
shows stiffness changes with fixed storage.

`constraints` excludes a **known invariant space**, such as rigid-body modes.
Its rows must be M-orthonormal, or use `normalize_constraints=True` to normalize
a private copy. Independent rigid spaces are needed for disconnected bodies.
General linear equality constraints use
[`ConstrainedKrylovSchur`](api/constraints.md#constrainedkrylovschur) with SPD
physical mass and a KKT inverse; an augmented singular mass is not passed to
`BlockLOBPCG`. See the [skinning formulation](../benchmarks/applications/SKINNING.md).

For fast block preconditioning, use
[`BatchedCGPreconditioner`](api/operators.md#batchedcgpreconditioner) with the
[pinned upstream runtime](installation.md#fast-block-solves). Truncated CG is
an approximate preconditioning action, not an exact shift-invert operator.
Its width must match the block width. Advanced product, graph and bucket
options are documented individually in the reference; use the
[Anvil recipe](releases/v0.2.0.md#install-and-reproduce-the-fast-anvil-solve)
for that measured configuration.

## Standard symmetric eigenpairs

Use [`KrylovSchur`](api/symmetric.md#krylovschur) for a standard symmetric
problem and a selected spectrum end:

```python
from warpack import KrylovSchur, SparseOperator

solver = KrylovSchur(SparseOperator(A), k=20, ncv=48, which="LM", tol=1e-10)
graph = solver.capture(100)
wp.capture_launch(graph)
```

`LM`/`SM` select largest/smallest magnitude, `LA`/`SA` largest/smallest algebraic,
and `BE` both ends. Require `k < ncv <= n`. Small-magnitude or interior modes
usually call for an inverse transformation. `SymmetricEigensolver` is a separate
block inverse-iteration API; its `which` parameter alone does not make an
untransformed partial block iteration target arbitrary spectrum ends.

## Shift-invert and sparse inverse backends

[`GeneralizedKrylovSchur`](api/symmetric.md#generalizedkrylovschur) uses the SPD
mass inner product. For modes nearest a real `sigma`, assemble `Kshift=K-sigma*M`
and supply a **one-RHS** inverse of that exact shifted matrix:

```python
from warpack import CuDSSInverse, GeneralizedKrylovSchur, SparseOperator

sigma = 7.3
inverse = CuDSSInverse(Kshift, width=1, mtype="general", matching=True, refinement_steps=1)
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
# Keep inverse, solver and graph alive until execution has completed.
```

The caller assembles the shifted matrix and selects its factor type. Use
`CGInverse` for SPD shifted systems or `GMRESInverse` for indefinite/general
systems to stay pure Warp. An interior shift need not leave the matrix SPD.
Inner tolerances generally must be stricter than outer tolerances; check inner
statistics and fresh original residuals. `CGInverse(atol=0.0)` avoids an absolute
tolerance dominating small RHSs, subject to upstream numerical floors.

For largest generalized modes, supply an inverse of `M` and set
`mode="regular", which="LA"`. Do not pass `M^-1*K` to a Euclidean symmetric
solver: it is self-adjoint in the M metric. `mode="cayley"` and
[`BucklingEigensolver`](api/symmetric.md#bucklingeigensolver) have different
transformed ordering rules; neither should be assumed to select ordinary
nearest-shift modes.

cuDSS constructors own native factors. After changing matrix values, rebuild
those factors and dependent graphs unless an explicitly supported update path
is used. Iterative preconditioner validity remains caller-managed. Finish graph
execution and discard graph references before closing owned native resources.
See [ownership and streams](library_contracts.md#arrays-and-ownership).

## Other problem families

| Problem | Entry point and contract |
|---|---|
| Complex Hermitian standard | [HermitianEigensolver](api/complex.md#hermitianeigensolver); `ncv` counts **realified** dimensions |
| General real/complex standard | [GeneralEigensolver](api/complex.md#generaleigensolver); real data uses zero imaginary components; `ncv` counts complex dimensions |
| Complex/Hermitian interior | [ShiftInvertEigensolver](api/complex.md#shiftinverteigensolver); creates an owned shifted inverse; its `ncv` counts complex dimensions even with `hermitian=True` |
| Nonsymmetric generalized | [GeneralizedShiftInvertEigensolver](api/complex.md#generalizedshiftinverteigensolver); regular pencil, finite right modes, possibly singular B; borrowed shifted inverse |
| Real partial SVD | [PartialSVD](api/svd.md#partialsvd); augmented symmetric formulation |
| Resolved largest nonzero singular triplets | [NormalSVD](api/svd.md#normalsvd); normal products square conditioning; no requested null singular bases |

All these APIs expose device arrays. General right eigenvectors need not be
orthogonal; generalized symmetric vectors use their metric; singular vectors
are Euclidean-orthonormal on each side. See the
[runnable examples](../examples/api_quickstart.py) and each reference page for
exact signatures and output shapes.

## Results and acceptance

Outputs are overwritten on each solve/replay. Copy them into separate storage
if they must survive a later solve. Explicit `.numpy()` calls transfer data and
synchronize; perform them outside graph capture.

| API | Device fields to inspect |
|---|---|
| BlockLOBPCG, symmetric/generalized Krylov and block solvers, Cayley, GeneralEigensolver, both shift-invert wrappers | `status[0] == 0`, `converged[0] == k`, and original-problem residuals |
| HermitianEigensolver | `real.status[0] == 0`, `real.converged[0] == 2*k`, `count[0] == k`; independently check complex residuals |
| BucklingEigensolver | `core.status[0] == 0`, `converged[0] == k`, original buckling residuals |
| PartialSVD, NormalSVD | `status[0] == 0`, `count_left[0] == count_right[0] == k`, both residual arrays |
| ConstrainedKrylovSchur | `status[0] == 0`, `converged[0] == k`, equilibrium and `constraint_residuals` |

Zero status alone means no recorded numerical failure; a budget can expire with
an incomplete count. These tests do not prove global spectrum completeness.
For repeated eigenvalues, compare invariant spaces rather than individual
vector signs. An exact invariant warm start can miss newly lower modes after
crossings. [`LowestModeCertificate`](completeness.md) provides an optional
numerical coverage check below a supplied gap, using an extra cuDSS setup query.
[`NumericalNullspaceDiagnostics`](api/diagnostics.md#numericalnullspacediagnostics)
classifies supplied near-zero directions; it does not prove a Betti number.

Residual conventions differ:

| Problem | Normalized equation residual |
|---|---|
| Standard unit-norm eigenvector | `||A*x-lambda*x|| / max(1, abs(lambda))` |
| Symmetric generalized | `||A*x-lambda*M*x|| / max(1, ||A*x|| + abs(lambda)*||M*x||)` |
| Generalized nonsymmetric pencil | `||A*x-lambda*B*x|| / (||A*x|| + max(1,abs(lambda))*||B*x||)`; invalid denominators are rejected |
| SVD | Each of `||A*v-s*u||` and `||A.T*u-s*v||`, divided by `max(1,s)` |

Krylov cached-product residuals can be optimistic near the precision floor.
[`EigenpairEvaluation`](api/diagnostics.md#eigenpairevaluation) reapplies a real
standard operator to the returned unit-norm vectors. Supply `eigenvalues` to
check the returned pairs; omitting them recomputes Rayleigh values. Use
`rhs_width=1` for one-RHS operators. This diagnostic does not change solver
stopping. Generalized and constrained equations need their corresponding
physical checks, not this standard-problem diagnostic.

## Graph replay

Construct and call `capture()` outside active capture. It allocates/prepares
workspaces, compiles, warms and synchronizes as needed. Launch the returned
graph with `wp.capture_launch`. On the affected CUDA stack, complete its first
launch during setup before queueing changed input values:

```python
graph = solver.capture(100)
wp.capture_launch(graph)
wp.synchronize_device(solver.device)
# Now enqueue fixed-address input updates and subsequent launches on the stream.
wp.capture_launch(graph)
```

This setup workaround is described in the
[execution contract](library_contracts.md#graph-setup-and-synchronization).
Subsequent pure-Warp replay has no host convergence decision. cuDSS currently
records small vendor HtoD metadata copies and requires fixed outer capture.
CG/GMRES inverses also default to fixed outer capture; their opt-in adaptive
mode needs the qualified upstream preallocated-loop runtime.

Methods accepting `adaptive=None` choose a compatible policy. Not every wrapper
accepts that keyword: use its documented `capture` signature. Budgets count
outer iterations/restarts, not individual operator applications or inner CG
steps. `solve()` uses a fixed budget. For caller-owned capture, warm first and
record `solve()` or supported `initialize()` / `step()` / `finalize()` stages;
do not nest standalone `capture()`.

Where supported, `finalizer=callback` records device postprocessing.
`BlockLOBPCG` also accepts `initializer=callback` before initialization.
Callbacks run eagerly during setup/warmup and recording, not as Python callbacks
on replay. Use preallocated device buffers and retain their owners.

Keep input storage, sparse topology, operators, solver, graph and factors alive
and fixed. Values can change between replays; update dependent inverses and
preconditioners consistently. Reserve structural entries that may become
nonzero. Scratch is mutable: do not execute one instance, or shared inverse
resources, concurrently. cuDSS construction/application/capture must use its
construction stream; cross-stream replay requires caller ordering.

## Limitations

The current API primarily supports FP64 and complex128. FP32 inner CG does not
imply FP32 outer eigensolvers. Complex SVD, public left/infinite modes and
arbitrary singular mass in SPD-metric solvers are unsupported.

General nonsymmetric requested-spectrum coverage has a known
[#104 gap](https://github.com/alecjacobson/warpack/issues/104), and ill-scaled
generalized pencils have [strict residual limitations](https://github.com/alecjacobson/warpack/issues/123).
The small reciprocal conformal fixture has a
[native stopping floor](https://github.com/alecjacobson/warpack/issues/209).
The [graph-upload issue](https://github.com/alecjacobson/warpack/issues/61)
remains open. Representative tests and recorded comparisons are not full
SciPy/CuPy/Spectra API parity or guarantees for every matrix.

Optimization history and hardware-specific implementation notes are preserved
in [historical documentation](../historical/library-implementation-notes.md).
