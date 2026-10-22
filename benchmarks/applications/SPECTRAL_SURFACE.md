# Dominant generalized modes for spectral surface reconstruction

This independent procedural fixture covers [#136](https://github.com/alecjacobson/warpack/issues/136).
It follows the type of objective in Section 4.2 of
[Braune, Dumas and Thiery, *Efficient Multiscale Lanczos Eigenpair Extraction*](https://theofbraune.github.io/blog/publications/projects/publication_files/multiscaleEigvs/mainCompressed.pdf)
(SIGGRAPH 2026), which builds on variational reconstruction of unoriented points.
It is not a reproduction of the authors' discretization, solver or datasets.
All point samples, normals and tetrahedra are generated locally.

## Pencil and requested spectrum

A scalar P1 field lives on a regular six-tet-per-cell background cube. Fibonacci
samples lie on an analytic ellipsoid; exact analytic normals supply an initial
unoriented alignment field. At sample p, S evaluates the field, G evaluates its
constant-per-tet gradient, and C = I + eta*n*n.T favors normal alignment.
Flipping an individual normal does not change C.

The local discretization is

```
A = (1/N) sum_p G_p.T C_p G_p
B = (alpha/N) S.T S + beta L + gamma L.T diag(m)^-1 L
A f = lambda B f, requesting largest algebraic lambda
```

L is the P1 scalar stiffness with natural boundary conditions, and m is the
lumped volume mass. Dividing A by the sample count scales eigenvalues relative
to the unnormalized sum in the cited objective, without changing eigenvectors.
The biharmonic term is explicitly sparse because this fixture uses lumped m.
This is not the pointwise Laplacian of a piecewise-linear field: it uses the
weak discrete Laplacian followed by the volume metric.

A is symmetric PSD and singular (at least its constant direction).
**B is an SPD regularization metric, not the physical mass matrix.** On the
connected background, beta>0 makes the derivative energy positive away from
constants; alpha>0 with point samples controls the remaining constant. The
fixture rejects an unanchored configuration instead of silently declaring a
singular B to be SPD. gamma>=0 and eta>=0 are required. General consistent-mass
biharmonic composition would require a separate matrix-free inverse; this
fixture does not assemble a dense consistent-mass inverse.

The simple dominant field's zero set approximates the ellipsoid. k>1 also
exercises the largest spectral subspace. Disconnected identical copies supply
exact repeated dominant clusters: request complete multiples of the number
of copies. Individual vectors within a repeated eigenspace are arbitrary, so
we compare the complete B-weighted subspace and do not attach a unique surface
to an arbitrary member of that cluster.

## Library call

With Warp sparse matrices A and B already constructed:

```python
from warpack import CuDSSInverse, GeneralizedKrylovSchur, SparseOperator

inverse = CuDSSInverse(B, width=1, mtype="spd")
solver = GeneralizedKrylovSchur(
    SparseOperator(A),
    SparseOperator(B),
    inverse,
    k=1,
    ncv=32,
    mode="regular",
    which="LA",
    tol=1e-11,
)
graph = solver.capture(max_iterations=4)
wp.capture_launch(graph)
# Inspect device outputs/status outside timed replay; close inverse when done.
```

Regular mode applies B^-1 A while orthogonalizing in the B metric. Feeding
B^-1 A to a Euclidean symmetric eigensolver would generally be incorrect.
The pure-Warp composition substitutes upstream FSAI-preconditioned `CGInverse`
with `tol=1e-13, atol=0.0, adaptive_capture=True`; it also audits fresh linear
residuals. No custom CG or sparse factorization is implemented here.

cuDSS uses a fixed outer capture budget because its backend does not support
our nested adaptive capture. The CG route can stop on device before that cap.
cuDSS's vendor HtoD scalar metadata copies remain visible under
[#29](https://github.com/alecjacobson/warpack/issues/29); only the pure-Warp route
is expected to have no host transfers at all. Both routes keep eigen-iteration
arithmetic and physical residual computation in Warp.

## Validation and comparison protocol

Small tests use dense generalized LAPACK and independent energy expressions.
Tests cover affine evaluation/gradients, integrated stiffness/volume, PSD A,
SPD B, its constant-nullspace obstruction without anchors, largest objective,
zero-set error, changed-anisotropy replay and complete repeated clusters.
Regularization changes include gamma=0 and a stronger anchored/biharmonic metric.

Large references solve one independent copy using ARPACK with four extra guard
pairs, ncv>=64 and tol=1e-13. The measured ARPACK route independently factors B
with SuperLU and uses `eigsh(A, M=B, Minv=..., which="LA", tol=1e-11)`.
No timed solver receives reference eigenvectors. A guard gap prevents requesting
only part of a near cluster. Original-pencil residual, eigenvalue, metric
orthogonality and complete-subspace checks all use a common 1e-8 acceptance
bound; Warp additionally requires native zero status and the full requested
count under its tighter stopping rule.

Geometric error is reported separately: sampled point-to-zero-set distance
estimated by |f|/|grad f|, exact zero crossings of the P1 field on tet edges,
relative radial ellipsoid error at those crossings, and mean absolute normal
alignment. These measure the discretization/reconstruction objective, not an
extra eigen-equation residual. Excellent eigen accuracy can coexist with finite
geometric approximation error.

Every worker separates fixture assembly, independent reference plus disconnected
copy assembly, setup/factorization, graph capture, and synchronized warm solve.
Validation/geometry measurement and output copies are outside solve timing.
Reported setup+capture+first-solve sums exclude assembly/reference and are not
independently repeated cold-process measurements. Device free-memory changes
are context snapshots, not peak storage.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu \
  --out build/surface-cg.occupancy.json --log build/surface-cg.log -- \
  taskset -c 8-11 python -m benchmarks.applications.spectral_surface_compare \
    --side 16 --modes 1 --backend cg --repeats 3 --out build/surface-cg.json
python -m benchmarks.applications.spectral_surface_report build/surface-cg.json
```

Use the current pinned FSAI installation. Other backends are `arpack` and
`cudss`; `--copies 2 --modes 2` requests a complete repeated dominant cluster.
The reporter requires three complete uncontended trials and accurate full
spectra. `--baseline prior.summary.json` additionally rejects incompatible
configuration/provenance or a solve slowdown above 25%. Vendor graph copies
cannot be relabeled as a transfer-free graph.
