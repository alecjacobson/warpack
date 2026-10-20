# warpack

Sparse eigenvalue problems computed in **NVIDIA Warp**, with an optional **cuDSS** inverse backend. FP64 arithmetic, device-resident projected eigensolves and residuals, and reusable CUDA graphs. NumPy, SciPy, CuPy, and Spectra are references in the tests and benchmarks; they do not perform eigensolver computation inside the library.

This is a first implementation, not an API-compatible drop-in replacement for Spectra or ARPACK. It covers their main eigenproblem families, with performance work focused on **large sparse problems**. The 283,857-DOF anvil comparison below reports timings and accuracy against ARPACK, Spectra, and CuPy.

[Download the animated teaser and editable Blender scene](https://github.com/alecjacobson/warpack/releases/tag/v0.1.3)

![Anvil elastic modes 1–20: 5% bounding-box-diagonal peak displacement with continuous instantaneous coloring](results/anvil_teaser.gif)

Anvil adapted from [“Blacksmith's Anvil”](https://sketchfab.com/3d-models/blacksmiths-anvil-4b11b76fe069496ebd8ccd98f7d37ade) by [m4rcu5j4ck50n](https://sketchfab.com/m4rcu5j4ck50n), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Tetrahedral mesh and vibration visualization adaptations are described in [Acknowledgments](ACKNOWLEDGMENTS.md).

## Supported problems

| Problem | API | Notes |
|---|---|---|
| Real symmetric | `KrylovSchur` | Thick restart, two-pass reorthogonalization, LM/LA/SA/SM/BE selection |
| Preconditioned symmetric block iteration | `BlockLOBPCG` | Smallest algebraic modes; SPD mass; reusable partial/full device block starts; optional invariant constraints |
| Symmetric block inverse iteration | `SymmetricEigensolver` | CholeskyQR2 and Rayleigh–Ritz; supply an iteration operator that amplifies the requested spectrum |
| Symmetric generalized `A x = λ B x` | `GeneralizedKrylovSchur`, `GeneralizedEigensolver` | SPD mass, including non-diagonal sparse B; B-orthonormal eigenvectors; restarted Krylov or block inverse iteration; interior targeting |
| Complex Hermitian | `HermitianEigensolver` | Standard problems via realified restarted Krylov; complex basis and breakdown recovery; replayable complex starting vector; all 30 exported Spectra Hermitian fixtures pass |
| Real or complex nonsymmetric | `GeneralEigensolver` | Restarted complex Arnoldi, device Hessenberg reduction and shifted QR; LM/SM/LR/SR/LI/SI selection; replayable complex starting vector |
| Real or complex shift-invert | `ShiftInvertEigensolver` | Nearest eigenpairs using cuDSS or pure-Warp GMRES; replayable complex starts; `hermitian=True` selects Hermitian Krylov for real shifts |
| Buckling / Cayley | `BucklingEigensolver`, `CayleyEigensolver`, `GeneralizedKrylovSchur` | Block or restarted Krylov transforms; original-problem residuals |
| Partial rectangular SVD | `PartialSVD` | Largest real singular triplets through an augmented symmetric operator; replayable device starting vector |
| Resolved largest nonzero SVD / POD | `NormalSVD` | Explicit normal-product formulation; smaller search space, squared conditioning; original-equation and orthogonality diagnostics |
| Pure-Warp SPD inverse | `CGInverse` | Preallocated Warp CG, upstream preconditioning, optional fresh residual audits and corrections |
| Pure-Warp indefinite/nonsymmetric inverse | `GMRESInverse` | Preallocated upstream Warp GMRES; fixed linear right preconditioning; fresh device residuals and inner failure counts |
| Known invariant-space exclusion | `MassOrthogonalComplementInverse` | Correct primal/dual projections for SPD mass, optional device normalization; caller supplies the invariant space |
| Symmetric equality-constrained modes | `ConstrainedKrylovSchur` | SPD physical mass, independent equality constraints, finite modes via KKT shift-invert; multiplier and constraint residuals; [skinning example and comparisons](benchmarks/applications/SKINNING.md) |
| Numerical nullspace evidence | `NumericalNullspaceDiagnostics` | Captured physical/metric checks, weak-positive and capacity flags; completeness remains unverified |

`SparseOperator` accepts compact scalar FP64 CSR or 3×3 FP64 BSR matrices from `warp.sparse`. Complex inputs use `HermitianCSR` or `ComplexCSR`, with values stored as `wp.vec2d(real, imaginary)`; `SparseOperator` accepts real matrices only. The general and Hermitian paths have less performance coverage than the real symmetric path. The [cooperative complex Schur solve](benchmarks/experimental/TILED_SCHUR.md) accelerates CUDA general projections of size 16–64 while retaining device-controlled iteration; smaller/larger projections retain the serial reference. The SVD path is validated primarily for leading nonzero singular triplets. Complete augmented-basis tests include requested zeros; arbitrary nullspace-basis recovery is not established. `NormalSVD` rejects unresolved/zero requested values.

The nonsymmetric solver has a known [requested-spectrum gap (#104)](https://github.com/alecjacobson/warpack/issues/104): full residual counts can accompany missed extremal modes. The strengthened general benchmark inventory currently accepts 11 of 20 cases. Residual convergence alone does not certify the requested set. The [native selection audit](benchmarks/upstream/GENERAL_SELECTION.md) also finds missed modes in the original Spectra/ARPACK benchmark settings and records explicit wider profiles that pass all 20 reference cases.

[Fixed-budget general solves preserve completed eigenpairs](benchmarks/upstream/GENERAL_CHECKPOINT.md) while continuing the search. General shift-invert candidates must pass the original equation before saving, and all final residuals are recomputed. Set `preserve_converged=False` to opt out. The explicit wider fixed-80 profile passes all 20 Spectra cases and 60 replays; the compact adaptive inventory above remains unchanged. This prevents losing a completed set under extra restarts without certifying unseen modes.

The [fused projected QR path](benchmarks/experimental/FUSED_SCHUR.md) keeps the small general eigensolve in a cooperative Warp kernel, with a fallback for GPUs with less shared memory. The optimization preserves the original requested-spectrum inventory; it does not resolve the selection gap above.

Constrained replay refreshes current constraint transposes and row norms on device. Keep sparse topology and addresses fixed, and update the KKT inverse when coefficients change. See the [constraint replay repair and regression controls](benchmarks/upstream/CONSTRAINED_REFRESH.md).

The installed `warpack` package contains the numerical solvers, inverse backends, spectral transforms, constraint operations, and diagnostics. Application matrix construction lives in [`examples/problems/`](examples/problems/README.md), shared by the showcase, benchmarks, and tests. This includes tetrahedral elasticity, mesh loading, rigid-motion bases, and skinning weight/rig matrices. These helpers require a source checkout and are not installed as library APIs.

The [singular boundary-mass conformal example](benchmarks/applications/CONFORMAL.md) removes the common translation kernel and solves an exact reciprocal pencil in an SPD metric. Flat and curved patches retain their finite modes without energy regularization; this is an explicit benchmark adapter, not a general singular-mass API.

Its [experimental block-preconditioned follow-up](benchmarks/experimental/CONFORMAL_BLOCK.md)
solves 20 modes at 32,766 unknowns in **0.519 s**, versus **6.801 s** for accurate
CG inverses and **0.384 s** for cuDSS. Short upstream FP32 CG applications
precondition FP64 block iteration; all original-pencil and full-count checks
pass. The report preserves the failed large native-CuPy comparison separately.

## Install

```bash
python -m pip install -e '.[test]'
# Optional sparse direct inverse:
python -m pip install -e '.[cudss]'
# Reference GPU/CPU benchmarks:
python -m pip install -e '.[benchmark]'
```

The latest full validation used Python 3.10, Warp 1.18.0+cu12 with the pinned FSAI overlay below, CUDA driver 570.158.01, cuDSS 0.8.0.10, CuPy 14.2.0, and an NVIDIA L40. The package requires Warp >=1.15, but stock Warp 1.15 has only the limited compatibility check described below; installing the extras above does not install the FSAI overlay. Conditional graphs require CUDA driver support for CUDA 12.4 or later. Complex data and solver workspaces use double precision. The production FSAI table uses FP32 factors with FP64 CG; the experimental block profiles use FP32 CG with FP64 outer iteration. Their pinned Warp PR environment is documented in [Validation and benchmarking](#validation-and-benchmarking).

## Symmetric solve and graph replay

```python
import warp as wp
from warpack import SparseOperator, KrylovSchur

# A is a compact, square, FP64 warp.sparse.BsrMatrix already on the GPU.
solver = KrylovSchur(SparseOperator(A), k=20, ncv=48, which="LM", tol=1e-10)
graph = solver.capture(max_iterations=100)
wp.capture_launch(graph)

# Device arrays: no implicit readback.
values = solver.eigenvalues       # (k,)
vectors = solver.eigenvectors     # (k, n), eigenvectors are ROWS
errors = solver.residuals
converged = solver.converged      # number of converged requested eigenpairs
iterations = solver.iterations
```

For symmetric solves, the reported residual is `||A x - λ x||₂ / max(1, |λ|)`, with unit-norm vectors. A normalized tolerance does not guarantee the same absolute residual bound when eigenvalues are large. `status == 0` means no internal factorization failure; it does **not** mean all eigenpairs converged. Check `converged == k` and the residuals. A fixed iteration budget can expire without convergence. `SM` targets are generally better addressed with shift-invert than an untransformed Krylov iteration.

When `KrylovSchur` rotates cached operator images (including inverse and generic callback operators), these are residual estimates; a fresh application can differ near the precision floor. `EigenpairEvaluation(op, solver.eigenvectors, eigenvalues=solver.eigenvalues, rhs_width=1)` checks the returned values with fresh one-RHS products and can run once as a capture finalizer. Omit `rhs_width` for ordinary block operators; omit `eigenvalues` to recompute Rayleigh values. Its independent count does not change Krylov stopping. See the [shifted-cycle precision and fresh-check report](benchmarks/upstream/CYCLE_RESIDUAL_FLOOR.md).

`capture()` compiles and warms kernels, allocates setup state, and constructs a reusable graph. Ordinary symmetric and general iterations use a device-controlled conditional loop. cuDSS requires a **fixed outer graph**. `CGInverse` also defaults to fixed outer capture for compatibility; `CGInverse(..., adaptive_capture=True)` enables nested device-controlled loops when upstream CG preallocates its loop storage (tested on the pinned Warp 1.18 environment). Stock Warp 1.15 rejects this opt-in because it allocates storage inside each solve. See the [adaptive-CG report](benchmarks/experimental/ADAPTIVE_CG.md). The compatible capture policy is selected automatically; `capture(iterations, adaptive=False)` explicitly selects a fixed graph. Fixed outer loops still perform all numerical work on device. There are no hidden host eigensolves or residual readbacks.

Matrix **values** can change between replays with stable array addresses. Changing a sparse topology requires a new operator/solver setup. An inverse's factors must also be updated or recreated when matrix values change. The complex GMRES backend refreshes its realified values on device, including stored zeros that become nonzero; see the [replay repair and validation](benchmarks/upstream/COMPLEX_REFRESH.md). A solver instance owns mutable workspaces and is not thread-safe. Destroy graphs before closing their inverse backend; never replay a graph after its factors have been released.

## Inverse and generalized problems

`BlockLOBPCG` directly targets the smallest algebraic eigenpairs, including
`K x = λ M x` with SPD mass. It accepts caller-owned FP64 block starts and
upstream Warp preconditioners; vectors are stored as rows:

```python
from warpack import BlockLOBPCG
import warp.optim.linear as linear

preconditioner = linear.preconditioner(K, ptype="diag")
block = BlockLOBPCG(
    SparseOperator(K), k=20, width=24, mass=SparseOperator(M),
    preconditioner=preconditioner, initial_vectors=previous_modes,
    initial_perturbation=1e-3, tol=1e-8,
)
graph = block.capture(200)
wp.capture_launch(graph)
```

Here `previous_modes` has 1–24 independent rows; missing rows are initialized
randomly. Omit it for a random start. `3*width <= n` is required. The optional
perturbation adds seeded exploration relative to each supplied row's norm;
it defaults to zero. This matters after mode crossings: an exact old invariant
block can omit newly lower modes even when every returned residual is zero.
Neither small residuals nor a full count certify global spectral completeness.
`constraints` may exclude a known invariant space, such as rigid modes;
`normalize_constraints=True` normalizes a private copy in the current mass
metric. Arbitrary equality constraints use `ConstrainedKrylovSchur` instead.
The [computed-start example](examples/block_reuse.py) assembles and updates a
small FEM bar on device; [block validation](benchmarks/upstream/BLOCK_API.md)
documents the supported CuPy fixtures and remaining reference mismatch.
Upstream FP64 BSR block-Jacobi and FSAI preconditioners can also be passed
directly; see the [adapter validation](benchmarks/experimental/UPSTREAM_PRECONDITIONERS.md)
for required upstream features and captured application checks.
`block.capture(..., finalizer=callback)` also records device postprocessing once
after the loop; preallocate its outputs and record Warp launches in the callback.
The [finalization and largest-mode example](benchmarks/experimental/LARGEST_BLOCK.md)
checks changed inputs and failure recovery on graph replay.


```python
from warpack import CuDSSInverse, GeneralizedKrylovSchur

# Kshift is A - sigma*B, assembled on device before factor setup.
sigma = 7.3
inverse = CuDSSInverse(
    Kshift, width=1, mtype="general", matching=True, refinement_steps=1,
)
solver = GeneralizedKrylovSchur(
    SparseOperator(A), SparseOperator(B), inverse,
    k=20, ncv=48, target=sigma, tol=1e-10,
)
graph = solver.capture(40, adaptive=False)
wp.capture_launch(graph)
```

`GeneralizedKrylovSchur` uses `which="LM"` in shift-invert mode to select modes nearest `target`; with SPD A and a zero target, these are the lowest modes. For an interior shift, pass the matching `target=sigma`; sorting by smallest algebraic value is not equivalent to sorting by distance to an interior shift. The generalized residual uses `||Ax-λBx||₂ / max(1, ||Ax||₂+|λ| ||Bx||₂)`. Inner iterative inverses need tighter tolerances than the outer eigensolve. Verify the residual in the **original** problem after any spectral transformation; `EigenpairEvaluation` provides this for real standard problems, and `ShiftInvertEigensolver` does so automatically for complex shift-invert.

For SPD inverse solves, `CGInverse(..., verify_residuals=True)` adds fresh device-side `A*x-b` checks. `stats` retains upstream recursive CG counters; `fresh_stats` separately counts fresh residual failures, and `residual_maxima` stores maximum absolute/relative linear residual norms. Check these alongside the original eigenproblem residual. The audit is optional, preserves the upstream stopping rule, and adds no replay work when disabled. Exact in-place application is supported; partially overlapping views are rejected. See the [CG audit and regression report](benchmarks/experimental/CG_RESIDUAL_AUDIT.md).

`CGInverse(..., refinement_steps=1, refinement_tol=0.01)` adds a fixed fresh-residual
correction using upstream CG and the same preconditioner. With corrections,
`stats` totals initial and correction iterations per original RHS;
`initial_stats` and `refinement_stats` expose the separate native stages.
Optional fresh auditing checks the final original equation against the initial
solve's tolerance. The [correction report](benchmarks/experimental/CG_CORRECTION.md)
records accepted connection-mode settings, their cost, and tighter requests
that still fail. The default correction budget is zero.

For a pure-Warp real indefinite inverse, import `GMRESInverse` from `warpack` and replace the factor setup with `GMRESInverse(Kshift, tol=1e-12, restart=128, maxiter=768, preconditioner=P)`. Here `P` is an optional **fixed linear** upstream Warp right preconditioner; it must be valid for the supplied shifted system. For FEM, the comparison uses FSAI built from SPD `K+M`, not from indefinite `K-sigma*M`. Default outer capture is fixed for Warp 1.15 compatibility; `adaptive_capture=True` requires preallocated upstream loop control, as in the pinned Warp 1.18 environment. `inverse.stats` reports actual iterations and fresh linear residual failures on device; check these alongside the outer original-pencil residual and requested spectrum. Changing matrix values in place is supported with stable topology, while preconditioner updates remain caller-managed. The [interior FEM report](benchmarks/experimental/INTERIOR_GMRES.md) records accepted 10,890- and 41,616-DOF comparisons, a near-eigenvalue shift, and preserved failed controls. This backend remains substantially slower than cuDSS.

For sparse indefinite factors, `mtype="general", matching=True` enables cuDSS column matching and scaling during setup. Matching defaults to zero correction; the example adds one fixed correction for tighter inverse accuracy. Without matching, symmetric-indefinite/general factors retain one correction by default, and SPD retains zero. Set `refinement_steps` explicitly to choose the budget. Each correction forms `b - A*x` in Warp, applies the existing cuDSS factors, and updates `x` in Warp. No host convergence check occurs. Check original-problem residuals for the chosen settings; the [factorization report](benchmarks/upstream/MATCHING.md) records accuracy, speed, and rejected alternatives.

`BucklingEigensolver(..., method="krylov")` uses a one-RHS inverse and stiffness-orthogonal Krylov iteration. `GeneralizedKrylovSchur(..., mode="cayley", target=sigma)` ranks Cayley-transformed magnitudes with an SPD mass. Both return original eigenpairs; their selection differs from ordinary nearest-shift targeting. See the [generalized-transform validation and comparisons](benchmarks/upstream/TRANSFORMS.md). The [signed tetrahedral buckling comparison](benchmarks/applications/BUCKLING.md) adds SPD elastic stiffness / indefinite geometric-stiffness pencils, complete stiffness-weighted mode checks, and 41,616-DOF direct/CG/reference results.

cuDSS analysis and first factorization are setup operations outside graph capture. cuDSS may use host work during that setup. A retained Warp workspace allocator avoids `cudaMalloc` inside captured cuDSS solves. Its setup also finalizes Warp's exact BSR nonzero count before scalar CSR expansion. Captured cuDSS solves retain small vendor HtoD metadata copies. The [CUDA API and graph audit](benchmarks/applications/CUDSS_GRAPH.md) finds no explicit synchronization calls or residual readbacks during warm replay, but measures nonzero host enqueue cost; the strict zero-host-transfer gap remains [#29](https://github.com/alecjacobson/warpack/issues/29).

`CGInverse` provides a pure-Warp SPD inverse using upstream CG and either an upstream preconditioner name or a prebuilt Warp preconditioner. For free-body elasticity, the example helper `examples.problems.elasticity.rigid_body_basis` constructs six rigid motions on device. `OrthogonalComplementOperator` handles Euclidean-orthonormal rows in mass-normalized coordinates. For a generalized pencil with consistent mass, `MassOrthogonalComplementInverse` applies the correct primal/dual mass projections, with optional device basis normalization. See the [API and validation](benchmarks/applications/MASS_PROJECTION.md). See the [anvil settings](#timing-and-accuracy) for the best recorded FSAI configuration.

`CGInverse(..., atol=0.0)` requests relative stopping for small right-hand sides, subject to upstream Warp's dtype safety floor. The default `atol=None` preserves Warp's use of `tol` as both the relative and absolute tolerance. This distinction matters when using a loose inner tolerance for residual preconditioning; always check the original eigenproblem residual.

`GeneralEigensolver` and `ShiftInvertEigensolver` accept `initial_vector=start`, a device `wp.vec2d` array of shape `(n,)`. Captured initialization reads its current contents and normalizes safely on device. Zero/nonfinite starts produce a nonzero status; check status before using modes. See the [general starting-vector contract and replay tests](benchmarks/upstream/GENERAL_STARTS.md).

`PartialSVD` accepts `initial_vector=start`, a device FP64 array of length `min(A.shape)`: a right-side start for tall/square matrices and a left-side start for wide matrices. Each captured replay reads the current start, so a computed previous mode can be reused. Zero/nonfinite starts set a nonzero status and clear both physical convergence counts. See the [SVD starting-vector contract, validation and default-path performance controls](benchmarks/upstream/SVD_STARTS.md).

For sequences of related systems, the block solvers `SymmetricEigensolver` and `GeneralizedEigensolver` accept `initial_vectors=starts`: a device FP64 array with independent rows and shape `(r, n)`, where `1 <= r <= ncv`. Remaining rows are seeded randomly. Captured initialization reads the current contents of `starts`, so callers can copy previous modes into it before replay. Check `status`, original residuals, and requested-spectrum coverage; a good starting subspace does not certify completeness. The inverse/factors must match the current matrix. This is block inverse iteration, not LOBPCG; Krylov solvers retain their separate single-vector interface. See [related-FEM validation and remaining block-method work](benchmarks/applications/BLOCK_STARTS.md).

## Anvil showcase

[`examples/problems/elasticity.py`](examples/problems/elasticity.py) assembles the stable Neo-Hookean rest Hessian and lumped mass in Warp; its `RestElasticity` helper supports graph-captured value updates. The example then supplies these matrices to the library's eigensolver.

The current README GIF uses the supplied [`anvil for export-ftetwild.mesh`](anvil%20for%20export-ftetwild.mesh), read directly from disk: **94,619 vertices, 508,903 tetrahedra, and 283,857 degrees of freedom**. No mesher is invoked. The longest rest extent is normalized to one metre; the illustrative material uses `E = 100,000 Pa`, `ν = 0.3`, and `ρ = 1,000 kg/m³`, with a free body, stable Neo-Hookean rest Hessian, and lumped mass. Six rigid modes are checked and omitted, then the first 20 elastic modes are shown in order.

Each mode's maximum displacement is **5% of the rest bounding-box diagonal**, checked over **all vertices**, including the interior. The normalized diagonal is 1.18974049 m, so every peak displacement is 0.05948702 m. The smooth, continuous gptoolbox `okloop(256,-4*pi/3,-pi/2)` palette has **no isoline stripes**. Color follows instantaneous displacement with one fixed scale for the entire sequence. Each mode eases from rest to peak and back with `squease`; the camera stays fixed against a white studio background. The GIF has 660 frames at 20 fps, lasting 33 seconds.

The maximum original-problem normalized residual is **2.62e-9**, and the mass-orthogonality error is **1.91e-14**. Frequencies span **1.811–9.166 Hz** under the illustrative material parameters. These are exaggerated linear mode shapes. See the [numerical report](results/anvil_modes.json), [scale and style metadata](results/anvil_teaser.json), and [evaluated scene validation](results/anvil_teaser_validation.json).

Reproduce with Blender 4.5 and [gifski](https://gif.ski/):

```bash
python -m examples.dragon_modes --mesh "anvil for export-ftetwild.mesh" \
  --out results/anvil_modes.npz
blender -b --python examples/render_modes.py -- \
  --input results/anvil_modes.npz --output results/anvil_base.blend \
  --name Anvil --up-axis Y --yaw-degrees 90
blender -b --python examples/render_teaser.py -- \
  --base results/anvil_base.blend --input results/anvil_modes.npz \
  --output results/anvil_teaser.blend --name Anvil --up-axis Y --yaw-degrees 90 \
  --amplitude-bbd 0.05 --frames build/anvil_frames --render
blender -b results/anvil_teaser.blend --python-exit-code 1 --python examples/validate_teaser.py
gifski --fps 20 --quality 85 --width 720 --repeat 0 \
  --output results/anvil_teaser.gif build/anvil_frames/frame_*.png
```

The mesh name, orientation, and displacement cap are rendering options; the eigensolver and Hessian are unchanged. The [`v0.1.3` release](https://github.com/alecjacobson/warpack/releases/tag/v0.1.3) includes the input mesh, modes, editable Blender scene, GIF, and validation metadata.

### Timing and accuracy

Best recorded accepted configurations for the same **283,857-DOF anvil**, including six rigid and 20 elastic modes. The CPU rows were rechecked with MKL PARDISO on 2026-10-15; the GPU rows retain the earlier measurements. This is not a fresh matched rerun of every backend, and newer experimental block results are reported separately below. Solve times are medians of three synchronized trials after warmup, on an **NVIDIA L40** or **Intel Xeon Platinum 8362 (four BLAS threads)**. Each trial restores its starting vector. Setup is a single recorded factorization or preconditioner construction, separate from the median solve time; solve excludes mesh I/O/assembly, transfers, JIT/warmup/graph capture, and final original-problem validation.

| Method | Setup (s) | Solve (s) | Max original residual, all 26 |
|---|---:|---:|---:|
| SciPy ARPACK + MKL PARDISO | 9.5567 | 16.5791 | 4.011e-9 |
| Spectra + MKL PARDISO | 13.4716 | 9.9369 | 2.737e-8 |
| CuPy eigsh + shared cuDSS | 2.1534 | 0.4587 | 2.852e-9 |
| Warp + cuDSS | 2.2063 | 0.4669 | 2.610e-9 |
| Warp + FSAI-preconditioned CG (64-entry rows) | 5.0477 | 9.6082 | 3.036e-9 |

All five pass the common original-problem residual cutoff of **1e-7**, orthogonality cutoff of **1e-8**, and elastic mode-shape checks. Residuals are `||Av−λv||₂ / max(1,|λ|)` for unit-norm vectors and the original mass-normalized operator. Elastic frequencies agree with the reference to within **6.56e-14 relative error**. CPU and Warp rows report the worst residual across their three timed trials.

The direct-inverse rows solve with **`A + I`** and `ncv=64`, where `A = M^(-1/2) K M^(-1/2)` is the mass-normalized stiffness. This is `sigma = -1` in the `A - sigma*I` shift-invert convention. CuPy's row explicitly wraps shared cuDSS factors; cuDSS is supplied by this benchmark. Both CPU rows use MKL PARDISO SPD Cholesky through Eigen, METIS ordering, a four-thread limit, and internal refinement disabled. ARPACK runs through SciPy; Spectra remains native C++ without Python matvec callbacks. These are separate runs of the same factor configuration: their setup differences and solve-time variation should not be interpreted as intrinsic eigensolver overhead. The [CPU factorization audit](benchmarks/ANVIL_FACTORIZATION.md) preserves the former SuperLU/Eigen timings, explains their large fill-in and numerical factorization costs, and records the rejected PARDISO refinement profile. Factor setup is reusable for the same matrix and shift; changed values generally require numerical refactorization. CPU/GPU inverse implementations differ.

The production pure-Warp setting in this table solves with **`A + 10,000 I`** (`sigma = -10,000`), `ncv=120`, one outer cycle, and inner CG tolerance **1e-12** with a 10,000-iteration limit. Six rigid motions are constructed analytically in Warp and projected out; the 20 elastic modes are computed numerically, and all 26 modes are validated. Rigid-basis setup costs an additional **0.0155 s**, excluded from the setup column. FSAI uses `max_row_size=64`, `max_step_size=3`, `apply_lanes=8`, and FP32 factor storage with FP64 products and CG. These settings are tuned for this anvil. Reference eigensolutions are used only for validation.

The [experimental block solve with adjacent RHS sparse operations](benchmarks/experimental/COALESCED_OPERATOR.md) records **1.8292 s** for Warp versus **2.1680 s** for CuPy using the same upstream FSAI32/FP32-CG25 preconditioner of `A + 100 I`. A replication records **1.8432 s / 2.1421 s**. Warp is about **20% faster** than the fresh 2.2915 s original-layout control. Every trial passes all 20 elastic modes; Warp's maximum original residual across both profiles is **1.8213e-8** and returned-pair residual **9.1062e-9**, below its normalized tolerance **1e-8**. CuPy meets its fresh full-block absolute tolerance **1e-5**; the stopping formulas differ. Warp uses width 24, device-selected active CG/search rows and full history; CuPy uses width 20. Warp takes 16–17 outer steps and **7,800–8,000 scheduled inner vector iterations**. Replication setup is **1.2185 s**, capture including branch warmup **1.1266 s**, and setup/capture device free-memory decrease **4.454 GB** (not peak memory). These settings are Anvil-specific. `BlockLOBPCG` is public; the batching, memory-layout, search-compaction and cached-image adapters remain experimental. The gap to the earlier cuDSS control is still about four times. The [factor-layout](benchmarks/experimental/COALESCED_FSAI.md), [active-search](benchmarks/experimental/ACTIVE_SEARCH.md), [budget](benchmarks/experimental/ANVIL_BUDGET.md), [shift](benchmarks/experimental/ANVIL_SHIFT.md) and [projected-kernel](benchmarks/experimental/FUSED_PROJECTED.md) reports retain earlier and rejected profiles.

FSAI comes from the unmodified [Warp PR #16](https://github.com/alecjacobson/warp/pull/16), pinned to `04427b997fe7ff07bc6ed1dde5ca9813496c6b55`. Its construction synchronizes outside capture; preconditioner application and the complete eigensolve are graph capturable. The recorded CG graph has no host-copy or host-callback nodes; median host enqueue is **1.311 ms**.

Sources: [current ARPACK + PARDISO](results/anvil_pardiso_noir_confirmed.json), [current Spectra + PARDISO](results/anvil_pardiso_spectra_confirmed.json), [original CPU and retained CuPy comparison](results/anvil_benchmarks.json), [latest Warp + cuDSS recheck](results/anvil_cudss_recheck.json), and [best CG settings and per-replay accuracy](results/anvil_cg_optimized.json). Earlier configurations and unsuccessful trials are in the [historical experiments](historical/README.md).

## Validation and benchmarking

The [complex connection comparison](benchmarks/applications/CONNECTION.md) adds
positive lumped and complex consistent mass, repeated modes, independent
Fourier/element-energy checks, and native-complex reference comparisons. Its
generalized complex recovery is a benchmark adapter outside the installed API.
At 16,384 complex unknowns and 20 modes, Warp+cuDSS passes in 0.330 s versus
0.620 s SciPy and 0.746 s Spectra. The report separately labels modified-CuPy
diagnostics and preserves native CuPy and strict inner-CG failures.

Application comparison commands and the normal-POD driver fail when any requested method misses its recorded numerical contract. Use their explicit `--diagnostic` option for inventories containing known failures or unsupported interfaces; the JSON keeps those failures. See the [comparison process contract](benchmarks/COMPARISON_CONTRACTS.md).

The [application coverage suite](benchmarks/applications/README.md) adds physics/geometry fixtures for nine API classes, independent spectrum and original-equation checks, and accuracy-gated ARPACK/Spectra/CuPy comparisons. [Coverage issues and priorities](https://github.com/alecjacobson/warpack/issues/18) track the remaining gaps. `GeneralizedKrylovSchur` adds mass-orthogonal restarted Krylov iteration for SPD-mass pencils; it supports nearest-shift solves and regular mass-inverse LM/LA/SA selection. Its required inverse has one RHS.

The [latest full local candidate validation](results/performance_followup/svd-start-validation.json), recorded on **2026-10-20**, has **1,365 passing tests** in the pinned FSAI environment, with passing lint and formatting checks. The separately merged upstream-status harness adds eight CPU tests, validated outside that count. The [clean-source wheel check](results/performance_followup/svd-start-wheel.json) confirms all 26 core modules install without benchmark/application modules. The 63 SVD/replay checks also pass on stock Warp 1.15. The general/projected/shifted solver subset passes 172 focused checks on stock Warp 1.15; the earlier public CG refinement/audit/tolerance subset passes 57. The projected, block and complete normal-basis paths also pass 158 focused tests on stock Warp 1.15; all 35 real GMRES component and interior-FEM tests passed there in their preceding validation. The [full CuPy refresh](benchmarks/upstream/CUPY_REFRESH.md) passes all 31 original ordinary failures with no lost passes: **860 ordinary passes, one known Fiedler failure, 3,015 skips, 72 expected failures and 36 unexpected passes**. This complete 3,984-case rerun includes all 39 newly enabled eigenproblem/SVD starting-vector cases; it is not a composite of focused results. The unchanged native CuPy inventory has no ordinary failures, and all 31 native Spectra executables pass. The [complete frozen Spectra numerical refresh](benchmarks/upstream/SPECTRA_REFRESH.md) records **170 passes, 14 failures and 176 unsupported cases** across all 360 fixtures: 52 former failures/timeouts now pass, with no lost passes. It checks the final of two replays; per-replay validation and 27 additional generalized mappings are separate follow-ups. Unsupported contracts and unchanged-profile failures remain explicit. [Warp residual correction of cuDSS inverses](benchmarks/upstream/REFINEMENT.md) fixes all 24 real CuPy SM eigenvector failures and the local damped-chain regression. That report records the full upstream rerun, strict-count regressions and correction overhead separately. The [projected Jacobi repair](benchmarks/upstream/PROJECTED_CONVERGENCE.md) fixes an insufficient sweep budget, adds bounded device convergence for larger even projections, and preserves Anvil and POD timing/accuracy gates. The [complete Hermitian follow-up](benchmarks/upstream/HERMITIAN_LARGE.md) passes all 30 Spectra Hermitian fixtures through 1,000 complex unknowns, with unchanged original tolerances and full counts, and corrects basis-width reporting. The [matched-factor follow-up](benchmarks/upstream/MATCHING.md) adds sparse-indefinite regressions and recovers the interior-mode timing gap, with explicit accuracy profiles and preserved failed experiments. The [inverse-selection profiles](benchmarks/upstream/SHIFTED_SELECTION.md) resolve the shifted-symmetric investigation with verified farthest-from-shift targets, wider explicit SM bases and retained compact nonconvergence controls. The [generalized-transform follow-up](benchmarks/upstream/TRANSFORMS.md) passes all 21 supported generalized Spectra cases, with independent spectrum/subspace regressions and native comparisons. The [Hodge coverage report](benchmarks/applications/HODGE.md) adds independent topology/nullspace checks, weak-component and disconnected fixtures, and separately labeled native CuPy eigsh and generalized LOBPCG comparisons. Its [CG follow-up](benchmarks/applications/HODGE_CG.md) validates faster explicit Krylov profiles, with rejected short budgets and strict-count distinctions retained. The [elastic ROM coverage](benchmarks/applications/FEM.md) adds clamped/free and disconnected tetrahedral bodies, material contrast, consistent mass and complete repeated-mode checks, including upstream block-Jacobi/FSAI comparisons. The [interior-band coverage](benchmarks/applications/INTERIOR.md) checks coupled FEM pencils, repeated nearest-mode sets and large frequency-band comparisons. The [indefinite FEM follow-up](benchmarks/applications/NEGATIVE_FEM.md) covers SA/SM/LM with repeated modes and both mass types, records absolute versus relative CG tolerance policies, and adds pure-Warp GMRES for indefinite nearest-zero solves with mandatory fresh inner checks. Its [regular-mode follow-up](benchmarks/applications/REGULAR_FEM.md) validates shorter explicit LM budgets and exact diagonal-mass routes, with native comparisons and retained rejected experiments. The [inner-CG accuracy follow-up](benchmarks/applications/CG_ACCURACY.md) records profiles that also meet strict outer counts at large sizes and 20/40 modes. The [device nullspace diagnostic](benchmarks/applications/NULLSPACE.md) distinguishes returned near-zero modes, weak positives, invalid data and capacity limits without claiming a certified Betti count. The [metric-QR recovery](benchmarks/applications/WEAK_BLOCK.md) fixes non-finite generalized block modes on extreme weak components, with full requested-space checks and measured recovery overhead. The broader [pinned CuPy and Spectra conformance runs](benchmarks/upstream/RESULTS.md) expose additional numerical failures, unsupported interfaces, and timeouts; the local suite is not full upstream conformance. The [Krylov restart regression report](benchmarks/upstream/MULTIPLICITY.md) records the multiplicity fixes and adaptive-performance tradeoffs. The [large POD comparison](benchmarks/applications/POD.md) adds tall/wide and repeated singular subspaces, vibrating-string snapshots, native ARPACK/Spectra/CuPy comparisons, and pure-Warp long-row product improvements. The fifteen [constrained skinning tests](benchmarks/applications/SKINNING.md) also pass on stock Warp 1.15. [Cached multiplier recovery](benchmarks/applications/SKINNING_PERFORMANCE.md) removes redundant KKT solves, bringing the large skinning replay to 288 ms alongside adapted CuPy+cuDSS at 288 ms. Earlier Anvil compatibility checks passed six tests, with four upstream block-Jacobi cases skipped. See the [original Anvil validation and source provenance](results/anvil_cg_validation.json) for that compatibility check.

The [experimental block eigensolver report](benchmarks/experimental/README.md) adds captured LOBPCG-style iteration, changed-pencil replay, and direct-FSAI/truncated-CG Anvil accuracy checks. Its numerical iteration is now the public `BlockLOBPCG`; batched CG and sparse-product experiments remain outside the installed package. Its original timings were contended. The [batched-CG follow-up](benchmarks/experimental/BATCHED_CG.md) adds uncontended timings, independent mixed-precision regression coverage and explicit workspace setup accounting. The [adaptive follow-up](benchmarks/experimental/ADAPTIVE_CG.md) adds nested upstream CG, stock-version rejection tests and a 79.08 ms large Hodge solve; the production table above retains its recorded configurations. The new `CGInverse.atol` regressions also pass on stock Warp 1.15.

The [active-CG follow-up](benchmarks/experimental/ACTIVE_CG.md) adds device-selected preallocated batch sizes, row reactivation and captured changed-matrix regressions. Its focused validation is separate from the full local suite above.

The explicit [`NormalSVD` API](benchmarks/upstream/NORMAL_SVD.md) uses normal products for well-resolved largest nonzero triplets, with device-side original-equation and orthogonality checks. On a 100,000×128 clustered snapshot matrix, a full 128-vector normal basis takes **74.96 ms**, alongside CuPy **159.70 ms** and augmented Warp **378.41 ms**. A 64-vector normal basis takes **434.73 ms**: basis size matters. The [guard and wider-snapshot report](benchmarks/experimental/NORMAL_GUARDS.md) preserves all profiles. Normal equations square conditioning and reject unresolved requested values; `PartialSVD` retains its augmented formulation. Both refresh current transpose values on captured replay at fixed topology. The [duplicate-CSR repair](benchmarks/upstream/DUPLICATE_CSR.md) preserves summed coefficients and constrained row norms for duplicate entries, including changed captured inputs. Complete normal search spaces of at most 64 dimensions use batched coordinate projection; larger and partial spaces retain Krylov expansion. The [small complete-basis comparison](benchmarks/experimental/FULL_NORMAL_BASIS.md) reduces the 100,000×64 replay from **27.21 to 14.58 ms**, alongside native CuPy **93.49 ms**. This explicitly forms the small normal matrix and retains the same conditioning safeguards.

The [Hermitian breakdown follow-up](benchmarks/upstream/HERMITIAN.md) fixes the seven reported complex identity/low-rank CuPy failures and adds changed-matrix replay and complex subspace checks. The [complex-start follow-up](benchmarks/upstream/HERMITIAN_STARTS.md) adds replayable device starting vectors, invalid-input status and 15 newly enabled CuPy cases. The [Hermitian shift-invert follow-up](benchmarks/upstream/HERMITIAN_SHIFT.md) fixes inaccurate sparse complex inverse solves, passes all 20 small Spectra Hermitian cases, and enables the remaining 48 CuPy SM starting-vector cases. The [general SM follow-up](benchmarks/upstream/GENERAL_SM.md) resolves the five listed nonsymmetric failures with matched factors and an explicit 200-cycle budget for compact cases; it does not claim success at the old 30-cycle budget. The [residual-directed general restart](benchmarks/experimental/GENERAL_RESTART.md) fixes the two larger tight-count cases at their original tolerances and budgets, with repeated full-count/spectrum checks and matched timing controls. These focused runs do not establish full upstream conformance.

The [Davidson-fixture and SVD follow-up](benchmarks/upstream/DAVIDSON_SVD.md) passes all eight cases with explicit wider Krylov spaces at unchanged tolerances: `ncv=40` for the symmetric fixtures and `ncv=20` for SVD. It adds paired SVD equations, physical convergence counts, and independent spectrum/subspace checks; 69 focused tests pass in the pinned environment and 46 on stock Warp 1.15. Production defaults and timing tables remain unchanged, and failed narrow-space controls are preserved.

The tests cover rectangular SVD, buckling/Cayley transforms, FP64 fractional shifts, real symmetric and nonsymmetric spectra, complex Hermitian and general spectra, repeated/zero eigenvalues, both spectrum ends, interior real/complex shifts, non-diagonal mass, device convergence limits, graph replay with changed matrix values, FEM energy derivatives, rigid modes, and graph-captured FEM assembly. They also check the incremental projected matrix against `Q A Qᵀ`, captured breakdown recovery, small projected eigenproblems, orthogonal-complement projection, and pure-Warp graphs for host nodes/transfers. These are representative Spectra-style tests, not a complete port of every upstream test.

```bash
OPENBLAS_NUM_THREADS=4 python -m pytest -q
python -m pip install ruff
ruff check warpack tests benchmarks examples

git clone https://github.com/yixuan/spectra.git build/spectra
# Pin the revision used for the published comparison:
git -C build/spectra checkout db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95
g++ -O3 -DNDEBUG -I/usr/include/eigen3 -Ibuild/spectra/include \
  benchmarks/spectra.cpp -o build/spectra_bench
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python benchmarks/compare.py
```

<details>
<summary>Pinned environment and commands for the current anvil comparison</summary>

The recorded GPU runs use the official **Warp 1.18.0+cu12 native runtime** for CUDA driver 570.158.01. The original SuperLU/Eigen/CuPy comparison used upstream linear-solver revision `5a0c33d17d847d0be07ae1003263b572aae0729a`; its exact environment is retained in the [baseline notes](historical/anvil-baseline.md). The recorded Anvil Warp rows and their original 111-test validation used three unmodified FSAI PR modules overlaid on that runtime. Subsequent validation uses the same pinned environment. The current CPU PARDISO rows use MKL 2020.4.304 and Eigen 3.4.0; their build and run commands are in the [factorization audit](benchmarks/ANVIL_FACTORIZATION.md#reproduction). Install the Warp overlay without changing the default Warp installation:

```bash
python -m pip install --no-deps --target build/warp-fsai \
  'https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
for f in warp/_src/optim/linear.py warp/_src/optim/fsai.py warp/optim/linear.py; do
  curl --fail --location \
    "https://raw.githubusercontent.com/alecjacobson/warp/04427b997fe7ff07bc6ed1dde5ca9813496c6b55/$f" \
    --output "build/warp-fsai/$f"
done
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 python -m pytest -q

# Original Eigen LDLT/SuperLU CPU references and retained direct GPU comparison.
# Requires the pinned Spectra checkout above and Eigen headers.
g++ -O3 -DNDEBUG -shared -fPIC -I/usr/include/eigen3 -Ibuild/spectra/include \
  benchmarks/spectra_native.cpp -o build/spectra_native.so
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.anvil --part direct

# Creates a cuDSS validation reference if one is not cached in build/.
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.cg_experiments --mode eigen --shifts 10000 \
  --preconditioners fsai64f --ncv 120 --tol 1e-12 \
  --project-rigid --repeats 3 --profile --out results/anvil_cg_optimized.json
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.cg_experiments --mode eigen --shifts 1 \
  --preconditioners cudss --ncv 64 --repeats 3 --out results/anvil_cudss_recheck.json
```

</details>

[Historical experiments](historical/README.md) preserve the older release tables, dragon showcases and constitutive audits, synchronization investigations, and CG/warm-start tuning, including failed configurations and their reproduction commands.
