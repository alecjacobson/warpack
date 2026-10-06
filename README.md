# warpack

Sparse eigenvalue problems computed in **NVIDIA Warp**, with an optional **cuDSS** inverse backend. FP64 arithmetic, device-resident projected eigensolves and residuals, and reusable CUDA graphs. NumPy, SciPy, CuPy, and Spectra are references in the tests and benchmarks; they do not perform eigensolver computation inside the library.

This is a first implementation, not an API-compatible drop-in replacement for Spectra or ARPACK. It covers their main eigenproblem families. Performance varies by problem: the measurements below include cases where Spectra or CuPy wins.

[Download the animated teaser and editable Blender scene](https://github.com/alecjacobson/warpack/releases/tag/v0.1.3)

![Anvil elastic modes 1–20: 5% bounding-box-diagonal peak displacement with continuous instantaneous coloring](results/anvil_teaser.gif)

## Supported problems

| Problem | API | Notes |
|---|---|---|
| Real symmetric | `KrylovSchur` | Thick restart, two-pass reorthogonalization, LM/LA/SA/SM/BE selection |
| Symmetric block inverse iteration | `SymmetricEigensolver` | CholeskyQR2 and Rayleigh–Ritz; supply an iteration operator that amplifies the requested spectrum |
| Symmetric generalized `A x = λ B x` | `GeneralizedEigensolver` | SPD mass, including non-diagonal sparse B; B-orthonormal eigenvectors; inverse iteration and interior targeting |
| Complex Hermitian | `HermitianEigensolver` | Realification, block iteration, and complex basis recovery; avoids duplicate realified eigenvectors |
| Real or complex nonsymmetric | `GeneralEigensolver` | Restarted complex Arnoldi, device Hessenberg reduction and shifted QR; LM/SM/LR/SR/LI/SI selection |
| Real or complex shift-invert | `ShiftInvertEigensolver` | Eigenpairs nearest a real or complex shift, using cuDSS or pure-Warp GMRES |
| Buckling / Cayley | `BucklingEigensolver`, `CayleyEigensolver` | Generalized spectral transforms, with residuals in the original problem |
| Partial rectangular SVD | `PartialSVD` | Largest nonzero real singular triplets through an augmented symmetric operator |
| Pure-Warp SPD inverse | `CGInverse` | Preallocated Warp CG, optional upstream preconditioning, device convergence checks |
| Tetrahedral rest elasticity | `RestElasticity` | Stable Neo-Hookean rest Hessian, lumped mass, graph-capturable value assembly |

`SparseOperator` accepts compact scalar FP64 CSR or 3×3 FP64 BSR matrices from `warp.sparse`. Complex CSR values use `wp.vec2d(real, imaginary)`. The general and Hermitian paths are currently reference implementations; their performance has not been tuned to the level of the real symmetric path. The SVD path is validated for leading nonzero singular triplets; recovery of nullspace bases is not currently supported.

## Install

```bash
python -m pip install -e '.[test]'
# Optional sparse direct inverse:
python -m pip install -e '.[cudss]'
# Reference GPU/CPU benchmarks:
python -m pip install -e '.[benchmark]'
```

Tested with Python 3.10, Warp 1.15.0, CUDA driver 570.158.01, cuDSS 0.8.0.10, CuPy 14.2.0, and an NVIDIA L40. Conditional graphs require CUDA driver support for CUDA 12.4 or later. Complex data and solver workspaces use double precision. The best FSAI result below uses FP32 preconditioner factors with FP64 products and CG; its pinned Warp PR environment is documented in [Validation and benchmarking](#validation-and-benchmarking).

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

For symmetric solves, the reported residual is `||A x - λ x||₂ / max(1, |λ|)`, with unit-norm vectors. `status == 0` means no internal factorization failure; it does **not** mean all eigenpairs converged. Check `converged == k` and the residuals. A fixed iteration budget can expire without convergence. `SM` targets are generally better addressed with shift-invert than an untransformed Krylov iteration.

`capture()` compiles and warms kernels, allocates setup state, and constructs a reusable graph. Ordinary symmetric and general iterations use a device-controlled conditional loop. cuDSS and the current Warp iterative inverse functors require a **fixed outer graph** because their internal graph operations cannot be nested inside a conditional body. This is selected automatically; `capture(iterations, adaptive=False)` makes it explicit. Fixed outer loops still perform all numerical work on device. There are no hidden host eigensolves or residual readbacks.

Matrix **values** can change between replays with stable array addresses. Changing a sparse topology requires a new operator/solver setup. An inverse's factors must also be updated or recreated when matrix values change. A solver instance owns mutable workspaces and is not thread-safe. Destroy graphs before closing their inverse backend; never replay a graph after its factors have been released.

## Inverse and generalized problems

```python
from warpack import CuDSSInverse, GeneralizedEigensolver

# Kshift is A - sigma*B, assembled on device before factor setup.
sigma = 7.3
inverse = CuDSSInverse(Kshift, width=48, mtype="symmetric")
solver = GeneralizedEigensolver(
    SparseOperator(A), SparseOperator(B), inverse,
    k=20, ncv=48, target=sigma, tol=1e-10,
)
graph = solver.capture(40, adaptive=False)
wp.capture_launch(graph)
```

For a zero shift and SPD A, `which="SA"` selects the lowest modes. For an interior shift, pass the matching `target=sigma`; sorting by smallest algebraic value is not equivalent to sorting by distance to an interior shift. The generalized residual uses `||Ax-λBx||₂ / max(1, ||Ax||₂+|λ| ||Bx||₂)`. Inner iterative inverses need tighter tolerances than the outer eigensolve. Verify the residual in the **original** problem after any spectral transformation; `EigenpairEvaluation` provides this for real standard problems, and `ShiftInvertEigensolver` does so automatically for complex shift-invert.

cuDSS analysis and first factorization are setup operations outside graph capture. cuDSS may use host work during that setup. A retained Warp workspace allocator avoids `cudaMalloc` inside captured cuDSS solves. Its setup also finalizes Warp's exact BSR nonzero count before scalar CSR expansion.

`CGInverse` provides a pure-Warp SPD inverse using upstream CG and either an upstream preconditioner name or a prebuilt Warp preconditioner. For free-body elasticity, `rigid_body_basis` constructs the six rigid motions on device, and `OrthogonalComplementOperator` projects them out of the iteration operator. See the [anvil settings](#timing-and-accuracy) for the best recorded FSAI configuration.

## Anvil showcase

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

Best recorded accepted configurations for the same **283,857-DOF anvil**, including six rigid and 20 elastic modes. Solve times are medians of three synchronized trials after warmup, on an **NVIDIA L40** or **Intel Xeon Platinum 8362 (four BLAS threads)**. Each trial restores its starting vector. Setup covers factorization or preconditioning; solve excludes mesh I/O/assembly, transfers, JIT/warmup/graph capture, and final original-problem validation.

| Method | Setup (s) | Solve (s) | Max original residual, all 26 |
|---|---:|---:|---:|
| SciPy ARPACK + SuperLU | 449.2954 | 42.6216 | 4.519e-9 |
| Spectra + Eigen LDLT | 799.6008 | 62.4199 | 3.078e-8 |
| CuPy eigsh + shared cuDSS | 2.1534 | 0.4587 | 2.852e-9 |
| Warp + cuDSS | 2.2063 | 0.4669 | 2.610e-9 |
| Warp + FSAI-preconditioned CG (64-entry rows) | 5.0477 | 9.6082 | 3.036e-9 |

All five pass the common original-problem residual cutoff of **1e-7**, orthogonality cutoff of **1e-8**, and elastic mode-shape checks. Residuals are `||Av−λv||₂ / max(1,|λ|)` for unit-norm vectors and the original mass-normalized operator. Elastic frequencies agree with the reference to within **6.56e-14 relative error**. Warp rows report the worst residual across their three replays.

The direct-inverse rows use shift **+1** and `ncv=64`. CuPy's row explicitly wraps shared cuDSS factors; cuDSS is supplied by this benchmark. ARPACK uses SuperLU (`MMD_AT_PLUS_A`, zero diagonal-pivot threshold, symmetric mode), while Spectra uses native C++ Eigen AMD-ordered sparse LDLᵀ. These timings include different CPU/GPU inverse implementations.

The fastest pure-Warp setting uses shift **+10,000**, `ncv=120`, one outer cycle, and inner CG tolerance **1e-12** with a 10,000-iteration limit. Six rigid motions are constructed analytically in Warp and projected out; the 20 elastic modes are computed numerically, and all 26 modes are validated. Rigid-basis setup costs an additional **0.0155 s**, excluded from the setup column. FSAI uses `max_row_size=64`, `max_step_size=3`, `apply_lanes=8`, and FP32 factor storage with FP64 products and CG. These settings are tuned for this anvil. Reference eigensolutions are used only for validation.

FSAI comes from the unmodified [Warp PR #16](https://github.com/alecjacobson/warp/pull/16), pinned to `04427b997fe7ff07bc6ed1dde5ca9813496c6b55`. Its construction synchronizes outside capture; preconditioner application and the complete eigensolve are graph capturable. The recorded CG graph has no host-copy or host-callback nodes; median host enqueue is **1.311 ms**.

Sources: [ARPACK, Spectra, and CuPy comparison](results/anvil_benchmarks.json), [latest Warp + cuDSS recheck](results/anvil_cudss_recheck.json), and [best CG settings and per-replay accuracy](results/anvil_cg_optimized.json). Earlier configurations and unsuccessful trials are in the [historical experiments](historical/README.md).

## Validation and benchmarking

The latest recorded validation has **111 passing tests** in the pinned FSAI environment, plus passing lint, formatting, and wheel-build checks. Stock Warp 1.15 compatibility checks passed six tests, with four upstream block-Jacobi cases skipped. See [validation and source provenance](results/anvil_cg_validation.json).

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

The recorded GPU runs use the official **Warp 1.18.0+cu12 native runtime** for CUDA driver 570.158.01. The CPU/CuPy comparison used upstream linear-solver revision `5a0c33d17d847d0be07ae1003263b572aae0729a`; its exact environment is retained in the [baseline notes](historical/anvil-baseline.md). The latest Warp rows and 111-test validation use three unmodified FSAI PR modules overlaid on that runtime. Install the latter without changing the default Warp installation:

```bash
python -m pip install --no-deps --target build/warp-fsai \
  'https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
for f in warp/_src/optim/linear.py warp/_src/optim/fsai.py warp/optim/linear.py; do
  curl --fail --location \
    "https://raw.githubusercontent.com/alecjacobson/warp/04427b997fe7ff07bc6ed1dde5ca9813496c6b55/$f" \
    --output "build/warp-fsai/$f"
done
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 python -m pytest -q

# Native Spectra reference; requires the pinned checkout above and Eigen headers.
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
