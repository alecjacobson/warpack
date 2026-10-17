# CuPy and Spectra conformance, 2026-10-13

Subsequent fixes and focused reruns are recorded in the
[Krylov multiplicity report](MULTIPLICITY.md) and
[cuDSS residual-correction report](REFINEMENT.md), and
[Hermitian breakdown report](HERMITIAN.md), and
[Hermitian shift-invert report](HERMITIAN_SHIFT.md). The counts below preserve this
original baseline.

This run found correctness and coverage gaps that the 111-test local suite did not expose. Production solver code stayed at `98f10fa`; the changes in this commit are test adapters, benchmark runners, and recorded results. Every inverse used cuDSS. No CG preconditioner or inner CG stopping criterion was involved.

The scope is CuPy's sparse eigensolver/SVD tests, Spectra's complete native CTest suite, the exported numerical problems from its 16 solver/example source files, and its complete 40-dataset benchmark. Private Spectra factorization/implementation tests ran natively; they do not have direct Warpack equivalents. [Methodology and reproduction](README.md) describe the adapters and unsupported interfaces.

## Environment and sources

One NVIDIA L40 (48 GB), driver 570.158.01, Xeon Platinum 8362, four BLAS/OpenMP threads, Python 3.10. The native Spectra tests use CMake Release; the original native benchmark is compiled with `g++ -O2 -std=c++11`. The pinned FSAI compatibility environment uses Warp 1.18.0+cu12 native runtime; these runs do not exercise FSAI or CG.

- [Spectra source](https://github.com/yixuan/spectra/tree/db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95).
- [CuPy sparse test source](https://github.com/cupy/cupy/blob/2316a02d70c7e39d6a7c035529df8358d794256e/tests/cupyx_tests/scipy_tests/sparse_tests/test_linalg.py).
- The CuPy reference loads that revision's unmodified `_eigen.py` over installed CuPy 14.2.0. It is a compatibility overlay, not a complete build of CuPy main.
- [Provenance, source hashes, native test log, and native benchmark samples](../../results/upstream_provenance.json).

## CuPy assertions

| Backend | Passed | Failed | Skipped | Expected failures | Unexpected passes |
|---|---:|---:|---:|---:|---:|
| Native CuPy overlay | 2,500 | 0 | 1,195 | 173 | 116 |
| Warpack adapter | 673 | 31 | 3,172 | 72 | 36 |

Each row accounts for all 3,984 selected parameter cases. Another 7,745 tests of unrelated sparse operations were deselected. Expected failures and unexpected passes retain upstream markers and are separate from ordinary passes. Different unsupported-feature skips mean the two rows have different effective coverage.

All **31 ordinary Warpack failures pass in the native CuPy overlay**:

- **Six complex identity cases, `which='SA'`:** returned zero eigenvalues instead of ones.
- **One complex low-rank Gram case, `which='LM'`:** returned zeros instead of the nonzero spectrum plus a null-space eigenvalue.
- **Twenty-four real near-singular `SM` cases:** eigenvalue-only checks pass, but the matrix-norm-scaled eigenvector residual is approximately `9.71e-10`, against a required `1e-12`. These are repeated parameterizations of the same underlying accuracy problem across input representations and requested eigenpair counts. They use the same automatic shift rule as the CuPy reference and a cuDSS inverse.

The complex failures make rank-deficient basis handling a priority. Source inspection suggests that the Hermitian block iteration can annihilate or lose rank in its basis: for the identity with `SA`, its `bound*I-A` iteration operator is zero. This is a diagnosis to test, not a verified production fix. The real inverse failure still needs isolation between inverse accuracy, the projected solve, and reconstruction of the physical eigenvectors.

Of the 3,172 skips, 580 come from upstream fixture conditions. The remaining 2,592 reflect unavailable precision or adapter/API coverage: FP32/complex64 (1,618), LOBPCG (588), complex SVD (256), complex starting vectors (63), CuPy's `ncv` clamping (36), complex shifted selection other than nearest (16), RNG interfaces (11), and additional `mode`/`M`/`Minv` keyword contracts (4). These are not passes and should not all be interpreted as missing mathematical problem classes: some are differences in API validation or input contracts.

Raw reports: [Warpack](../../results/upstream_cupy.json), [native CuPy overlay](../../results/upstream_cupy_native.json). The [first adapter pass](../../results/upstream_cupy_first_pass.json) is retained for audit; its larger failure count includes adapter bugs subsequently corrected, not additional confirmed solver defects.

## Spectra assertions

All **31 native CTest executables passed** (63.42 seconds). Warpack's own **111 tests passed** (13.12 seconds). Spectra's native success includes sections that explicitly allow nonconvergence; 75 of the 360 exported numerical cases carry that flag.

The adapted run accounts for **360 cases: 118 passed, 63 failed, 176 unsupported, and 3 timed out**. There were no runtime errors. Of the 63 failures, 19 meet the original numerical bounds but miss the adapter's internal convergence count; another case meets the residual bound but returns the wrong requested spectrum. Eighteen failed cases belong to sections where upstream permits nonconvergence (these categories overlap).

| Family | Passed | Failed | Unsupported | Timed out |
|---|---:|---:|---:|---:|
| ComplexEigs | 18 | 5 | 12 | 1 |
| DavidsonSymEigs | 0 | 4 | 0 | 0 |
| Example1 | 0 | 3 | 0 | 0 |
| Example1Shift | 0 | 3 | 0 | 0 |
| Example2 | 3 | 0 | 0 | 0 |
| Example3 | 0 | 0 | 2 | 0 |
| Example4 | 1 | 1 | 0 | 0 |
| GenEigs | 18 | 4 | 12 | 2 |
| GenEigsComplexShift | 0 | 0 | 36 | 0 |
| GenEigsRealShift | 4 | 2 | 30 | 0 |
| HermEigs | 14 | 16 | 0 | 0 |
| SVD | 0 | 4 | 0 | 0 |
| SymEigs | 26 | 4 | 0 | 0 |
| SymEigsShift | 24 | 6 | 0 | 0 |
| SymGEigsCholesky | 2 | 4 | 24 | 0 |
| SymGEigsRegInv | 0 | 3 | 12 | 0 |
| SymGEigsShift | 8 | 4 | 48 | 0 |

The final timeouts are `ComplexEigs_33`, `GenEigs_12`, and `GenEigs_33`, each after a 120-second whole-worker attempt. All general smallest-magnitude cases use cuDSS in this final report.


The adapter preserves generated matrices, requested eigenpair counts, selection rules where supported, and the original absolute residual/eigenvalue/vector assertions. It uses Warpack APIs rather than reimplementing Spectra algorithms. Its internal `1e-12` convergence request and outer-cycle budgets differ from Spectra's defaults. Results that meet the original residual bound but miss Warpack's stricter internal convergence count are identified separately.

Some important failures require checking the requested spectrum, not only residuals. In `Example1_0` (the repeated spectrum of a cycle Laplacian), Warpack reports all three pairs converged and a residual around `2.42e-13`, but the selected eigenvalues differ from the requested spectrum by about `0.142`. `Example4_1`, an explicit-start regression, also fails. These are useful targeted regressions before tuning performance.

All four partial-SVD fixtures fail at the recorded 300-cycle budget and original `ncv=10`: singular-value errors range from about `9.16e-6` to `2.64e-2`, against `1e-9`. This exposes a convergence gap for the augmented SVD formulation at those settings; it does not establish that more iterations or a larger subspace would fail.

Unsupported cases include Spectra's imaginary-part-magnitude selection (different from Warpack's signed-imaginary selection), its real-part complex-shift transform, non-nearest general shift-invert selection, and generalized extremal/transform selections not exposed by the current public API. [Per-case results](../../results/upstream_spectra_warp.json) give the exact reason for each case.

## Upstream benchmarks

Spectra's original benchmark contains **40 dense datasets**, with 100 or 1,000 unknowns, symmetric or nonsymmetric, ten random datasets per size/family, and ten trials per dataset. All **400 trials per CPU backend** ran for ARPACK and Spectra. The GPU adapter uses the same matrices, starts, `k`, `ncv`, and requested `tol=1e-10`. Different convergence definitions still matter.

These benchmark datasets request largest-magnitude eigenpairs and use direct Warp iteration. They do not exercise cuDSS; cuDSS is exercised by the inverse problems in the conformance suites.

These are small-problem diagnostics. They do not replace the large Anvil and dragon measurements in the main README. The [pinned CuPy performance repository](https://github.com/cupy/cupy-performance/tree/f014dacb4181d1b0a8b9c3923524f37509f7a59e) contains no eigensolver benchmark, so CuPy is measured on the shared Spectra datasets. CuPy has no general sparse `eigs` entry point.

Times below are **milliseconds**, using the median of each dataset's ten trial times, then the median across datasets. GPU cells include accepted datasets out of ten; a partial group uses a different subset from the native CPU columns and is not a matched speedup comparison.

| Problem | Size | ARPACK CPU | Spectra CPU | CuPy GPU | Warp GPU |
|---|---:|---:|---:|---:|---:|
| Symmetric | 100 | 0.864 | 0.715 | 38.423 (10/10) | 7.100 (10/10) |
| Symmetric | 1,000 | 54.997 | 45.151 | 64.446 (10/10) | 22.424 (10/10) |
| General | 100 | 4.177 | 3.067 | No sparse eigs API | 737.809 (10/10) |
| General | 1,000 | 466.043 | 408.320 | No sparse eigs API | 5679.324 (9/10) |

| Problem | Size | Maximum accepted Warp residual | Maximum accepted CuPy residual |
|---|---:|---:|---:|
| Symmetric | 100 | 3.617e-10 | 2.436e-11 |
| Symmetric | 1,000 | 3.960e-10 | 1.148e-11 |
| General | 100 | 1.625e-10 | — |
| General | 1,000 | 1.655e-10 | — |

Acceptance requires maximum absolute residual below `1e-8`, all requested pairs, and zero solver status when exposed by the API. Benchmark worker timeouts: `benchmark_gen_33`. Completed backend failures/errors: none.

The general solver has a substantial performance gap on this suite. These measurements do not establish a bottleneck; profiling its projected complex eigensolve is a next step.


GPU timings are medians of ten warm, synchronized wall-clock solves. Setup, conversion, JIT, graph capture, and validation are outside the GPU timing; their costs are recorded separately. Original native CPU timings include solver construction and extraction. A failed or timed-out result is never used as an accepted performance sample. The original general ARPACK benchmark does not calculate its residual (`-1` in the raw output), so its timing is not an independently accuracy-validated result here.

Spectra test workers initially allow 30 seconds, with timed-out cases retried for 120 seconds. Benchmark workers allow 180 seconds for preparation and all ten timed solves. These are whole-worker limits, not bounds on a single solve. Timeouts and unsupported cases prevent a claim of full upstream conformance.

[Machine-readable summary](../../results/upstream_summary.json) · [GPU benchmark samples](../../results/upstream_benchmarks.json)
