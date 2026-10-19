# Upstream conformance runs

This harness runs the eigensolver/SVD portion of CuPy's tests and exports Spectra's numerical solver fixtures. It also runs Spectra's own native suite and benchmarks. It does not claim that unrelated CuPy array operations or Spectra's private factorization classes have Warpack equivalents.

See [recorded results and known failures](RESULTS.md). Production solver code was held fixed during this run.

Pinned sources:

- Spectra: `db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95`
- CuPy: `2316a02d70c7e39d6a7c035529df8358d794256e`
- CuPy performance repository (inventory only): `f014dacb4181d1b0a8b9c3923524f37509f7a59e`

CuPy's tests are copied unchanged into `build/`. The pytest plugin substitutes the solver entry points and expands dtype decorators so each precision is reported individually. Fixtures, NumPy/SciPy references, and assertions retain their upstream definitions. CuPy remains a **test boundary and reference dependency**, not a production eigensolver backend. Dense/CSC/COO and matrix-backed LinearOperator inputs are materialized into Warp CSR by the adapter outside the solve. This tests their numerical problems, not native support for those input containers. Unsupported precision, interfaces, and methods are explicit skips.

Automatic `SM` shifts and output ordering follow the pinned CuPy adapter contract. Real shift-invert supports LM/LA/SA selection through Warp Krylov iteration on the cuDSS inverse. Complex Hermitian shift-invert remains limited to nearest eigenvalues and now accepts complex starting vectors; nonsymmetric starting-vector support is separate.

Standard complex Hermitian LM/LA/SA problems accept device starting vectors; the adapter forwards `v0` to this API. See the [complex-start follow-up](HERMITIAN_STARTS.md) for the direct coverage, and the [Hermitian shift-invert follow-up](HERMITIAN_SHIFT.md) for the subsequently enabled SM cases.

For the native CuPy comparison, the pinned, unmodified `_eigen.py` is loaded over installed CuPy 14.2.0. This is a compatibility overlay, **not a complete build of pinned CuPy main**. Both its source hash and test source hash are recorded. The adapter maps ordinary SVD fixtures' `rng=0` to Warpack's fixed seed; it does not claim CuPy RNG API compatibility. RNG-specific tests remain unsupported.

Spectra's C++ generators and Catch sections execute directly. `prepare_spectra.py` replaces only execution bodies in a generated build-directory copy to export exact matrices, parameters, and explicit starts. Self-adjoint lower-triangle semantics are preserved. Warpack then checks the original residual tolerances, and SVD/vector or regression eigenvalue assertions where present. The unmodified upstream native suite runs separately with CTest. Private implementation tests (QR, Givens rotations, etc.) run in that native suite, not against unrelated Warpack internals. Some upstream cases explicitly permit nonconvergence; that flag is preserved in the report.

Davidson fixtures exercise Warpack's symmetric Krylov solver; Warpack does not gain a Davidson implementation through this adapter. Partial SVD uses Warpack's augmented symmetric operator and the fixture's original `ncv`, which need not have the same convergence behavior as Spectra's SVD formulation.

All inverse operations in the adapted Spectra/CuPy runs use cuDSS, including smallest-magnitude selections for real and complex general problems. Standard extremal problems use direct Warp iteration. Warpack runs captured graphs; references and assertions run outside capture. The runner records its outer-cycle budgets separately because algorithms' iteration counts are not interchangeable.

## Reproduce

Run from the repository root with the pinned Warp compatibility environment described in the main README. Install the test/benchmark/cuDSS extras, Eigen headers, CMake, a C++ compiler, and gfortran; the native benchmark also links ARPACK, LAPACK, and BLAS. The current script uses `libarpack.so.2` where the ARPACK development symlink is absent.

```bash
git clone https://github.com/yixuan/spectra.git build/upstream-spectra
git -C build/upstream-spectra checkout db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95
git clone --filter=blob:none --sparse https://github.com/cupy/cupy.git build/upstream-cupy
git -C build/upstream-cupy sparse-checkout set tests/cupyx_tests/scipy_tests/sparse_tests
git -C build/upstream-cupy checkout 2316a02d70c7e39d6a7c035529df8358d794256e
cmake -S build/upstream-spectra -B build/upstream-spectra-build \
  -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/upstream-spectra-build -j 4
ctest --test-dir build/upstream-spectra-build --output-on-failure -j 4 --timeout 120
python -m benchmarks.upstream.prepare_spectra
python -m benchmarks.upstream.prepare_benchmarks
cp build/upstream-cupy/tests/cupyx_tests/scipy_tests/sparse_tests/test_linalg.py \
  build/test_upstream_cupy_linalg.py
git -C build/upstream-cupy show HEAD:cupyx/scipy/sparse/linalg/_eigen.py \
  > build/cupy_upstream_eigen.py

export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
python -m pytest -q -p benchmarks.upstream.cupy_plugin build/test_upstream_cupy_linalg.py
python -m benchmarks.upstream.bounded spectra --timeout 30 --fresh
python -m benchmarks.upstream.bounded spectra --timeout 120 --retry-timeouts
WARPACK_UPSTREAM_BACKEND=cupy-main-overlay \
  WARPACK_UPSTREAM_REPORT=results/upstream_cupy_native.json \
  python -m pytest -q -p benchmarks.upstream.cupy_plugin build/test_upstream_cupy_linalg.py
python -m benchmarks.upstream.bounded benchmarks --timeout 180 --fresh
python -m benchmarks.upstream.summarize
```

For untimed focused work, see the [Hermitian shift-invert report](HERMITIAN_SHIFT.md), [general SM convergence profile](GENERAL_SM.md), and [Davidson/SVD accuracy profiles](DAVIDSON_SVD.md). The exporter accepts `--families`, `--max-size` and `--out`; the runner accepts `--fixtures`, `--selection`, `--accuracy-only` and explicit `--iterations`, `--ncv` and `--tol` overrides. Original assertion bounds and historical default settings are unchanged; overrides and effective ncv are recorded in the output. New reports distinguish `requested_ncv` from the allocated `effective_ncv`, `basis_vector_length` and `basis_scalar_type`. Direct Hermitian realification doubles the fixture width and vector length; shifted Hermitian wrappers do likewise. Clamped and default widths come from the constructed solver. `outer_iterations` records executed cycles after the solve. Historical reports used `effective_ncv` for the requested width and are retained unchanged; see [the full Hermitian follow-up](HERMITIAN_LARGE.md).

Run GPU jobs sequentially. The upstream Spectra benchmark has 40 dense datasets (100 and 1,000 unknowns; symmetric and nonsymmetric), each repeated ten times. `prepare_benchmarks.py` executes its original native benchmark and exports the exact matrices and starting vectors. `run_benchmarks.py` uses them for Warp and CuPy, with warmup and ten synchronized trials, and reports setup separately. All benchmark backends request `tol=1e-10`, although their convergence criteria differ. General Warp's test-only initialization adapter accepts the original start using Warp copies/kernels. These are small-problem diagnostics, not replacements for the large Anvil/dragon benchmarks.

These commands overwrite recorded reports. Omit `--fresh` to resume an interrupted bounded run; without it, already completed cases are retained.

No eigensolver benchmark was found in the pinned CuPy performance repository; its suites cover array operations, reductions, kernels, and convolution. CuPy is measured on the shared Spectra datasets instead. CuPy has no general sparse `eigs` entry point, so those benchmark rows are marked unsupported.

Each bounded worker's wall limit includes import, setup, capture, validation, and replays. Spectra cases that exceeded the initial 30-second limit receive a 120-second retry; benchmark workers allow 180 seconds for ten timed solves plus preparation. A timeout is reported separately from numerical failure and is not an accepted timing or a lower bound on a single solve. Completed rows are retained on resume, and retry records preserve earlier timeout metadata. The first Spectra pass ran in one process; its completed rows are retained, while the remaining cases use isolated workers.

The Spectra adapter requests an internal tolerance of `1e-12` and separately checks the original absolute residual assertions (usually `1e-9`, with family-specific exceptions). It also requires the requested number of converged pairs when the solver exposes that count. Consequently, some reported failures meet the upstream residual assertion but fail this stricter internal convergence check; the summary counts these separately. This is a numerical conformance adapter, not an identical algorithm, iteration budget, or convergence criterion.

The recorded run repeats the Hermitian family after correcting its real-to-complex convergence count, and the general `SM` cases after routing them through cuDSS. The commands above use those corrected adapters from the start.

The first CuPy report is retained separately to document adapter corrections (skip propagation, return ordering, validation precedence, and the upstream automatic shift for singular smallest-magnitude problems). Full-space eigsh adapters use one cycle when no explicit budget is supplied; accuracy assertions are unchanged.

## Numerical assertions and convergence counts

The Spectra adapter retains its strict `passed` policy: all original numerical
bounds must pass **and** the internal reported count must reach `k`. It does not
relax assertions for an upstream section that permits nonconvergence.
`checked_numerical_bounds_pass`, `reported_count_pass`, and
`upstream_allows_nonconvergence` record these three separate facts. A numerical
pass with a short count is not relabeled a full pass. Missing results,
unsupported cases, errors and timeouts are not numerical successes.

The absolute upstream bounds differ from the internal `1e-12` tolerance.
`reported_convergence_space` distinguishes the original equation from inverse
iteration, SVD's augmented problem, and Hermitian realification/extraction.
`original_evaluation_converged` records the separate physical counter where
available; it does not change the upstream absolute assertions. For SVD this
is the minimum of the left and right residual counts. New SVD reports also
record `left_converged` and `right_converged`; strict success requires both to
reach k when recorded. The comparison reports this as `physical_count_pass`;
older reports lack this additional gate. No independent complex
physical counter exists for the Hermitian realification adapter, so its field
is null. The complex shift-invert API already reports physical convergence.

A residual alone does not certify the requested spectrum. Original Spectra
Examples 1/1Shift/2/4 also compare the eigenvalues, and SVD checks singular
values and both sets of vectors. Generic upstream tests check residuals and
counts only; `requested_spectrum_checked` makes that limitation explicit.
Application benchmarks separately require their known requested spectrum.
The focused Hermitian and Davidson adapters add spectrum and subspace checks.
New SVD reports also check both paired equations, orthogonality and leading
subspaces; absolute-vector comparisons alone cannot detect inconsistent signs.

Compare complete reports, overlaying focused runs onto an older baseline:

```bash
python -m benchmarks.upstream.compare_reports \
  --baseline results/upstream_spectra_warp.json results/multiplicity_spectra.json \
  --current results/refinement_spectra.json --out results/refinement_contracts.json
```

This command fails for lost passes, missing cases, contradictory success
labels, or loss of any previously passing set of numerical assertions—even
when the overall case remains `failed` in both reports. Its CPU-only regression
tests include small residuals with wrong spectra, short counts with correct
numerics, allowed nonconvergence, invalid numbers, missing evidence, and SVD
vector failures. Raw per-case assertions and outcome labels remain available.
