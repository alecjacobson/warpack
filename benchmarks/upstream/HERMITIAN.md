# Hermitian breakdown recovery, 2026-10-16

This correctness-only follow-up resolves [#3](https://github.com/alecjacobson/warpack/issues/3).
It uses matrices with at most 100 complex unknowns, plus the existing small real
Krylov/Arnoldi regressions. No performance comparison or large FEM run was made;
the Anvil timings remain unchanged.

The former Hermitian implementation applied block orthogonal iteration to
`bound*I-A` for smallest-algebraic selection. For the identity, this operator is
exactly zero: the entire basis disappears and the returned eigenvalues are zero
instead of one. Direct iteration on a rank-five Gram matrix likewise cannot
preserve a block wide enough to return the five nonzero modes and a null mode.
The new identity regression was observed failing before the change.

`HermitianEigensolver` now uses the existing real `KrylovSchur` implementation on
the realified Hermitian operator. Its on-device breakdown recovery supplies new
independent chains; complex Gram–Schmidt continues to remove duplicate realified
directions. The shift-bound kernels and wrapper are removed. Numerical work stays
in Warp, without an inverse backend or new host convergence checks. Capture still
warms up and synchronizes before recording; replay has no added host synchronization.

The default real basis width is unchanged. Explicit `ncv` counts **real** vectors
and must now exceed `2*k` to leave Krylov expansion room; values above `2*n` are
clamped. Previously `ncv=2*k` was accepted by block iteration. `solve(iterations)`
now counts restart cycles and requires at least one. The real convergence count
is out of `2*k`, while `count` counts extracted complex vectors. Neither alone
certifies complex residuals or spectrum completeness.

Validation on the pinned Warp 1.18.0+cu12 environment and NVIDIA L40:

- **25 Hermitian tests pass**: random indefinite matrices; identity for k=3,6,12
  and LM/LA/SA; rank-five complex Gram matrices with and without a positive shift;
  changed values in one graph from identity to zero, unitarily rotated repeated
  spectra, negative identity and back; invalid basis widths. Checks include
  requested eigenvalues, original residuals, conjugate orthogonality, allowed
  repeated-eigenspace membership and device counts/status. Identity is checked
  in eager execution and replay. All four inspected changed-matrix graphs contain
  no host nodes or HtoD/DtoH transfers.
- **79 related local tests pass** across Hermitian, multiplicity, real symmetric,
  Arnoldi and complex Schur tests. This is a focused run, not a new full-suite run.
- The same **25 Hermitian tests pass on stock Warp 1.15**.
- **22 pinned CuPy tests pass; 18 skip** under their original LinearOperator
  fixture condition. All seven failures listed in #3 are included and pass the
  unchanged assertions. [Raw test report](../../results/hermitian_cupy_regression.json).
  The recorded pytest durations are test-run metadata, not benchmark results.
- Repository lint, changed-file formatting and a clean wheel build pass; the
  wheel contains 18 core Python modules and excludes examples and benchmarks.

[Spectra convergence issue #9](https://github.com/alecjacobson/warpack/issues/9)
remains open: its complete exported family has not been rerun in this follow-up.
The original upstream summary counts remain historical. Generalized complex mass,
complex starting-vector APIs and broader connection-Laplacian coverage are not
established by this fix. The subsequent [starting-vector follow-up](HERMITIAN_STARTS.md) adds the standard Hermitian starting-vector API and records its focused coverage.

Reproduce after preparing the pinned upstream test copy as described in
[README.md](README.md):

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_hermitian.py tests/test_multiplicity.py \
  tests/test_solver.py tests/test_arnoldi.py tests/test_schur.py
WARPACK_UPSTREAM_REPORT=/tmp/hermitian-cupy-regression.json \
  python -m pytest -q -p benchmarks.upstream.cupy_plugin \
  build/test_upstream_cupy_linalg.py \
  -k 'complex128 and (test_degenerate or test_low_rank_gram)'
```
