# Hermitian shift-invert accuracy, 2026-10-17

All **20 Spectra Hermitian fixtures with n ≤ 100** now pass original residuals,
requested spectrum, complex orthogonality, invariant-subspace checks and strict
convergence counts. This is a correctness-only follow-up to #9 and #17, with
no performance measurements or new Anvil timings. The ten 1,000-variable
Hermitian fixtures were not rerun; #9 remains open.

## Cause and changes

After the earlier Krylov fixes, the small-family baseline had 18 passes and two
failures: sparse smallest-magnitude cases `HermEigs_17` and `HermEigs_22`. Their
realified matrices and factor inputs agree exactly with independent assembly.
The unmatched general cuDSS factors nevertheless leave large solve errors after
the existing one-step FP64 residual correction. Matching/scaling during setup,
while retaining that correction, resolves those errors.

[Eight controls](../../results/hermitian_inverse_controls.json) separate factor
matching from the outer iteration. Both matching settings use one correction
and the same k, ncv and 30-cycle budget:

| n | Outer iteration | Matching | Max original eigenpair residual | Converged |
|---:|---|---|---:|---:|
| 10 | General Arnoldi | Off | 5.07e-8 | 0/3 |
| 10 | Hermitian Krylov | Off | 2.91e-8 | 0/3 |
| 10 | General Arnoldi | On | 3.30e-14 | 3/3 |
| 10 | Hermitian Krylov | On | 8.94e-16 | 3/3 |
| 100 | General Arnoldi | Off | 3.75e-6 | 0/10 |
| 100 | Hermitian Krylov | Off | 2.64e-1 | 0/10 |
| 100 | General Arnoldi | On | 5.24e-16 | 10/10 |
| 100 | Hermitian Krylov | On | 5.26e-16 | 10/10 |

Matching alone fixes the observed numerical failures with general Arnoldi too.
The new Hermitian path is not the principal cause of that improvement. The
controls also show why an inaccurate inverse must not be treated as an exact
symmetric operator. Unmatched errors vary between runs; the failed controls are
retained rather than presented as successful solves.

`ComplexShiftInverse` now enables general cuDSS matching/scaling and retains one
Warp residual-correction step by default. This applies to both Hermitian and
nonsymmetric complex shift-invert. The GMRES backend is unchanged. Matching is
setup work; its cost has not been benchmarked in this follow-up.

`ShiftInvertEigensolver(..., hermitian=True)` adds an explicit Hermitian path:

- Accepts `HermitianCSR` or `ComplexCSR`, with the caller asserting Hermitian
  symmetry. Requires a real shift. Default `hermitian=False` retains general
  Arnoldi and support for complex shifts.
- Applies the realified inverse with symmetric Krylov, then removes duplicate
  complex directions. Numerical computation remains Warp except cuDSS solves.
- Supports both cuDSS and pure-Warp GMRES, and replayable `initial_vector` data
  of shape `(n,)` with dtype `wp.vec2d`. Invalid-start diagnostics are exposed as
  `status`; outputs are invalid when it is nonzero.
- `ncv` counts complex dimensions here and is doubled internally. This differs
  from the direct `HermitianEigensolver`, whose ncv counts real dimensions.
- Returns `wp.vec2d` eigenvalues with zero imaginary parts, retaining the general
  shift-invert API's storage. Physical residuals and `converged` use the original
  complex equation. The shift and factorized matrix stay fixed across replay;
  changed matrices require new factor setup.

```python
solver = ShiftInvertEigensolver(
    A, k=10, sigma=0.25, ncv=24, backend="cudss",
    hermitian=True, initial_vector=start, tol=1e-10,
)
graph = solver.capture(30)
wp.capture_launch(graph)
# Check solver.status, solver.converged and solver.residuals after completion.
# Update start in place and reuse graph; release graph before closing solver.
```

## Validation and limits

The small-family run preserves the exported k/ncv, selectors and iteration
budgets. Original absolute residual bounds remain 1e-9; independent spectrum,
orthogonality and subspace gates also use 1e-9. The runner now includes those
diagnostics in its success contract rather than accepting residuals alone.
The maximum over all 20 cases is **1.77e-13 residual**, **5.48e-13 eigenvalue
error**, **7.78e-16 orthogonality error**, and **6.10e-14 subspace error**.
[Before](../../results/hermitian_small_before.json),
[after](../../results/hermitian_small_after.json), and
[contract comparison](../../results/hermitian_small_comparison.json) preserve
the two fixes and show no lost passes within this small-family scope.

The CuPy adapter selects the Hermitian path for complex eigsh shift-invert.
**56 selected cases pass**, including all **48 previously unsupported SM
starting-vector cases** and eight shifted cases. The 16 non-nearest shifted
selector cases remain explicitly unsupported. [Raw report](../../results/hermitian_shift_cupy.json).
Together with the earlier 15 direct cases, all 63 historical complex-start skips
in #17 have now been exercised successfully. This does not implement general
nonsymmetric starts, complex generalized mass, or full CuPy API compatibility.

**95 focused local tests pass**, excluding two 1,000-variable real-pivoting cases
from that run. They cover shifted and direct complex solvers, general Arnoldi,
refinement, matching, and report contracts. The **14 shift tests also pass on
stock Warp 1.15**. The new 17 KiB fixture preserves the exact two failing sparse
matrices, with provenance in [the fixture notes](../../tests/fixtures/README.md).
Tests independently check inverse residuals for changed RHS and in-place batched
application, repeated nearest modes, changed starts, and conjugate orthogonality.
Lint, formatting and a clean wheel build pass; 18 core modules are packaged,
without tests, examples or benchmark code.

The supplied-start GMRES graph contains no host nodes or HtoD/DtoH transfers.
The cuDSS graph has no host callback or DtoH nodes, but retains vendor HtoD scalar
copies tracked in #29. A first audit assertion incorrectly matched the word
`HOST` inside an HtoD node's memory description; the test now distinguishes
actual host callback nodes. No vendor-copy compliance claim is made.

Environment: pinned Warp 1.18.0+cu12, NVIDIA L40, driver 570.158.01. Original
upstream totals and accepted timing tables are historical and unchanged.

## Reproduce without timing

The exporter can now select families and limit exported matrix sizes while
preserving upstream case IDs. Larger generators still execute on the CPU to
preserve case numbering, but their matrices are not exported or solved. Use an
isolated output directory to preserve full-suite exports.

```bash
python -m benchmarks.upstream.prepare_spectra --families HermEigs \
  --max-size 100 --out build/hermitian-followup/fixtures
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.upstream.run_spectra --family HermEigs \
  --fixtures build/hermitian-followup/fixtures --accuracy-only \
  --out build/hermitian-followup/current.json
python -m benchmarks.upstream.hermitian_inverse_audit \
  --out build/hermitian-followup/controls.json
WARPACK_UPSTREAM_REPORT=build/hermitian-followup/cupy.json \
  python -m pytest -q -p benchmarks.upstream.cupy_plugin \
  build/test_upstream_cupy_linalg.py \
  -k 'complex128 and (TestEigshSM or TestEigshSigma)'
python -m pytest -q tests/test_shift.py tests/test_hermitian.py \
  tests/test_arnoldi.py tests/test_refinement.py tests/test_upstream_contracts.py \
  tests/test_pivoting.py -k 'not SymEigs_27'
```

Accuracy-only mode captures, replays twice and checks the final result; it emits
no setup or solve timings. Local replay regressions check each replay separately.
The full-size #9 cases and nonsymmetric #12 cases still need their own reruns.
