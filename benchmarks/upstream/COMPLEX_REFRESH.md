# Complex GMRES coefficient replay repair

Issue [#73](https://github.com/alecjacobson/warpack/issues/73) exposed a stale
representation in `ComplexShiftInverse(..., backend="gmres")`. Construction
realified `A-sigma I` once. Later source-value updates reached the original
complex operator but not the matrix used by GMRES, so replay solved the old
linear system. This differs from cuDSS's explicit factor lifetime contract.

For a diagonal six-dimensional complex matrix, shift `.25+.1j`, and a fixed
realified all-ones RHS, doubling A while preserving its topology produces:

| Implementation / execution | Before update | After A → 2 A |
|---|---:|---:|
| Original, CPU eager | 1.397e-16 | 1.1292 |
| Repaired, CPU eager | 1.397e-16 | 1.469e-16 |
| Original, captured L40 replay | 4.934e-16 | 1.1292 |
| Repaired, same captured replay | 4.934e-16 | 3.331e-16 |

Entries are fresh relative residuals against the **current original** shifted
matrix. [Original CPU](../../results/performance_followup/complex-gmres-stale-cpu.json),
[repaired CPU](../../results/performance_followup/complex-gmres-refreshed-cpu.json),
[original GPU](../../results/performance_followup/complex-gmres-stale-gpu.json),
[repaired GPU](../../results/performance_followup/complex-gmres-refreshed-gpu.json).

## Implementation

The GMRES representation now preserves structural real/imaginary zeros at
setup and builds a device mapping from each realified source/shift contribution
to the compact real CSR. Before each multi-RHS apply, a Warp kernel refreshes
the values using the current complex source. It includes diagonal shift terms
and sums duplicate coordinates. No sparse topology is rebuilt on replay and
there is no numerical host readback, convergence decision or synchronization.
Changing source topology or addresses still requires new setup.

Unused trailing source capacity is zero-initialized in setup triplets and
excluded from the refresh map. Regression padding deliberately contains NaNs
and invalid column indices; it must never enter the active operator. Explicit
zeros in active entries can become nonzero without rebuilding the graph.

Both `apply` (realified rows) and `apply_complex` share the refresh. The same
updated matrix serves all RHSs in a call. cuDSS continues to prune initial
numerical zeros and keeps its factors tied to construction-time coefficients;
changing its source requires recreating the inverse and corresponding graphs.
This repair adds no cuDSS refactorization API and does not change #29's vendor
metadata-copy limitation.

The GMRES path retains up to all four realified structural components per
complex source entry plus shift positions, even when some values are zero.
That is an explicit storage/product tradeoff for changing values. This report
makes no speedup claim.

## Validation

Sixteen new cases cover real and complex shifts, both apply entry points,
multiple/zero/in-place RHSs, changing and restored coefficients, initially real
matrices becoming complex, duplicate unsorted source coordinates, explicit
stored zeros, and unused capacity. They independently validate the entire
realified matrix and the current original equation, preserve caller input
values, and audit GPU graphs for HOST callbacks, HtoD and DtoH transfers.

All **16 CPU cases pass in 82.45 s**. The focused pinned GPU suite—including
the new cases and all existing shift/Hermitian tests—passes **63 tests in
11.36 s**. The same **63 tests pass on stock Warp 1.15 in 14.08 s**.
[Validation and source hashes](../../results/performance_followup/complex-refresh-validation.json).
This is focused validation after the preceding full 664-test run,
not a new full or upstream conformance rerun.

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m pytest -q tests/test_complex_shift_refresh.py tests/test_shift.py tests/test_hermitian.py
```

## Unchanged-value overhead control

A 20,000-by-20,000 complex-storage banded matrix, initially real, exercises
the extra imaginary structural zeros. Three samples each average 20 captured
linear solves after warmup. The previous implementation takes **4.6274 ms**
per apply and the repaired implementation **4.6445 ms**, a roughly **0.4%**
difference, not a material regression or speedup claim. Every sample passes
the current original linear equation at relative residual **3.429e-16**.

The realified matrices have 159,996 versus 239,992 active stored entries; that
real storage/work tradeoff is not hidden. This is one well-conditioned linear
inverse-application control, not a general complex eigenproblem performance
claim. Both sequential runs have clean occupancy samples and graph audits.
[Old control](../../results/performance_followup/complex-gmres-control-old.json),
[repaired control](../../results/performance_followup/complex-gmres-control-fixed.json),
[driver](../experimental/complex_gmres_profile.py). Use the driver against
pre-fix commit `3bef1b3` for the old control; setup/capture and validation are
excluded from replay timing and recorded separately.
