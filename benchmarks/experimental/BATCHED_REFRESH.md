# Refreshing the FP32 batched-CG working matrix

2026-10-19 fix for [#70](https://github.com/alecjacobson/warpack/issues/70),
found while preparing the supported block API in #68. This adapter remains
outside the installed package; production `CGInverse` is unaffected.

Previously, `BatchedCGInverse(..., working_dtype=wp.float32)` cast its FP64
matrix only during construction. A captured solve therefore kept using old
coefficients after the caller updated the source values. On a 45-by-45 SPD
tridiagonal matrix, changing `A` to `2A` left an independent relative residual
of **1.000000016**, despite zero reported unconverged inner RHSs against the
cached old matrix. The initial unchanged-matrix residual was **4.011e-7**.

The adapter now refreshes working values with a Warp cast kernel before every
application. The FP64 route already uses source storage directly. Shared
active-width states continue to share one working matrix and upstream FSAI
factor, and each application refreshes from current source values. The same
changed-matrix reproducer now has residual **4.011e-7** for both `A` and `2A`.

Sparse topology and captured buffer identities must remain fixed. Changing
the values in those buffers is supported; replacing arrays or changing the
sparsity structure requires reconstructing the relevant state and recapturing.
Updating a supplied preconditioner remains the caller's responsibility. An
older SPD FSAI factor can still precondition a new SPD matrix; the operator
matrix itself must not silently remain old. There is no host synchronization
or allocation in the refresh/replay path.

Existing constant-matrix results remain valid. The related-FEM drivers
constructed new inverses for their changed pencils, so those records are also
unaffected by the original limitation.

## Validation and timing

All **33 batched-CG tests pass in 2.73 s**. Four new cases cover captured
nonuniform matrix changes in CSR and 3x3 BSR, FP32 and FP64, in-place RHS use,
independent dense solutions/current-equation residuals, and zero host graph
nodes/transfers. The existing four active/padded adapter cases now also change
source coefficients between shrinking/growing RHS blocks, checking shared
storage and exact work accounting.

A fresh three-trial Anvil comparison, with the same accepted FSAI32/FP32 CG15,
symmetric/fused/paired outer updates and 200-step budget, records:

| Method | Median warm solve (s) | Requested-pair acceptance |
|---|---:|---|
| Warp block, width 24 | **3.7232** | All 20 counts and physical gates, every trial |
| CuPy LOBPCG + same supplied Warp CG, width 20 | **3.0469** | Physical and native absolute tolerance, every trial |

Warp takes 29–30 outer steps. Worst returned-pair residual is 7.5101e-9 and
worst original residual including the rigid block is 1.5021e-8. Setup is
1.1438 s and capture 2.0904 s, single observations including compilation where
encountered. The graph contains two conditional nodes, 338 kernel nodes and
20 DtoD copies; no HOST/HtoD/DtoH nodes appear in its DOT audit. The additional
kernel relative to the previous profile is the device refresh.

A separate single-step diagnostic measures that refresh at **.206 ms** within
a **126.25 ms** step, about .16% of the instrumented step. At 30 applications
the cast work is roughly 6 ms, before cache interactions. Profiling categories
overlap; this is not an additive timing decomposition. Both its before/after
physical checks pass. The lower whole-solve medians than the preceding run
are not evidence that the refresh makes eigensolving faster: outer convergence,
floating-point reductions and run-to-run timing vary. The remaining matched-
preconditioner gap is still tracked in #67.

Records under [`results/performance_followup/`](../../results/performance_followup/):
`batched-cg-stale-matrix.json`, `batched-cg-refreshed-matrix.json`,
`anvil-refreshed-cg15.json`, and `anvil-refreshed-cg15-profile.json`. Anvil runs
have separate occupancy records with no sampled competing compute process.
The full suite most recently passed 570 tests before this fix; the focused
33-test result above validates the subsequent changed-source behavior.

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m pytest -q tests/test_batched_cg_experimental.py
```

The Anvil command is the [outer-products command](OUTER_PRODUCTS.md#validation-and-records)
with `--paired-rotation`; use the current adapter and a fresh output path.
