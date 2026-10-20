# Direct upstream BSR preconditioners for block iteration

Fix for [#86](https://github.com/alecjacobson/warpack/issues/86). The public
`BlockLOBPCG` adapter previously required `preconditioner.dtype == wp.float64`.
Warp BSR operators expose scalar matrix dimensions but retain a matrix element
dtype such as `wp.mat33d`, so this rejected valid upstream block Jacobi and
FSAI before any solve. Their `scalar_type` is FP64 and their existing matvecs
already accept scalar vectors.

The adapter now checks scalar precision, retaining square/diagonal shape checks
and compatibility with scalar matvec metadata. It still delegates every
application to the unmodified upstream implementation. No factorization,
preconditioning kernels, new buffers, synchronization or numerical policy are
introduced. Updating coefficients still requires caller-managed preconditioner
reconstruction/refitting.

```python
P = linear.preconditioner(K_bsr, ptype="block_jacobi_sequential")
solver = BlockLOBPCG(SparseOperator(K), k=20, width=24,
                    mass=SparseOperator(M), preconditioner=P)
```

`linear.FSAI(K_bsr, ...)` is also accepted when available. Upstream block-Jacobi
and FSAI availability depends on the installed Warp version; these runs use the
pinned overlay documented in the main README. Stock Warp 1.15 compatibility
explicitly skips unavailable implementations.

## Validation

**40 targeted tests pass in 3.26s** on pinned Warp. Stock Warp 1.15 passes
**30 tests in 2.92s**, with 10 explicit feature skips (six new adapter cases and
four existing CG/block-Jacobi cases). Tests cover 2×2/3×3 BSR blocks, batches of
changed RHSs on captured replay against independent dense block solves, input
preservation, complete repeated eigenspaces, scalar metadata compatibility,
non-FP64/nonsquare rejection, existing public block/CG behavior and exact
diagonal-mass routes. Captured applications have no HOST/HtoD/DtoH nodes.
Lint and formatting pass. These focused tests follow the preceding 839-test
full local run; they are not a new full-suite run.

[Machine-readable validation](../../results/performance_followup/upstream-block-validation.json).
The previous setup errors remain preserved in the
[fused projected report](FUSED_PROJECTED.md#validation-and-retained-failures).
Both original one-step reproduction commands now reach a solve, report 0/4,
retain their numerical failure rows and exit 1 as required:
[block Jacobi](../../results/performance_followup/upstream-block-bj-short.json),
[FSAI](../../results/performance_followup/upstream-block-fsai-short.json).

## Application comparisons

Consistent-mass tetrahedral beams, three warm uncontended trials on L40,
600-iteration maximum, normalized Warp tolerance 1e-8 and native CuPy absolute
tolerance 1e-8. CuPy receives the same upstream Warp preconditioner; this is an
adapted reference, not native CuPy factor construction. Setup/capture and
validation are outside replay. Both backends use the listed starting width,
but Warp stops on the requested k and CuPy on its entire block.

| Upstream preconditioner | DOFs / k / width | Warp median ms | CuPy median ms |
|---|---|---:|---:|
| Block Jacobi | 2646 / 6 / 12 | 271.335 | 1861.080 |
| FSAI32 | 10890 / 20 / 24 | 639.997 | 1371.850 |

All requested original residuals, spectra, complete spaces and orthogonality
pass; Warp reports full counts/status 0, and CuPy meets the fresh full-block
absolute bound. Warp uses117/74 outer iterations; CuPy history lengths are
406/277. These are different stopping definitions and different workloads
between rows; do not interpret this as an isolated preconditioner comparison.
There is no speedup claim against the former adapter, which failed at setup.

[Block-Jacobi data](../../results/performance_followup/upstream-block-bj-fem600.json),
[FSAI data](../../results/performance_followup/upstream-block-fsai-fem600.json),
with companion occupancy records. Both Warp graphs contain only device work
and device-to-device copies. Source hashes identify the final adapter.

The initial 300-iteration block-Jacobi comparison is
[retained as rejected](../../results/performance_followup/upstream-block-bj-fem.json):
Warp's six requested modes passed, as did CuPy's first six, but CuPy's twelfth
padding vector had absolute residual ~1.26e-6, above 1e-8. Increasing only the
budget to 600 lets the native full block converge; tolerances are unchanged.
The rejected native time is not used as an accepted performance baseline.
