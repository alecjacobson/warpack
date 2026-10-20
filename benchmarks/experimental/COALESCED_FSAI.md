# Adjacent right-hand sides for FSAI application

The subsequent [matrix-application report](COALESCED_OPERATOR.md) improves this
configuration to 1.83–1.84s. This report retains the factor-only comparison.

Changing the memory layout used to apply upstream Warp's FSAI factors reduces
Anvil's experimental block solve from **2.2915 to 1.8976 seconds**, about **17%**.
CuPy using the same captured upstream Warp CG preconditioner improves from
**2.4469 to 2.1686 seconds**. These are medians of three accepted warm trials on
an L40. The previous cuDSS control remains substantially faster at about .466s;
this change does not close issue #15.

## Implementation

The adapter in [coalesced_fsai.py](coalesced_fsai.py) applies the unchanged
upstream factors as `G.T @ (G @ x)`. Factor construction, sparsity, precision and
CG are still upstream Warp. Independent CG right-hand sides retain their own
coefficients and convergence tests. Only the factor application changes:

1. Transpose the CG vectors from `(rhs, dof)` to `(dof, rhs)`.
2. Apply both sparse factors with adjacent threads loading adjacent RHS values.
3. Transpose back and apply the requested alpha/beta output update.

Each warp handles eight RHS values and four sparse entries, reducing across the
latter. Two working arrays replace the two previous arrays without increasing
their element count. Padded shared tiles implement the transposes. All of these
operations are Warp kernels, including the transposes, and are captured. The
adapter supports in-place input/output and skips reading the old output when
beta is zero. Existing captured factor values may change in place; rebuilding
factor topology requires rebuilding the adapter/graph.

`BatchedCGInverse(..., fsai_layout="columns")` opts in. The default remains
`"rows"`. The option propagates through the preallocated active-width CG
adapters, including the adapter supplied to CuPy. This is benchmark code, not an
installed public API. Application-specific active search, cached operator images
and CG scheduling also remain experimental.

## Whole Anvil comparison

The comparison retains the 283,857-DOF mesh, 20 elastic modes and six analytic
rigid constraints, shift 100, upstream FSAI32 FP32 factors, FP32 CG limit 25 and
relative inner tolerance .01. Outer arithmetic is FP64. Warp uses width 24,
active CG/search buckets of four, and preserves the full history block. CuPy
uses width 20 and the same factors and CG settings.

| Layout | Warp median, s | CuPy median, s | Warp outer steps | Scheduled CG vector iterations |
|---|---:|---:|---:|---:|
| Previous rows, fresh control | 2.291474 | 2.446894 | 16–17 | 7,800–8,000 |
| Adjacent RHS columns | **1.897630** | **2.168628** | 17 | 8,000 |
| Adjacent RHS columns, replication | 1.935807 | 2.174956 | 17 | 8,000 |

The replication gives a 15.5% improvement against the fresh row-layout control;
the first run gives 17.2%. The profiles are sequential, not alternating
whole-solve measurements. All three occupancy records are clean.

Every timed result passes the existing spectrum, original residual, mode-shape
and orthogonality checks. Warp reports all 20 pairs and status zero. Its first
column-layout profile has maximum original residual **7.9651e-9** and maximum
returned-pair normalized residual **3.9826e-9**, below the unchanged 1e-8 bound.
CuPy also meets its native freshly recomputed full-block absolute 1e-5 bound;
the methods' stopping formulas differ. Setup/capture/warmup and CPU validation
are outside solve timing. Each process reassembles the same mesh, so bitwise
identical assembled matrices across processes are not asserted.

The first column profile uses **1.2172s** for factor/workspace setup and
**1.1523s** for capture including branch warmup. Device free-memory decrease
through setup/capture is **4.387GB**, not peak memory. Its graph contains 30
conditional nodes and 110 DtoD copies, with no HOST, HtoD or DtoH nodes. This
matches the earlier search adapter's transfer behavior; the improvement comes
from device work, not removal of a host synchronization.

## Component controls and tests

[compare_fsai_layout.py](compare_fsai_layout.py) reuses exactly the same Anvil
factors and random inputs for both implementations. It alternates execution
order and validates every timed output against independent SciPy FP64
accumulation. Timings include the transposes and output update. Width four is a
retained slower control; this is not a universal improvement for narrow batches.

The final component profile has 15 samples per layout/width, all accepted:

| RHS width | Previous rows, ms | Columns including transposes, ms |
|---|---:|---:|
| 4 | **0.4437** | 0.5052 |
| 8 | 0.8475 | **0.5137** |
| 16 | 1.6341 | **0.8742** |
| 24 | 2.4245 | **1.3894** |

Maximum relative output error against FP64 accumulation is below 8.1e-8.
The component graph contains four kernels and no copies or host nodes.

**128 related tests pass in 6.06s** on the pinned Warp environment, including
active-search regressions. This is focused validation; the preceding full-suite
record in the README remains separate. All **32 standalone factor tests pass
on stock Warp 1.15 in 1.35s**.

The tests cover FP32/FP64 factors and working arithmetic, odd and tile-crossing
widths, aliases, alpha/beta, changed inputs and factor values on replay, upstream
FSAI equivalence, zero/scaled RHSs, changing matrix values, active bucket
reactivation, and complete generalized eigenspace acceptance. DOT inspections
check that replay contains no host nodes or host/device copies.

## Reproduction

Use the pinned upstream Warp/FSAI environment from the README and the same
Anvil mesh/reference cache as the previous [active-search report](ACTIVE_SEARCH.md).
Run GPU benchmarks sequentially under `benchmarks.monitor_gpu`.

```bash
python -m benchmarks.experimental.compare_block --case anvil --width 24 \
  --steps 200 --shift 100 --fsai-width 32 --fsai-layout columns \
  --repeats 3 --tol 1e-8 --cupy --cupy-width 20 --gram-chunk 4096 \
  --symmetric-products --fused-updates --paired-rotation --reuse-x-image \
  --batched-products --inner-cg 25 --batched-cg --active-cg-bucket-size 4 \
  --inner-fp32 --inner-tol .01 --active-search-bucket-size 4 \
  --active-search-keep-history --out build/anvil-columns.json
python -m benchmarks.experimental.compare_fsai_layout --out build/fsai-layout.json
```

Use `--fsai-layout rows` for the unchanged control. The complete raw trials,
including occupancy, setup, stopping diagnostics and source hashes, are kept in
`results/performance_followup/coalesced-fsai-*`.
