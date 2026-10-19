# Reducing experimental block-iteration work

Latest: [FP32 working-matrix refresh](BATCHED_REFRESH.md) fixes changed-source replay and records a fresh accepted 3.7232 s Anvil profile.

2026-10-19 follow-up to [#67](https://github.com/alecjacobson/warpack/issues/67).
The numerical code remains experimental and outside the installed package.
Production Krylov defaults and the main production timing table are unchanged.

## Changes

Two opt-in settings reduce FP64 work without removing stability checks:

- `--symmetric-products` computes one triangle of Gram matrices known to be
  symmetric (`X M X^T` and `S K S^T`). Rectangular projection products retain
  their full evaluation. Diagonal tiles retain both products for symmetrization.
- `--fused-updates` combines projection rotation and subtraction, avoiding
  intermediate writes/copies. The history-direction rotation omits the known
  zero coefficient block associated with current eigenvectors.

Both metric orthogonalization passes, rank-revealing whitening, constraint
projection and fresh original-equation residual products remain. The small
eigensolves and all outer arithmetic are still FP64 Warp; the inner solver is
upstream Warp CG with upstream FSAI factors. There are no host numerical
decisions or CPU small eigensolves.

An instrumented Anvil single-step screen drops from the prior 119.85 ms to
107.30 ms. This is diagnostic evidence, not a whole-solve speed ratio. The
profiling categories overlap, and fused projections are no longer included in
the standalone `rotate` category. The complete solve is measured separately.

## Anvil and Hodge

NVIDIA L40, three restored-start warm trials, no sampled competing compute
process. All trials pass the unchanged independent original-equation,
eigenvalue, subspace/mode-shape and orthogonality gates. Warp additionally
returns zero status and all requested convergence counts. Setup, graph capture
and validation are outside replay timings.

| Configuration | Warp block median (s) | CuPy LOBPCG + same supplied Warp CG (s) |
|---|---:|---:|
| Anvil, optimized outer products, FSAI32/FP32 CG15 | **3.9028** | **3.2999** |
| Hodge, optimized outer products, FSAI32/FP32 CG10; CuPy tol 5e-11 | **.07654** | .11856 |

Anvil uses 283,857 DOFs, 20 elastic modes, Warp width 24 / CuPy width 20,
shift 10000, Gram chunk 4096, normalized Warp tolerance 1e-8 and absolute CuPy
tolerance 1e-5. The maximum outer budget is 200 for both. Warp takes 30–31
steps; CuPy histories contain 67–92 entries. **Every CuPy Anvil trial also
meets its native absolute tolerance**, unlike the limited-budget profile in
[MATCHED_CG.md](MATCHED_CG.md). Worst Warp returned-pair residual is 7.148e-9;
worst original residual including the separately supplied rigid block is
1.430e-8. Scheduled Warp inner vector iterations are 10,800–11,160, versus
9,525–10,035 for CuPy. These include masked RHS iteration slots.

Warp setup is 1.1168 s including the preconditioner and solver workspaces,
with .5977 s capture; these are single observations with cached compilation,
not repeated cold-start measurements. The graph has two conditional nodes,
339 kernel nodes and 20 DtoD copies, with no HOST/HtoD/DtoH nodes in its DOT
audit. The earlier graph had 361 kernel nodes and 40 DtoD copies.

The previous accepted Anvil profile took 4.3531 s with ten inner CG steps and
the original products. The new profile reduces that recorded time by about
10%, but also changes the inner budget; it does not isolate the kernels' whole-
solve contribution. Accurate single-trial controls retain the original
products: CG15 takes 4.2527 s / 30 steps and CG20 takes 4.5315 s / 28 steps.
Fewer outer steps do not necessarily mean a faster solve. The optimized CG10
screen takes 4.1214 s / 38 steps. All screens are retained; none is substituted
for a three-trial median.

Hodge has 39,366 DOFs, four harmonic modes plus two positive guards, width 6,
shift .001 and Gram chunk 256. Warp takes 24 steps. CuPy with absolute tolerance
1e-10 stops after 22 history entries but returns maximum absolute residual
1.06436e-10 in each trial. All shared physical gates pass, but **its native
tolerance does not**. [#69](https://github.com/alecjacobson/warpack/issues/69)
tracks this soft-locking/reference limitation separately. A native 128-by-128
positive diagonal reproducer, with no Warp or preconditioner, was filed as
[CuPy #10402](https://github.com/cupy/cupy/issues/10402). The 113.59 ms figure
must not be described as a native-tolerance success. A separately recorded
profile tightens CuPy to 5e-11: all three trials meet that native tolerance
(maximum 4.59435e-11) at a 118.56 ms median. The table uses this fully accepted
profile and the paired 76.54 ms Warp replay. Both profiles remain available.

## Related consistent-mass FEM

The 10,890-DOF beam uses the same 2% regional stiffness change and independent
reference construction as the [matched-CG report](MATCHED_CG.md). All methods
use the same changed-pencil FSAI32/FP32 CG10 preconditioner, width 24 for both outer methods,
and identical starting columns within each paired comparison. The nominal tolerance is
1e-9, with each solver's documented stopping normalization.

| Start on the changed matrix | Warp block (s) | CuPy + supplied Warp CG (s) |
|---|---:|---:|
| Random | .32798 | .40964 |
| Previous 20 + four random | .25719 | .31976 |
| Previous full 24 | .17789 | .27050 |

All three trials of every row pass the shared physical gates, all 20 Warp
counts, and CuPy's native returned-vector tolerance. The source Warp solve
costs .34245 s. Reference eigenvectors never enter a start. The full-block
reuse benefit is preserved, but this smaller fixture shows little Warp timing
change from the previous product implementation. Computed starts can differ
in roundoff and rotations between implementations; cross-report CuPy timing
differences are not attributed to Warp outer kernels.

## Validation and records

The 44 focused tests pass on both the pinned Warp 1.18 environment (3.85 s)
and stock Warp 1.15 (18.27 s including compilation). They include independent
scaled/changed-input products with partial tiles, coupled mass, indefinite
stiffness, repeated modes, rigid constraints, dependent search spaces, continued
iteration after convergence, and captured changes to the full pencil/start.
The immediately preceding full suite passed 522 tests; that run predates these
new experimental kernels and is not presented as their full-suite validation.

Raw reports are in [`results/performance_followup/`](../../results/performance_followup/):
`anvil-outer-products-cg15.json`, `hodge-outer-products-cg10.json`,
`hodge-outer-products-cg10-tight-cupy.json`,
`related-consistent-outer-products.json`, the `anvil-active-cg15-screen` /
`anvil-active-cg20-screen` controls, and `anvil-symmetric-fused-cg10-screen`.
Each timing run has a separate occupancy record. Sampling is not a GPU
reservation and can miss brief jobs. Reports contain per-trial accuracy,
native-tolerance flags, setup/capture scopes, versions and source hashes.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu --out build/outer.occupancy.json --log build/outer.log -- \
  python -m benchmarks.experimental.compare_block --case anvil --width 24 \
  --cupy --cupy-width 20 --steps 200 --shift 10000 --tol 1e-8 --fsai-width 32 \
  --inner-cg 15 --batched-cg --inner-fp32 --inner-tol .01 --gram-chunk 4096 \
  --batched-products --symmetric-products --fused-updates --repeats 3 \
  --out build/anvil-outer-products.json
```

The outer-method gap remains open in #67, the larger iterative-versus-cuDSS
gap in #15, and supported block-API promotion in #68. The evaluation milestone
#59 is complete; that closure does not promote this prototype into the API.

## Paired Ritz-rotation follow-up

The opt-in `--paired-rotation` computes the new current block and history block
in one kernel. If the search basis is `[X,U]` with coefficient blocks `[Cx,Cu]`,
the two outputs are `X_new = Cx X + Cu U` and `P_new = Cu U`. Reusing `Cu U`
reduces arithmetic; bounded input views preserve partial inner tiles. FP64
precision, orthogonalization and fresh residual evaluation remain unchanged.

A three-trial Anvil CG15 run, otherwise matching the profile above, takes
**3.8442 s** with Warp versus **3.2612 s** with the same-preconditioner CuPy
reference. All physical gates, Warp counts and CuPy native tolerances pass.
Warp takes 29–30 steps, with worst returned-pair residual 9.8595e-9. The roughly
1.5% difference from the earlier 3.9028 s median is modest and includes ordinary
run-to-run convergence variation; it does not establish a robust 1.5% speedup.
The setting remains opt-in. Three additional product tests cover partial
inner tiles, changed inputs and separate independent references for both
outputs. All 47 focused tests pass in the pinned environment (12.04 s) and stock Warp
1.15 (11.50 s).
Raw record: `anvil-paired-rotation-cg15.json` and its occupancy record.

## Sparse lane-layout screen

A separate microbenchmark varies the number of lanes cooperating on each row
of the FP32 shifted stiffness and upstream FSAI factor products. It performs
one `A X` product and one `G.T (G X)` application on the actual Anvil matrix,
with width 24. All candidates pass independent FP64 product checks using the
actual FP32 coefficients (relative Frobenius errors below 1.11e-7).

| Lanes per row | RHSs per warp | Median of five graph replays (ms) |
|---:|---:|---:|
| 8 (current layout) | 4 | 2.9410 |
| 4 | 8 | 2.9036 |
| 2 | 16 | 3.1581 |
| 1 | 32 | 5.0804 |
| 16 | 2 | 4.4678 |
| 32 | 1 | 8.6125 |

The four-lane difference is small and isolated from CG/eigensolver work; it
is not a demonstrated end-to-end gain. The current layout remains unchanged.
The diagnostic does not replace upstream CG or construct different FSAI
factors. Twenty-four additional tests cover both product precisions, CSR/3x3
BSR, empty rows, partial RHS groups and changed/scaled captured input. All
31 sparse-product tests pass (7.23 s). Raw data and occupancy are in
`anvil-sparse-layout.json` and `anvil-sparse-layout.occupancy.json`.

```bash
python -m benchmarks.experimental.tune_sparse_layout --out build/sparse-layout.json
```

After these follow-ups, the complete local suite passes **570 tests in 256.74 s**,
including the symmetric/fused/paired kernels and sparse-layout tests.
[Validation provenance](../../results/performance_followup/sparse-layout-validation.json)
records source hashes and checks; lint/format also pass.
