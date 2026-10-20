# Anvil block width and CG budget follow-up

The existing width-24/FSAI32/CG25 configuration remains the best accepted
configuration in this sweep. Its fresh control takes **1.7784 s Warp** and
**2.1676 s CuPy**, with all 20 elastic modes accepted. None of the six alternate
settings improves the accepted comparison. No numerical code or default was
changed. Issue #15 remains open: the earlier cuDSS control is still much faster.

This follows [the sparse-operation layout optimization](COALESCED_OPERATOR.md).
The new control has 16 outer iterations in all three trials, versus 16–17 in
that report's 1.8292/1.8432 s profiles. The difference is not a new algorithmic
speedup. Floating-point assembly/reduction variation can affect termination,
and GPU clocks are unlocked.

## Same-device comparisons

Each row is a separate process on the L40, with three synchronized warm trials
per backend. The problem has 283,857 unknowns, 20 elastic modes and six analytic
rigid constraints. Shift is 100, inner tolerance .01, both sparse layouts use
adjacent RHS columns, and the outer method is FP64 with FP32 upstream Warp CG
and upstream FSAI factors. Active CG/search buckets have width four and retain
full history. CuPy uses width 20 and the same captured CG/factors within each
process; changing Warp's width also changes its random starting block.

| Warp width / FSAI row limit / CG limit | Warp median, s | CuPy median, s | Warp outer iterations | Scheduled CG vector iterations | Aggregate result |
|---|---:|---:|---:|---:|---|
| 24 / 32 / 25, control | **1.778412** | **2.167555** | 16 | 7,800 | Accepted |
| 28 / 32 / 25 | 2.194270 | 2.192301 | 14 | 8,500 | Accepted |
| 32 / 32 / 25 | 2.144009 | 2.178209 | 12 | 8,700 | Accepted |
| 24 / 16 / 25 | 1.963566 | 2.438138 | 19 | 9,400 | Rejected; diagnostic timings |
| 24 / 16 / 35 | 1.921642 | 2.260309 | 16 | 10,640 | Rejected; diagnostic timings |
| 24 / 32 / 20 | 1.965887 | 2.264150 | 19 | 7,520 | Accepted |
| 24 / 32 / 30 | 2.047537 | 2.426419 | 17 | 9,240 | Rejected; diagnostic timings |

Wider blocks reduce outer iterations, but increase the scheduled vector work
and cost of each block operation. CG20 reduces scheduled inner vector iterations
slightly, but requires three more outer iterations and takes longer. Fewer
iterations by one accounting measure do not establish a faster eigensolve.
Scheduled vector iterations include masked/padded work within active buckets;
they are not the sum of independent first-convergence iteration counts.

All **21 Warp outputs** pass status zero, all 20 physical counts, original
residuals, requested eigenvalues, orthogonality and elastic mode-shape checks.
Across all seven profiles, Warp's largest normalized returned-pair residual
is below 1e-8. The accepted control's setup is 1.2914 s and capture 1.1515 s;
device free-memory decrease through setup/capture is 4.454 GB, not peak memory.
Setup/capture/warmup and host validation are outside solve times.

CuPy meets its declared fresh full-block absolute 1e-5 stopping criterion in
all 21 trials. However, five outputs miss the additional shared normalized
returned-pair bound of 1e-8: one FSAI16/CG25, all three FSAI16/CG35, and one
FSAI32/CG30. The worst is 1.261e-8. Those three aggregate comparisons are
rejected and their monitor reports are not timing-eligible. They are retained
as diagnostics, not used to claim a speedup. This is a mismatch between two
different stopping/acceptance formulas, not evidence that CuPy violated its
own declared tolerance. No tolerance was relaxed or failed output discarded.

All seven occupancy records contain no competing GPU process. Every Warp graph
contains device work and DtoD copies only: no HOST/HtoD/DtoH nodes. The cached
Anvil reference was reused; no CPU sparse factorization was rerun. The prior
CPU and direct-GPU timings were not refreshed, so this is not a new all-backend
comparison. The latest full local suite remains the separately recorded
1,434-test run; this change publishes benchmark evidence and documentation only.

## Reproduction and records

Use the pinned Warp/FSAI environment from the main README, four BLAS/OMP threads,
and the existing mesh/reference cache. Run GPU jobs serially:

```bash
python -m benchmarks.monitor_gpu --out build/anvil-settings.occupancy.json \
  --log build/anvil-settings.log -- \
  python -m benchmarks.experimental.compare_block --case anvil --width 24 \
  --steps 200 --shift 100 --fsai-width 32 --fsai-layout columns \
  --matrix-layout columns --repeats 3 --tol 1e-8 --cupy --cupy-width 20 \
  --gram-chunk 4096 --symmetric-products --fused-updates --paired-rotation \
  --reuse-x-image --batched-products --inner-cg 25 --batched-cg \
  --active-cg-bucket-size 4 --inner-fp32 --inner-tol .01 \
  --active-search-bucket-size 4 --active-search-keep-history \
  --out build/anvil-settings.json
```

Change only width, FSAI row limit and CG limit as listed in the table.
[Summary, source commit, hashes and acceptance](../../results/performance_followup/anvil-tuning-summary.json)
links the seven full reports and occupancy records in the same directory.
Source commit is e7f901a; the benchmark code is unchanged from the preceding
layout implementation. Factors, matrix assembly and starts are regenerated in
each process; bitwise equality across those separate processes is not asserted.
