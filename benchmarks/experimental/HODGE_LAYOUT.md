# Hodge follow-up: active work and coalesced inner products

The Anvil optimizations also help the disconnected Hodge problem. On the L40,
39,366 unknowns and six requested modes, warm replay falls from a fresh
**76.60 ms** control to **62.77 ms**, with a fresh repeat at **65.08 ms**.
That is an observed 15–18% improvement. All five trials in each large profile
pass the unchanged original-equation, full-spectrum, orthogonality, four-mode
harmonic-subspace, positive-guard, status and full-count checks.

This remains the benchmark batched-CG route around the public block solver,
not a new production default. Upstream Warp constructs FSAI32 and runs CG;
all iteration and layout kernels execute in Warp. The historical production
cuDSS comparison was 23.65 ms, so #40 remains open. That number was not rerun
here and is not a matched regression baseline for these measurements.

## Fresh comparisons

Each entry is the median of five warm synchronized solves; setup, capture,
reference construction and validation are excluded. Identity mass is implicit,
width is six, shift magnitude .001, outer budget 100, Warp tol1e-11, inner FP32
CG tol.01. CuPy receives the same shifted matrix, upstream FSAI factor and
captured CG settings, with exact active RHS widths. Its absolute tol5e-11 is
checked on fresh returned vectors as well as its reported history.

| Configuration | Warp, ms | CuPy + supplied Warp CG, ms | Warp outer steps |
|---|---:|---:|---|
| Prior symmetric/fused/batched-products control, CG10 | 76.60 | 119.82 | 24 |
| Paired rotation, cached A X, active search and CG buckets of 2, row layouts | 71.71 | 119.54 | 23–24 |
| Same active-work configuration, both column layouts, CG10 | 62.77 | 112.56 | 23–24 |
| Column layouts, CG25 | 63.03 | 89.60 | 14–15 |
| Fresh column-layout CG10 repeat | 65.08 | 112.23 | 23–24 |

CG25 cuts outer work but does not improve total Warp time here. It does improve
CuPy time; the inner budget is therefore stated for each comparison. Reordering
floating-point reductions can change iteration counts, so the complete-solve
layout comparison is not a pure attribution to memory bandwidth. The first
CG10 column run contains three 23-step trials; its repeat contains one.

Worst Warp normalized original residual across the accepted large profiles is
9.77e-12; worst harmonic-subspace error is below 4.95e-10. All large CuPy trials
also meet their native stopping tolerance. No competitor process or monitoring
error was sampled during any run; sampling is not a GPU reservation.

The fresh column-layout replication has .1163 s setup and .3482 s capture.
The first column run includes additional JIT work in its .8418 s setup. Device
free-memory decrease through setup/capture is 1,933,574,144 bytes (about 1.80 GiB),
including allocator/runtime allocations; this is not peak memory or live solver
workspace. The graph has ten conditional nodes and 45 DtoD copies, with no
HOST callbacks, HtoD or DtoH nodes. Numerical convergence is tested independently
of this graph audit.

## Broader topology and mass checks

Without retuning the CG10 column configuration, side-six fixtures exercise
one torus (two harmonics), disconnected tori with coupled SPD mass (four), and
a weakly weighted annulus plus torus (three). All 15 Warp trials pass complete
physical/count/status checks. The torus and weak-positive CuPy comparisons pass
all ten native and physical checks.

The 216-unknown coupled-mass CuPy row fails its native stopping test on all five
trials: its largest returned absolute residual is **9.6951e-11**, exceeding the
requested **5e-11** despite a history length of 88 below the budget of 100.
The shared physical checks pass, including all four harmonic modes. This is
additional evidence for #69; it is not counted as an accepted comparison or
used for a speedup. Its raw timings remain diagnostics. Warp takes 94 steps on
this small case, which also illustrates why these settings are not universal
small-problem defaults.

## Evidence and reproduction

All eight reports and occupancy records are in
[`results/performance_followup/`](../../results/performance_followup/), named
`hodge-layout-control`, `hodge-modern-rows`, `hodge-modern-columns`,
`hodge-modern-columns-cg25`, `hodge-modern-columns-repeat`, `hodge-modern-torus`,
`hodge-modern-disconnected`, and `hodge-modern-weak`.
They retain 80 trials: 75 accepted, five rejected native CuPy stopping tests.
The disconnected whole-run occupancy wrapper reports `timing_eligible=false`
because numerical rejection gives a nonzero exit, not because competition was
observed. Do not override that combined profile rejection.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu \
  --out build/hodge.occupancy.json --log build/hodge.log -- \
  python -m benchmarks.experimental.compare_block \
  --case hodge_large_disconnected --side 81 --width 6 --steps 100 \
  --shift .001 --fsai-width 32 --fsai-layout columns --matrix-layout columns \
  --repeats 5 --tol 1e-11 --cupy --cupy-tol 5e-11 \
  --symmetric-products --fused-updates --batched-products --paired-rotation \
  --reuse-x-image --active-search-bucket-size 2 --active-search-keep-history \
  --active-cg-bucket-size 2 --inner-cg 10 --batched-cg --inner-fp32 \
  --inner-tol .01 --out build/hodge.json
```

The fresh control omits column layouts, paired rotation, cached A X and both
active bucket settings. The row-layout modern control omits only the column
layout flags. The CG25 profile changes only `--inner-cg`. Small cases change
`--case` and `--side 6`; no solver code or tolerance changes were made.
