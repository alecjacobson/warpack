# Device-selected block search spaces

A subsequent [factor-application layout optimization](COALESCED_FSAI.md) lowers
the same configuration to 1.90–1.94 seconds. This report retains the preceding
search-space experiment and controls.

An experimental adapter reduces the Anvil block solve to **2.270 seconds**,
versus a fresh **2.356-second** control: about **3.7%** faster. All 20 elastic
modes and the six analytic rigid modes pass the existing checks. This is a
modest improvement, not a closure of the roughly 4.9x gap to the preceding
accepted .4656-second cuDSS control.

The public `BlockLOBPCG` implementation is unchanged. The adapter in
[`active_search.py`](active_search.py) remains outside the installed package.
It uses private block-solver hooks and is tied to this checkout.

## What is reduced

The ordinary block method keeps a trial space `[X, R, P]`: current Ritz vectors,
preconditioned residuals and previous search directions, each of width b.
Inner CG already packs only residual rows whose freshly evaluated errors exceed
tolerance. The outer method previously still processed the full 3b trial space.

The new adapter keeps **every X vector**, selects active residual rows on device,
and dispatches to preallocated search-width buckets through CUDA conditional
nodes. No residual or rank is read back to the host. Rows can reactivate after
each fresh Ritz residual evaluation. Fixed storage, device reductions, two
orthogonalization/whitening passes and original-pencil checks are retained.

Two explicit policies are available:

- Compact R and the corresponding P rows: trial width `b + 2*q`.
- `--active-search-keep-history`: compact R but retain every P row, giving
  width `2*b + q`. This is the best tested Anvil setting.

Here q is a bucket width at least as large as the active residual count. All
buffers are allocated before capture, with large arrays shared through bounded
views. Each branch's small projected solver is distinct. Setup warms every
shape and synchronizes before capture, consistent with the existing graph-upload
workaround. Replay contains no host synchronization or allocation decision.

This is residual-based soft locking, not a certificate of the globally requested
spectrum. The public block start/exploration limitations still apply. Tests
include perturbed old modes after eigenvalue crossings; reference vectors never
initialize the Anvil comparison.

## Accepted Anvil comparisons

Same 283,857-DOF mesh-based problem, 20 elastic modes, width 24, shift 100, FSAI32,
FP32 inner CG and FP64 outer iteration. The driver reassembles the pencil in each
process; no bitwise identity of assembled coefficients across processes is
asserted. Input mesh/reference hashes are retained. Every profile has three
synchronized warm trials on the L40, no sampled competing GPU processes, and
unchanged original spectrum/residual/mode-shape/orthogonality, full Warp count
and status, and fresh native CuPy stopping gates.

| Search policy | CG limit | Warp median, s | Outer steps | Scheduled CG vector iterations | CuPy median, s |
|---|---:|---:|---:|---:|---:|
| Original, fresh control | 25 | 2.35636 | 16–17 | 7,800–8,000 | — |
| Compact R/P, bucket 4 | 15 | 2.51265 | 26 | 7,320 | 2.70431 |
| Compact R/P, bucket 4 | 25 | 2.29747 | 19 | 8,400 | 2.43499 |
| Compact R, keep all P, bucket 4 | 25 | **2.27028** | 16–17 | 7,800–8,000 | **2.45485** |
| Compact R, keep all P, full-width bucket 24 | 25 | 2.36348 | 16–17 | 7,800–8,000 | — |

The full-width control keeps the adapter and packing but removes dimension
reduction. Its timing is close to the original control. Compacting both R and P
saves work per step but needs more steps. Retaining P restores the outer-step
range while preserving part of the dimensional saving. These are measured
whole-solve results; no component profiler attribution is claimed. The existing
single-step profiling hook does not account for these branches, so the driver
rejects `--profile` together with active search.

The best profile's Warp samples are 2.27331/2.27028/2.20081s. The fastest trial
stops one step earlier and has a larger final error; it is not substituted for
the median. Maximum original elastic residual is **1.9377e-8**, and maximum
returned-pair normalized residual is **9.6887e-9**, below the declared 1e-8 bound.
All trials report 20/20 and status 0. CuPy uses width 20 and the same supplied
upstream Warp FSAI/CG preconditioner; its native fresh full-block absolute 1e-5
bound also passes every trial. The stopping formulas and final errors differ.

Total factor/workspace setup is **1.1989s**; capture, including warming new branch
shapes, is **2.7578s** in this run and depends on JIT cache state. Setup/capture
device free-memory decrease is **4.387GB**, not peak allocation. The graph has
30 conditional nodes, 110 DtoD copies and no HOST/HtoD/DtoH nodes. The additional
branch storage and capture cost are tradeoffs for repeated replay. The earlier
[CG-budget report](ANVIL_BUDGET.md) remains available, including unsuccessful
settings and its tighter final residuals.

## Validation and reproduction

**167 related tests pass** in 9.02 seconds on pinned Warp 1.18. All 27 new tests
also pass stock Warp 1.15 in 6.42 seconds. Coverage includes coupled metrics,
repeated eigenvalues, updated constraints/matrices/starts on the same graph,
both history policies and both update paths, odd widths, zero-search fixed and
adaptive iterations, reactivation, nonfinite failure/recovery, perturbed prior
modes after crossings, and graph host-transfer audits. No installed core source
was changed; the preceding 941-test full validation remains separate.

All five profiles and their occupancy records are retained:
[original control](../../results/performance_followup/anvil-active-search-baseline.json),
[compact R/P CG15](../../results/performance_followup/anvil-active-search-cg15.json),
[compact R/P CG25](../../results/performance_followup/anvil-active-search-cg25.json),
[compact R with full history](../../results/performance_followup/anvil-active-search-keep-history-cg25.json),
[full-width adapter control](../../results/performance_followup/anvil-active-search-full-width.json).
All commands exit zero and all numerical/occupancy gates pass.
[Validation, input hashes and source provenance](../../results/performance_followup/active-search-validation.json).

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.experimental.compare_block --case anvil --width 24 \
  --steps 200 --shift 100 --fsai-width 32 --repeats 3 --tol 1e-8 \
  --cupy --cupy-width 20 --gram-chunk 4096 --symmetric-products \
  --fused-updates --paired-rotation --reuse-x-image --batched-products \
  --inner-cg 25 --batched-cg --active-cg-bucket-size 4 --inner-fp32 \
  --inner-tol .01 --active-search-bucket-size 4 --active-search-keep-history \
  --out build/anvil-active-search.json
```
