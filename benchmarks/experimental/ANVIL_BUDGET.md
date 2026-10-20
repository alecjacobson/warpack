# Anvil block width and inner-CG budget

A subsequent [device search-width experiment](ACTIVE_SEARCH.md) records an
accepted 2.270s profile; the budget-only results below are retained.

Increasing the truncated-CG preconditioner budget from 15 to 25 steps improves
the accepted Anvil block solve from **2.545 to 2.385 seconds** (about 6.3%).
The matrix, shift 100, FSAI32, FP32 inner arithmetic, normalized outer tolerance,
random start policy and 20 requested modes are unchanged. This is an explicit
Anvil setting, not a new default or a numerical solver change.

The earlier [shift follow-up](ANVIL_SHIFT.md) explains the formulation and
previous controls. Here all profiles use the same L40, three synchronized warm
trials, and full original residual/spectrum/mode-shape/orthogonality, Warp count
and status, and fresh native CuPy stopping checks. Reference vectors never
initialize the solve. No Anvil CPU factorization was rerun.

| Warp width | FSAI width | CG limit | Warp median, s | Warp outer steps | CuPy median, s | Aggregate |
|---:|---:|---:|---:|---:|---:|---|
| 24 | 32 | 15 | 2.54485 | 22–23 | 2.71816 | Previous accepted baseline |
| 20 | 32 | 15 | 2.87791 | 35–37 | 2.83820 | Accepted, slower |
| 24 | 32 | 25 | **2.38536** | 17 | **2.45376** | Accepted |
| 24 | 64 | 15 | 2.49325 | 20 | 2.57013 | Rejected comparison |
| 24 | 32 | 35 | 2.64549 | 16 | 2.70737 | Rejected comparison |

CuPy uses width 20 in all rows. The last two aggregate comparisons fail because
some CuPy returned pairs miss the shared normalized physical residual bound of
1e-8, despite passing their native fresh absolute tolerance 1e-5. Their maximum
returned normalized residuals are 1.405e-8 and 2.574e-8. This is a distinction
between stopping formulas, not evidence that CuPy misreports its native
criterion. Warp individually passes those profiles, but neither rejected
aggregate is an accepted cross-backend timing baseline. All raw trials remain.

For the accepted CG25 profile, Warp samples are 2.38536/2.37995/2.39283s.
All trials take 17 outer steps and schedule 8,000 inner vector iterations,
versus 6,840–6,960 with CG15. More inner work reduces expensive outer work and
is faster overall. The denser factor and CG35 do not improve total solve cost.
The denser factor also costs 4.812s to construct versus 1.061s for FSAI32.

CG25's maximum original elastic residual is **7.9751e-9**, and its maximum
returned-pair normalized residual is **3.9876e-9**. It retains all 20 modes with
zero status. CuPy's fresh full-block absolute bound passes every trial.
These tolerances differ from the tighter production shift-invert results.

Total factor/workspace setup is **1.2017s**, capture **.6373s**, and setup/capture
device free-memory decrease **4.308GB**, not peak allocation. The graph has
19 conditional nodes, 50 DtoD copies and no HOST/HtoD/DtoH nodes. Post-profile
checks pass. Device-selected batched CG and cached operator images remain
benchmark adapters around the public block solver. The preceding accepted
cuDSS control is .4656s, leaving a roughly 5.1x gap; issue #15 stays open.

## Evidence and reproduction

[Width20](../../results/performance_followup/anvil-shift100-width20.json),
[CG25](../../results/performance_followup/anvil-shift100-cg25.json),
[rejected FSAI64 comparison](../../results/performance_followup/anvil-shift100-fsai64.json),
[rejected CG35 comparison](../../results/performance_followup/anvil-shift100-cg35.json).
Sibling `.occupancy.json` records show sequential execution with no sampled
competing processes. Accepted commands exit 0; rejected aggregates exit 1.
The benchmark's `timing_eligible` occupancy flag alone is not numerical acceptance.
[Validation and input hashes](../../results/performance_followup/anvil-budget-validation.json).

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.experimental.compare_block --case anvil --width 24 \
  --steps 200 --shift 100 --fsai-width 32 --repeats 3 --tol 1e-8 \
  --cupy --cupy-width 20 --gram-chunk 4096 --symmetric-products \
  --fused-updates --paired-rotation --reuse-x-image --batched-products \
  --inner-cg 25 --batched-cg --active-cg-bucket-size 4 --inner-fp32 \
  --inner-tol .01 --profile --out build/anvil-cg25.json
```
