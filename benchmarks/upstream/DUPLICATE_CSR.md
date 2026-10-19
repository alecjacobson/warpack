# Duplicate sparse coefficients on replay

Follow-up to [#78](https://github.com/alecjacobson/warpack/issues/78).
Compact scalar CSR can store several entries for one logical coefficient. Their
sum defines the matrix, including cancellation to zero. Source rows may be
unsorted; unused storage beyond the final logical offset is not matrix data.

The old shared transpose refresh mapped duplicates to the same destination and
assigned competing values, leaving other duplicate transpose slots unchanged.
For offsets `[0,2,3]`, columns `[1,1,0]` and values `[1,3,2]`, the logical matrix
is `[[0,4],[2,0]]`. A CPU reproduction produced a transpose coefficient of 6
instead of 4; after changing values to `[2,6,4]`, it produced 9 instead of 8.
The [failed evidence](../../results/performance_followup/duplicate-transpose-before.json)
is retained. GPU execution additionally permits competing-write races.

The shared refresh now clears the cached transpose values and atomically sums
source entries into the first destination slot for each coordinate. Any other
duplicate slots remain zero. The mapping still depends only on fixed topology;
all numeric refresh work is Warp and runs inside capture.

Constrained residual normalization also needs logical coefficients: summing the
squares of individually stored duplicates gives the wrong row norm. A setup-time
source ownership map identifies one entry per logical coefficient. The existing
parallel row reduction then sums squares of the refreshed, coalesced values
exactly once. Only constrained solves allocate this additional map.

No caller matrix is changed. Fixed topology, logical dimensions and array
addresses remain required for replay. A changed constrained pencil still needs
an updated caller-supplied KKT inverse. Floating-point duplicate summation has
ordinary rounding/order dependence; this is not an exact arithmetic guarantee.

## Regression scope

Tests cover sorted and unsorted duplicate coordinates, tall and wide matrices,
explicit zeros, cancellation, padded NaN capacity, scaling, entry changes and
restoration. CPU tests check transpose products; CUDA tests additionally check
cooperative constraint norms and graph transfer audits. Full captured SVD tests
exercise both augmented and normal formulations against independent dense
singular values, complete requested spaces and both original equations.
Constrained tests exercise coupled SPD mass, repeated modes, non-invariant
constraints and interior shifts using a small exact test-fixture inverse; they
check equilibrium, feasibility, mass orthogonality and norm diagnostics against
independently coalesced matrices. These are library regressions, not claims about
cuDSS accepting arbitrary raw duplicate storage itself.

Focused checks pass **87 tests in 13.18 s** on pinned Warp 1.18 and **87 in
16.64 s** on stock Warp 1.15. Four additional CPU-only transpose checks pass.
After integration with the small complete-basis SVD optimization, the
[combined local suite](../../results/performance_followup/duplicate-validation.json)
passes **770 tests in 350.82 s**. Lint and formatting pass, and the
[isolated wheel check](../../results/performance_followup/duplicate-refresh-wheel.json)
imports all 23 core modules without benchmark dependencies.

## Canonical-input overhead controls

Sequential clean-occupancy L40 runs, three warm trials each, with the preceding
accepted control using the same fixture and settings. All current trials pass
original physical acceptance. POD records full paired counts on every trial;
the skinning driver records a full final device count. Setup/capture and
validation are recorded separately. Clocks are unlocked, so small differences
are not speedup claims.

| Fixture | Before ms | Repaired ms |
|---|---:|---:|
| POD 100,000×48, k=20 | 8.567 | 8.543 |
| POD 100,000×128, k=20 | 75.391 | 75.660 |
| Skinning 2,025 DOFs, k=20 | 53.323 | 53.348 |
| Skinning 45,625 DOFs, k=20 | 287.779 | 287.440 |

[48-column POD](../../results/performance_followup/duplicate-refresh-pod-48.json),
[128-column POD](../../results/performance_followup/duplicate-refresh-pod-128.json),
[small skinning](../../results/performance_followup/duplicate-refresh-skinning-8.json),
[large skinning](../../results/performance_followup/duplicate-refresh-skinning-24.json).
Sibling occupancy files record the monitors. POD predecessors are the
`small-normal-api-clustered-48/128.json` records; skinning predecessors are
`constraint-refresh-side8/24-fixed.json`. Matrix hashes match, and cached
skinning host references were reused. No large CPU factorization was run.

The refresh adds one device memset. Pure-Warp POD graphs still contain no host
callbacks or host/device transfers. Skinning retains cuDSS's 420 captured HtoD
metadata copies, tracked separately in #29; there are no HOST or DtoH nodes.
Only constrained solves allocate an additional ownership array (one int32 per
cached transpose capacity entry), at setup. No replay allocation or host
convergence decision is added.

Reproduce the profiles with the normal POD comparison's `--case clustered
--size 100000 --modes 20 --snapshots 48 --ncv 48 --steps 100 --repeats 3
--compare ''` (or both widths 128), and the skinning comparison's `--side 8`
(or 24), `--steps 2 --repeats 3 --methods warp-cudss`. The retained JSON records
include exact arguments and source hashes.
