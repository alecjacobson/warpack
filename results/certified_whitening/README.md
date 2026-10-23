# Certified second metric whitening (#162)

This benchmark-only subclass preserves the first canonical rank-revealing
whitening and both orthogonalization passes. In the second whitening, it
computes a fresh symmetric search Gram matrix and checks a conservative
certificate entirely on device. Production dispatch is unchanged.

For the previously active rows, let δ bound the maximum absolute row sum of
`G_active − I`. Require δ ≤ 0.25 and `1 − δ > rank_tol * (1 + δ)`. Gershgorin
bounds then put every active eigenvalue in `[1 − δ, 1 + δ]`, above the existing
relative rank threshold. Every inactive Gram row/column must be exactly zero;
all entries and flags must be valid. Thus this branch changes no retained rank.
It applies Cholesky QR on the active subspace, filling inactive diagonal entries
with one only in separate factorization scratch. Inactive vectors remain zero.

An unsuccessful certificate uses the existing canonical eigensolve. A separate
factorization status also selects this fallback if Cholesky fails, preserving
the original Gram matrix and the solver's prior status. Three device graph
conditions select factorization, successful application and fallback. Eager
setup retains the canonical implementation. All numerical work remains Warp;
extra factor/matrix/work arrays are allocated for each search branch at setup.

Ten focused tests pass on pinned Warp; seven pass on stock Warp with three
upstream-FSAI-only skips. Coverage includes changing ranks, zero/partially inactive
search, nearly dependent directions, incorrect inactive flags, scaled rows,
stringent rank thresholds, nonfinite recovery, deliberately failed factorization,
complete clamped-consistent/free/heterogeneous FEM replays, and changing coupled
mass/normalized constraints with repeated clusters. The latter independently
checks complete requested subspaces and device-only graph replay.

## Complete prototype controls

Fresh independent workers compare identical inputs, references, starts,
source manifests and dependencies within each pair. Three measured Warp and
three measured CuPy solves per worker all pass unchanged physical/native gates:
36 accepted measurements, with clean occupancy. Times below are seconds.

| Problem | Second whitening | Warp setup + capture | Warp replay median | CuPy replay median |
| --- | --- | ---: | ---: | ---: |
| 100 modes, width104 | Canonical | 4.2733 | 3.2356 | 3.7524 |
| 100 modes, width104 | Certified | 4.2746 | 2.9738 | 3.8565 |
| Anvil, width24 | Canonical | 1.6169 | 1.6625 | 2.4451 |
| Anvil, width24 | Certified | 1.6199 | 1.5914 | 2.3743 |
| 200 modes, width204 | Canonical | 18.7805 | 11.0964 | 6.4161 |
| 200 modes, width204 | Certified | 19.0982 | 10.4679 | 5.7546 |

Observed Warp median reductions are 8.1%, 4.3% and 5.7%. Trajectories also
change with basis rounding: these are whole-method observations, not isolated
kernel speedups. Anvil's candidate takes16 steps versus16–17 for its control;
a separate narrow-size qualification is needed before changing the small default.
CuPy uses the same supplied Warp FSAI/CG within each worker; its variability is
visible and these results do not establish a universal ranking. Its independent
setup is not represented by the Warp setup column.

The final candidate replays use35/17/37 certified passes and zero fallbacks
for100/Anvil/200 respectively. Both whitening passes remain present; only the
second eigensolve is avoided. Setup and capture are also available separately
in the [summary](full/summary.json), alongside raw results/logs/occupancy.
[Focused tests](tests.log), [stock tests](stock-tests.log).

Promotion review will add a conservative floating-point margin to the row-sum
certificate and share scratch across mutually exclusive branches, then run fresh
regression controls. The diagnostic event-profiler integration failure discovered
here is tracked separately as #163; uninstrumented full solves pass. Production
was unchanged during these prototype measurements.
