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
| 100 modes, width 104 | Canonical | 4.2733 | 3.2356 | 3.7524 |
| 100 modes, width 104 | Certified | 4.2746 | 2.9738 | 3.8565 |
| Anvil, width 24 | Canonical | 1.6169 | 1.6625 | 2.4451 |
| Anvil, width 24 | Certified | 1.6199 | 1.5914 | 2.3743 |
| 200 modes, width 204 | Canonical | 18.7805 | 11.0964 | 6.4161 |
| 200 modes, width 204 | Certified | 19.0982 | 10.4679 | 5.7546 |

Observed Warp median reductions are 8.1%, 4.3% and 5.7%. Trajectories also
change with basis rounding: these are whole-method observations, not isolated
kernel speedups. Anvil's candidate takes 16 steps versus 16–17 for its control;
a separate narrow-size qualification is needed before changing the small default.
CuPy uses the same supplied Warp FSAI/CG within each worker; its variability is
visible and these results do not establish a universal ranking. Its independent
setup is not represented by the Warp setup column.

The final candidate replays use 35/17/37 certified passes and zero fallbacks
for 100/Anvil/200 respectively. Both whitening passes remain present; only the
second eigensolve is avoided. Setup and capture are also available separately
in the [summary](full/summary.json), alongside raw results/logs/occupancy.
[Focused tests](tests.log), [stock tests](stock-tests.log).

Promotion review will add a conservative floating-point margin to the row-sum
certificate and share scratch across mutually exclusive branches, then run fresh
regression controls. The diagnostic event-profiler integration failure discovered
here is tracked separately as #163; uninstrumented full solves pass. Production
was unchanged during these prototype measurements.

## Production integration

The production default is limited to Ada CUDA search metrics of size 96–512.
Anvil's smaller metrics retain canonical whitening, as do other architectures
and eager solves. The row-sum certificate includes a conservative floating-point
roundoff margin; borderline cases use the canonical fallback. The three square
scratch arrays are shared across mutually exclusive search branches, costing
`24 * largest_eligible_metric_size**2` bytes plus small flags/counters.

The [full pinned suite](production-full-suite.log) passes **2,214 tests with 11
skips** (742.24 s). [Sixteen focused whitening/profiler checks](production-focused-tests.log)
pass, including shared strided scratch and conservative borderline fallback.
The conditional profiler regression now times the complete second-pass stage
outside child graphs, suppresses nested event records there, and verifies
before/after numerical acceptance and wrapper restoration (#163).
A [clean staging wheel](production-wheel.json) contains all 37 core Python
modules with exact source-byte matches and excludes experiments/examples/tests.
Repository-wide lint and changed-file formatting checks pass.

[Stock Warp checks](production-stock-tests.log) pass 53 tests with 13 upstream-FSAI
skips. An additional two-case odd/even branch check passes on both runtimes,
covering strided shared factor scratch at odd metric dimensions. It was added
after the full-suite run above; no numerical source changed afterward.
[Additional pinned check](production-odd-branch-tests.log),
[additional stock check](production-odd-branch-stock-tests.log).

Fresh production controls retain exact inputs/reference/starts/source/dependencies
within each pair and clean occupancy. The qualifying comparison contains 30
accepted measured solves: paired Warp/CuPy controls for 100 modes and Anvil,
and a separate Warp-only 200-mode control pair. Times are seconds.

| Problem | Second whitening | Warp setup | Capture | Warp replay median | CuPy replay median |
| --- | --- | ---: | ---: | ---: | ---: |

| 100 | canonical | 0.3660 | 3.8533 | 3.1863 | 3.4874 |
| 100 | default | 0.3691 | 3.9089 | 3.0406 | 2.7687 |
| anvil | canonical | 0.7886 | 0.8179 | 1.6199 | 2.3833 |
| anvil | default | 0.8152 | 0.8170 | 1.6202 | 2.3885 |
| 200 | canonical | 0.6433 | 18.3401 | 11.1309 | — |
| 200 | default | 0.6360 | 18.6927 | 10.4733 | — |

The production median improves 4.6% at 100 modes and 5.9% at 200 modes. Anvil
uses the same canonical path in both workers and differs by only 0.013%; this
is a regression control, not an Anvil optimization claim. CuPy varies
substantially between trials; narrower-width controls remain separate work.

The original paired 200-mode canonical worker is **rejected**: all three Warp
trials and all physical checks pass, but one CuPy trial has a fresh absolute
residual 1.0000331425977916e-10 against 1e-10 (#69). Its two other CuPy trials pass;
the separate production-default worker passes all three. No ratio is taken
from the rejected worker, and its raw evidence remains in `production/`.
The predeclared separate Warp-only pair in `production-warp-only/` qualifies the
Warpack change without weakening or relabeling that reference failure.
[Qualification summary](production/qualification.json).

## Updated wide-step profile

The independent profiler passes complete original-equation, spectrum, subspace
and count checks before and after instrumentation. It records retained rank
outside the CUDA event intervals. All three samples per checkpoint have the
same rank counts, and occupancy is clean. Median milliseconds:

| Prefix steps | Selected step | CG | X orthogonalization | First metric eigensolve | Certified second whitening | Ritz eigensolve | Retained search / metric size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 136.774 | 26.135 | 7.237 | 12.765 | 8.080 | 24.700 | 208 / 208 |
| 10 | 103.790 | 7.807 | 6.956 | 13.216 | 6.879 | 17.952 | 191 / 192 |
| 20 | 67.003 | 2.394 | 6.953 | 8.977 | 3.140 | 13.083 | 128 / 128 |
| 30 | 57.381 | 1.221 | 6.961 | 7.058 | 2.749 | 10.788 | 112 / 112 |

Nested categories overlap and cannot be added. The complete second-whitening
stage includes mass/Gram work and factor application; its internal conditional
nodes contain no timing events. The diagnostic host branch selection is outside
production replay. These stage samples are not whole-solve speed ratios.
Almost every search direction remains active, so further rank compaction would
save little here. The first metric and Ritz eigensolves still matter; CG becomes
a small fraction of late steps. [Raw profile](profile/profile-a1.json) and
[occupancy](profile/profile-a1.occupancy.json).
