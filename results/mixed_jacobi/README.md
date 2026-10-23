# Mixed projected-basis experiment (#160)

This benchmark-only pipeline scales the projected matrix on device, computes
a complete FP32 candidate basis, restores its orthogonality with FP64
Cholesky QR, projects the **original FP64 matrix**, finishes with the existing
FP64 Jacobi solver, and lifts the vectors in FP64. No column is discarded in
FP32 and no FP32 eigenvalue determines final values or rank. All workspaces
are allocated before capture. Production is unchanged.

Nineteen tests pass on both the pinned Warp 1.18 overlay (11.77 s) and stock
Warp 1.15 (59.73 s, including first compilation). They cover dimensions 2–208,
indefinite and repeated spectra, eigenvalues around the FP64 rank threshold,
scales 1e-80..1e80, strided buffers, changed-input graph replay, independent
equations/full spectra/orthogonality, and zero/nonfinite status/recovery.

## First initializer: accurate, but too slow

The first version uses the single-block global-memory FP32 initializer derived
from #159. The FP64 finishing solve consistently drops to two sweeps, but the
initializer remains expensive. At 208 and above, its FP32 stopping floor also
exhausts the 30-sweep bound. It supplies a complete usable basis and final FP64
checks pass, but it is not a useful performance choice.

Uncontended L40 medians in milliseconds, three measured replays plus a validated
warmup for each backend/size. Setup/capture excluded; device input reset included.

| Dimension | Existing FP64 global | Mixed pipeline | FP32 / FP64 finishing sweeps |
| ---: | ---: | ---: | ---: |
| 48 | 1.649 | 1.808 | 7 / 2 |
| 72 | 2.537 | 4.744 | 7 / 2 |
| 104 | 4.200 | 15.602 | 7 / 2 |
| 128 | 5.302 | 31.428 | 8 / 2 |
| 208 | 11.545 | 487.109 | 30 / 2 |
| 312 | 22.079 | 1618.916 | 30 / 2 |
| 512 | 60.390 | 7283.469 | 30 / 2 |

Every row passes independent original-equation, orthogonality and full-spectrum
checks after warmup and every measured replay. Exact input hashes match within
each pair; graphs have no host callbacks or host transfers. This version loses
at every size and is not promoted. The reduction in FP64 finishing work motivates
testing a cheaper initializer, with the same final-accuracy requirements.

[Raw evidence](isolated/jacobi-a1.json),
[occupancy](isolated/jacobi-a1.occupancy.json),
[log](isolated/jacobi-a1.log).

## Shared-memory initializer

A second benchmark-only variant keeps the complete FP32 matrix and basis in
shared memory and stops its coarse solve at 2e-5. It uses the same FP64 recovery
and final acceptance as above. The full pipeline beats the existing global
FP64 path at 48–72 columns, but loses at 96 and above. Production is unchanged.

Uncontended L40 medians in milliseconds, three measured replays plus a validated
warmup; reset included, setup/capture excluded:

| Dimension | Existing FP64 global | Shared-initialized mixed | FP32 / FP64 finishing sweeps |
| ---: | ---: | ---: | ---: |
| 36 | 1.1013 | 0.6566 | 6 / 2 |
| 48 | 1.6583 | 0.9514 | 6 / 2 |
| 64 | 2.2389 | 1.5427 | 7 / 2 |
| 72 | 2.5288 | 2.3370 | 7 / 2 |
| 96 | 3.8232 | 3.9475 | 7 / 2 |
| 104 | 4.1835 | 4.8977 | 7 / 2 |
| 108 | 4.3552 | 5.6163 | 7 / 2 |

The 36-column control forces the global solver; it is **not** the production
36-column default. The complete FEM experiment therefore replaces only the
existing Ada global path at 48–64 or 48–72 columns, checking shared capacity.
Other sizes and architectures retain their original dispatch.

The expanded isolated suite passes 39 tests on pinned Warp 1.18 and stock Warp
1.15, with six shared-capacity skips on this L40. It includes scales, strided
buffers, rank-threshold spectra, repeated spectra, changed-input graph replay,
and zero/nonfinite recovery. Six additional pinned-runtime FEM tests pass:
clamped consistent mass, free-body, and heterogeneous systems, each using both
candidate size limits and two validated graph replays. Complete Anvil timing
and accuracy controls are required before considering production use.

[Shared initializer raw evidence](shared-initializer/jacobi-a1.json),
[occupancy](shared-initializer/jacobi-a1.occupancy.json),
[log](shared-initializer/jacobi-a1.log).

## Complete Anvil control: no useful improvement

Fresh independent processes use exactly the same Anvil inputs, references,
starts, source manifest and dependencies, with unchanged outer/native/physical
criteria. Each worker has three measured Warp and three measured CuPy solves;
all 18 pass. Occupancy is clean and captured graphs contain only device copies
and kernels/conditionals. Setup includes each candidate's extra workspaces;
graph capture is reported separately. All times below are seconds.

| Projected backend | Warp setup | Capture | Warp replay median | CuPy replay median |
| --- | ---: | ---: | ---: | ---: |
| Default | 0.8428 | 0.8455 | 1.5903 | 2.3411 |
| Shared FP32 start, 48–64 | 0.7851 | 0.9162 | 1.6078 | 2.3823 |
| Shared FP32 start, 48–72 | 0.7905 | 0.9243 | 1.6113 | 2.3840 |

Both candidates take 16–17 outer steps, as does the default. The 1.1–1.3%
replay increase supplies no evidence of a whole-Anvil improvement, despite
the isolated gains. Neither is promoted. The worst original residual across
Warp runs is 1.99e-8 under the existing Anvil physical check; the returned-pair
normalization and native 1e-8 stopping check also pass. CuPy keeps its existing
1e-6 absolute stopping bound and passes fresh returned-vector checks.

CuPy uses the same supplied Warp FSAI/CG preconditioner within each worker;
its setup is not represented by the Warp setup column. These are controlled
block-method experiments, not a replacement for the README's six-backend
standalone setup/timing table. See [summary](anvil/summary.json) and the adjacent
raw reports, logs and occupancy records for the full setup scopes and checks.
