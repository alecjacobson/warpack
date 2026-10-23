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
