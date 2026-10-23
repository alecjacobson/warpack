# Guarded first whitening: complete solve controls

The benchmark-only composition attempts certified first whitening on CUDA
metric sizes 96–512. A finite diagonal in [0.5, 2] enforces normalized input scale
and rejects zero padding before factorization. Uncertain certificates or failed
factors select canonical rank revelation on the original normalized Gram.
Existing second projection/whitening and outer/native/physical gates remain.

Factor, work and congruence-product scratch share the second pass's buffers.
The existing first transform and metric eigenvector buffers hold the inverse
and transformed Gram; canonical fallback reinitializes the eigenvectors.
No additional dense scratch is needed on the qualified wide Ada composition.
All numerical work and conditional decisions stay in Warp on device.

## Same-source controls

L40, pinned Warp/FSAI, four Xeon 8362 cores, 2026-10-23. Fresh processes per
backend/problem; three measured replays plus the first captured solve.

| Problem | Backend | Setup s | Capture s | First-solve compute s | Warm median s |
|---|---|---:|---:|---:|---:|
|100 modes, compact history|canonical|0.4127|1.4126|4.6034|2.6687|
|100 modes, compact history|certified|0.4188|1.4092|4.5366|2.6190|
|200 modes, compact history|canonical|0.5939|3.8971|13.7991|8.9249|
|200 modes, compact history|certified|0.6230|3.9523|13.7423|8.6785|
|Anvil, retained history|canonical|0.7928|0.6208|3.0975|1.6168|
|Anvil, retained history|certified|0.7889|0.6101|3.0831|1.6225|

The wide replay gains are **1.9% / 2.8%**. Anvil's metrics are below the size
guard, so its numerical path does not change. First-solve gains are smaller,
especially at 200 modes; capture/setup are single observations per process.
These results do not justify a universal speed claim or replace the six-backend
Anvil table. The first-whitening composition remains experimental.

All **24 complete first/later outputs pass**. Every pair has identical
input/reference/start, source/version/dependency identities and clean external
occupancy. All native and independent physical/spectrum/subspace gates remain.
The worst original normalized residual for the certified wide workers is
8.99e-12 / 9.98e-12 at 100/200 modes, under the unchanged 1e-11 tolerance.
Captured graph DOTs contain no host nodes and only device-to-device copies.

Last-replay fast/fallback counts are 6/12 at 100 modes and 10/14 at 200; all
fallbacks in these two observations are caught by the diagonal guard. Smaller
metrics retain the existing path and are not included in these counters. Anvil
records zero attempts. These counts are observations, not assumptions about
future input rank or failure patterns.

First-solve compute = independently built preconditioner/solver setup + capture
+ validated first captured solve. Cached kernels; assembly, reference work,
imports and host validation excluded. Reproduction commands, all phases and
samples are retained in each raw JSON/occupancy record.

## Regression scope

81 affected first/second-whitening tests pass on the pinned runtime. On stock
Warp, 68 pass and 13 skip because upstream FSAI/batched-CG features are absent.
The same stock worker also passes the separate 33 paired-Jacobi tests; the
combined log therefore reports 101 passes / 13 skips. Checks cover both history
policies, consistent/lumped mass, free/disconnected/heterogeneous FEM, rigid
nullspaces, both whitening passes, first-pass failure/fallback, extreme row
scaling, changed coupled mass/stiffness/constraints and repeated modes on the
same graph. The standalone component's rank/congruence checks also remain.

[Validation logs](../validation/) include the earlier component-only controls
and the later composition runs. Core dispatch remains unchanged; broader large
family controls and performance follow-ups are tracked in #171.
