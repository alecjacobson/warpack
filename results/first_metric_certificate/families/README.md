# Large family controls for guarded first whitening

The same experimental composition as the [clamped/Anvil controls](../whole/README.md)
passes all 24 first/later outputs across these six fresh workers. Every case
requests 100 positive modes, plus the known rigid nullspace where present.
Compact history, ordinary projected graphs, pinned Warp/FSAI, L40, four Xeon
8362 cores, 2026-10-23. All pairs have matching input/start/reference, source,
version and dependency identities; all occupancy records are clean.

| Case | DOFs | Returned / null pairs | Width | Replay s, canonical / certified | First-compute s, canonical / certified | Certified worst original residual |
|---|---:|---:|---:|---:|---:|---:|
|free|35,328|106 / 6|112|2.5773 / 2.4955|4.6840 / 4.6311|8.51e-12|
|disconnected|70,656|112 / 12|120|4.6295 / 4.5324|7.1552 / 7.1117|9.56e-12|
|heterogeneous|34,560|100 / 0|104|2.9164 / 2.8246|4.8762 / 4.7904|9.86e-12|

Warm medians use three solves; improvements are 3.2%, 2.1%, 3.1%. First-compute
includes independent setup, capture and the validated first captured solve,
excluding assembly/reference/imports/host checks with cached compiled kernels.
All phases, samples and exact commands remain in the raw JSON. These gains
remain modest and do not establish a CuPy speed ratio.

Native counts/status/stopping and independent original equations, spectrum,
mass orthogonality, rigid nullspace and requested-subspace checks all pass.
The last-replay fast/fallback counts are 8/10, 9/9 and 9/9; those fallbacks all
come from early padding rejection. This is not an assumption about other inputs.
Together with the prior clamped/Anvil controls, 48 complete first/later outputs
qualify this experimental composition. Core first whitening remains canonical.
