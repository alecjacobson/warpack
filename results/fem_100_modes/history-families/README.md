# Compact history across FEM families (#128, #147)

These large procedural FEM cases each request 100 positive elastic modes.
Free and disconnected cases also return their six/twelve rigid modes, and must
converge the entire native count. Compact history uses the existing public
`keep_history=False`; all current Ritz vectors remain in the search space.

| Case | DOFs | Returned / null pairs | Width | Replay s, retained / compact | First-solve compute s, retained / compact | Compact worst original residual |
|---|---:|---:|---:|---:|---:|---:|
|free|35,328|106 / 6|112|2.9956 / 2.5762|5.4798 / 4.7070|8.36e-12|
|disconnected|70,656|112 / 12|120|5.3649 / 4.6573|8.2431 / 7.1259|9.53e-12|
|heterogeneous|34,560|100 / 0|104|3.3924 / 2.9100|5.5185 / 4.8419|9.85e-12|

Replay improves 14.0%, 13.2%, 14.2%, respectively. Outer iterations increase from
28 → 36, 27 → 32–33, 37 → 51. Smaller projected problems outweigh the additional steps
on these fixtures. This is a measured configuration tradeoff, not a change to
rank/convergence tolerances or a universal recommendation for every mode count.

All 24 complete first/later outputs pass, comprising 12 new compact-policy outputs
and 12 reused [retained-history controls](../../projected_symmetry/families/README.md).
They include native status/count/stopping, original-equation residuals, spectrum,
mass orthogonality, known nullspace and requested invariant-subspace checks.
Every pair has matching input/reference/start, core source, version and dependency
identities. All occupancy records are clean. Captured graphs contain no host
nodes and only device-to-device copies.

Measured 2026-10-23 on L40/driver 570.158.01, pinned Warp/FSAI, Xeon 8362 cores 8–11.
Replay medians use three solves. First-solve compute includes independently built
preconditioner/solver setup, capture and the validated first captured solve,
with cached kernels; it excludes assembly, imports and host validation. Raw JSON
retains separate setup/capture phases, all samples and exact commands. Other
settings match the [clamped history controls](../history-current/README.md),
including FSAI32, FP32 CG25 and outer tolerance 1e-11.
