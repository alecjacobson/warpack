# Projected eigenvector norm drift (#174)

The expanded 1,024-dimensional repeated/null regression fails the strict
full-basis Frobenius orthogonality gate in the existing global Jacobi solver.
It is predominantly **column-length drift**, not a comparable loss of mutual
orthogonality. Independent extended-precision column norms confirm the diagnosis.

| Input scale | Original Frobenius error | Original column-norm contribution | Original off-diagonal contribution | After column normalization |
|---|---:|---:|---:|---:|
|1|1.18009e-11|1.17756e-11|7.66201e-13|7.66207e-13|
|1e-80|1.15122e-11|1.14874e-11|7.53491e-13|7.53570e-13|
|1e80|1.17934e-11|1.17687e-11|7.64787e-13|7.64808e-13|

The unchanged gate is 1e-11. Final normalization uses one pure-Warp tiled
reduction per column after the entire bounded Jacobi loop. It changes neither
H, eigenvalues nor sweep counts. Zero/nonfinite columns remain unchanged for
existing failure detection. It does not attempt to repair off-diagonal loss.
No buffers, host decisions, vendor calls or new synchronization are needed.

[Diagnostic JSON](diagnostic.json) retains all original failures, corrected
checks, scales, native stopping and exact source/dependency identities. The
612-dimensional companion also passes. This diagnostic makes no timing claim.

## Complete prototype controls

Only existing even global projections of size 256–1024 normalize in this
wrapper. L40, pinned Warp/FSAI, four Xeon 8362 cores, 2026-10-23; three measured
replays plus the first captured solve per fresh worker.

| Problem | Setup s, original / normalized | Capture s, original / normalized | First-compute s, original / normalized | Warm median s, original / normalized |
|---|---:|---:|---:|---:|
|100 modes, compact history|0.4142 / 0.4439|1.4223 / 1.4451|4.6276 / 4.6871|2.6668 / 2.6631|
|200 modes, compact history|0.6048 / 0.6124|3.9399 / 3.8978|13.9151 / 13.8251|8.9138 / 8.8987|
|Anvil, retained history|0.8002 / 0.7884|0.6306 / 0.6069|3.0578 / 3.0345|1.6255 / 1.6806|

**All 24 complete outputs pass** native counts/status/stopping and independent
physical/spectrum/subspace checks. Each pair has matching inputs, starts,
source/version/dependencies and clean occupancy. Anvil's smaller numerical path
is unchanged; outer counts are 16/17/16 versus 16/17/17, explaining why its
median varies despite the guard. This is an accuracy fix, not a speed claim.

First-compute includes independent preconditioner/solver setup, capture and the
validated first captured solve, excluding assembly/reference/imports/host
validation with cached kernels. Setup/capture are single observations per
process. [Raw reports and commands](whole/) retain all phases and samples.
These are narrow controls, not a replacement six-backend showcase comparison.

Eight prototype tests pass on both pinned and stock Warp. They cover
104/612/1024 dimensions, indefinite and repeated/null spectra, scaling by
1e±80, zero/changed replay, original equations, independent complete spectra,
full-basis orthogonality, graph audits and invalid-column/strided-storage
preservation. The stock log also runs four added paired-Jacobi cases: three
pass and the existing #174 regression remains strict xfail before integration.
[Logs](validation/) preserve that distinction.

The installed large projected path now includes this fix, shared with compact
graphs. [Current validation](VALIDATION.md) records 2,566 full-suite passes,
focused checks on both runtimes and exact-source wheel verification. The tables
above remain prototype controls; no test threshold is loosened.
