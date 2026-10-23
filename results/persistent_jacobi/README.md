# Persistent projected Jacobi prototype (#159)

This benchmark-only prototype keeps the global Jacobi solver's pair schedule,
right-then-left update arithmetic, 2e-15 convergence threshold and 30-sweep
bound, but executes the rounds in one CUDA block. Matrices and rotations live
in caller-supplied global workspaces. Tile collective barriers order dependent
global writes between rounds; the same tested Warp synchronization dependency
is documented in the [global-factor study](../global_cholesky/README.md).

Production dispatch is unchanged. A single block may lose bandwidth and
parallelism at large projected dimensions, so no speedup is assumed.

Eighteen pinned-overlay tests pass: even sizes 2,48,104,128,208,312,512,
indefinite random and repeated-eigenvalue spectra, scales 1e-80 through 1e80,
strided input/output/workspace, changed-input graph replay, independent SciPy
spectra, original equations, orthogonality, nonfinite/zero input and recovery,
and rejection of unsupported dimensions. The first development test attempt
failed compilation because Warp disallows tuple assignment to array elements;
the corrected kernel assigns each element separately. No numerical comparison
was run from that failed attempt.

`benchmarks.experimental.persistent_jacobi_compare` compares the existing
multi-kernel global solver against this prototype with exact input hashes,
validated warmup and three measured repeats per dimension, and graph audits.
Isolated timing qualification and complete FEM/Anvil regressions are pending.
