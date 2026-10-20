# Full CuPy conformance refresh and boundary-contract follow-up

The full pinned CuPy eigensolver/SVD suite has been rerun after the solver fixes.
All **31 ordinary failures from the original inventory now pass**, and every
previous pass is retained. One ordinary failure remains: the newly enabled
Fiedler comparison tracked in #72. Its reference starts in the wrong invariant
space; the adapter's 20-step Warp budget also fails its strict count. This is
not a claim of complete upstream conformance.

| Run | Passed | Failed | Skipped | Expected failures | Unexpected passes |
|---|---:|---:|---:|---:|---:|
| Full Warp adapter, before the boundary follow-up | 821 | 1 | 3,054 | 72 | 36 |
| Full unchanged native CuPy overlay | 2,500 | 0 | 1,195 | 173 | 116 |
| Composite Warp inventory after the 40-case focused follow-up | 857 | 1 | 3,018 | 72 | 36 |

Every inventory accounts for 3,984 selected cases; 7,745 unrelated tests are
excluded. Expected failures and unexpected passes retain upstream markers.
The composite overlays 40 exact node IDs onto the complete run; it is not a
second full-suite run. The native overlay's counts match its original inventory.

The full Warp run takes 453.08 s and native CuPy 191.90 s, including setup,
capture, assertions and test overhead. These are **not eigensolver performance
comparisons**. Spectra's unchanged native suite also passes all 31 executables;
its numerical-fixture refresh is separate ongoing work.

## Boundary contracts enabled

The adapter now clamps explicit `ncv` to `min(n, max(ncv, k+2))`, matching the
pinned CuPy entry point. Its default width remains the documented Warp adapter
width. CuPy performs an initial Lanczos sweep even when `maxiter=0`; the adapter
maps that to one complete Warp cycle, since Warp budgets count the initial
cycle. The 36 formerly skipped `test_starting_vector` parameterizations now
pass their original assertions. Neither the production solver's strict width
validation nor its positive-cycle requirement changes.

The adapter also rejects non-null `M`/`Minv` and non-normal `mode` with
`NotImplementedError`, and accepts explicit `mode='normal', M=None, Minv=None`.
This is validation parity, not generalized CuPy eigsh support. The four original
keyword fixtures **still skip** at their subsequent normal-mode FP32 solve.
They are not relabeled as passes merely because the exception check succeeds.

Ten local regression tests pass on pinned Warp 1.18 and stock 1.15. They check
unsupported-keyword error precedence, supported FP64/complex128 normal-mode
results against an independent diagonal spectrum, zero-budget/clamped-width
exact starts, physical equations and unchanged input vectors.

## Remaining unsupported contracts

After the focused overlay, 580 skips are upstream fixture conditions and
**2,438** are unsupported precision or adapter contracts:

| Contract | Skipped cases | Meaning |
|---|---:|---|
| FP32/complex64 eigensolver inputs | 1,622 | Includes four keyword fixtures whose later normal solve is FP32 |
| Block precision combinations | 528 | Public block solve requires FP64 |
| Complex partial SVD | 256 | Current partial SVD is real FP64 |
| Complex shifted selection beyond nearest | 16 | Generalized complex selection API work remains separate |
| CuPy RNG/internal API | 8 | No drop-in RNG contract |
| SVD v0/RNG passthrough | 3 | Distinct from supported eigenproblem starts |
| Full-space block fallback | 2 | No dense k=n fallback |
| FP32 block starts/matrices in fixed fixtures | 2 | Not covered by parameterized precision rows |
| Block per-iteration history | 1 | No equivalent public history API |

Matrix-backed input containers are still materialized outside capture. Passing
these boundary tests does not certify arbitrary native matrix-free callbacks.
The Fiedler reference mismatch, compact general-spectrum discovery, unsupported
precision/features and external vendor limitations remain visible in their
issues. No skip, expected failure or wrong-spectrum result is counted as a
numerical success.

## Evidence

- [Full Warp run](../../results/performance_followup/upstream-refresh-cupy-warp.json)
- [Full unchanged native CuPy run](../../results/performance_followup/upstream-refresh-cupy-native.json)
- [Environment and exact source hashes](../../results/performance_followup/upstream-refresh-manifest.json)
- [Focused 36 passes / four precision skips](../../results/performance_followup/cupy-width-contracts.json)
- [Inventory transition and local validation](../../results/performance_followup/cupy-width-contracts-validation.json)

Sources remain CuPy `2316a02d70c7e39d6a7c035529df8358d794256e` and Spectra
`db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95`, on the L40/driver570.158.01 with
CuPy14.2.0 and the pinned Warp1.18 Python/native overlay. Installed package
metadata says Warp1.15; the manifest separately identifies the actual imported
1.18 module and runtime hashes. CuPy references, conversion and validation stay
outside the Warp solver. Direct problems use Warp iteration; inverse problems
use cuDSS; LOBPCG uses only supplied preconditioners.

Reproduce full runs with the commands in [README](README.md), setting
`WARPACK_UPSTREAM_REPORT` to a new output path. Do not overwrite the historical
baseline. The focused follow-up selects the 36 starting-vector node IDs and
four unsupported-keyword node IDs recorded in its raw report. GPU jobs run
sequentially. No installed CuPy source is patched.
