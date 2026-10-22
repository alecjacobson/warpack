# Dominant modes without strict metric inverses

[#144](https://github.com/alecjacobson/warpack/issues/144): composing the public
`BlockLOBPCG` on `-A` with SPD metric B substantially reduces the pure-Warp cost
for this spectral surface fixture. Truncated upstream Warp FSAI/CG supplies a
preconditioner; it does not claim to compute an accurate B inverse at each
application. Eigenvalue sign recovery is part of the captured device graph.
No installed solver arithmetic or default changed.

See the [pencil, API and reproduction guide](../../benchmarks/applications/SPECTRAL_SURFACE.md).
These are independent procedural P1 reconstruction problems. B is a nonmass
regularization metric; A is singular PSD. No external meshes or reference
vectors enter timed solver initialization.

## Matched measurements

NVIDIA L40 / Xeon Platinum 8362, 2026-10-22, four BLAS/OMP threads pinned to
cores 8–11, verified current Warp 1.18.0/FSAI overlay. Each worker independently
constructs its matrix representation, factor and solver, then runs one warmup
and three measured solves. All rows below have uncontended GPU occupancy.

| Variables / modes | Method | Setup / factor (s) | Capture (s) | Warm median (s) |
|---|---|---:|---:|---:|
| 4,913 / 1 | Warp + cuDSS Krylov | 0.2727 | 0.2502 | 0.01749 |
| 4,913 / 1 | Warp + strict FSAI/CG Krylov | 0.3399 | 0.5056 | 0.15918 |
| 4,913 / 1 | Warp block + truncated FSAI/CG | 0.3398 | 0.3172 | 0.02187 |
| 35,937 / 1 | Warp + cuDSS Krylov | 0.9249 | 0.2581 | 0.05660 |
| 35,937 / 1 | Warp + strict FSAI/CG Krylov | 0.3199 | 1.0432 | 0.71755 |
| 35,937 / 1 | Warp block + truncated FSAI/CG | 0.3509 | 0.2807 | 0.09813 |
| 35,937 / 4 | Warp + cuDSS Krylov | 0.9236 | 0.3452 | 0.19032 |
| 35,937 / 4 | Warp + strict FSAI/CG Krylov | 0.3039 | 1.0552 | 2.39245 |
| 35,937 / 4 | Warp block + truncated FSAI/CG | 0.3895 | 0.2995 | 0.14433 |
| 9,826 / complete 2-fold cluster | Warp + cuDSS Krylov | 0.3093 | 0.2775 | 0.05515 |
| 9,826 / complete 2-fold cluster | Warp + strict FSAI/CG Krylov | 0.3043 | 0.5198 | 0.43361 |
| 9,826 / complete 2-fold cluster | Warp block + truncated FSAI/CG | 0.3493 | 0.3421 | 0.03548 |

The two 4,913-variable regularization controls also pass:

| Control | Method | Setup (s) | Capture (s) | Warm median (s) |
|---|---|---:|---:|---:|
| Gradient-only | cuDSS | 0.2201 | 0.2654 | 0.02020 |
| Gradient-only | Strict CG | 0.2071 | 0.4151 | 0.11976 |
| Gradient-only | Truncated block | 0.2152 | 0.3332 | 0.02780 |
| Stronger regularization | cuDSS | 0.2292 | 0.2363 | 0.01748 |
| Stronger regularization | Strict CG | 0.2923 | 0.5577 | 0.21047 |
| Stronger regularization | Truncated block | 0.3446 | 0.3201 | 0.02322 |

For the large single-mode problem, block is **7.3x faster than strict CG**, but
cuDSS remains 1.73x faster on warm replay. For four modes, block is 16.6x faster
than strict CG and 1.32x faster than this direct profile. These are problem and
configuration-specific results, not a universal block-versus-direct claim.

Setup excludes procedural assembly and independent CPU references; both are
recorded separately. Capture includes warmup/JIT work. Setup is one observed
construction per worker, not a repeated cold-process median. Output copies,
validation and geometric reconstruction measurements are outside solve timing.
CPU references are immutable, fingerprinted cached validation data; each load
rechecks array checksums, original equations and B-orthogonality. They are
never starting vectors, preconditioner data or shared factors.

Block uses width=max(4,2*k), CG cap 25, relative preconditioner tolerance 0.01,
atol=0, FP32 upstream factors/CG, FP64 outer iteration and a 200-step adaptive
cap. Large one/four-mode runs take 22/24 outer steps. Krylov uses ncv=32.
Strict CG uses tol=1e-13, atol=0 with fresh residual audits and adaptive outer
capture. cuDSS uses a fixed budget: one cycle for default single modes,
four for large-four, three for the repeated cluster, two for gradient-only,
and one for stronger regularization. These explicit budgets improve on the
initial four-cycle direct baseline; they are not proposed universal defaults.

A one-cycle gradient-only cuDSS trial is retained as a rejection: its residual
passes the common 1e-8 physical gate but its native 1e-11 convergence count is
zero. It is not an accepted speed result. Strict CG's recursive and fresh
inner residual diagnostics remain separate; outer accuracy does not imply
that all requested tight inverse tolerances passed.

## Validation and retained evidence

All **18 final configurations / 72 solves including warmups** pass fresh
original-equation residual, B-orthogonality, requested eigenvalues and complete
requested-subspace checks, plus native zero status/full count. Maximum
normalized residual is `8.53e-12`; maximum B-weighted subspace error is
`3.62e-11`. Geometric reconstruction metrics agree across methods within
`1e-8`. Matrices, numerical source hashes, versions and environment match
within each case. [Validation record](validation.json).

[Final raw workers](final/) include setup, capture, synchronized timing,
original-pencil residual, metric orthogonality, requested spectrum/subspace,
geometry, native counts, graph audits, source/dependency identities and
occupancy. The first large-four strict-CG worker encountered another GPU
process; its raw evidence remains, but its timing is excluded. An uncontended
repeat supplies the table row.

Pure-Warp graphs have no HOST/HtoD/DtoH nodes, including the sign finalizer.
cuDSS's vendor scalar HtoD copies remain visible under
[#29](https://github.com/alecjacobson/warpack/issues/29); direct graphs are not
claimed to be transfer-free. DOT inspection is not a CUDA API trace.

The [earlier exploration](pre-device-sign/) recovered positive values in its
host output accessor; those timings are not promoted as complete device-output
replays. Width-one/two exploratory workers in [width-screening/](width-screening/)
and [width-confirm/](width-confirm/) encountered other GPU work. Their accuracy
is useful, but they cannot establish a timing ranking. Width four remains the
qualified default for this application experiment.

The focused checks include dense generalized references, complete repeated
clusters, changed-anisotropy graph replay, block widths 1/2/4, gradient-only
and biharmonic metrics, positive device output, cache corruption/revalidation,
and rejection of inaccurate, contended, incomplete or mislabelled reports.
[Combined 78-test run](tests.log); [56-test reporter/cache follow-up](report-tests.log).
The follow-up adds 17 new reporter cases and repeats 39 existing checks.
No installed core changed, so the full library suite was not rerun.
