# Sparse geometric coarse correction: qualified but not recommended yet

Follow-up to [#134](https://github.com/alecjacobson/warpack/issues/134) and the
[rank-24 correction experiment](../fem_coarse_correction/README.md). This static
hierarchy removes the coarse eigenbasis/source solve entirely. It reduces CG
work, but does not beat the existing FP32 bucketed FSAI/CG configuration.

The independent procedural clamped tet beam has 34,560 free variables,
consistent mass, 20 requested modes and block width 24. P1 transfers coarsen
15→5→1 grid subdivisions: 34,560→1,620→36 variables. At each level Warp forms
the sparse Galerkin shifted operator `Hc=P.T H P`, with `H=K+0.01 M`.
The additive inverse approximation is `S + P Bc P.T`, where S applies upstream
FSAI factors and Bc recurses. The 36-variable bottom uses Warp Cholesky.
Galerkin operators preserve the fine mass contribution; direct coarse mass
assembly is not substituted. No cuDSS or CPU sparse/dense solve appears in
preconditioner construction or application. Host mesh/transfer assembly and
independent references stay in benchmark code outside `warpack/`.

## Fresh matched timing

NVIDIA L40 / Xeon Platinum 8362, 2026-10-22, current verified Warp 1.18.0/FSAI
overlay, four BLAS/OMP threads on cores 8–11. One warmup and three measured
solves per configuration; no sampled competing GPU process. Same fine input
hash and source hashes across all four workers.

| Preconditioner | Total setup including assembly/capture (s) | Warm solve median (s) | Outer steps | Scheduled inner vector iterations |
|---|---:|---:|---|---|
| Existing FP32 FSAI/CG25, active buckets | 1.5937 | 0.33553 | 26 | 6,460–6,492 |
| FP64 FSAI/CG25, full width | 1.5229 | 0.51781 | 26 | 14,400 |
| Geometric additive, no inner CG | 2.7628 | 0.69595 | 68–69 | — |
| Geometric additive inside FP64 CG25 | 2.8173 | 0.48541 | 25–26 | 7,200–7,560 |

The geometric CG route cuts scheduled inner work by about half compared with
the full-width FP64 control, but its repeated restriction/prolongation and
coarse solves leave only a 6% observed warm-time improvement. It is **45%
slower than the current FP32 bucketed baseline**, and setup is also higher.
Using the hierarchy directly avoids inner CG but requires many more outer
steps. Neither experimental route is promoted as the new default.

Setup includes fine assembly, P1 transfers, all coarse operators/factors,
solver workspaces and graph capture. Independent reference computation is
excluded. Individual setup components and context-memory changes are recorded
in the [raw workers](final/); the latter are not peak memory. Setup numbers
are individual constructions, not repeated cold-process medians. Warm solve
starts from the same random block each time. There is no hidden coarse source
solve in any row.

All 16 final solves (including warmups) pass native zero status/full count,
original-equation residual, metric orthogonality, requested eigenvalues and
independent complete-subspace checks. Maximum normalized residual is
`8.79e-12`, maximum subspace error `5.90e-12`. Every graph audit has no
HOST/HtoD/DtoH nodes. Earlier [screening](screening/) records are retained
separately; the contended inner-CG screen is not a timing comparison.

The hierarchy uses FP64 applications with upstream FP32 FSAI factors. Its
inner CG uses relative tol=.01, atol=0 and at most 25 iterations. Truncated
preconditioning is not a strict inverse guarantee. Current geometric CG
trials reach the inner tolerance, while baseline truncated trials can miss
it; acceptance still requires independently correct outer modes.

## Tests and next steps

### Mixed precision and active-width follow-up

The [fresh mixed-precision workers](mixed/) keep the same three-level
operators/factors but use FP32 inner application and the public active-CG
buckets. `GeometricCG` is a benchmark subclass of `BatchedCGPreconditioner`:
it substitutes the hierarchy's linear application while retaining public
matrix refresh, conversion, statistics and upstream Warp CG recurrence.
Width variants share coarse operators/factors and own distinct workspaces.
There is no new custom CG implementation.

| Profile | Setup including assembly/capture (s) | Warm solve median (s) | Scheduled inner vector iterations |
|---|---:|---:|---|
| Fresh FP32 bucketed FSAI/CG25 baseline | 1.5218 | 0.33602 | 6,472–6,620 |
| Geometric FP32 CG25, full width | 4.0798 | 0.41822 | 7,272–7,464 |
| Geometric FP32 CG25, buckets of 4 | 3.4193 | 0.36925 | 4,528–4,628 |

All three workers are uncontended. All 12 solves pass original/native checks,
with maximum residual `9.21e-12` and subspace error `6.66e-12`. The bucketed
hierarchy improves on the earlier FP64 hierarchy, but remains about 10% slower
than the fresh existing baseline. Setup includes compilation of the new
working-precision kernels; these individual setup measurements are not stable
cold-process medians. The new profiles do not change recommendations.

[59 focused regression tests pass](mixed/tests.log), including shrinking and
growing active widths, non-leading packed RHSs, zero and scaled RHSs, in-place
application, changed fine matrix values, shared-factor storage, fresh linear
residuals, and both mass variants through captured eigen-solves. The generic
FP64 additive identity remains independently checked, and the FP32 application
is checked against that identity at working-precision accuracy.

### Remaining work

Four new tests verify the additive SPD identity and independent dense
Galerkin matrices across three levels, exact in-place application, changed
bottom-factor graph replay, rejection of an indefinite bottom and recovery,
and complete FEM solves with both lumped and consistent mass. Two reporter
regressions require complete setup accounting without inventing a coarse
source eigensolve. See [combined tests](../spectral_surface_block/tests.log)
and [reporter follow-up](../spectral_surface_block/report-tests.log).

The hierarchy is fixed during each solve. It does not automatically refit all
levels when the fine matrix changes. Transfers, factors and workspaces stay
alive with the hierarchy. An SPD fixed preconditioner can remain mathematically
valid for another SPD fine matrix without remaining effective.

Next work under #134 is to reduce transfer/application overhead in these
mixed-precision active-width hierarchies, before extending to irregular meshes
or stronger contrast. This experiment establishes neither a multiscale Lanczos
implementation nor a general multigrid performance claim. No installed core
code or default changed.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu \
  --out build/geometric.occupancy.json --log build/geometric.log -- \
  taskset -c 8-11 python -m benchmarks.applications.fem_coarse_correction \
    --profile geometric-inner-cg --coarse 5 --fine 15 --modes 20 --width 24 \
    --repeats 3 --out build/geometric.json
python -m benchmarks.applications.fem_coarse_correction_report build/geometric.json
```

Use the current pinned overlay. Other profiles are `geometric`, `cg64`, and
`cg`. The reporter rejects failed/contended results and compares only matching
configuration/provenance when supplied `--baseline`.
