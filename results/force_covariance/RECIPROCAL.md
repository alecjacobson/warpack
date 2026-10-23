# Reciprocal force-covariance formulation

[Application and usage](../../benchmarks/applications/FORCE_COVARIANCE.md#reciprocal-recipe)
· [implicit comparison](README.md#qualified-implicit-comparison)
· [tracking issue #152](https://github.com/alecjacobson/warpack/issues/152)
· [remaining explicit-reference conditioning issue #153](https://github.com/alecjacobson/warpack/issues/153)

For invertible S, `Q = H S^-1 H` has covariance modes as its lowest generalized
eigenvectors, with covariance values `lambda = 1/mu`. The GPU application keeps
both Q and its inverse implicit. It reuses H solves and avoids M factorization.
The CPU alternative explicitly assembles Q and factors it with SuperLU. This
recreates the reciprocal formulation, not the authors' UMFPACK implementation.

## Accuracy and cost

The same 10,890-DOF procedural cantilever, full-rank nonuniform diagonal prior,
lumped mass and 20-mode request are used as in the implicit comparison. Matrix,
prior and dependency hashes match. Every common imported source hash matches;
only the additional reciprocal composition differs. Hardware and thread/cache
settings are identical. GPU solves use one fixed Krylov cycle with width 64;
polished rows then take one fixed block inverse step. Independent full-subspace,
spectrum, energy, covariance residual and reciprocal residual gates are all
`1e-8`, with complete final native counts on every warmup and timed repeat.

Fresh 2026-10-23 results, each with one validated warmup and three validated,
uncontended repeats:

| GPU recipe | Native Q tolerance | Complete setup (s) | Median warm solve (s) | Worst Q residual |
|---|---:|---:|---:|---:|
| cuDSS, no polishing | 1e-8 | 1.1546 | 0.06796 | 4.97e-9 |
| cuDSS, one block polish | 1e-9 | 1.1236 | 0.08827 | 3.89e-10 |
| Pure Warp CG, one block polish | 1e-9 | 4.3338 | 1.30192 | 7.22e-10 |
| Pure Warp CG, one inner correction and one block polish | 1e-9 | 5.2871 | 1.45378 | 3.10e-10 |

Setup charges assembly, uploads, inverse/preconditioner construction, capture
and stage initialization. The four rows' worst covariance residuals are below
6e-13 and requested-subspace errors below 1.15e-12. Native Q counts refer to the
final output stage; the raw files separately retain initial Krylov counts and
polishing iterations. A smaller initial physical count can be repaired by
polishing, but a smaller final count always fails acceptance.

Against the implicit table's 0.08820 s cuDSS and 1.29935 s CG medians, unpolished
reciprocal cuDSS reduces warm solve time about 23%. It has the weaker Q residual
shown explicitly above, while preserving covariance accuracy. Polishing brings
cuDSS back to the implicit runtime. Pure Warp shows no material replay benefit;
the additional inner correction is slower. Setup variation alone is not a
claimed speedup. The implicit route requests native covariance tolerance 1e-11;
this comparison shares independent physical gates, not identical native stopping
definitions on the two different pencils.

Final timing rows are recorded under `reciprocal-final/`; only strict reporter
outputs may be used as regression baselines. The initial qualified pass is
retained under `reciprocal-qualified/`; the final pass follows an API cleanup
making `capture()` return all stage graphs, with no numerical algorithm change.
The final `cg-polish-a1` attempt is contended and excluded; `a2` supplies its
accepted timing. The explicit CPU final row remains rejected for accuracy.

One block polishing step repairs the large pure-Warp reciprocal equation while
preserving the requested covariance subspace. Adding an inner CG residual
correction improves fresh H residuals but does not provide a performance win.
The implicit covariance route remains the general choice, especially for
singular priors. A fixed-budget reciprocal run's small covariance residual is
insufficient to establish its native reciprocal count or original Q accuracy.

## Rejected controls

`reciprocal-screen/` retains the initial `tol=1e-9` profiles. cuDSS reports only
13/20 native reciprocal pairs; CG reports 11/20 and has a fresh independent Q
residual about 1.97e-8. Their covariance subspaces are accurate, but the runs are
rejected. `reciprocal-refinement/` requests native `tol=1e-8`; one inner CG
correction alone still leaves only 17/20 pairs and Q residual 2.30e-8. The accepted
polished method addresses the eigenvector's reciprocal equation, not just the
fresh H solve residual. No rejected row is rounded into a pass.

The explicit CPU reciprocal solve also fails, even with ARPACK tolerance
`1e-12`: worst covariance residual 2.54e-8, covariance eigenvalue error 4.53e-8,
and Q residual 1.09e-8. Its requested subspace error is only 1.38e-11, illustrating
the squared-conditioning sensitivity of the eigenvalues and equations. The
accepted CPU implicit covariance route remains the comparison baseline; this
is not evidence that ARPACK generically fails the problem.

Explicit Q has 1,702,170 nonzeros versus 431,294 in H, and its SuperLU L/U factors
have 14,668,536 combined nonzeros. GPU Q applications do not construct that
matrix or its factors. CPU explicit assembly and factorization are charged to
its setup, although its failed timing is not eligible for performance claims.

## Reproduction

Use the pinned FSAI environment and record GPU occupancy and environment as in
the [implicit protocol](../../benchmarks/applications/FORCE_COVARIANCE.md).

```bash
python -m benchmarks.monitor_gpu --out reciprocal.occupancy.json --log reciprocal.log -- \
  python -m benchmarks.applications.force_reciprocal_compare \
  --backend cg --side 10 --modes 20 --ncv 64 --cycles 1 --fixed-cycles \
  --tol 1e-9 --polish-steps 1 --repeats 3 --out reciprocal.json
python -m benchmarks.applications.force_covariance_report reciprocal.json
python -m pytest -q tests/test_force_reciprocal.py tests/test_force_covariance_report.py
```

For the unpolished cuDSS profile use `--backend cudss --tol 1e-8` and omit
`--polish-steps`. `--backend arpack --tol 1e-12` without `--fixed-cycles` runs
the explicit CPU reciprocal formulation; the preserved large-case failure is
expected. Native tolerances differ, and the shared independent physical gates
are unchanged.
