# Regular generalized FEM modes with diagonal mass

Follow-up to [#31](https://github.com/alecjacobson/warpack/issues/31), using the
[negative-foundation Q1 fixture](NEGATIVE_FEM.md): symmetric indefinite K, SPD
mass M, 27,648 unknowns, 20 requested largest-magnitude modes, and two repeated
polarizations. Analytic tensor-product eigenpairs independently establish the
complete requested space. The LM set coincides with LA for this fixture; no
reference vectors enter the start. This is an accuracy-gated settings and
benchmark-adapter improvement, with no production numerical source changes.

## Supported routes and API meaning

- `warp`: `GeneralizedKrylovSchur(..., mode="regular")`, cuDSS applies M inverse.
- `warp-cg-fsai`: the same metric formulation, upstream Warp FSAI/CG applies M
  inverse. Both diagonal and consistent SPD mass are supported.
- `warp-diagonal`: the same generalized solver, using the existing upstream
  Warp diagonal preconditioner as an **exact** mass inverse. Host fixture checks
  require finite positive diagonal M; construction and application of its
  reciprocal are performed by upstream Warp on device.
- `warp-scaled`: explicit standard form `D^-1/2 K D^-1/2`, with D=M. Matrix
  scaling, standard Krylov solve and physical vector recovery are all Warp
  operations. This benchmark adapter lives outside the installed library.
  Its input values are fixed; changing them requires rebuilding the setup.
  It reports standard-problem counts/residuals, while comparison checks use
  the original pencil and mass-normalized physical vectors.

The two diagonal routes explicitly reject consistent mass, singular/nonpositive
mass and shift-invert selection. They do not approximate a coupled mass inverse
by its diagonal. Generalized mass orthogonality is retained; no symmetric
Euclidean eigensolver is applied directly to nonsymmetric `M^-1 K`.

## Accepted profiles and remaining gap

Warm median milliseconds, three sequential uncontended samples, with setup,
capture, validation and reference assembly outside replay timing:

| Mass / route | Earlier profile | Current profile | ncv / cycle budget |
|---|---:|---:|---:|
| Lumped / Warp + cuDSS | 639.6 | 243.3 | 96 / 12, previously 32 cycles |
| Consistent / Warp + cuDSS | 1593.0 | 606.5 | 96 / 12, previously 32 cycles |
| Lumped / Warp + FSAI-CG | 694.3 | 263.5 | 96 / 12, previously 32 cycles |
| Consistent / Warp + FSAI-CG | 2040.1 | 775.5 | 96 / 12, previously 32 cycles |
| Lumped / Warp exact diagonal inverse | — | 201.6 | 96 / 12 |
| Lumped / Warp scaled standard form | — | 192.0 | 96 / 12 |
| Lumped / native CuPy eigsh, scaled standard form | — | 118.5 | 96 / native adaptive stopping |

Every entry passes original residual, complete requested spectrum/subspace and
mass orthogonality at the common 1e-8 bounds, plus all requested Warp counts at
1e-11 and zero CG failures where applicable. Native CuPy also passes its own
fresh scaled Frobenius residual bound 1e-8. These are different internal
stopping definitions, not identical tolerances.

The exact-diagonal Warp route additionally meets that absolute CuPy threshold:
its maximum fresh scaled Frobenius residual is **6.771e-10**. The scaled Warp
profile reaches **1.419e-8**, slightly above that additional threshold despite
passing its declared normalized tolerance and all common physical gates.
Thus 201.6 ms is the conservative Warp comparison if both the common and that
absolute stopping bound are required. No core counts are relabeled or tolerance
relaxed. Native CuPy remains faster.

[Direct 12-cycle trials](../../results/performance_followup/negative-lm-direct12.json),
[CG 12-cycle trials](../../results/performance_followup/negative-lm-cg12.json),
[latest diagonal/scaled/native comparison](../../results/performance_followup/negative-lm-regular-regression.json).
The scaled profile's same-setting regression passes (about 0.5% variation).
Exact-diagonal replay changes by under 0.01% from its prior 201.645 ms control.
The latest CuPy samples are 118.384, 118.453 and 119.321 ms; an earlier matching
ncv96 invocation recorded 165.244, 170.102 and 138.875 ms. Both complete blocks
are retained; this is reference timing variation, not a claimed CuPy code
improvement. Starts remain Warp seed 42 versus CuPy seed 2026.

The pure-Warp graphs have zero HOST nodes and no HtoD/DtoH transfers. The
cuDSS path retains its separately tracked vendor metadata copies (#29).
Setup/capture for the latest exact-diagonal route is .188/.377 s and for the
scaled route .154/.274 s; these include setup/JIT/runtime effects and are not
silently folded into replay timings.

## Exploration and rejected settings

The default adaptive exact-diagonal profile takes **31 cycles**, with 19
independent-direction recoveries and a 516.7 ms replay. The conservative
exploration guard remains unchanged. The 12-cycle profiles exhaust an explicit
budget with zero recoveries; their complete requested spaces are established
by this fixture's independent analytic checks. They are **not a universal
replacement for the adaptive guard**, nor proof that every arbitrary input is
complete when residuals pass.

Eight-cycle profiles for both diagonal routes fail, returning only 15/20
converged pairs and missing a requested direction. Ten cycles give 18/20 and
requested-subspace errors 4.60e-7 / 4.55e-6. These faster times are rejected:
[eight cycles](../../results/performance_followup/negative-lm-diagonal8.json),
[ten cycles](../../results/performance_followup/negative-lm-diagonal10.json).
The earlier six-cycle direct/CG failures remain in NEGATIVE_FEM.md.

Wider/narrower accepted controls do not improve Warp replay here:
[ncv128 / eight cycles](../../results/performance_followup/negative-lm-scaled128.json)
takes 212.0 ms (CuPy 127.7 ms), and
[ncv64 / 24 cycles](../../results/performance_followup/negative-lm-scaled64.json)
takes 203.8 ms (CuPy 134.6 ms). The
[proposed shared-memory 72-column dispatch](../../historical/regular-shared72-rejected.md)
passes accuracy but regresses performance by 31%, so it was fully reverted.
All failed budgets and candidate results remain available; no favorable trial
is selected out of a failed run.

## Validation and reproduction

Seven new tests cover both diagonal routes with nonuniform coordinate scaling,
unit factors 1e-8/1/1e8, captured repeated replay, original metric recovery and
explicit invalid/unsupported mass rejection. The final focused pinned suite
passes **60 tests in 15.28 s**; all seven new tests also pass on stock Warp 1.15
in **2.96 s**. This is a focused follow-up, not a new full local/upstream run.
[Validation and final source hashes](../../results/performance_followup/regular-validation.json).

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.applications.negative_compare --side 24 --modes 20 \
  --selections LM --masses lumped --methods warp-diagonal,warp-scaled,cupy-eigsh \
  --ncv 96 --steps 12 --repeats 3 --out build/regular-diagonal.json
```

Broader adaptive exploration and regular Krylov performance remain in #31.
These controls do not change the Anvil showcase timings or imply a measured
Anvil improvement.
