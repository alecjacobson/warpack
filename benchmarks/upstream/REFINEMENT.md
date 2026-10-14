# Warp correction of cuDSS inverses

The CuPy SM eigenvector failures in [#2](https://github.com/alecjacobson/warpack/issues/2)
come from inverse accuracy. On the exact exported 30-by-30 fixture, an
unrefined symmetric cuDSS replay has relative linear residual `2.07e-8`;
one residual correction reduces it below `4.47e-16`. The general factorization
has the same issue (`1.26e-8` to `4.85e-16`). The shifted matrix's 2-norm
condition number is approximately 63; despite the original issue's title,
this particular matrix is not near-singular. Its smallest absolute eigenvalue
is approximately 0.281. The targeting shift is near zero.

`CuDSSInverse` now preserves the RHS, forms `r = b - A*x` in Warp, solves
`A*delta = r` using the existing cuDSS factors, and adds the correction in
Warp. General and symmetric-indefinite matrices default to one correction;
SPD matrices retain the original zero-correction default. `refinement_steps`
selects a fixed nonnegative budget. This is not an accuracy certificate for
arbitrary conditioning, scaling or factorization error.

All correction buffers are allocated at setup, and iteration can be captured.
There is no host residual check or host numerical decision. cuDSS retains its
known captured host-to-device scalar copies, tracked in
[#29](https://github.com/alecjacobson/warpack/issues/29); an extra solve adds
another set of those vendor copies. No claim of zero host transfers is made.
The correction uses the wrapper's scalar CSR expansion and supports block CSR,
multiple right-hand sides, in-place calls, and replay with changed device RHS.

## Accuracy contracts

The full adapted CuPy inventory now has **697 passed, 7 failed, 3,172 skipped,
72 expected failures and 36 unexpected passes** (3,984 selected cases).
All 24 transitions are the original real SM eigenvector failures becoming
passes. There are no lost passes or marker changes. The original assertions
remain unchanged. The seven remaining failures are the complex Hermitian
identity/low-rank cases in [#3](https://github.com/alecjacobson/warpack/issues/3).
These are Warpack results under the upstream adapter, not native CuPy timings.

Permanent eigen tests check residual, requested eigenvalues, orthogonality and
invariant subspace for `k=3,6,12`. Separate generalized fixtures contain two
exact zeros and two near-zero modes, a non-diagonal SPD mass, and two shifts.
Inverse tests cover both factorization types, multiple RHS, in-place replay,
and arbitrary 2-by-2 BSR blocks. The damped-chain fixture now passes its
physical residual and spectrum checks, so its strict expected-failure marker
has been removed.

The Spectra harness now separates original numerical bounds, internal counts,
and upstream-permitted nonconvergence. It keeps its strict outcome policy.
Its reporting tests explicitly reject small residuals with a wrong spectrum,
nonfinite metrics, SVD vector failures, and absent evidence. The comparison
command also detects numerical regressions hidden behind unchanged failed
labels. See [the contract definitions](README.md#numerical-assertions-and-convergence-counts).

## Full Spectra inventory

The full 360-case rerun records **135 passed, 47 failed, 176 unsupported and
2 timed out**. Of the failures, 15 meet every original numerical assertion but
miss the stricter internal count; 32 fail numerical assertions. No previously
passing set of numerical bounds was lost. No unsupported case is counted as a
pass, and generic residual-only cases do not establish requested-spectrum
correctness.

The comparison overlays the prior 102-case multiplicity run onto the original
360-case report: 130 passed, 51 failed, 176 unsupported and 3 timed out.
Seven failed labels become passes, but two strict passes become failures:
`ComplexEigs_16` and `GenEigsRealShift_12`, both 19/20 counts while satisfying
the original numerical bounds. **The strict regression gate therefore fails.**
They are tracked in [#35](https://github.com/alecjacobson/warpack/issues/35).
Their maximum absolute residuals are `3.58e-12` and `1.86e-12`, respectively,
against upstream bounds of `1e-9` and `1e-8`. These results are retained.

Not every transition can be attributed to this PR. The Davidson direct path
uses the earlier Krylov fix and has no cuDSS inverse. The direct nonsymmetric
`ComplexEigs_16` path is also unchanged by this correction. The worker wall
limit increased from up to 120 to 150 seconds; `GenEigs_12` now completes with
correct numerical bounds and a short count. That is additional evidence from
a longer budget, not a solver speedup. The two remaining timeouts are
`ComplexEigs_33` and `GenEigs_33`.

One correction improves but does not resolve the three sparse symmetric SM
failures in #4. Their condition numbers are about 17, 134 and 5,480, so those
failures still need factorization/refinement investigation. The other numerical
failures, unsupported interfaces and timeout work remain tracked.

## Shifted cycle convergence

All three shifted-cycle cases satisfy the original eigenvalue and eigenvector
bounds. Their physical residuals are `1.7e-16`–`2.4e-16`; physical counters
report 3/3, 5/5 and 6/6 at the harness's `1e-12` tolerance. The inverse-space
counter remains 4/5 for `Example1Shift_3`, so the strict upstream adapter still
labels that case failed. This is now an explicit reporting classification,
not an assertion that the physical eigenpairs are inaccurate.

The shift `-1e-6` gives the zero mode an inverse eigenvalue of `1e6`.
Roundoff in that transformed operator can therefore matter at a tolerance
where the original cycle equation is already accurate. The physical and
inverse residuals have different scaling; neither counter is substituted
for the other. A captured regression checks all three requested spectra,
including the zero mode and repeated eigenspaces, on two replays.

## Application and performance checks

All **179 local tests pass**, with no expected-failure marker remaining.
All 18 Warp application configurations (nine classes, requested sizes 80 and
1024) pass every trial's original residual/spectrum/orthogonality gate.
The damped-chain residual is at most `3.26e-13` at n=80 and `1.49e-10` at
n=1024, against `1e-8`; this fixes the demonstrated fixture failure but does
not complete generalized damped-system support in #27.

The isolated inverse sweep covers scales `1e-6`, `1`, `1e6`, one/three RHS,
and symmetric/general factors. All 12 corrected configurations pass every
trial with relative residual at most `1.05e-15`; none of the 12 uncorrected
configurations meets `1e-12`. These rejected uncorrected timings are not a
matched-accuracy performance baseline. Captured corrected applies have zero
host callback/DtoH nodes and four vendor scalar HtoD copies (two per solve).

Correction has a measured cost on already-accurate interior bar fixtures:

| Interior modes | Before | One correction | Ratio |
|---|---:|---:|---:|
| n=80 | 12.774 ms | 18.205 ms | 1.425× |
| n=1024 | 17.199 ms | 24.183 ms | 1.406× |
| n=10000 | 21.228 ms | 31.531 ms | 1.485× |

These rows fail the unchanged 1.25× timing gate, tracked in
[P1 #34](https://github.com/alecjacobson/warpack/issues/34). All accuracy gates pass;
all application/isolated-inverse trials report no competing GPU process.
Other previously accepted small application rows remain within that timing
gate. H has no accepted prior timing baseline. POD is unchanged by this PR
and retains the earlier #31 performance gap relative to its original baseline.

The default SPD path is unchanged. The Anvil cuDSS median is **0.467083 s**
(previous 0.467120 s), with all three existing physical/modal checks passing.
The Anvil runner uses warm synchronized samples but does not contain the
application runner's per-sample contention poll. GPU jobs were serialized.
The 10k-DOF lowest/most-negative bar paths also pass their regression gates.
CG/FSAI numerical code is unchanged and its earlier timing is not presented
as a new measurement here.

Evidence: [validation and transitions](../../results/refinement_validation.json),
[CuPy](../../results/refinement_cupy.json),
[Spectra](../../results/refinement_spectra.json),
[separate contracts](../../results/refinement_contracts.json),
[inverse sweep](../../results/refinement_inverse.json),
[applications](../../results/refinement_applications.json),
[10k bars](../../results/refinement_large.json),
and [Anvil](../../results/refinement_anvil.json).

## Reproduction

Use the pinned Warp/cuDSS environment in [the upstream protocol](README.md).
GPU workloads must run sequentially. Setup and capture are outside replay
timings; the isolated inverse benchmark checks every measured sample.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q
WARPACK_UPSTREAM_REPORT=results/refinement_cupy.json python -m pytest -q \
  -p benchmarks.upstream.cupy_plugin build/test_upstream_cupy_linalg.py
python -m benchmarks.upstream.bounded spectra --fresh --timeout 150 \
  --out results/refinement_spectra.json
python -m benchmarks.upstream.compare_reports \
  --baseline results/upstream_spectra_warp.json results/multiplicity_spectra.json \
  --current results/refinement_spectra.json --out results/refinement_contracts.json
python -m benchmarks.applications.refinement
python -m benchmarks.applications.compare --methods warp --sizes 80,1024 --repeats 3 \
  --baseline results/multiplicity_applications.json --out results/refinement_applications.json
python -m benchmarks.applications.compare --classes ABC --methods warp --sizes 10000 \
  --steps 5 --repeats 3 --baseline results/multiplicity_large_regression.json \
  --out results/refinement_large.json
python -m benchmarks.cg_experiments --mode eigen --shifts 1 --preconditioners cudss \
  --cycles 1 --ncv 64 --repeats 3 --out results/refinement_anvil.json
```

A separate early-exploration prototype for #31 was rejected because the
existing cycle test recovered the wrong multiplicity. Its
[patch and failed regression](../../historical/adaptive-exploration-rejected.md)
are retained; no production change or performance claim came from it.
