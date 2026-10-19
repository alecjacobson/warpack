# Matrix-free normal-product POD and replay repair

Follow-up on 2026-10-19 to adaptive-throughput [#31](https://github.com/alecjacobson/warpack/issues/31)
and changed-input replay [#63](https://github.com/alecjacobson/warpack/issues/63).
The normal-product SVD is **experimental**, outside the installed package.
Production `PartialSVD` retains its augmented symmetric formulation.

## Formulation and limits

For a tall matrix, the prototype applies `A.T @ (A @ x)` without forming a Gram
matrix; for a wide matrix it applies `A @ (A.T @ x)`. Sparse products, Krylov
iteration, small eigensolves, vector lifting and both original-equation residuals
are Warp operations. There is no cuDSS, CuPy or host numerical computation in the
solver. Reference libraries are only called by the comparison driver.

The snapshot fixtures have just 40 or 48 columns (rows for the wide transpose).
Using a full basis in that small normal space makes one restart sufficient and
covers repeated eigenspaces. The existing independent-chain stopping guard is
unchanged for partial bases. This does not weaken the augmented solver's guard
or establish a general adaptive eigensolver throughput fix.

Normal equations square the condition number. The prototype supports well-resolved
largest **nonzero** triplets, explicitly rejecting requested eigenvalues at or
below 1e-14 of the largest normal Ritz value, nonfinite data, or zero lifted
vectors (`status=2`, incomplete counts). This cutoff is a conservative diagnostic,
not a rank certificate or a guarantee of accuracy. Both original SVD equations,
both orthogonalities and complete requested spaces still need validation. General
weighted POD, complex SVD and recovery of null singular bases are not implemented.
The augmented method remains useful when squaring the conditioning is undesirable.

## Large comparisons

NVIDIA L40, pinned Warp 1.18/FSAI environment, CuPy 14.2, SciPy 1.15,
pinned native Spectra, four BLAS threads. Three synchronized warm solves,
median milliseconds; setup/capture/validation are excluded and recorded separately.
Each method restores its start. Every accepted row passes the same independent
1e-8 requested-value, paired-equation, orthogonality, requested-subspace and
reconstruction-energy gates. The normal prototype also requires full left/right
counts at its unchanged physical tolerance 1e-11 and zero device status.

| Snapshot fixture | k | Normal Warp, ncv=48 | Augmented Warp | CuPy | SciPy ARPACK | Spectra |
|---|---:|---:|---:|---:|---:|---:|
| Clustered, 100,000×48 | 20 | **15.408** | 198.796 | 74.776 | 478.290 | 552.997 |
| String, 100,000×48 | 20 | **15.811** | 197.849 | rejected | 461.316 | 702.055 |
| Wide, 48×100,000 | 4 | **14.145** | 35.787 | 15.565 | 416.970 | 292.486 |

Reference ncv is 40 for the 20-mode rows and 32 for the four-mode row. The normal
prototype explicitly uses the full 48-dimensional basis; these are different
formulations and internal work budgets under matched external accuracy gates.
The clustered result is 12.90× faster than augmented Warp and 4.85× faster than
CuPy in warm solve time. The wide improvement over CuPy is much smaller.

Normal-product residuals stay below 1.65e-14 and orthogonality errors below 1e-13
in these trials. The clustered fixture includes a threefold leading singular
value and two exact zero singular values outside the requested set. The string
fixture is rank 22; CuPy's native ncv=40 failure remains rejected and tracked in
[#52](https://github.com/alecjacobson/warpack/issues/52). It is not a speed comparison.

The clustered/string native references come from the initial same-session runs;
the two Warp formulations were rerun after the transpose-refresh repair. The wide
row runs all methods after that repair. All runs were sequential with no observed
competing GPU process. Per-run occupancy sampling, graph audits and raw validations
are retained. Pure-Warp graphs contain only device-to-device copies and no HOST
callbacks; this is a graph-DOT audit, not a CUDA API trace.

The clustered prototype's setup is 153 ms and capture 224 ms; string setup 174 ms and
capture 224 ms; wide setup 199 ms and capture 224 ms. Setup includes sparse upload,
transpose/mapping construction and workspace allocation. Compilation is counted
where encountered. These are single process-local measurements, not warm setup
medians; replay ratios are not first-use or end-to-end speedups.

Artifacts: [clustered final Warp comparison](../../results/performance_followup/normal-clustered-final.json),
[string final Warp comparison](../../results/performance_followup/normal-wave-final.json),
[wide all-method comparison](../../results/performance_followup/normal-wide-final.json),
[clustered initial references](../../results/performance_followup/normal-clustered-large.json),
[string initial references and rejected CuPy](../../results/performance_followup/normal-wave-large.json).

The original small application fixtures also pass using the full smaller basis:
80×40 takes **2.899 ms** (ncv=40), versus 7.294 ms augmented;
1024×48 takes **4.716 ms** (ncv=48), versus 10.766 ms augmented.
These explicit alternative profiles fit inside the original pre-multiplicity
1.25× allowances of 4.531 ms and 11.595 ms. The production default formulation
is unchanged; this does not make its old identical-settings regression pass.
[80×40](../../results/performance_followup/normal-small80-final.json) ·
[1024×48](../../results/performance_followup/normal-small1024-final.json).

## Rejected early chain exploration

An alternative attempted to seed a fresh orthogonal direction at the first
expansion of each restart once all requested residuals passed, while retaining
the k-chain exploration guard. The existing cycle-Laplacian regression rejected
it: n=20/k=3/ncv=6 returned `[1.809017, 1.951057, 2]` instead of
`[1.951057, 1.951057, 2]`, eigenvalue error 0.14203952. Another 71 focused tests
passed. The patch was reverted and is not part of the library.

A newly seeded chain can be discarded before it explores enough to recover a
missing repeated eigenvector. Counting seeds alone does not fix that. Keep the
[preserved failing patch](../../results/performance_followup/rejected-early-exploration.patch)
and [regression output](../../results/performance_followup/rejected-early-exploration.log)
as evidence against that optimization, rather than relaxing the spectrum check.

## Production SVD replay fix

Review uncovered a separate existing defect: `PartialSVD` copied A's transpose
at construction but read live A values during replay. A changed matrix therefore
made the augmented operator nonsymmetric. In the minimized 50×30 fixture, scaling
A by 4 and replaying the same graph produced 17.69% singular-value error and a
left-equation residual 16.72; both reported counts correctly fell to 0/5.
The [failure record](../../results/performance_followup/svd-stale-transpose.json)
is retained.

The fix builds a transpose-entry mapping on device at setup and refreshes only
transpose values at the start of every eager solve/captured replay. This adds one
Warp kernel, with no host readback or sparse topology reconstruction. The same
helper serves the normal prototype. Fixed sparse topology and stable array
addresses are required; topology changes still require new setup.

The [corrected record](../../results/performance_followup/svd-refreshed-transpose.json)
returns 5/5 counts, relative singular-value error 1.60e-14 and physical residual
7.27e-13 after the fourfold change. Regression coverage includes tall/wide sparse
matrices, empty/uneven rows, explicit stored zeros becoming nonzero, overallocated
offset storage, multiple value changes and restoration on the same graph.

The unchanged-input clustered augmented control changes 198.827→198.796 ms after
the refresh; string changes 197.856→197.849 ms. These differences are timing noise,
not speedups. The normal prototype adds approximately 0.063 ms in each paired run
(under 0.5%). No material large-case throughput regression was observed.

## Reproduction

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu --out build/pod.occupancy.json --log build/pod.log -- \
  python -m benchmarks.experimental.normal_pod --case clustered --size 100000 \
  --modes 20 --ncv 48 --reference-ncv 40 --out build/pod.json
python -m pytest -q tests/test_svd.py tests/test_svd_replay.py tests/test_normal_svd_experimental.py tests/test_multiplicity.py
```

Use `--case wave` for the string; `--case wide --modes 4 --reference-ncv 32` for
the wide case. `--compare warp` reruns only
the two Warp formulations; an empty `--compare ''` measures just the prototype.
No normal-product method is exported from `warpack` yet. Public API promotion,
conditioning/orthogonality safeguards across more demanding spectra, and wider
snapshot spaces remain follow-up work under #31/#17.

Validation: **496 full-suite tests pass** in 196.63 s in the pinned environment;
**37 SVD tests pass** in 8.79 s on stock Warp 1.15. Lint, formatting and isolated
wheel build pass, with only the 18 core modules packaged. The experimental code
remains excluded. [Validation provenance](../../results/performance_followup/normal-validation.json).
