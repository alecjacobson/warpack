# Normal-product POD: orthogonality and wider snapshots

Follow-up to [#31](https://github.com/alecjacobson/warpack/issues/31) and
[#76](https://github.com/alecjacobson/warpack/issues/76). The normal-product
formulation squares conditioning; small original-equation residuals alone do
not certify an orthonormal POD basis.

The guarded implementation is now exposed through the explicit [NormalSVD API](../upstream/NORMAL_SVD.md); measurements below record the pre-move implementation and the API report links its unchanged-arithmetic controls.

## Reproducer and repair

A 91×32 fixture has independently generated orthogonal left/right factors,
largest singular value 1e-3, and eight requested singular values decreasing
geometrically to 2e-10. Twenty-four smaller values are unrequested. A full
normal basis, three-cycle budget and original-equation tolerance 1e-11 yield:

| Diagnostic | Before | With guard |
|---|---:|---:|
| Maximum original SVD-equation residual | 3.907e-13 | 3.907e-13 |
| Maximum entrywise left Gram error | 3.841e-8 | 3.841e-8 |
| Left/right accepted counts | 8 / 8 | 0 / 0 |
| Status | 0 | 3 |

The old result violates the external 1e-8 orthogonality gate, although it lies
above the existing normal-eigenvalue rejection floor. This was not an accepted
historical benchmark: the external validator already checks orthogonality.
[Original sweep](../../results/performance_followup/normal-conditioning-old.json),
[guarded sweep](../../results/performance_followup/normal-conditioning-fixed.json).
The exact fixture is in `test_normal_svd_rejects_ill_conditioned_lift_and_recovers`.

Finalization now computes both current Gram matrices in Warp. Device array
`orthogonality_errors` contains their maximum entrywise deviations from identity.
`orthogonality_tol` defaults to 1e-8 and must be finite and positive. Nonfinite
Gram entries or errors exceeding that tolerance set status 3 and clear both
accepted-triplet counts. Existing status 2 for unresolved requested normal
values takes precedence. Equation residual arrays remain available. This is
rejection, not a repair of ill-conditioned singular vectors or a rank estimate.
Initialization resets status, so later well-resolved input can recover on the
same captured graph. No host numerical decisions, synchronization or fallback
are introduced; production augmented `PartialSVD` is unchanged.

Seventeen new cases cover tall/wide 96-snapshot spaces with full/partial bases,
complete repeated requested spaces, scaled ill-conditioned lifts, restoration
of valid inputs, nonfinite Gram diagnostics and invalid tolerances. Together
with prior normal-product and transpose replay coverage, **39 tests pass in
4.33 s** on pinned Warp and **39 in 5.53 s** on stock Warp 1.15. Graph tests reject
HOST callbacks and host/device transfers. This is focused validation after the
full 691-test run, not a full upstream conformance rerun.

## Timing controls and wider comparison

Sequential NVIDIA L40 runs, pinned Warp environment, three warm replays per
method, same independent spectrum/subspace/orthogonality/reconstruction and
paired-equation gates. Every recorded row passes all gates; Warp reports full
counts and status 0 on every trial. Occupancy samples are clean. Setup,
capture, references and validation are outside replay timings and recorded
separately. These are warm-solve comparisons, not first-use speedups.

The unchanged 100,000×48 clustered k=20 fixture takes **15.416 ms** before the
guard and **15.763 ms** after, about **2.3%** overhead. Both use the full
48-vector normal basis. [Before](../../results/performance_followup/normal-guard-old-control.json),
[after](../../results/performance_followup/normal-guard-fixed-control.json).

A wider 100,000×128 fixture retains the complete threefold leading singular
space and two unrequested zero singular values:

| Method | Basis width | Median ms |
|---|---:|---:|
| Guarded normal Warp, partial basis | 64 | 434.726 |
| Guarded normal Warp, full basis | 128 | 74.955 |
| Augmented Warp | 48 | 378.414 |
| Native CuPy svds | 48 | 159.695 |
| SciPy ARPACK svds | 48 | 1968.636 |
| Spectra | 48 | 1803.602 |

The partial normal profile takes 20 cycles and 19 independent recoveries.
Full normal space avoids restart exploration; the guard is unchanged. Thus
normal products alone do not fix adaptive-throughput regressions: appropriate
explicit basis width is consequential. A full basis costs more storage and
setup and is useful only when the smaller matrix dimension is manageable.
The full-basis row is a subsequent same-session run on the identical matrix;
native rows come from the preceding all-method comparison. The reports include
matrix hashes, actual cycles, strict counts and original equation checks.
[All-method comparison](../../results/performance_followup/normal-guard-wide-snapshots.json),
[full-basis follow-up](../../results/performance_followup/normal-guard-full128.json).
Sibling `.occupancy.json` files preserve monitoring evidence.

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.experimental.normal_pod --case clustered --size 100000 \
  --modes 20 --snapshots 128 --ncv 64 --reference-ncv 48 --steps 100 \
  --repeats 3 --compare cupy,scipy,spectra,warp --out build/normal128.json
# Repeat with --ncv 128 --compare '' for the full-basis follow-up.
```

The original augmented/adaptive regression and general sparse throughput work
remain open in #31. No default formulation or independent-chain guard changes.
