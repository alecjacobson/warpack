# General largest-magnitude selection audit

A small residual verifies an eigenpair; it does not establish that every requested
extremal eigenvalue was found. This affects the original Spectra benchmark
settings as well as Warpack. The native audit below supplements the earlier
residual-only CPU timing harness; it does not retroactively certify those timings.

All 20 original nonsymmetric benchmark matrices and original starts are checked
against a dense SciPy eigendecomposition. Native Spectra uses the pinned revision
`db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95` and its C++ `DenseGenMatProd` without
Python operator callbacks. SciPy 1.15.0 exercises both real ARPACK and complex
ARPACK with identical real inputs cast to complex128. CuPy has no sparse general
`eigs` implementation.

Each run requests ten LM eigenpairs, tolerance `1e-10`, and at most 1,000 native
restart iterations. These native iterations are not equivalent to Warp cycles.
The original dimensions are n=100/ncv=20 and n=1,000/ncv=30, ten matrices each.
Wider search spaces are explicit alternate profiles, with unchanged inputs,
starts, requested count, tolerance, and iteration cap.

| Profile | ncv at n=100 / n=1,000 | Spectra | Real ARPACK | Complex ARPACK |
| --- | --- | --- | --- | --- |
| Original | 20 / 30 | 14/20 | 15/20 | 14/20 |
| Wide | 40 / 60 | 19/20 | 19/20 | 20/20 |
| Very wide | 80 / 120 | 20/20 | 20/20 | 20/20 |

These are accuracy pass counts, not performance measurements or guarantees for
arbitrary inputs. All original-profile failures have tiny residuals and full
reported convergence counts. The very-wide profile supplies an accepted native
reference configuration for these 20 fixtures; a timing comparison using it
must separately measure those settings and validate every timed replay.

The validator checks original residuals, eigenvalue assignment, unit vector
norms, numerical independence, and the allowed invariant subspace. At a
conjugate tie at the cutoff, either member is allowed, but every eigenvalue
strictly larger than that cutoff must be represented. This catches a subtle
case where complex ARPACK on `benchmark_gen_30` returns both boundary members
and omits a larger conjugate partner: all returned eigenvalues individually
belong to the allowed spectrum. Independent, nonorthogonal eigenvectors are
accepted; duplicate vectors for repeated eigenvalues are rejected. Numerical
independence uses the standard dimension-scaled machine-precision singular-value
threshold, not an orthogonality requirement. Defective problems require a
separate invariant-subspace contract.

On `benchmark_gen_24`, native Spectra and real ARPACK omit the real eigenvalue
-5.6551521684 and include a complex eigenvalue of magnitude 5.651714 instead.
Both report all ten pairs converged, with maximum absolute residual about
1.40e-10. On `benchmark_gen_30`, both real native libraries pass at original
settings while Warp's original profile fails. The native limitations therefore
do not account for all of Warp's failures in
[#104](https://github.com/alecjacobson/warpack/issues/104).

The [complete results](../../results/general-native-selection-audit.json) record
all 180 case/profile/backend combinations, eigenvalues, numerical gates, matrix
and start hashes, source hashes, native build flags and revision, and runtime
versions. No CPU or GPU timings are claimed. This closes the reference-coverage
work in [#105](https://github.com/alecjacobson/warpack/issues/105), while the
benchmark independence fix is tracked in
[#106](https://github.com/alecjacobson/warpack/issues/106). No upstream bug report
asserts that residual convergence can guarantee global spectral completeness.

From the repository root, after exporting the original fixtures as described in
[the harness instructions](README.md):

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.upstream.check_general_references \
  --profiles original wide very-wide \
  --out build/general-native-selection-audit.json
python -m pytest -q tests/test_general_reference.py
```

This accuracy-only command needs NumPy, SciPy, Eigen headers and a C++ compiler;
it builds a small native Spectra extraction library locally. It does not run the
old Fortran timing harness. `--fixtures`, `--spectra-source`, `--library`, repeated
`--case`, and `--backends` allow a focused reproduction without overwriting the
published result. The two widening policies are `min(n, max(2*ncv, 4*k))` and
`min(n, max(4*ncv, 4*k))`.
