# Inverse-spectrum SM selection and explicit accuracy profiles

2026-10-19 follow-up to [#6](https://github.com/alecjacobson/warpack/issues/6).
This is an accuracy-policy result using the existing solver, not a new harmonic
Ritz algorithm or a faster default. Production arithmetic and default basis
sizes are unchanged.

For a physical eigenvalue lambda and shift sigma, the iteration eigenvalue is
`theta = 1/(lambda-sigma)`. Selection acts on theta:

| Inverse selection | Physical request |
|---|---|
| LM | Nearest sigma |
| SM | Farthest from sigma |
| LA | Largest algebraic theta; normally closest above sigma |
| SA | Smallest algebraic theta; normally closest below sigma |
| BE | Both algebraic ends of theta, with the odd extra at the larger end |

The LA/SA descriptions depend on which sides of the shift contain eigenvalues;
the theta ordering is the actual contract. This is distinct from the primary
low-mode FEM shift-invert request, which selects inverse **LM**. Original-system
SM and inverse-system SM must not be interchanged.

## What passes, and what remains a numerical limit

The two required-success failures, `SymEigsShift_3` (SA) and `_14` (BE), pass
with original k/ncv, 100 cycles, requested internal tolerance 1e-12 and the
original 1e-9 absolute assertion. This confirms the previously merged restart
fixes. The two compact transformed-SM cases still fail; upstream explicitly
allows nonconvergence for those sections. Their strict failure labels remain.

Explicit wider profiles recover the requested farthest physical modes:
`_2` uses ncv8 rather than 6; `_7` uses ncv48 rather than 20. Both retain 100
cycles and unchanged tolerances. All inverse applications use cuDSS with the
established symmetric factor profile and one Warp residual correction; CG is
not involved. The final runner additionally checks independent physical
spectrum, invariant subspace and orthogonality, which the generic upstream
residual assertion alone did not establish.

| Case / selection | ncv | Outcome | Absolute residual | Spectrum error | Inverse / physical count |
|---|---:|---|---:|---:|---:|
| SymEigsShift_2 / SM | 6 | failed | 9.214e-02 | 4.956e-03 | 0 / 0 of 3 |
| SymEigsShift_2 / SM | 8 | passed | 1.821e-14 | 2.309e-14 | 3 / 3 of 3 |
| SymEigsShift_7 / SM | 20 | failed | 2.052e+00 | 9.854e+00 | 0 / 0 of 10 |
| SymEigsShift_7 / SM | 48 | passed | 5.160e-13 | 1.414e-12 | 10 / 10 of 10 |
| SymEigsShift_3 / SA | 6 | passed | 1.898e-14 | 3.864e-14 | 3 / 3 of 3 |
| SymEigsShift_14 / BE | 50 | passed | 6.257e-13 | 4.505e-12 | 20 / 20 of 20 |


The wider profiles are validated against independent dense references and
captured replay with two changed starting vectors. Tests independently define
the physical target by distance from sigma, rather than reusing the reporting
selector. They also check status, full inverse and original-equation counts,
original residuals, orthogonality and complete requested subspaces. Graph DOT
audits find no HOST callback or DtoH node; vendor cuDSS HtoD metadata copies
remain tracked in #29.

## Rejected narrower attempts and native controls

A width40 screen on `_7` passes the default start at 100 cycles, but a changed
start returns only 9/10 inverse and physical counts, residual 2.696 and one
wrong requested mode. It is **incomplete**, not a false full-count success.
Continuing that diagnostic to 200 cycles recovers all ten. Width48 passes all
three diagnostic starts at 100 cycles; the permanent tests then validate two
changed starts in one public fixed-budget capture. The final profile therefore
uses width48, not the favorable width40/default-start result.

Width32 remains incomplete. A width60 capture was interrupted after width40
had passed the initial default-start screen; width100 was never attempted.
Neither unfinished profile is classified as a numerical failure or accepted
result. The diagnostic ten-cycle continuation graphs are not performance
measurements or a new host-controlled production stopping mechanism.

The unmodified native Spectra SM section passes at n=10 with ncv6. At n=100,
the dense ncv20 section reports only 3/10 after 501 iterations and returns
through its explicit allowed-failure branch. Catch's overall exit code is zero
because that nonconvergence is permitted, not because the dense solve passed.
The sparse sections at both sizes pass. These controls do not imply identical
starting vectors, iterations or behavior between algorithms.

No claim is made that ncv8/48 guarantees arbitrary-matrix convergence. Interior
Ritz selection can be slow or depend on the start; increasing the basis/budget
and checking the original equations remains necessary. Existing failures are
preserved rather than relabeled under a looser tolerance.

## Records and reproduction

- [Final six profiles / four fixtures](../../results/performance_followup/shifted-selection-final.json).
- [Initial width screen](../../results/performance_followup/shifted-selection-screen.json).
- [Changed-start continuation screen](../../results/performance_followup/shifted-start-screen.json)
  and [rejected width40 replay](../../results/performance_followup/shifted-changed-start-diagnostic.log).
- [Native small](../../results/performance_followup/shifted-native-small.log)
  and [native medium](../../results/performance_followup/shifted-native-medium.log) controls.
- [Validation and provenance](../../results/performance_followup/shifted-validation.json).

The final family adapter captures and replays twice, then validates the last
replay. Permanent regressions check each changed-start replay. All results here
are accuracy diagnostics, not accepted timing comparisons or a new full
CuPy/Spectra conformance run. **58 focused tests pass** across transformed
selection, multiplicity and reporting contracts. Five additional CPU selector
checks preserve unshifted Hermitian ordering; lint and formatting pass.

```bash
python -m benchmarks.upstream.prepare_spectra --families SymEigsShift \
  --out build/shifted-selection/fixtures
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.upstream.shifted_selection --out build/shifted-selection.json
python -m pytest -q tests/test_transformed_selection.py tests/test_multiplicity.py \
  tests/test_upstream_contracts.py
build/upstream-spectra-build/test/SymEigsShift '*10x10*' -c 'Smallest Magnitude'
build/upstream-spectra-build/test/SymEigsShift '*100x100*' -c 'Smallest Magnitude'
```

The 60 KB generated fixture archive preserves the exact two small matrices,
with source provenance in [the fixture notes](../../tests/fixtures/README.md).
Large exported matrices remain local build artifacts.
