# Current library validation after projected symmetry recovery

This historical baseline is superseded by the [projected column-normalization
validation](../normalized_jacobi/VALIDATION.md).

On 2026-10-23, pinned Warp/FSAI runtime, L40 GPU and four Xeon cores:

- **Full suite: 2,445 passed, 11 skipped**, 805.37 seconds.
- Affected production suite: **171 passed**, 59.21 seconds.
- Stock Warp affected suite: **168 passed, 3 expected upstream-feature skips**,
  56.13 seconds.
- A clean wheel contains exactly 39 core modules matching source. An isolated
  installed-wheel smoke passes changed-input graph replays while forbidding
  imports of benchmarks, examples, SciPy, CuPy and the optional cuDSS package.

[Full transcript](validation/full-tests.log),
[pinned focused transcript](validation/pinned-focused-tests.log),
[stock focused transcript](validation/stock-focused-tests.log),
[wheel verification](validation/wheel.json),
[isolated wheel replay](validation/wheel-smoke.json),
[source identities](validation/source.json).

The production regressions cover ordinary and compact projected graphs on
indefinite and repeated/null spectra, changed shifted input, extreme rescaling,
strided buffers and convergence of the recorded 612-dimensional failure under
the unchanged 30-sweep bound and 2e-15 diagonal-scaled criterion. Compact and
ordinary graph policies still produce identical results with identical round
order. Existing complete-suite checks also cover original sparse equations,
requested spectra/subspaces, mass orthogonality, repeated/null modes, changed
input/mass/constraints, callbacks, nonfinite recovery and graph residency.

This replaces the earlier 2,214-pass baseline as the current full-suite evidence.
It does not claim complete CuPy/Spectra API conformance, eliminate known open
issues, or turn correctness tests into performance comparisons. Fresh full-FEM
performance and accuracy controls are recorded separately in this result family.
