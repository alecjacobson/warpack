# Nonfinite aggregate acceptance (#188)

Twelve benchmark/validation drivers contained gates that reduced error lists with Python `max`. A later NaN can be overlooked (`max([0.0, float("nan")]) == 0.0`). Every measurement is now explicitly required to be finite, nonnegative, and below the unchanged threshold (preserving strict versus inclusive comparisons), with nonempty evidence and a finite positive threshold. This changes host-side validation only; no solver, timing region, numerical tolerance or historical result was changed.

Affected drivers: `experimental/compare_paired_rotation_tiles.py` (5e-13), `experimental/tune_sparse_layout.py` (5e-6), and `applications/refinement.py` (1e-12). Their saved evidence was audited independently: 360 paired checks (720 scalar errors), 6 sparse-layout checks (12 scalar errors), and 24 inverse-refinement checks (72 scalar errors). All 804 scalar errors were finite and nonnegative. Every acceptance decision agrees with the corrected gate, including previously rejected inverse solves. Original hashes, thresholds, error values and decisions are retained in `audit.json`.

74 benchmark acceptance/process tests pass, including NaN, both infinities, negative errors, exact-boundary/over-bound errors at every position, empty evidence, invalid thresholds and valid zero/small errors. No performance rerun is needed for this validation-only correction.

The paired screen's original source is committed at `af3731a`. Its source hashes describe that as-run code, not this follow-up fix. Other historical sources and results remain unchanged.


The broader direct-gate scan also corrected triangular rotation checks, canonical first-metric checks, fresh-eigenpair checks, the block comparison's returned/full CuPy residual gates, both conformal CuPy stopping gates, the negative-spectrum CuPy reproduction, cycle-floor physical residuals, and the dragon validator's elastic spectrum check. The skinning validator already checks all metrics for finiteness before its maximum and needed no change. Matrix construction cutoffs and diagnostic maxima were not reinterpreted as acceptance gates.

`extended-audit.json` checks the corresponding raw recorded fields in 230 retained JSON files: 36,230 scalar measurements are finite and nonnegative. This supports the absence of a hidden NaN in those recorded groups; it does not certify unrecorded intermediates or make rejected runs valid. The original results remain intact. Inclusive-gate regression tests additionally preserve equality while rejecting nonfinite, negative and empty evidence.
