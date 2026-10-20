# Cooperative complex projected Schur solve

Projected-solve optimization for [#14](https://github.com/alecjacobson/warpack/issues/14).
The implementation uses pure-Warp tiled Householder reduction, Givens QR and right
back substitution. Shifts, deflation tolerance and bounded QR iteration policy
follow the serial reference. It preallocates all workspaces and supports nested
device-conditional capture. It contains no vendor dense eigensolver or host
convergence test. The public general solver selects it for CUDA projection sizes 16–64 when
conditional graphs are supported; smaller/larger projections and other
devices retain the serial reference. The standalone component supports 1–64.
Eager solves use a prebuilt device QR loop; captured solves record that loop
inside the outer graph. Setup synchronization precedes graph creation to
contain the driver-570 upload issue tracked in #61.

Twenty tests pass on pinned Warp 1.18 (2.72 s) and stock Warp 1.15 (22.44 s,
including compilation). Tests cover scaled real/complex matrices, changed
replay, identity/zero/repeated/defective spectra, bounded QR failure, nonfinite
recovery, and saved 50-by-50 projections inside an outer conditional loop.
Independent SciPy references and residual/Schur-similarity/unitarity checks are
outside timed solves. Captured graphs have no host nodes or host transfers.

| Projection | Serial median ms | Tiled median ms |
|---|---:|---:|
| Saved direct, n50 | 199.707 | 17.990 |
| Saved shifted, n50 | 199.635 | 17.990 |
| Random complex n6 | 0.633 | 0.539 |
| Random complex n16 | 8.474 | 2.515 |
| Random complex n32 | 60.773 | 8.495 |
| Random complex n50 | 222.827 | 21.755 |
| Random complex n64 | 458.646 | 31.850 |

These are **component timings**, not whole sparse solves. Every case passes
independent spectrum, backward-error, normalization and Schur checks. Each
method uses the same input, one warmup, three timed batches of three replays,
resetting its projection each replay. Timing excludes setup/capture/reference
work. GPU process monitoring observed no competing work. Numerical gates and
raw samples are in [results](../../results/schur-components.json) and
[occupancy](../../results/schur-components.occupancy.json).

## Integrated general results

The original #35 matrices, k20/ncv50, requested tolerance 1e-12 and cycle
budgets are unchanged. Three warmed synchronized replays pass all physical
and transformed counts, original residuals, requested-spectrum and vector-norm
gates. These are matched **accepted** comparisons to the public residual-directed
restart with the serial Schur component, not to the earlier failed baseline.

| Problem | Serial Schur, full solve s | Tiled Schur, full solve s | Speedup |
|---|---:|---:|---:|
| Direct complex SR, n1000 | 17.4965 | 3.2161 | 5.44× |
| General shift-invert, sigma100, n1000 | 6.6888 | 1.2877 | 5.19× |

Direct trials use 80–82 of the original 300 adaptive cycles, with worst
normalized original residual 9.77e-13. Shifted trials use the original 30 fixed
cycles and reach 1.44e-14. The numerical stopping rule is unchanged; parallel
rounding changes the precise cycle counts. Direct replay has no host nodes or
host transfers. Shifted replay retains exactly the same 6,172 vendor HtoD copies
tracked in #29, with no host nodes or device-to-host transfers.

[Direct raw results](../../results/schur-general-direct.json),
[shifted raw results](../../results/schur-general-shift.json), and
[serial accepted baselines](GENERAL_RESTART.md) include source/runtime hashes.
All 20 upstream general benchmark workers now finish within their original
180-second worker limits. The stronger per-trial checks accept 11 and reject
9; eight rejected rows report full residual counts but miss requested LM
modes, and one also has a short count. Failed rows are not performance wins.
[Complete inventory](../../results/schur-upstream-benchmarks.json).
The native upstream CPU diagnostic is retained separately: its ARPACK wrapper
reports residual=-1, and Spectra reports residuals without full requested-set
checks. Those timings are not certified matched-accuracy comparisons.

The exact n100 `benchmark_gen_24` also misses the requested negative real mode
with the serial projection and the old restart (129b82d), so this is not caused
by tiled Schur. Its basis orthogonality error is about 1e-15 and projected
backward error below 1e-15. Three variants stop after 46 cycles with ten genuine,
independent eigenvectors, but return the 11th/12th-magnitude eigenvalue instead
of the 10th. Fixed 80 cycles still miss it; fixed 300 recovers it. These controls
identify premature target selection, not a defective projected eigensolve.
[Serial control](../../results/general-selection24-serial.json),
[old restart](../../results/general-selection24-prior.json),
[fixed 300 control](../../results/general-selection24-fixed300.json).
The remaining selection work is tracked in #104. All three formerly timed-out Spectra workers also finish and pass their
original assertions and strict counts: `GenEigs_12`, `GenEigs_33`, and
`ComplexEigs_33`. Both SM cases additionally check the requested spectrum and
subspace; the generic direct LM upstream assertion does not. These are bounded
accuracy-only runs, not matched timing comparisons. A representative large
sparse performance comparison remains required under #14. These
two accepted timing comparisons do not establish universal speedups.
The implementation's experimental checkpoint is commit `2cdd51d`.

```bash
python -m pytest -q tests/test_tiled_schur.py
python -m benchmarks.experimental.compare_schur --out build/schur-components.json
```

## Final local validation

After the separate [Hodge test-budget correction](HODGE_COUNT_RESERVE.md), the
full pinned suite passes **1,167 tests** (462.34 s). The general/projected
subset passes 119 on stock Warp 1.15 (30.26 s). Lint, formatting and a clean-source
wheel check pass; all 24 installed core modules match source, including tiled
Schur, while benchmark/example/test modules remain outside the wheel. This
local result does not reclassify the nine stronger upstream selection failures.
[Validation](../../results/performance_followup/tiled-schur-validation.json),
[wheel](../../results/performance_followup/tiled-schur-wheel.json).
