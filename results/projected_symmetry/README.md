# Restoring projected symmetry (#170)

The repeated/null-spectrum component discovered in #169 is a 612-square PSD
matrix with a large nullspace. The installed scalar Jacobi path exhausts its
30-sweep budget with normalized original-equation residual **1.14203e-11**, above
the unchanged 1e-11 component gate. This does not establish an accepted wrong
public FEM result: the outer physical and native stopping checks remain required.

Two controlled pure-Warp experiments distinguish a larger budget from restoring
symmetry during iteration. Both retain the original rotation schedule and
2e-15 diagonal-scaled stopping criterion:

| Policy, seed159 matrix | Sweeps | Original relative residual | Max off-diagonal | Internal target |
|---|---:|---:|---:|---:|
| Original, limit30 | 30 (budget) | 1.1420e-11 | See raw diagnostic | 3.04e-13 |
| Original, limit60 | 60 (budget) | 1.8936e-13 | 3.45e-12 | 3.04e-13 |
| Restore symmetry after each sweep, limit30 | 20 | 1.8928e-13 | 2.92e-14 | 3.04e-13 |

Simply doubling the budget improves the externally measured residual but still
does not meet the internal off-diagonal stopping criterion. Restoring symmetry
reaches that criterion within the existing budget, including rescaled inputs
1e-80 and1e80. This is consistent with roundoff skew interfering with the
symmetric iteration. It does not change the requested outer tolerance.

**24 tests pass on each of pinned and stock Warp.** They cover the failing
matrix, changed shifted input, rescaling, indefinite and repeated/null spectra,
strided workspaces, nonfinite recovery, and matching arithmetic between the
30/60-budget controls on already-converged cases. The symmetry-only candidate
keeps the30-sweep bound. Prototype code remains outside the installed library.

The diagnostics retain source/dependency hashes, exact procedural input
identities and all measurements. Their single replay durations are diagnostic,
not qualified performance comparisons. `budget-only-source.py.txt` reproduces
the helper hash before the symmetry option was added.

Fresh monitored component controls are in `components-*.json`. The indefinite
worker passes both backends. The repeated/null worker remains rejected because
of the old baseline miss at612; no speed ratio uses that rejected worker.
Complete-FEM and first-solve controls are being qualified separately in `fem/`.
No main-library behavior or six-backend Anvil table has changed in this experiment.

Reproduce components with:

```bash
python -m benchmarks.experimental.projected_symmetry_compare \
  --kind rank_deficient --sizes 104,208,416,612 --out <new-report.json>
```

Use the pinned runtime and `benchmarks.monitor_gpu` for timing. Each raw worker
records source identities, dependency hashes, graph inspection and all original
first/later accuracy checks. The complete-solve wrapper is
`benchmarks.experimental.compare_projected_symmetry --projected-backend default|symmetric`
with the usual public `compare_block` options.
