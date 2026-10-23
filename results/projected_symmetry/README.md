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
| Original, limit30 | 30 (budget) | 1.1420e-11 | 4.15e-9 | 3.04e-13 |
| Original, limit60 | 60 (budget) | 1.8936e-13 | 3.45e-12 | 3.04e-13 |
| Restore symmetry after each sweep, limit30 | 20 | 1.8928e-13 | 2.92e-14 | 3.04e-13 |

Simply doubling the budget improves the externally measured residual but still
does not meet the internal off-diagonal stopping criterion. Restoring symmetry
reaches that criterion within the existing budget, including rescaled inputs
1e-80 and 1e80. This is consistent with roundoff skew interfering with the
symmetric iteration. It does not change the requested outer tolerance.

**24 tests pass on each of pinned and stock Warp.** They cover the failing
matrix, changed shifted input, rescaling, indefinite and repeated/null spectra,
strided workspaces, nonfinite recovery, and matching arithmetic between the
30/60-budget controls on already-converged cases. The symmetry-only candidate
keeps the 30-sweep bound. Prototype code remains outside the installed library.

The diagnostics retain source/dependency hashes, exact procedural input
identities and all measurements. Their single replay durations are diagnostic,
not qualified performance comparisons. `budget-only-source.py.txt` reproduces
the helper hash before the symmetry option was added.

Fresh monitored component controls are in `components-*.json`. The indefinite
worker passes both backends. The repeated/null worker remains rejected because
of the old baseline miss at 612; no speed ratio uses that rejected worker.
Complete-FEM controls are recorded separately in `fem/` (see below).
The prototype did not change main-library behavior or the six-backend Anvil table.
The subsequently qualified [production integration](production/README.md) is now
covered by the current full-suite validation.

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


## Complete prototype controls

All six fresh workers pass the first captured output and three subsequent
replays: **24 accepted complete solves**, with exact paired input/source/dependency
identities and clean external occupancy. All native stopping and original
physical/spectrum/subspace gates remain enforced.

| Problem | Capture s, baseline / symmetry | First-solve compute s, baseline / symmetry | Warm replay median s, baseline / symmetry |
|---|---:|---:|---:|
| Anvil | 0.6032 / 0.6276 | 3.0044 / 3.0803 | 1.6068 / 1.6122 |
| 100 modes | 1.6393 / 1.6429 | 5.2224 / 5.2066 | 3.0309 / 2.9746 |
| 200 modes | 4.7800 / 4.7794 | 16.4518 / 16.6340 | 10.4664 / 10.6381 |

The benefit is fixing projected convergence; these timings do not establish a
universal warm-solve speedup. Finite-precision changes also affect iteration
trajectories. Anvil's global projected dimensions 48..72 participate in this
control, unlike the unchanged Anvil path in the earlier block-initializer study.

First-solve compute = setup + capture + validated first captured solve, with
cached kernels and excluding mesh/reference work, imports and host validation.
It is not total process wall time. Setup/capture are single observations per
fresh worker and warm medians contain three solves. This does not replace the
six-backend Anvil table or provide a new CuPy ratio. The [production controls](production/README.md) qualify both ordinary and compact
graph paths with the same accuracy criteria.
