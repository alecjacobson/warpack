# Constraint-value replay repair

Issue [#74](https://github.com/alecjacobson/warpack/issues/74) exposed stale
constraint transposes and row norms in `ConstrainedKrylovSchur`. Forward
constraint products used current values, but multiplier equilibrium checks
used construction-time values. On a 12-dimensional fixture, doubling C and
updating the KKT inverse left the true equilibrium residual at 2.119e-15,
yet the solver reported residuals up to 0.146 and zero of three converged modes.
The repaired captured replay reports all three converged, with true equilibrium
residual 4.777e-15. Independent references validate the returned eigenpairs;
they are not supplied as starting vectors.

[Original captured reproducer](../../results/performance_followup/constrained-stale-gpu.json),
[repaired replay](../../results/performance_followup/constrained-refreshed-gpu.json),
and [isolated CPU finalizer diagnostic](../../results/performance_followup/constrained-stale-cpu.json).
The CPU diagnostic is not a full CPU eigensolver test.

Initialization now refreshes the fixed-topology transpose and constraint row
norms on device. The transpose mapping is shared with SVD through the new
internal `sparse_utils` module; its algorithm is unchanged. Row norms use
parallel partial sums to avoid serial work over long skinning constraints.
There is no replay topology reconstruction or host readback. Values may change
at fixed addresses and topology; callers must update their KKT inverse too.

## Regression coverage

Eleven new cases exercise captured value changes and restoration, non-invariant
constraints, coupled SPD mass, repeated requested eigenspaces, ordinary and
interior shifts, explicit zeros becoming nonzero, irregular and empty rows,
long rows, and unused NaN padding. Full solves check the nearest spectrum,
complete metric subspace, orthogonality, original equilibrium and feasibility,
reported residual agreement, full convergence counts, and unchanged caller
inputs. The small exact dense inverse is a test fixture exposed as a Warp
operator, not a new production backend. Pure-Warp graphs contain no HOST,
HtoD or DtoH nodes.

The pinned focused suite passes **48 tests in 6.75 s**. Stock Warp 1.15 passes
**63 tests in 12.70 s**, including the additional ordinary SVD suite.

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m pytest -q tests/test_constrained_replay.py tests/test_constrained.py \
  tests/test_svd_replay.py tests/test_normal_svd_experimental.py
```

## Unchanged-input skinning controls

Sequential L40 runs use 20 requested modes, two fixed cycles and three timed
replays, with matching settings and cached host references. Both sizes pass
all original accuracy gates and report 20 converged modes. Occupancy samples
show no competing GPU processes. These controls measure repair overhead;
they do not replace the earlier all-backend showcase comparison.

| Physical dimension | Before, median ms | Repaired, median ms | Ratio |
|---|---:|---:|---:|
| 2,025 | 53.3474 | 53.3225 | 0.9995 |
| 45,625 | 287.2308 | 287.7785 | 1.0019 |

The larger repaired case reports equilibrium residual 3.479e-13 and constraint
residual 5.138e-17. Graphs add three kernels and retain the same transfer counts.
cuDSS still contributes its known 420 captured HtoD metadata transfers; this
repair does not resolve [#29](https://github.com/alecjacobson/warpack/issues/29).
There are no HOST callback or DtoH nodes.

[Small before](../../results/performance_followup/constraint-refresh-side8-old.json),
[small repaired](../../results/performance_followup/constraint-refresh-side8-fixed.json),
[large before](../../results/performance_followup/constraint-refresh-side24-old.json),
[large repaired](../../results/performance_followup/constraint-refresh-side24-fixed.json).
Sibling `.occupancy.json` files preserve the monitor records. Runs use
`benchmarks.applications.skinning_compare --side 8` (or `24`), `--steps 2
--repeats 3 --methods warp-cudss`, with the old report passed as `--baseline`
to the repaired run. No new large CPU factorization was performed.
