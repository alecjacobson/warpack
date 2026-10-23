# Force-aware reduced bases

This checkout-only application composes public Warpack operators to extract
dominant displacement covariance modes. Physical assembly and CPU references
remain outside the installed solver. See [#135](https://github.com/alecjacobson/warpack/issues/135)
for coverage and [the snapshot comparison](WEIGHTED_SNAPSHOT_SVD.md) for low-rank
force ensembles.

For symmetric positive-definite stiffness `H` and mass `M`, and a symmetric
positive-semidefinite force prior `S`, displacement covariance is
`C = H^-1 S H^-1`. Its mass-weighted optimal basis solves

```
(M C M) u = lambda M u,    U.T M U = I,    largest lambda first.
```

`C M` alone is generally not Euclidean symmetric. The application presents the
symmetric generalized pencil to `GeneralizedKrylovSchur`; it does not construct
a dense covariance. Each application uses sparse products and two stiffness
solves. Regular generalized iteration additionally solves with the mass matrix.
Both coupled and diagonal SPD mass are supported. A singular force prior is
allowed and is never inverted; the requested positive mode count must fit its
effective rank. A singular mass requires a different formulation.

## Usage

Given assembled device FP64 scalar sparse matrices `H`, `M`, and `S`:

```python
import warp as wp
from benchmarks.applications.force_covariance import ForceCovarianceSolver

driver = ForceCovarianceSolver(H, M, S, k=20, backend="cudss", ncv=64)
try:
    driver.capture(cycles=4)
    driver.solve()
    values = driver.solver.eigenvalues
    modes = driver.solver.eigenvectors  # rows, in physical coordinates
    # Inspect status, count and original physical accuracy outside replay.
finally:
    wp.synchronize_device(H.device)
    driver.close()
```

Use `backend="cg"` for pure Warp, with upstream FSAI from the
[pinned overlay](../../docs/installation.md#fast-block-solves). Inner CG uses
FP64 arithmetic and FP32 FSAI factors. H and M are fixed for this recipe; S's
values can change at fixed topology/storage addresses. Capture initializes the
first graph upload during setup. Replay has no host numerical decisions or
residual transfers. cuDSS retains the documented vendor metadata HtoD copies.

cuDSS requires a fixed outer budget. Pure Warp defaults to adaptive capture;
`capture(cycles=..., adaptive=False)` explicitly chooses a fixed budget. Native
residual counts do not alone establish requested-spectrum completeness. The
benchmark records the Krylov exploration guard separately: a budgeted run may
pass an independent full requested-subspace reference while the adaptive guard
would still demand exploration. Such a run is labeled **fixed**, never reported
as an adaptively completed solve. No exploration safeguard is disabled in core.

## Choose the formulation

| Force prior | Useful route |
|---|---|
| `S = F F.T`, few force columns | Solve `G = H^-1 F` once, then use [Gram reduction or weighted snapshot SVD](WEIGHTED_SNAPSHOT_SVD.md). Repeated implicit applications usually waste those solves. |
| `S = M` | Ordinary elastic modes of `(H,M)` already give this basis; covariance eigenvalues are inverse squared elastic eigenvalues. The implicit case is a cross-check. |
| Full-rank nonuniform prior | The implicit generalized covariance route handles sparse S without constructing C. |
| Invertible S with cheap inverse | The reciprocal pencil `(H S^-1 H,M)` is another candidate, requesting the smallest eigenvalues. It squares stiffness conditioning and requires its own accuracy/performance qualification. |

Small tests check the complete requested subspace against dense generalized
`eigh`, spectrum, M-orthogonality, expected captured energy, and operator
self-adjointness on probes spanning ten orders of magnitude. They exercise
changed-prior first replay, rank exhaustion and recovery, both mass categories,
both inverse backends, and mixed cuDSS-H/CG-M solves. The last combination
exposed and now regression-tests [capture capability propagation](https://github.com/alecjacobson/warpack/issues/150).

The large procedural cantilever comparison uses 10,890 unknowns and 20 modes.
Every warmup and repeat faces physical and independent subspace gates at `1e-8`,
plus native status/count and fresh inner residual checks. CG requests recursive
relative tolerance `1e-13`; large runs can miss that strict tolerance when the
residual is recomputed (about `2.7e-11` in the screening case). Fresh misses are
retained, and acceptance explicitly uses `1e-8` for the fresh inner residual
and original covariance checks. This is not a claim of fresh `1e-13` accuracy.

Run the comparator and strict report on an idle GPU; setup, capture and solve
are separate fields. The reporter also needs `environment.json` beside the raw
file, recording hardware, runtime, thread count and affinity:

```bash
python -m benchmarks.monitor_gpu --out run.occupancy.json --log run.log -- \
  python -m benchmarks.applications.force_covariance_compare \
  --backend cg --prior diagonal --side 10 --modes 20 --ncv 64 \
  --cycles 1 --fixed-cycles --repeats 3 --out run.json
python -m benchmarks.applications.force_covariance_report run.json
python -m pytest -q tests/test_force_covariance.py tests/test_force_covariance_report.py
```

Use `--backend arpack` without `--fixed-cycles` for the CPU implicit comparator.
It constructs its own SuperLU factors; reference factors are not reused for
measured setup. This is a matched implicit-operator baseline, **not execution
of the paper authors' implementation**. [Evidence and qualification status](../../results/force_covariance/README.md)
distinguish screening, fixed budgets and completed adaptive exploration.

## Relation to the published method

[Force-Dual Modes: Subspace Design from Stochastic Forces](https://arxiv.org/html/2505.23969v1)
derives force-aware mass-weighted bases and low-rank snapshot reductions. The
author's public [implementation](https://github.com/otmanon/simkit/blob/262db01a944214d83376fee285a471af72e14ac1/examples/force_dual_modes/force_dual_modes.py)
has a diagonal-prior reciprocal-pencil path and a weighted dense SVD path.
Its [eigensolver helper](https://github.com/otmanon/simkit/blob/262db01a944214d83376fee285a471af72e14ac1/simkit/eigs.py)
calls SciPy ARPACK with an explicit UMFPACK inverse at zero shift. The repository
contains an [MIT license](https://github.com/otmanon/simkit/blob/262db01a944214d83376fee285a471af72e14ac1/LICENSE.MIT).
These links pin the inspected revision; no implementation, mesh or force data
was copied. Our consistent-mass validation uses the full coupled mass, whereas
that snapshot routine's explicit weighting uses its diagonal.
