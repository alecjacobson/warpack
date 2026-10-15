# Interior frequency bands in coupled FEM pencils

This extends [class B / issue #21](https://github.com/alecjacobson/warpack/issues/21)
from the analytic bar to coupled elastic stiffness/mass matrices. The requested
operation is `eigs(K, M, k, sigma)`: return the physical eigenvalues nearest
`sigma`, including multiplicity. For vibration problems, sigma has units of
squared angular frequency, not frequency.

K is symmetric positive definite after clamping, or positive semidefinite for
free/disconnected bodies. M is SPD and generally non-diagonal. The selected
interior shift makes `K - sigma*M` **indefinite**, even when K itself is SPD.
CG's SPD assumptions therefore do not apply to this inverse.

## API and solver contracts

```python
from warpack import CuDSSInverse, GeneralizedKrylovSchur, SparseOperator

# Kshift = K - sigma*M is assembled before factorization.
inverse = CuDSSInverse(Kshift, width=1, mtype="general", matching=True)
solver = GeneralizedKrylovSchur(
    SparseOperator(K),
    SparseOperator(M),
    inverse,
    k=6,
    ncv=32,
    target=sigma,
    tol=1e-11,
)
graph = solver.capture(2, adaptive=False)
```

The iteration count above is a measured profile, not a universal convergence
budget. All numerical work outside the cuDSS sparse inverse remains Warp.
Returned eigenvalues and residuals refer to `Kx = lambda Mx`; largest magnitude
in the resolvent is a **nearest physical eigenvalue** request, not physical LM.
The shift, matrix, dimensions and basis budget must match the factored/captured
configuration. Accuracy checks and explicit device-status readback belong
outside timed replay.

| Comparison | Contract |
|---|---|
| SciPy ARPACK | Generalized shift-invert with explicit sparse inverse and restored start |
| Spectra | Native generalized shift-invert, physical nearest selection |
| Warp + cuDSS | Mass-orthogonal generalized Krylov, matched general factors; block iteration separately labeled |
| CuPy | The current native comparison has no generalized interior-shift API; explicitly unsupported |
| Warp + block-Jacobi/FSAI CG | Explicitly unsupported for the indefinite shifted system |

No Euclidean Hermitian eigensolve is applied to `M^-1 K` for a non-diagonal
mass. The missing CuPy comparison is not a timing win, and this coverage adds
no general-purpose indefinite iterative inverse; [#49](https://github.com/alecjacobson/warpack/issues/49)
tracks that optional backend separately.

## Fixtures and independent references

Five shared test/benchmark fixtures cover clamped consistent and lumped mass,
material/density contrast, two disconnected free bodies, and two identical
Hermite bending polarizations. The assembly and low-frequency references come
from the [elastic fixture suite](FEM.md). Small matrices use dense generalized
LAPACK; larger references use ARPACK with seed 913, tighter tolerance and an
expanded basis. Timed CPU starts use seed 2026; Warp uses seed 42. Reference
vectors never enter eigensolver starts.

The target lies at a recorded fraction of a gap between distinct positive
reference eigenvalues. The default gap fraction 0.37 exercises both sides;
0.001 stresses a shift close to an eigenvalue, where the nearest set may all
lie on one side. The fixture generator extends k across a tied distance-boundary cluster; the
production solver still returns the k requested by its caller.
The reference window must contain a guard beyond every selected direction.
Free-body rigid modes stay in the original pencil but are excluded from the
requested interior set.

Every trial checks original residual, M-orthogonality, a one-to-one match to the
nearest physical spectrum, and projection onto the entire independently
referenced requested subspace. The tests deliberately submit accurate low
modes from the wrong band and require rejection. A small residual or a reported
convergence count alone cannot pass this selection contract.

## Measurements

All measurements use an NVIDIA L40 and Intel Xeon Platinum 8362 (BLAS/OMP
limited to four threads), with the pinned Warp/cuDSS environment from the
[weak-block report](WEAK_BLOCK.md). GPU jobs run sequentially with three warm,
synchronized trials and contention checks. Factor setup and graph capture are
separate from replay; CPU reference assembly is excluded. Numerical acceptance,
strict internal counts and graph-transfer audits remain separate fields.

The [small comparison](../../results/interior_small.json) retains four rejected
five-step block profiles. All four pass at [twenty block steps](../../results/interior_small_block20.json),
including strict counts. This is a convergence-budget classification, not a
new factorization defect or a successful five-step timing. The primary Krylov
profiles and native CPU solvers pass the small fixtures. Small timings are
preserved for regression, not used as large-problem performance claims.

The large replay medians below all pass every numerical trial and Warp's strict
convergence count. Entries are milliseconds; they exclude setup and capture.

| Fixture | n / k / ncv | ARPACK | Spectra | Warp + cuDSS |
|---|---|---:|---:|---:|
| Consistent beam | 10,890 / 6 / 32 | 344.960 | 314.605 | 31.216 |
| Material contrast | 10,890 / 6 / 32 | 346.843 | 311.385 | 31.138 |
| Disconnected bodies | 22,506 / 6 / 32 | 770.587 | 720.607 | 37.660 |
| Consistent beam | 41,616 / 6 / 32 | 2663.843 | 2493.204 | 72.376 |
| Interior 20-mode band | 10,890 / 20 / 64 | 983.555 | 947.295 | 68.473 |

Warp uses two fixed Krylov cycles in these profiles. Native CPU solvers stop
adaptively with tolerance `1e-11`; starts and algorithms differ. Sources:
[three large fixtures](../../results/interior_large_k6.json),
[larger matrix](../../results/interior_larger_k6.json),
[20-mode band](../../results/interior_large_k20.json).

Setup matters: on the 41,616-DOF case, sparse-factor setup is 17.192 s for ARPACK's
explicit inverse, 16.681 s for the Spectra adapter and 0.520 s for Warp+cuDSS;
Warp graph capture takes another 0.264 s. The CPU adapters use sparse LU and
cuDSS uses matched general factors. Setup includes each adapter's construction
work and is not repeated in replay timing. These figures do not include process
startup or independent reference construction.

The [near-eigenvalue shift](../../results/interior_large_near.json) also passes
all three methods on the 10,890-DOF pencil: Warp 31.143 ms, Spectra 319.760 ms,
ARPACK 348.967 ms. The target is 0.001 of the reference gap above its lower
endpoint. The [saved matched replay gate](../../results/interior_regression.json)
passes at a timing ratio of 0.99977, with no accuracy or graph regression.

The [coverage index](../../results/interior_coverage.json) contains **67 profiles: 39 numerical passes, four preserved
insufficient-budget failures and 24 explicit unsupported profiles**. Every
accepted Warp trial reaches its full strict count. The unsupported profiles
are CuPy generalized interior shift and BJ/FSAI CG on indefinite systems.

Graph audits include nested captured bodies. They contain no host callbacks or
DtoH output copies, but cuDSS's vendor HtoD nodes remain tracked in #29. DOT
inspection is not a CUDA API synchronization trace. No full upstream or Anvil
rerun is implied by these application comparisons.

The full local suite passes **311 tests in 170.76 seconds**, including 19 new
interior-band regressions. Lint, Python formatting and wheel build pass.
Validation and exact source provenance are recorded in
[interior_validation.json](../../results/interior_validation.json); the raw
reports retain every timed sample and acceptance check.

## Reproduction

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_interior_fem.py
python -m benchmarks.applications.interior_compare
python -m benchmarks.applications.interior_compare --methods warp-block --steps 20 \
  --cases clamped_consistent,clamped_lumped,heterogeneous,disconnected \
  --out /tmp/interior-block20.json
python -m benchmarks.applications.interior_compare --side 10 \
  --cases clamped_consistent,heterogeneous,disconnected \
  --methods scipy,spectra,warp --ncv 32 --steps 2 --out /tmp/interior-large.json
```

The default comparison selects the primary Krylov path. To reproduce the
original rejected five-step block profiles, pass `--methods warp-block` with
`--steps 5`; that command preserves results and exits nonzero. The worker
timeout includes setup/reference work; it is not a solve-time measurement.
