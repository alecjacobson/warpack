# Signed buckling pencils on tetrahedral elastic beams

This follow-up to [#26](https://github.com/alecjacobson/warpack/issues/26)
adds clamped P1 tetrahedral beams with an SPD elastic stiffness `K` and a
symmetric **indefinite** geometric stiffness `G`. Both positive and negative
load-factor targets are tested, with 20 requested modes at 10,890 and 41,616
unknowns. The input is `K x = lambda G x`; its eigenvectors use the **K metric**,
not an indefinite G norm.

## Matrix construction and API

For each tet, `G_ia,jb = delta_ab V grad(N_i)^T S grad(N_j)`. The prescribed
uniform initial stress is `-S`, so the tangent is `K-lambda G`. S has mixed
signs and off-diagonal entries. K uses the independent P1 strain-displacement
elasticity assembler; the clamped face's DOFs are eliminated from both matrices.
The initial-stress form follows the [FEniCSx solid-buckling derivation](https://bleyerj.github.io/comet-fenicsx/tours/eigenvalue_problems/buckling_3d_solid/buckling_3d_solid.html).
This fixture prescribes stress to test the eigensolver; it does not solve a
nonlinear equilibrium or produce a calibrated engineering buckling prediction.

```python
# Device matrices K, G and K_minus_sigma_G are supplied by the application.
inverse = CuDSSInverse(K_minus_sigma_G, width=1, mtype="spd", refinement_steps=1)
solver = BucklingEigensolver(
    SparseOperator(K),
    SparseOperator(G),
    inverse,
    k=20,
    sigma=sigma,
    ncv=64,
    tol=1e-9,
    method="krylov",
)
graph = solver.capture(8)
wp.capture_launch(graph)
```

The SPD factor choice above is valid for the tested shifts below the first
signed load factor. For an indefinite shifted matrix, use an appropriate
indefinite inverse instead. Pure-Warp comparisons use unmodified upstream
FSAI32 (FP32 factors) and FP64 CG with relative tolerance 1e-14, zero requested
absolute tolerance and a 10,000-step limit. `CGInverse.stats` reports upstream
recursive convergence. Optional `verify_residuals=True` adds fresh linear-residual
diagnostics; see the [CG audit report](../experimental/CG_RESIDUAL_AUDIT.md).
The timings here leave that optional audit disabled.

Buckling ranks `abs(lambda/(lambda-sigma))`. Dense regressions explicitly
prove that the requested set differs from ordinary nearest-physical-shift
selection. Independent references solve the reciprocal symmetric pencil
`G x = mu K x`, with `lambda=1/mu`: dense LAPACK/QZ on small cases and ARPACK
with seed 913, tolerance 1e-12, expanded basis and a boundary guard on large
cases. Native timed starts use seed 2026; Warp uses seed 42. Reference modes
never enter solver starts. The requested boundary cluster is retained whole.

## Accepted large comparisons

The 41,616-unknown, 20-mode profiles use signed shifts +/-0.1, ncv64 and eight
fixed Warp cycles. CPU methods use their native adaptive stopping at 1e-11;
Warp uses 1e-9 with its stricter reciprocal-core tolerance. Every trial passes
independent original residual/spectrum, complete requested K-space and
K-orthogonality checks. Warp additionally reaches all physical and reciprocal
counts with status zero; every upstream CG recursive solve reports convergence.
These are different stopping formulas, not identical final residuals.

Three warm synchronized trials on the L40, no sampled competing GPU process:

| Target | Backend | Factor/workspace setup, s | Warm solve median, s |
|---|---|---:|---:|
| Positive | SciPy ARPACK + SuperLU | 17.6074 | 13.2219 |
| Positive | Spectra + Eigen LDLT | 17.0958 | 14.6820 |
| Positive | Warp + cuDSS | .4491 | .8108 |
| Positive | Warp + upstream FSAI-CG | .2316 | 5.4658 |
| Negative | SciPy ARPACK + SuperLU | 17.5892 | 23.7950 |
| Negative | Spectra + Eigen LDLT | 16.9152 | 25.9791 |
| Negative | Warp + cuDSS | .2345 | .8048 |
| Negative | Warp + upstream FSAI-CG | .1886 | 5.3031 |

Fixture assembly and independent reference construction are outside those
columns. Capture and warmup are recorded separately. Setup includes compilation
where encountered, so the first and second GPU rows are not intrinsic setup
comparisons. CPU threads are limited to four; no Anvil CPU factorization was
rerun. Eight cycles are a conservative validated profile, not a demonstrated
optimum. Pure-Warp CG remains substantially slower than the cuDSS inverse;
[#92](https://github.com/alecjacobson/warpack/issues/92) tracks that performance work.

Worst original normalized residual is 2.70e-10 for CPU references and 7.10e-12
for the large Warp rows; worst requested-space error is 1.15e-9 and 2.28e-10,
respectively. Pure-CG graphs have no HOST/HtoD/DtoH nodes. cuDSS graphs retain
its known vendor metadata HtoD copies ([#29](https://github.com/alecjacobson/warpack/issues/29));
no device-only claim is made for those copies.
[Accepted raw results](../../results/performance_followup/buckling-tet-41616.json)
and [occupancy record](../../results/performance_followup/buckling-tet-41616.occupancy.json).

## CuPy and retained failures

CuPy has no native buckling-mode eigsh API. The explicit reciprocal comparison
uses `lobpcg(sign*G, X, B=K, M=supplied_K_inverse, largest=True)` and recovers
`lambda=sign/mu`. The supplied K inverse is cuDSS; CuPy is not selecting cuDSS
itself. The fresh reciprocal absolute residual bound is 1e-10 in addition to
all original physical and requested-space gates.

At 162 unknowns all five methods pass both signs in three trials.
[Small comparison](../../results/performance_followup/buckling-tet-small.json).
At 10,890 unknowns, positive CuPy passes and both signs of ARPACK, Spectra,
Warp+cuDSS and Warp+CG pass individually, but negative CuPy crashes during
residual orthogonalization. The entire initial comparison is therefore
**rejected**, and its row times are diagnostic rather than accepted speedups.
[Initial rows](../../results/performance_followup/buckling-tet-large-initial.json),
[retained exception](../../results/performance_followup/buckling-initial-run-failure.json),
[isolated follow-up with explicit error row](../../results/performance_followup/buckling-negative-cupy-followup.json).

[#90](https://github.com/alecjacobson/warpack/issues/90) tracks the exception.
A native-only control imports no Warp/Warpack/cuDSS and uses no external stream
or capture. CuPy's own dense Cholesky/triangular preconditioner reproduces the
same `NoneType` exception once; two further trials exhaust 400 iterations with
physical residuals .994 and K-orthogonality errors 300–354. This isolates the
failure from the supplied Warp/cuDSS adapter. Native diagonal and
unpreconditioned controls avoid that exception but miss convergence; none is an
accepted timing baseline. The native sparse-LU control completed one rejected trial (physical residual
.9942, K-orthogonality error 307.8), then exceeded its 180-second budget and
was terminated after 192 seconds.
[Dense control](../../results/performance_followup/buckling-native-cupy-dense.json),
[diagonal/none controls](../../results/performance_followup/buckling-native-cupy-diagonal.json),
[partial sparse-LU result](../../results/performance_followup/buckling-native-cupy-probe.json),
[bounded sparse-LU diagnostic](../../results/performance_followup/buckling-native-superlu-timeout.json).
Completed diagnostic probes exit zero to preserve observations; failed
`accepted` fields remain explicit. The interrupted sparse-LU probe has a
nonzero exit and a separate timeout record.

A separate small CG tolerance screen retains physical passes with incomplete
reciprocal-core counts at inner tolerance 1e-13. Tightening only the inner
solve to 1e-14 resolves those counts; outer tolerances are unchanged.
[Full retained screen](../../results/performance_followup/buckling-inner-tolerance-screen.json).

The native residual-orthogonalization exception is also reported upstream as
[CuPy #10404](https://github.com/cupy/cupy/issues/10404).

## Validation and reproduction

Tests cover independently evaluated element energies, translations, symmetric
indefinite G/SPD K, signed QZ spectra, non-nearest buckling ordering, complete
K-spaces, invalid shifts, direct/CG graph replays and native exception retention.
The validator's former indefinite subspace norm was fixed separately in
[#88](https://github.com/alecjacobson/warpack/issues/88).
[Validation record](../../results/performance_followup/buckling-tet-validation.json).

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.applications.buckling_compare --side 16 --modes 20 \
  --ncv 64 --steps 8 --shift .1 --repeats 3 \
  --methods scipy,spectra,warp-cudss,warp-cg --out build/buckling.json

# Native-only failure isolation; no Warp environment needed.
PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.upstream.cupy_buckling_probe --side 10 --modes 20 \
  --methods dense --repeats 3 --out build/buckling-cupy.json
```

Add `--baseline results/performance_followup/buckling-tet-41616.json` to compare
against the saved accepted large profile. The checker matches `(sign, method)`,
input hashes, numerical settings and runtime; mismatches and solve-time ratios
above 1.25 fail the command. Requested subsets of methods/signs are supported.
The [post-CG-change regression](../../results/performance_followup/buckling-cg-default-regression.json)
passes all four GPU rows with ratios 0.9972–1.0027 and unchanged graph audits.
