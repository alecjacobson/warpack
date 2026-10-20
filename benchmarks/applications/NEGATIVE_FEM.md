# Indefinite FEM pencils: algebraic and magnitude selection

Coverage follow-up for [#22](https://github.com/alecjacobson/warpack/issues/22).
The fixture is a negative-foundation curvature surrogate: two repeated vector
polarizations of anisotropic diffusion on a unit cube, discretized with
trilinear Q1 finite elements and homogeneous Dirichlet boundary conditions.
It is not geometric prestress or a buckling load-factor model. The matrix
`K = K0 - tau M` is symmetric indefinite, and `M` is either diagonal lumped or
sparse consistent SPD mass. There are separated negative, near-zero and positive
modes, with no exact zero eigenvalue.

Tensor-product sine modes give an independent complete spectrum and requested
mass-orthonormal subspaces. Each scalar mode has two polarizations; full clusters
at the requested boundary are retained. The foundation places zero between two
low scalar eigenvalues. No reference vectors are used as starts. Small fixtures
also check the analytic spectrum against independent dense generalized LAPACK.
Fixture/reference assembly lives in `benchmarks/applications/`, outside the
installed numerical library.

## API and matrix contracts

| Target | Warp generalized operation | Inverse properties |
|---|---|---|
| SA: most negative | Shift-invert with `sigma=-tau-1`, below the spectrum | `K-sigma*M = K0+M` is SPD; cuDSS or upstream CG is valid |
| SM: nearest zero, both signs | Shift-invert at `sigma=0` | K is nonsingular indefinite; general cuDSS factors or upstream GMRES; CG is invalid |
| LM: largest magnitude | `mode="regular", which="LM"`, supply `M^-1` | SPD mass inverse; cuDSS or upstream CG is valid |

`GeneralizedKrylovSchur(SparseOperator(K), SparseOperator(M), inverse, ...)`
keeps the mass inner product. It does not pass the nonsymmetric Euclidean
operator `M^-1 K` to a symmetric solver. `CGInverse` uses unmodified upstream
Warp CG; block Jacobi and FSAI also come directly from the pinned Warp source.
The comparison adapter now permits the already-supported regular mass-inverse
CG route. The separate `benchmarks.experimental.interior_gmres` driver exercises
public `GMRESInverse` for SM, with fixed upstream FSAI built from SPD K0+M
before subtracting the foundation; the solve matrix itself remains indefinite. This does not add a new production eigensolver or preconditioner.

Native CuPy `eigsh` has no generalized mass argument. Its separate diagonal-mass
SA/LM comparison explicitly scales by `M^-1/2`. Native `eigsh` does not accept
SM or a shift argument. Generalized CuPy comparisons use LOBPCG with a supplied
cuDSS SPD preconditioner. LOBPCG's largest-algebraic selection is compared to LM
only when the fixture proves that the two requested sets coincide. Consistent
mass with native `eigsh`, and generalized SM in the present CuPy adapter, are
explicit unsupported routes. These are not claims that custom reformulations
are impossible.

## Acceptance and timing

Every timed trial must pass the original pencil residual, spectrum, complete
requested eigenspace and mass orthogonality at the common 1e-8 bounds. In
addition, this driver requires full Warp counts at its 1e-11 tolerance, zero
reported inner CG/GMRES failures, and fresh native CuPy stopping checks. CuPy LOBPCG
uses per-vector absolute residual norms; native `eigsh` additionally checks the
Frobenius norm of the residual block in the correctly scaled standard problem.
Setup, reference construction, capture and validation are outside warm solve
timing and recorded separately. Solves restore their starts, and GPU methods
run sequentially under an occupancy monitor. cuDSS's known metadata HtoD copies
remain tracked separately in #29; pure-Warp graph audits reject host callbacks
and host/device transfers.

The initial 1,024-DOF sweep is retained in
[negative-foundation-side8.json](../../results/performance_followup/negative-foundation-side8.json).
Its 20-cycle budget is a diagnostic control, not a recommended setting. Three
CG profiles with inner tolerance 1e-13 pass the common physical gates but miss
full outer counts: diagonal/lumped 11/12, FSAI/lumped 6/12, FSAI/consistent 6/12.
The worst reported physical residuals are 1.020e-11, 3.105e-11 and 5.017e-11,
respectively, against the stricter outer tolerance 1e-11. No inner iteration
budget is exhausted. Those timings are rejected by the strict profile.

A separate [calibrated SA profile](../../results/performance_followup/negative-foundation-sa-calibrated.json)
uses inner tolerance 1e-14 and three outer cycles with unchanged outer/physical
bounds. All direct, diagonal-CG, upstream block-Jacobi and upstream FSAI cases
pass both mass types, with full 12/12 counts and no failed inner solves. Native
CuPy eigsh passes the lumped case; the consistent case is explicitly unsupported.
This changes a public accuracy setting and a fixed work budget; it does not
reinterpret convergence counts or change the iteration algorithm.

The initial CuPy LOBPCG LM/consistent profile also passes common physical gates
but misses its requested absolute tolerance 1e-8: fresh maximum residuals are
8.048e-8, 2.071e-8 and 1.746e-8 across three trials. This is additional evidence
for the existing native tolerance tracker #69, not a valid strict timing win.
All rejected data are retained. In this preliminary record only, the
`largest_magnitude_is_largest_algebraic` field outside LM compares the selected
SA/SM target with LA and should be ignored; the revised driver computes actual
LM/LA equivalence for every selection. This metadata issue does not affect the
LM eligibility check or numerical acceptance.

## 27,648-DOF, 20-mode comparison

Accepted tested warm solve profiles, milliseconds. ARPACK/Spectra rows use the
analytic reference for acceptance; neither is treated as ground truth merely
because it returned eigenvectors. The CuPy column names the accepted route.
Every numeric table entry passes all three trials and the recorded stopping
gates. Source batches retain rejected rows and may exit nonzero; the table uses
only successfully completed, accepted workers with uncontended samples, not
failed trials or a blanket passing claim about a mixed batch.

| Target / mass | ARPACK | Spectra | CuPy | Warp + cuDSS | Warp + upstream FSAI-CG |
|---|---:|---:|---:|---:|---:|
| SA / lumped | 3234.1 | 3313.5 | eigsh 129.1 | 106.6 | 483.9 |
| SA / consistent | 5705.5 | 6090.6 | LOBPCG 390.5 | 110.4 | 514.1 |
| SM / lumped | 1939.0 | 2054.5 | unsupported | 59.4 | CG invalid |
| SM / consistent | 3533.6 | 3879.2 | unsupported | 68.6 | CG invalid |
| LM / lumped | 1316.2 | 2392.7 | eigsh 165.2 | 639.6 | 694.3 |
| LM / consistent | 18022.9 | 18816.9 | rejected | 1593.0 | 2040.1 |

Direct SA uses two cycles and SM one; regular LM uses 32. The SA FSAI profile uses three
cycles, relative inner tolerance 1e-13 and `atol=0`. LM FSAI uses the same inner
policy and 32 cycles. Native CuPy eigsh uses tolerance 1e-8; the accepted
LOBPCG SA follow-up uses 1e-9 with a 1,000-iteration cap. CPU references use
1e-11, and Warp outer counts use 1e-11. These are explicit, separately checked
algorithm settings, not identical iteration budgets or identical definitions
of an internal tolerance. All use the same original physical acceptance gates.

Setup, capture and warmup are separate in the raw records. For example, the
LM/consistent CPU factor setup is 4.465 s for SciPy; its 18.023 s median is the
subsequent eigensolve, not setup mislabeled as a solve. The CPU work here is for
this synthetic cube; no Anvil SuperLU/Eigen factorization was launched.

[Initial large sweep](../../results/performance_followup/negative-foundation-side24.json),
[direct regression control](../../results/performance_followup/negative-direct-regression.json),
[relative-CG regression control](../../results/performance_followup/negative-relative-regression.json),
[accepted CuPy SA control](../../results/performance_followup/negative-sa-cupy-tight.json),
[extended LM profiles](../../results/performance_followup/negative-lm-extended.json).
The same-setting direct and relative-CG regression replays differ by less than
0.2%, pass every numerical/stopping gate, and have clean occupancy records.

The six-cycle LM rows are rejected for insufficient work; all returned pairs
are not claimed converged. With 32 cycles both Warp inverse routes pass all
20 modes for both masses. Native diagonal-mass CuPy eigsh is faster on the LM
case; the remaining regular/adaptive throughput work belongs to #31. The
longer CuPy LOBPCG runs remain rejected, including the independent instability
tracked in #80. They are not used as timing wins for Warp.

## Direct work-budget calibration

The original six-cycle SA/SM profile used more work than needed. A shorter
profile retains full 20/20 counts and all physical/requested-space gates in
three trials: SA takes 106.6/110.4 ms for lumped/consistent mass with two cycles;
SM takes 59.4/68.6 ms with one cycle. The prior accepted medians were
296.6/307.8 ms for SA and 298.3/343.3 ms for SM. This is a validated work-budget
change, not a production kernel or tolerance change.

One-cycle SA is **rejected**: only 16/20 counts and original residuals up to
1.52e-5. Those trials remain in
[negative-direct-cycle1.json](../../results/performance_followup/negative-direct-cycle1.json)
alongside the accepted SM trials. The accepted
[two-cycle SA profile](../../results/performance_followup/negative-direct-cycle2.json),
[SM regression replay](../../results/performance_followup/negative-direct-sm-regression.json)
and [SA regression replay](../../results/performance_followup/negative-direct-sa-regression.json)
retain all samples and graph audits. Independent same-setting replays differ
by less than 0.2%, with uncontended occupancy records. No failed shorter run is
counted as a performance improvement.

## Pure-Warp GMRES for the indefinite nearest-zero matrix

`GMRESInverse` supplies the missing iterative SM route, using the unmodified
upstream Warp implementation and a fixed FP64 FSAI32 right preconditioner of
SPD K0+M. The preconditioner is built before subtracting the foundation; the
actual solve is still with indefinite K. No reference vectors enter the start
or preconditioner. The outer tolerance is 1e-11 and inner relative tolerance
1e-13 with `atol=0`.

At 27,648 DOFs and 20 modes, ncv96 and one outer cycle:

| Mass | Pure-Warp FSAI32 / GMRES128 | cuDSS control | Inner iterations / failed RHSs |
|---|---:|---:|---:|
| Lumped | 2.5077 s | .05957 s | 24,576 / 0 of 96 |
| Consistent | 2.5720 s | .06831 s | 24,576 / 0 of 96 |

All three trials pass full 20/20 counts, original equations, spectrum, complete
requested subspace, mass orthogonality and fresh inner checks. Each RHS takes
256 GMRES iterations within the 512 budget. Maximum physical residuals are
2.83e-14 (lumped) and 2.13e-14 (consistent). Pure-Warp graphs contain only device
work and DtoD copies. The separate cuDSS graphs retain 192 vendor metadata HtoD
copies and no host callbacks/DtoH copies. Setup/capture free-memory decrease is
about 1.86–1.89 GB for this FSAI32 profile, including runtime/pool allocations;
it is not a live or peak memory measurement.

[Large lumped trials](../../results/performance_followup/negative-gmres-large-lumped.json),
[large consistent trials](../../results/performance_followup/negative-gmres-large-consistent.json),
and a [1,024-DOF / 12-mode control](../../results/performance_followup/negative-gmres-medium.json)
retain all samples, setup/capture costs, diagnostics and occupancy companions.
The medium consistent control also passes (359.85 ms GMRES, 39.62 ms cuDSS).
A [same-setting consistent-mass replay](../../results/performance_followup/negative-gmres-consistent-regression.json)
passes all gates and differs by less than 0.04% for both methods, with clean
occupancy. Configuration checks reject changed matrices, tolerances, budgets,
runtimes, failed accuracy and contended profiles. The remaining iterative cost
is tracked in #75; this is coverage, not timing parity with direct solves.

The revised driver fixes [#81](https://github.com/alecjacobson/warpack/issues/81):
`physical_accepted`, `outer_stopping_accepted` and `inner_solve_accepted` are
separate, and all are mandatory for overall acceptance. Rejected profiles now
return nonzero. Earlier interior data are retained verbatim; their outer
acceptance field must be read with the already-recorded inner failure counts.
[Focused validation](../../results/performance_followup/negative-gmres-validation.json)
passes 43 GMRES/component tests initially and 39 final foundation/CG/GMRES tests
(overlapping suites). The eight new cases pass on the pinned runtime; stock
Warp 1.15 passes six and explicitly skips two unavailable FSAI cases. Lint and
formatting pass. The deliberate exhausted-budget CLI returns exit 1 and keeps
its [failed row](../../results/performance_followup/negative-gmres-rejected.json).
This benchmark-only follow-up is not a new full local/upstream suite run.

Eight new captured tests cover both masses with no preconditioner, upstream
diagonal scaling, and upstream FSAI, plus exhausted inner budgets. The failure
regression also verifies that a physically exact full-count outer result cannot
hide recorded inner failures. Reference vectors are used only for that gate
check, never as input to an eigensolve.

```bash
python -m benchmarks.experimental.interior_gmres --case negative_consistent \
  --side 24 --modes 20 --ncv 96 --steps 1 --restart 128 --maxiter 512 \
  --inner-tol 1e-13 --outer-tol 1e-11 --fsai-width 32 --repeats 3 \
  --out build/negative-gmres.json
```

## Absolute versus relative-only CG stopping

Follow-up to [#79](https://github.com/alecjacobson/warpack/issues/79). The shared,
FEM and negative-foundation comparison CLIs now expose `--cg-atol`. Omission
preserves `None`: upstream uses `cg_tol` as both relative and absolute tolerance.
Its stopping bound is `max(atol, tol*||rhs||)`, subject to the upstream dtype
floor. `--cg-atol 0` requests relative-only stopping. This matters for small
mass-weighted right-hand sides, mesh refinement and changes of units.
`CGInverse` already supported the option; production defaults and upstream CG
are unchanged. The raw records and baseline-configuration checks now include
`cg_absolute_tolerance`, so these policies cannot silently count as identical.

Matched 27,648-DOF SA controls, inner tolerance 1e-14 and six outer cycles:

| Mass | Default absolute policy: count / ms | Relative-only: count / ms | Relative-only max physical residual |
|---|---:|---:|---:|
| Lumped | 10/20, 860.831 (rejected) | 20/20, 969.632 | 2.464e-13 |
| Consistent | 13/20, 912.356 (rejected) | 20/20, 1028.258 | 1.476e-13 |

[Default-policy control](../../results/performance_followup/negative-atol-default.json),
[relative-only control](../../results/performance_followup/negative-atol-relative.json).
All inner solves report convergence in both profiles. Relative-only stopping
requires about 13% more solve time here, but removes the premature absolute
cutoff. Separately reducing the outer budget to three cycles with relative
inner tolerance 1e-13 still passes all 20 modes at 483.243/513.338 ms;
[raw profile](../../results/performance_followup/negative-sa-relative13.json).
This is a validated settings choice, not a universal inner/outer error bound.

Regression tests scale both K and M by 1e-8, 1 and 1e8, retaining eigenvalues and
checking the appropriately mass-normalized requested spaces. They cover both
mass structures and SA/LM, full counts, original equations and no failed inner
solves. CLI probes separately exercise the shared runner and the tetrahedral
FEM driver. Those one-replay probes are correctness checks, not timing claims.

## Independent native CuPy failure

[The isolation driver](negative_cupy_repro.py) imports no Warpack, launches no
Warp kernels, and uses no cuDSS, interop stream or graph capture. Twelve trials
cover generalized exact scalar preconditioning, generalized identity
preconditioning, and the equivalent standard problem. At 1,000 iterations the
standard CuPy LOBPCG case repeatedly loses orthogonality (error 462.99), has
normalized residual 0.99569 and native absolute residual 1.148e8. Input matrix
data remain unchanged. The recorded history has 1,001 entries: this is
instability at budget exhaustion, not a claim of a reported converged result.
The finite generalized controls also miss the requested native tolerance.

[Raw isolation trials and histories](../../results/performance_followup/negative-cupy-isolation.json)
are tracked in [#80](https://github.com/alecjacobson/warpack/issues/80). The
smaller native stopping misses remain in #69. Native eigsh and the independent
analytic/CPU/Warp checks demonstrate that this requested spectrum is resolvable.

## Validation and reproduction

The [combined local suite](../../results/performance_followup/negative-validation.json)
passes **801 tests in 359.41 s**; the final focused stock-Warp 1.15 suite passes
**49 in 41.49 s**. Lint and formatting pass. Thirty-one new test cases cover the
analytic/dense fixture checks, captured selections with both inverse routes,
native diagonal-mass CuPy comparisons, unsupported routes, unit scaling and
regression-configuration identity. Production solver source and package contents
are unchanged. This is not a new complete upstream conformance run.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.applications.negative_compare --side 24 --modes 20 \
  --selections SA --methods warp-cg-fsai --ncv 96 --steps 3 \
  --cg-tol 1e-13 --cg-atol 0 --repeats 3 --out build/negative-sa.json
python -m benchmarks.applications.negative_cupy_repro \
  --out build/negative-cupy-isolation.json
```

Exact arguments, input fingerprints, numerical source hashes, setup/capture,
all samples, original acceptance diagnostics and graph audits are in the raw
reports. The native comparison binary must be built from the existing
`benchmarks/applications/spectra.cpp`; its hash is recorded by the current driver.
