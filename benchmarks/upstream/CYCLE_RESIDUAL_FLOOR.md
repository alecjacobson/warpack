# Shifted-cycle residual estimates and fresh evaluation

The shifted-cycle investigation distinguishes three quantities: cached Krylov
residual estimates, fresh application of the implemented inverse, and residuals
against an 80-digit reference inverse of the stored matrix. They differ near
the attainable precision floor. Small physical residuals do not imply that a
very tightly requested inverse-space tolerance is met.

The probe uses the 20-node cycle with diagonal 1 and neighboring entries −.5,
shift −1e-6, k=3/5/6 with ncv=6/12/12, 100 fixed cycles and inverse tolerance 1e-12.
The shifted matrix has condition number approximately 2,000,001. For each of
zero through three fixed cuDSS corrections, it creates two fresh factors and
replays each captured solve twice, retaining every result. Starts always use
Warp seed 42. The high-precision inverse only validates outputs; it never
initializes the solver. All arithmetic in the tested solve and fresh device
check is Warp or cuDSS.

## Findings and retained controls

The initial 48-run screen on c295594 finds 22 runs where cached estimates report
all requested pairs passing but a fresh inverse application exceeds 1e-12.
For example, k=6/ncv=12 with no correction reports 6/6 and maximum cached residual
5.54e-13, while fresh application gives 1.74e-11 and the high-precision reference
gives 3.28e-11. A default one-correction k=3 run reports 3/3 with cached residual 1.40e-15, but
fresh residual 3.63e-12 and reference residual 2.36e-11. All original physical modes remain accurate.

More corrections do not reliably fix this. Across the initial screen, maximum
fresh residuals for 0/1/2/3 corrections are 7.69e-11/6.72e-11/5.85e-11/4.27e-11;
maximum physical residuals stay below 1.18e-15. Results vary across both
factor instances and replays; this does not isolate factorization as the cause
of the changing last digits and cached counts. The report does not select
a favorable retry or relabel the inverse-space 1e-12 contract as passed.

Two effects matter. Rotating previously computed operator images introduces
rounding differences from a fresh application to the returned vectors. Also,
a near-null inverse amplifies small physical-vector errors. If `B=A-sigma*I`
and `mu=1/(lambda-sigma)`, then
`B^-1*x-mu*x = -mu*B^-1*(A*x-lambda*x)`.
The inverse norm is about 1e6 here. Even an exact inverse applied to the stored
FP64 vectors can have an inverse-space residual above 1e-12. Thus additional
sparse-solve refinement is not a general cure for either effect.

[Initial evidence](../../results/performance_followup/cycle-floor-initial.json),
[retained initial script](../../historical/cycle-floor-initial-probe.md),
and [first captured fresh-evaluation follow-up](../../results/performance_followup/cycle-floor-first-fresh.json).
The first follow-up still evaluates physical Rayleigh values; the final probe
below additionally checks physical values obtained from the returned inverse
values and retains the vectors for independent verification.

The final 48-run probe checks both Rayleigh and returned transformed physical
values and membership in the allowed eigenspaces, including complete
boundary clusters when the requested count splits a repeated eigenvalue. All physical gates pass;
maximum returned physical residual is 1.427e-15, spectrum error
1.166e-15, and subspace error 3.252e-15.
In 14 runs, cached full counts coexist with incomplete fresh counts.
Maximum fresh/reference inverse residuals are 6.704e-11/5.820e-11.
The changed last digits versus earlier probes are retained, not used to claim
that a count became stable. [Final probe with returned vectors and values](../../results/performance_followup/cycle-floor-fresh.json).

## Explicit fresh returned-pair checks

`EigenpairEvaluation` now accepts optional `eigenvalues` and `rhs_width`.
Supplying values checks those actual values instead of recomputing Rayleigh
values. A fixed RHS width supports inverse adapters configured for one RHS.
Input contents can change on replay while array identities and topology stay
fixed. The residual formula for unit-norm real standard-problem vectors is
`||A*x-lambda*x|| / max(1,abs(lambda))`.

```python
fresh = EigenpairEvaluation(
    inverse,
    solver.eigenvectors,
    tol=solver.tol,
    eigenvalues=solver.eigenvalues,
    rhs_width=1,
)
physical = EigenpairEvaluation(original_operator, solver.eigenvectors, tol=1e-9)


def finalize():
    fresh.run()
    physical.run()


graph = solver.capture(100, finalizer=finalize)
wp.capture_launch(graph)
# Later, outside capture/timing, inspect each distinct contract:
cached_count = solver.converged.numpy()
fresh_count = fresh.converged.numpy()
physical_count = physical.converged.numpy()
```

`physical` in this example intentionally recomputes physical Rayleigh values.
Pass the actual transformed physical values through `eigenvalues` to validate
those as well. Generalized pencils retain their own original-pencil evaluators.
No residual count alone certifies that the requested spectrum/subspace was found.

This is an optional independent final check. Existing Krylov arithmetic,
iteration stopping, cached estimates and defaults remain unchanged. The API
and README now label cached inverse/dense-operator residuals explicitly.
The new check does not make an unmet inverse tolerance pass. Callers should
choose the relevant physical contract and inspect fresh inverse residuals when
that inverse-space contract matters.

## Cost of checking once after the solve

A 4,096-unknown Dirichlet 3D Poisson comparison uses eight low modes, ncv=32,
12 fixed cycles and the same cuDSS factors. Both graphs include the original
physical evaluator; one additionally checks all eight returned inverse values
with eight one-RHS solves at graph completion. Independent analytic sine modes
check membership in the allowed boundary eigenspace along with the requested
spectrum and orthogonality; reference modes never initialize the solver.
All counts, physical and transformed spectra, orthogonality, subspace and fresh
residual checks pass for every timing trial.

Five trials per graph, alternating order after three warmups each, give
**79.658ms without** and **81.455ms with** fresh checking: **1.798ms, about 2.3%**
additional replay cost for this fixture. The default graph receives an untimed
fresh check for validation. Capture and shared factor setup are separate. This
is not an Anvil timing or a universal overhead estimate.

The new graph adds exactly 16 vendor HtoD nodes for eight cuDSS applications,
consistent with the existing two-metadata-copy issue [#29](https://github.com/alecjacobson/warpack/issues/29).
There are no HOST or DtoH nodes. Pure-Warp evaluator/CG tests have no host
transfers. The evaluator itself introduces no host convergence decision.

[Timing and all acceptance checks](../../results/performance_followup/fresh-eigenpair-cost.json),
[occupancy](../../results/performance_followup/fresh-eigenpair-cost.occupancy.json).

## Validation

Full pinned suite: **941 passed in 415.67 seconds**. All 32 focused evaluator,
cycle and refinement checks also pass on stock Warp 1.15 in 9.40 seconds. Tests
cover provided wrong values versus recomputed Rayleigh values, changed matrix
values/vectors on the same graph, batched one/two-RHS CG/cuDSS application,
nonfinite failure/recovery, invalid shapes and counts, and physical versus
inverse accuracy. The existing shifted-cycle tests also exercise the fresh
check as a captured finalizer. Lint/format pass. The clean wheel imports the
new optional parameters outside the checkout and contains only 23 core modules.

[Validation and source hashes](../../results/performance_followup/fresh-evaluation-validation.json),
[wheel evidence](../../results/performance_followup/fresh-evaluation-wheel.json).
The original tight inverse-space failures remain recorded as failures; this
work does not claim a new full upstream conformance pass.

## Reproduction

The precision probe is explicitly diagnostic: it requires all physical gates,
but records rather than disguises failed inverse counts. Install optional
`mpmath` for its high-precision reference. No high-precision library is used by
Warpack itself.

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.upstream.cycle_floor --out build/cycle-floor.json
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.experimental.fresh_eigenpair_cost --out build/fresh-cost.json
```
