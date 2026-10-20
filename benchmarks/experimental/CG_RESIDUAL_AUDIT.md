# Optional fresh CG residual checks and in-place solves

`CGInverse(..., verify_residuals=True)` recomputes `A*x-b` after each
upstream Warp CG solve, entirely on device. It does not change CG's stopping
rule, solution, or recursive counters. This distinguishes recursive convergence
from the accuracy of the original linear equation, especially at tight tolerances.

```python
inverse = CGInverse(A, tol=1e-12, atol=0.0, verify_residuals=True)
# Inside the replay body:
inverse.reset_stats()
inverse.apply(rhs, solution)
# Inspect after replay when desired, outside the timed/captured computation:
recursive = inverse.stats.numpy()
fresh = inverse.fresh_stats.numpy()
max_absolute, max_relative = inverse.residual_maxima.numpy()
```

Both counter arrays contain `[total_iterations, max_iterations, failed_rhs,
rhs_count]`. Fresh checks use the effective tolerance returned by upstream CG,
including its numerical floor. Maxima are the largest fresh absolute and relative
linear residual norms since `reset_stats()`. These are inner-solve diagnostics;
users must also check the original eigenproblem residual and requested spectrum.
The audit does not refine a failed solution or change the outer convergence count.
With the default `verify_residuals=False`, the extra arrays are `None` and no
additional GPU operations are recorded for disjoint inputs and outputs.

Exact in-place `inverse.apply(x, x)` is now supported. Previously, clearing the
output erased the right-hand side: a diagonal test returned zero instead of one
while reporting no failed right-hand sides. The repair snapshots each RHS into
existing workspace before clearing the output. Nonidentical views with overlapping
address ranges are rejected before modifying data. This conservative rule includes
interleaved views whose address spans overlap. The exposed inverse `b` and `x`
workspaces can also be used for exact in-place application.

The audit uses upstream Warp's public linear operator and existing Warp reduction
kernels. Tests include scalar, 2x2 and 3x3 BSR matrices, changed RHS scales, zero
RHSs, changed matrix values on the same graph, nonfinite failure and recovery,
partial-overlap rejection, and an ill-conditioned case where the recursive
criterion passes while the fresh criterion correctly reports failure. Graphs
contain no host callbacks or host/device transfers.

## Default eigensolver regression

The signed tetrahedral buckling comparison was repeated at 41,616 unknowns and
20 modes, three trials per row, against the saved accepted baseline. Every row
passes physical spectrum/residual/subspace, full outer convergence, and inner
recursive CG checks. The saved-baseline checker also requires matching matrix
hashes, signed shifts, numerical budgets, tolerances, runtime and uncontended
execution; its regression threshold is 1.25x. The numerical and timing gates
remain separate.

| Signed branch | Backend | New median (s) | New / baseline |
|---|---|---:|---:|
| Positive | Warp + cuDSS | 0.811844 | 1.0013 |
| Positive | Warp + FSAI-CG | 5.465694 | 1.0000 |
| Negative | Warp + cuDSS | 0.806954 | 1.0027 |
| Negative | Warp + FSAI-CG | 5.288396 | 0.9972 |

All four graph audit dictionaries exactly match their baseline, including node
counts and copy kinds. CG has no host-transfer or callback nodes; cuDSS retains
its previously documented vendor metadata copies. These runs leave the optional
audit disabled. No large Anvil CPU factorization was rerun.

[Raw regression](../../results/performance_followup/buckling-cg-default-regression.json),
[occupancy](../../results/performance_followup/buckling-cg-default-regression.occupancy.json),
and [buckling formulation and references](../applications/BUCKLING.md).

## Component cost

The component driver solves a 32,768-unknown Dirichlet 3D Poisson matrix with
four RHSs, including one zero RHS. A manufactured random solution only constructs
and validates the RHS; CG starts at zero each replay. It uses the upstream diagonal
preconditioner, FP64 relative tolerance 1e-11 and zero requested absolute tolerance.
Every timed trial must pass independent fresh residual and solution-error checks;
audited trials must also pass all device fresh counters. Resets and necessary
in-place copying are included. Setup, JIT warmup and capture are separate.

The first screen used one graph warmup and five timed replays. Disjoint application
measured 10.451 ms without the audit and 11.203 ms with it; in-place application
measured 10.483 and 10.464 ms respectively. The reversed small difference for the
in-place pair and differing overhead across the two pairs motivated a follow-up
with 100 warmup replays. The initial screen remains available; it does not establish
that audit work is free. All four rows used 423 total CG iterations (142 maximum),
with maximum independent relative residual 9.90e-12.

[Initial component screen](../../results/performance_followup/cg-audit-component.json)
and [occupancy](../../results/performance_followup/cg-audit-component.occupancy.json).

With 100 warmup replays the result repeats: disjoint application takes
10.452 ms without the audit and 11.206 ms with it (about 7.2% overhead);
in-place takes 10.483 and 10.468 ms, effectively unchanged. The four paths have
different graph layouts and copying, so the latter pair does not remove the
extra audit operations or predict their cost in a larger eigensolve. All fresh
checks and all 423-iteration counters still pass. These are component timings,
not end-to-end eigensolver speedups.

[Warmed component follow-up](../../results/performance_followup/cg-audit-component-warmed.json)
and [occupancy](../../results/performance_followup/cg-audit-component-warmed.occupancy.json).

## Validation

Full pinned-Warp suite: **929 passed in 419.17 seconds**. The final fourteen
CG audit tests and eight saved-baseline tests pass on both pinned Warp 1.18
and stock Warp 1.15. An earlier 50-test compatibility run also passed on stock
Warp, covering CG absolute tolerance and real GMRES. Lint and formatting pass.
The clean wheel contains 23 core modules, excludes benchmarks/applications,
and imports the new optional API outside the checkout.

[Validation and source hashes](../../results/performance_followup/cg-audit-validation.json),
[wheel check](../../results/performance_followup/cg-audit-wheel.json),
and [retained in-place failure before the fix](../../results/performance_followup/cg-alias-before.json).
