# Explicit complex GMRES restart width

`ComplexShiftInverse(..., backend="gmres", inner_restart=96)` now exposes
upstream Warp's restart width. `ShiftInvertEigensolver` forwards the same option.
It counts **realified dimensions**: a complex n-vector uses 2n real entries.
The requested width is clipped to 2n; upstream may additionally clip it to its
iteration budget. The default remains `min(2*n, 64)`. No default replay kernels,
allocations or numerical decisions change. Nonpositive/noninteger widths are
rejected, as is supplying this option to the cuDSS backend.

```python
inverse = ComplexShiftInverse(shifted, backend="gmres",
                             inner_tol=1e-14, inner_maxiter=1000,
                             inner_restart=96)
```

This resolves a configuration gap found while testing consistent spatial mass
in the generalized-pencil candidate (#119). Its 48-dimensional complex system
has 96 realified dimensions. At restart64, tol1e-14, maxiter1000 and eight outer
cycles, two damped fixtures stagnate: original residuals4.168e-4 /2.683e-4 and
quadratic residuals6.417e-4 /5.376e-4 fail the unchanged1e-8 physical bounds.
With explicit restart96, both pass the physical checks and strict counts.
The failed64 run is retained; this is not a tolerance relaxation.

A complete tiny Krylov basis is not a scalable large-system preconditioner.
Increasing restart raises device workspace and projected work. Large complex
iterative convergence and throughput remain separate tasks (#75/#97).

Nine dedicated tests check width forwarding/clamping/defaults, invalid
configuration, current-value inverse graph replay, independent original linear
residuals and no HOST/HtoD/DtoH nodes. They pass on pinned Warp1.18 and stock1.15
as part of the recorded focused suites. The full candidate suite has1,409 passes;
that candidate also contains the still separate generalized-pencil work and is
not a claim of1,409 tests in the smaller main-branch restart-only change.

[Validation and source hashes](../../results/performance_followup/complex-gmres-restart-validation.json)
record that scope. The public eigensolver remains pure Warp; this option invokes
unmodified upstream Warp GMRES, not a local GMRES implementation.
