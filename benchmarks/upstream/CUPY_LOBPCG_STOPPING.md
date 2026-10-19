# Native CuPy LOBPCG stopping diagnostic

2026-10-19: [local #69](https://github.com/alecjacobson/warpack/issues/69),
reported upstream as [CuPy #10402](https://github.com/cupy/cupy/issues/10402).

CuPy 14.2.0 can terminate before its outer budget while a returned pair has
absolute residual above the requested tolerance. The large Hodge comparison
first exposed this with a supplied Warp preconditioner. It also reproduces
with **native CuPy alone**, no preconditioner, mass, constraints or custom
operator, on small 128-by-128 sparse matrices.

The diagonal fixture repeats each eigenvalue 1 through 32 four times. It is
strictly positive definite. The cycle fixture is the direct sum of four
32-vertex cycle Laplacians. Both request six pairs from the same NumPy seed-42
starting block, with `maxiter=200`. Fresh CPU matrix products validate the
returned vectors independently. Three restored-start trials agree in each row.

| Matrix | Requested absolute tol | History length | Maximum returned absolute residual | Native tol met |
|---|---:|---:|---:|---|
| Positive diagonal | 1e-10 | 76 | 1.402e-10 | No |
| Positive diagonal | 5e-11 | 80 | 9.213e-11 | No |
| Four cycles | 1e-10 | 98 | 3.769e-10 | No |

Eigenvalues and orthogonality remain accurate. The discrepancy concerns early
termination against the requested stopping criterion. The installed source
uses a monotone active mask, `activeMask = activeMask & ii`: once a residual
column passes, it cannot reactivate if it later grows. Every history column
has previously passed in these examples, but the last history and fresh
returned residuals do not all pass simultaneously. Repeated eigenspaces can
rotate, so column-history minima are not fixed physical mode identities.
Monotone soft locking is a plausible cause; no unvalidated replacement is
installed in CuPy or used in production Warpack.

The [CuPy API documentation](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.lobpcg.html)
describes `tol` as the stopping criterion. The upstream report asks whether this
behavior is intentional and requests simultaneous returned-pair convergence
or an explicit warning/contract. Current CuPy main also contains the monotone
mask; no open LOBPCG PR was found during the 2026-10-19 duplicate check.

Tightening the requested tolerance can meet an application's original target,
but is **not a guarantee that CuPy will meet the newly tightened native target**,
as the second row demonstrates. The separate large-Hodge profile at 5e-11
does meet its own native target (4.594e-11 maximum); both accepted and missed
profiles are retained in [OUTER_PRODUCTS.md](../experimental/OUTER_PRODUCTS.md).
Benchmark reports keep native-tolerance status separate from shared physical
accuracy gates. No timing claim is made from these small diagnostics.

```bash
python -m benchmarks.upstream.cupy_lobpcg_stopping --out build/cupy-diagonal.json
python -m benchmarks.upstream.cupy_lobpcg_stopping --case cycles --out build/cupy-cycles.json
python -m benchmarks.upstream.cupy_lobpcg_stopping --tol 5e-11 --out build/cupy-diagonal-tight.json
```

Records under [`results/performance_followup/`](../../results/performance_followup/):
`cupy-lobpcg-diagonal-stopping.json`, `cupy-lobpcg-diagonal-stopping-tight.json`,
and `cupy-lobpcg-cycle-stopping.json`. Initial Hodge/minimal screens are retained
separately. Runtime: NVIDIA L40, Python 3.10, NumPy 2.2.6, CuPy 14.2.0, CUDA
runtime 12.9, driver 570.158.01. The diagnostic CLI runs all three configurations
with three restored-start trials; lint/format checks pass. This is not a full
upstream test-suite rerun or a fix for CuPy's stopping behavior.
