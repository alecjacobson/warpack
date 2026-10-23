# Explicit reciprocal conditioning diagnostic (#153)

These are CPU accuracy diagnostics on the procedural 10,890-DOF cantilever,
20 modes, with both lumped and consistent mass. They investigate the rejected
explicit `Q = H S^-1 H` / ARPACK / SuperLU reference in
[the reciprocal comparison](../RECIPROCAL.md). The force prior is positive
diagonal; stiffness and mass are SPD. There is no timing claim.

The independent covariance reference uses sparse H solves without forming Q.
Every row is checked against both the original covariance pencil and the
reciprocal pencil, with the existing 1e-8 gates. Each correction is labeled;
none is presented as unmodified ARPACK.

| Mass | Method | Covariance residual | Explicit Q residual | Accepted |
| --- | --- | ---: | ---: | --- |
| Lumped | Native explicit Q | 2.544e-8 | 1.083e-8 | No |
| Lumped | Native Q with diagonal equilibration | 1.796e-8 | 1.263e-8 | No |
| Lumped | Wider product, then FP64 Q | 1.727e-8 | 1.008e-8 | No |
| Lumped | Wider product plus equilibration | 2.503e-8 | 1.464e-8 | No |
| Lumped | Covariance Rayleigh values only | 1.157e-8 | 1.082e-8 | No |
| Lumped | Covariance Ritz rotation | 1.038e-13 | 1.083e-8 | No |
| Lumped | Implicit inverse step, then covariance Ritz | 3.229e-14 | 5.106e-10 | Yes |
| Consistent | Native explicit Q | 2.199e-8 | 2.676e-8 | No |
| Consistent | Native Q with diagonal equilibration | 3.410e-8 | 2.461e-8 | No |
| Consistent | Wider product, then FP64 Q | 1.764e-8 | 3.020e-8 | No |
| Consistent | Wider product plus equilibration | 2.200e-8 | 2.445e-8 | No |
| Consistent | Covariance Rayleigh values only | 6.828e-9 | 2.676e-8 | No |
| Consistent | Covariance Ritz rotation | 9.349e-14 | 2.675e-8 | No |
| Consistent | Implicit inverse step, then covariance Ritz | 3.254e-14 | 1.271e-9 | Yes |

The CPU supports a 63-bit `longdouble` mantissa, versus FP64's 52 bits (excluding
the implicit leading bit). Wider-product controls assemble Q in that precision
and round to FP64 before SuperLU. Separate residual checks evaluate Q in wider
precision both as an assembled product and as `H (S^-1 (H V))`, starting from
the same FP64 FEM coefficients. This does not regenerate a higher-precision
physical model.

Native lumped Q residual is 1.083e-8 in both wider evaluations; consistent mass
also remains above the gate. The failed residual is therefore not just noise
from evaluating the matrix-vector product in FP64. Changing assembly precision
or diagonal scaling does not fix these controls. These experiments do not
uniquely apportion error between assembly, sparse factorization and Ritz
recovery.

The requested subspace is already accurate, but that alone is insufficient:
recovering covariance eigenvalues and rotating within the returned subspace
fixes the covariance equation while leaving Q's residual above tolerance.
One implicit inverse step

```
V <- H^-1 S H^-1 M V
```

followed by a small generalized covariance Rayleigh–Ritz solve passes both
equations. The two accepted rows have spectrum errors below 6.4e-14 and
requested-subspace errors below 6.3e-11. The inverse step costs two H solves per
column; covariance projection uses two further H solves, mass/prior products
and projected dense work. This accuracy diagnostic reuses its reference H
factor. A standalone timed adapter would need its own factorization charged to
setup. It is not a change to installed Warpack computation or a native SuperLU fix.
The existing implicit covariance recipe remains the preferred starting point.

Raw evidence: [lumped](side10-lumped.json), [consistent](side10-consistent.json).
They include every failed control, versions, source and matrix fingerprints,
all physical checks and actual mantissa widths. Six focused tests check
independent dense covariance equations for mixed nonorthogonal subspaces with
lumped/coupled mass, optional inverse recovery, and both report families.

Reproduce with the checkout's SciPy/NumPy reference environment:

```bash
python -m benchmarks.applications.force_reciprocal_conditioning \
  --out build/conditioning-lumped.json
python -m benchmarks.applications.force_reciprocal_conditioning --consistent \
  --out build/conditioning-consistent.json
python -m pytest -q tests/test_force_reciprocal_conditioning.py
```

#153 remains open for native-reference/backend improvements and any properly
qualified timing comparison of the corrected adapter. No accuracy gate was
relaxed and the original rejected CPU results remain visible.
