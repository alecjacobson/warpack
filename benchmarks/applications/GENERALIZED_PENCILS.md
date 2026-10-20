# Nonsymmetric generalized shift-invert pencils

`GeneralizedShiftInvertEigensolver` computes finite right modes of
`A x = lambda B x` nearest a real or complex shift. It applies
`(A-sigma B)^-1 B` in Warp, with a supplied inverse that can use cuDSS or
upstream Warp GMRES. It never forms `B^-1 A`. This extends the explicitly
standardized damped examples to consistent spatial mass.

A and B provide `complex_n`, `device`, and `apply_complex(x,y)` on row-major
`wp.vec2d` arrays. Real matrices use zero imaginary components. B need not be
SPD or invertible, but the pencil must be regular and A-sigma B invertible.
Only finite nearest right modes are supported. Infinite modes, singular
pencils, and a public left-mode/biorthogonality API remain outside this contract.
Returned vectors have unit Euclidean norm; nonsymmetric eigenvectors are not
promised to be B-orthogonal.

```python
# A, B and shifted represent device operators; shifted = A - sigma*B.
inverse = ComplexShiftInverse(shifted, backend="cudss")
solver = GeneralizedShiftInvertEigensolver(
    A,
    B,
    inverse,
    k=20,
    sigma=sigma,
    ncv=64,
    tol=1e-9,
)
graph = solver.capture(20)
wp.capture_launch(graph)
```

The inverse is borrowed. Release this solver's graph before closing the inverse;
`solver.close()` does not close a shared factorization. For iterative replay,
update A, B and the supplied shifted operator consistently at fixed topology.
cuDSS factors remain tied to their construction values: rebuild factors and
graphs when those values change. The solver does not silently reassemble or
refactor a stale supplied inverse.

An optional `initial_vector` is a device `wp.vec2d` array of length n. Replays
read its current contents. Invalid starts set nonzero status and zero counts;
valid later starts recover on the same graph. All products, inverse composition,
iteration, checkpoint decisions and residual evaluation execute on device.
Host conversion/reference validation belongs outside the solver. cuDSS retains
its known metadata HtoD transfers (#29); the GMRES graphs have no host nodes or
host transfers.

## Original-equation acceptance

The device residual is

```
||Ax-lambda Bx|| / (||Ax|| + max(1,|lambda|) * ||Bx||).
```

The denominator scales with common multiplication of A and B and remains useful
at a zero finite eigenvalue. Nonfinite values and zero/nonfinite denominators
are rejected. Check both zero `status` and `converged == k`; status zero alone
is not convergence. Complete-set checkpoints also require this original
residual, so a small transformed residual cannot bypass a wrong original
pencil. Finalization reevaluates the current original equation.

Neither a residual nor a complete count certifies unseen requested modes.
Independent spectrum/subspace checks remain necessary when validating a new
problem class. The known compact general-spectrum discovery limitation (#104)
is not solved by adding a generalized interface.

## Consistent spatial mass coverage

The benchmark assembles

```
A = [ 0   I ],   B = [ I  0 ],   x = [ displacement ].
    [-K -D-G]       [ 0  M ]         [ velocity     ]
```

M is a consistent P1 chain mass with adjacent-node coupling and a non-diagonal
two-polarization inertia block. K contains elastic and foundation coupling;
D is nonproportional and G optionally gyroscopic. This is different from the
older lumped-spatial examples with only a small polarization mass inverse.
Independent spatial sine modes reduce reference construction to 4x4 generalized
pencils. Small tests additionally compare the assembled dense pencil, original
quadratic equation, complete right-mode subspace and reference left/B-duality.
Those reference left vectors do not constitute a public left-mode solver.

The runtime checks both the original first-order pencil and
`(K + lambda*(D+G) + lambda**2*M) q = 0`, as well as velocity consistency,
requested values/subspace, vector norms/rank, status and strict physical counts.
SciPy/ARPACK receives A/B and a supplied shifted SuperLU inverse. The native
Spectra C++ adapter applies the same shifted inverse times B. CuPy has no sparse
nonsymmetric eigs and is recorded as unsupported. No Hermitian substitution or
Anvil SuperLU factorization is used.

The explicit GMRES restart 96 resolves the small restart 64 stagnation control;
see [the restart contract and retained rejected result](../upstream/COMPLEX_GMRES_RESTART.md).
It is a complete tiny realified basis, not a large iterative performance claim.

## Accepted comparisons and retained failures

All large rows below have 65,536 first-order unknowns, 425,968 A entries,
229,368 B entries, k=20, ncv=64, two fixed cycles and shift -0.15+6.3i.
Times are median synchronized warm solves in seconds, three trials each,
on the L40 with pinned Warp 1.18 and four CPU BLAS/OMP threads. Every trial
passes independent original-pencil/quadratic residuals, spectrum/subspace,
unit norms, independence, status and full physical counts. The common physical
bounds are 1e-8; backend internal tolerances have different definitions.

| Accepted profile | Warp+cuDSS | ARPACK+SuperLU | Spectra+Eigen | Worst Warp original residual |
|---|---:|---:|---:|---:|
| Public solver, tol=1e-8, native tol=1e-9 | 0.182122 | 1.007707 | 1.955093 | 1.641e-9 |
| Explicit inverse polish + velocity recovery, tol=1e-9 | 0.190870 | 1.161072 | 2.078369 | 7.711e-13 |
| Same recovery, independent repeat | 0.190901 | 1.207315 | 2.195423 | 7.716e-13 |
| Gyroscopic pencil, same recovery | 0.190405 | 1.162372 | 2.231607 | 7.614e-13 |

The recovery rows use benchmark-only postprocessing: apply
`(A-sigma B)^-1 B` once to the vectors, recover `velocity=lambda*displacement`,
normalize and recompute original A/B products and physical counts. All compared
backends perform the same operations inside their timed solves. No eigenvalue
quotient update is used in these rows. Across the three recovered profiles,
Warp's worst quadratic residual is 1.511e-12, requested-spectrum error 7.900e-10,
and complete subspace error 3.343e-11. The underlying Arnoldi transformed count
is separately retained as `core_count`; it does not replace the fresh physical
count after recovery. This mechanics-specific relation is not a core pencil
assumption or a general correction guarantee.

For the first recovery row, setup is 0.604 s Warp, 0.065 s ARPACK and 0.131 s
Spectra; Warp capture is 0.266 s. Setup/capture/warmup are excluded from solve
timings and recorded separately. The graph has no HOST or DtoH nodes; its HtoD
nodes include cuDSS metadata (#29), so it is not claimed to be transfer-free.
No competing GPU process was observed. Unlocked clocks and differences between
CPU/GPU algorithms limit interpretation of timing ratios.

A 1,024-unknown, eight-mode public control also passes at tol=1e-10:
Warp 0.091068 s, ARPACK 0.015485 s and Spectra 0.015133 s. CPU solvers win this
small case. CuPy has no supported sparse nonsymmetric comparison.

The initial large public tol=1e-9 profile returns the correct requested space
but only 18/20 physical counts in every replay, with original residual up to
2.136e-9. One inverse-polishing step alone leaves 18–19 counts; velocity
recovery alone and quotient-only updates also fail. Those timings are rejected.
Combining inverse polishing with an original-pencil quotient passes but worsens
the spectrum error to 4.295e-9. It remains a separately recorded experiment,
not the recommended recovery. The successful tol=1e-8 public profile does not
rewrite the stricter failed result. Generic strict-residual recovery remains
tracked in [#123](https://github.com/alecjacobson/warpack/issues/123).

[All accepted and rejected profiles, hashes and summary](../../results/performance_followup/pencil-profile-summary.json)
link to their complete raw reports and occupancy records in the same directory.
No failed profile is counted as a speedup.

## Validation

The core candidate at d848d30 passed 1,409 full local tests and a clean 27-module
wheel/import check. Its 36 focused public pencil/restart/replay controls passed
on stock Warp 1.15. The final 12 experimental recovery tests pass on both pinned
Warp (32.60 s) and stock Warp (32.43 s); they include an excess finite-mode
request on a regular singular-mass pencil, zero/repeated modes, and fresh
original-residual agreement after postprocessing. Ten CPU fixture/reference
controls and eight native ARPACK/Spectra standard/generalized controls pass.
The combined-main suite passes **1,434 tests in 552.80 s**, including the latest
upstream adapter checks. Python lint/format and whitespace checks pass. All 27
combined core modules match the clean tested wheel source byte for byte. See
[validation](../../results/performance_followup/pencil-validation.json) and
[wheel provenance](../../results/performance_followup/pencil-wheel.json).

```bash
# Build the native generalized reference; standardized ABI remains supported.
g++ -O3 -DNDEBUG -std=c++17 -shared -fPIC -I/usr/include/eigen3 \
  -Ibuild/upstream-spectra/include benchmarks/applications/damped_spectra.cpp \
  -o build/damped_spectra.so
python -m benchmarks.monitor_gpu --out build/pencil.occupancy.json \
  --log build/pencil.log -- python -m benchmarks.applications.damped_compare \
  --consistent-mass --nodes 16384 --k 20 --ncv 64 --cycles 2 \
  --tol 1e-9 --polish 1 --kinematic --repeats 3 --out build/pencil.json
```

Use the pinned environment from the README and run GPU jobs sequentially.
Add `--kind gyroscopic` for the gyro control. For the public-only profile,
remove `--polish 1 --kinematic` and use `--tol 1e-8 --reference-tol 1e-9`.
The public solver, optional benchmark postprocessing and their respective
validation/performance claims remain distinct. Large pure-Warp GMRES and
public left modes remain open coverage in #27.
