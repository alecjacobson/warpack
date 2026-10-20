# Singular boundary-mass conformal modes

This class-F benchmark solves the **unregularized** P1 conformal pencil
`K x = lambda M x`. `K` is the symmetric conformal-energy Hessian, and `M`
is a positive semidefinite boundary projector with zero interior rows. The
small regularized example in `problems.py` remains a separate historical fixture.
Application assembly, gauge selection and reciprocal recovery live in
`benchmarks/applications/`, outside the installed package.

The [block-preconditioned follow-up](../experimental/CONFORMAL_BLOCK.md) reduces
the large pure-Warp solve to 0.519 s with short CG preconditioning, retaining the
same original-pencil acceptance gates. The accurate-inverse profiles below
remain available and retain their fresh inner-solve audits.

The motivating formulation is [Mullen et al., Spectral Conformal
Parameterization, equations 5–7](https://www.geometry.caltech.edu/pubs/MTAD08.pdf):
a boundary-normalized conformal energy and a centered boundary projector.
Their numerical implementation regularizes the energy. This benchmark instead
removes the common translation kernel explicitly and uses a negative spectral
shift. It is a controlled flat/curved triangular-patch family, not a reproduction
of the paper's meshes or a guarantee of an injective parameterization.

## Matrix and API contract

For each oriented triangle, the assembly integrates
`(u_x-v_y)^2 + (u_y+v_x)^2` in an orthonormal tangent frame. Thus
`x.T K x` is twice the usual conformal energy. With boundary indicator `D`,
`C = D - b b.T / nb` and `M = diag(C,C)`. The two translations lie in the
kernel of **both** forms. The flat patch also has two finite zero modes,
corresponding to planar similarity maps. Curved patches have a positive first
finite pair. Every finite eigenvalue occurs in a real pair because rotating the
parameterization preserves both forms.

Setting the two coordinates of one vertex to zero chooses a representative
modulo translation. It imposes no shape constraint: translating any map to that
vertex preserves both forms. If `P` selects the remaining coordinates, define
`Kr=P.T K P`, `Mr=P.T M P`, and `A=Kr-sigma Mr`, with `sigma<0`.
On this family `A` is SPD, while `Mr` remains singular. The exact equivalent
problem is

```python
# All operators/inverse are already allocated on device.
core = GeneralizedKrylovSchur(
    SparseOperator(Mr),
    SparseOperator(A),
    inverse_A,
    k=20,
    ncv=64,
    mode="regular",
    which="LA",
    tol=1e-10,
)
# mu = 1 / (lambda - sigma)
# lambda = sigma + 1 / mu
# An A-normalized vector is M-normalized by dividing by sqrt(mu).
```

`ConformalEigensolver` lifts the reduced vectors, subtracts their boundary means,
converts eigenvalues and evaluates the **full original equations**, including
the rows omitted by the gauge. All of this finalizer is Warp and is captured.
cuDSS supplies only sparse solves; the iterative route uses unmodified upstream
Warp CG and FSAI with the public optional fresh-residual correction.

This is an explicit supported transformation, not arbitrary singular-mass
acceptance in `GeneralizedKrylovSchur`. Infinite modes map to zero reciprocal
values and are excluded by requesting the largest positive reciprocals. The
full common-kernel pencil is never passed directly to an SPD-metric solver.
For a `side x side` grid, the reduced dimension is `2 side^2 - 2`, the finite
spectrum has `2(nb-1)` values and there are `2(side-2)^2` infinite modes.

## Independent validation

Small homogeneous dense QZ tests classify finite versus infinite modes without
assuming an SPD mass. A separate large reference condenses the interior
Dirichlet stiffness, solves the centered boundary Schur complement with dense
LAPACK, then harmonically extends the eigenvectors. It never uses the shifted
reciprocal factorization. Reference vectors do not enter any starting space.
An independent 3D triangle-gradient/area calculation checks both assembled forms.

Every accepted trial needs original residual, spectrum, boundary-metric
orthogonality, full-vector subspace and centered-gauge errors below `1e-8`.
The subspace check includes interior Euclidean differences: the boundary mass
is only a seminorm and would hide interior corruption. Warp additionally needs
all physical and reciprocal counts at `1e-10`, status zero, and no failed native
or fresh CG audit. Short budgets, invalid reciprocals, changed starts, interior
corruption and alternative gauges have explicit regression tests.

Native SciPy ARPACK is compared in both formulations. The original route uses
`eigsh(Kr, M=Mr, sigma=sigma, OPinv=A_inverse)`; SciPy explicitly permits a
[semidefinite M when sigma is supplied](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html).
The reciprocal route uses an SPD metric and `Minv=A_inverse`. Spectra uses its
regular-inverse generalized symmetric engine with Eigen SparseLU. CuPy uses
native LOBPCG with `B=A`, largest modes and a supplied cuDSS preconditioner;
`B` is SPD, not the original singular boundary mass. The optional `cupy-scaled`
comparison scales the left matrix by `-sigma`, maps reciprocals into `(0,1]`,
and reverses that scaling in its output. It does not modify CuPy source.

The [implicit boundary-product follow-up](../experimental/BOUNDARY_PROJECTOR.md)
keeps these accuracy gates and factors, reducing large CG replay from 12.69 to
6.80 seconds and cuDSS from 0.419 to 0.384 seconds. The explicit controls below
remain the original coverage baseline; the follow-up has matched controls and
stronger implementation fingerprints.

## Large curved-patch comparison

NVIDIA L40, pinned Warp 1.18.0/FSAI commit
`04427b997fe7ff07bc6ed1dde5ca9813496c6b55`, CuPy 14.2.0, SciPy 1.15.0,
cuDSS 0.8.0.10; four BLAS threads. The patch has 16,384 vertices,
32,766 reduced unknowns, 20 requested finite modes, `ncv=64`, shift `-0.001`,
and four fixed Warp cycles. Three synchronized warm trials per method;
setup/factorization and capture are separate. CPU reference construction and
validation are excluded. All rows below pass every recorded acceptance gate.

| Method | Setup (s) | Capture (s) | Solve median (s) | Max original residual |
|---|---:|---:|---:|---:|
| ARPACK / reciprocal SPD metric + SuperLU | 2.7623 | — | 1.9047 | 2.53e-13 |
| ARPACK / original singular mass + SuperLU | 2.9287 | — | 1.8142 | 2.58e-13 |
| Spectra / reciprocal SPD metric + Eigen SparseLU | 2.6696 | — | 2.8723 | 2.54e-13 |
| Warp + cuDSS | 0.4920 | 0.6209 | 0.4183 | 3.27e-14 |
| Warp + upstream FSAI32 / corrected CG | 1.5047 | 5.9393 | 12.6861 | 2.19e-14 |

The iterative profile uses FP64 CG, an upstream FP32 FSAI32 factor,
`tol=1e-9`, `atol=0`, and one public residual correction at relative `0.01`.
Each replay performs 136,297 accumulated CG iterations over 187 original RHSs,
with zero native/fresh audit failures and max fresh relative residual about
`1.05e-11`. Both Warp routes recover all 20 original and reciprocal counts at
`1e-10`. The measured iterative gap remains [#98](https://github.com/alecjacobson/warpack/issues/98).

The pure-Warp graph contains no HOST, HtoD or DtoH nodes. cuDSS has no HOST/DtoH
nodes but retains 748 vendor HtoD metadata nodes with this correction/cycle
profile ([#29](https://github.com/alecjacobson/warpack/issues/29)); it therefore
does not pass the strict no-host-transfer graph contract.

The native CuPy row is excluded from accepted timings: the unscaled reciprocal
form crashes at residual-block orthogonalization on both the 1,150- and
32,766-unknown patches (the missing-None guard tracked in
[#90](https://github.com/alecjacobson/warpack/issues/90)). Scaling the reciprocal
values into `(0,1]` avoids that crash at 1,150 unknowns. At native tolerance
`1e-9` it still misses the original subspace gate (`1.26e-8`); at `1e-11` it
passes. At 32,766 unknowns the scaled `1e-11` run instead fails a projected
Cholesky during a timed trial ([#99](https://github.com/alecjacobson/warpack/issues/99)). None of these failed runs is a Warpack speedup.

The initial side-24 default-shift CG screen passes eigenpair gates but misses
3/101 fresh linear audits at `1e-11`; it remains rejected. The side-64 and
side-128 profiles explicitly use shift `-0.001` and CG tolerance `1e-9`, and
all fresh audits pass. This changes the conditioning and requested inner
accuracy; it does not retroactively validate the stricter attempt.

A native-only small control replaces the cuDSS preconditioner with cached
CuPy dense Cholesky and triangular solves, importing no Warp or cuDSS.
The unscaled route repeats the missing-None crash at sides 16, 24 and 32;
the scaled `1e-11` route passes both trials at each size. This isolates the
small crash from Warpack. It does not establish that every large projected
factorization failure has the same cause.

The identical-configuration baseline rerun passes: cuDSS median `0.418387 s`
(ratio `1.00012`) and CG `12.686125 s` (ratio `1.00000`). Both three-trial
records pass all numerical and strict stopping gates; sampled GPU inventories
are uncontended. These repeat timings check stability, not a new algorithm.

## Reproduction and retained evidence

Use the pinned FSAI environment described in the main README and compile the
existing native Spectra adapter:

```bash
g++ -O3 -DNDEBUG -std=c++17 -shared -fPIC -I/usr/include/eigen3 \
  -Ibuild/upstream-spectra/include benchmarks/applications/spectra.cpp \
  -o build/application_spectra.so

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m benchmarks.monitor_gpu \
  --out build/conformal.occupancy.json --log build/conformal.log -- \
  python -m benchmarks.applications.conformal_compare \
  --side 128 --modes 20 --ncv 64 --steps 4 --shift -0.001 --cg-tol 1e-9 \
  --repeats 3 --methods scipy,scipy-original,spectra,warp-cudss,warp-cg-fsai \
  --out build/conformal.json
```

For the current strict same-runtime/config 1.25 timing gate, use the fresh
[implicit-product baseline](../experimental/BOUNDARY_PROJECTOR.md). The original
coverage records predate the stronger implementation fingerprints and are no
longer automatically comparable. Methods run in isolated sequential workers with a bounded
240-second timeout; rejected rows are retained before a nonzero exit. Use
`--diagnostic` explicitly when collecting known failed native CuPy controls.
For the small scaled comparison, use `--side 24 --modes 6 --ncv 32 --steps 4
--methods cupy-scaled --cupy-tol 1e-11`. Native-only isolation is
`python -m benchmarks.applications.conformal_cupy_repro --out build/native.json`.

Raw evidence:

- [Initial side-24 screen](../../results/conformal-screen.json) and [occupancy](../../results/conformal-screen.occupancy.json): native CuPy crash and strict inner-CG rejection retained.
- [Side-64 screen](../../results/conformal-8190-screen.json): accepted larger conditioning/tolerance profile.
- [Side-128 screen](../../results/conformal-32766-screen.json): all CPU/Warp modes pass; native CuPy crashes.
- [Repeated large comparison](../../results/conformal-32766.json) and [occupancy](../../results/conformal-32766.occupancy.json).
- [Same-configuration regression](../../results/conformal-32766-regression.json) and [occupancy](../../results/conformal-32766-regression.occupancy.json).
- [Small scaled CuPy, rejected at 1e-9](../../results/conformal-scaled-cupy.json), [accepted screen at 1e-11](../../results/conformal-scaled-cupy-tight.json), and [large projected-factor failure](../../results/conformal-32766-cupy-scaled.json).
- [Native-only small CuPy isolation](../../results/conformal-cupy-native.json).

The explicit boundary-projector blocks contain 516,128 stored entries at the
largest size (stiffness: 328,988). Implicit diagonal-minus-rank-two products and
stronger iterative preconditioning are performance opportunities tracked in
[#98](https://github.com/alecjacobson/warpack/issues/98); no such optimization is
claimed in these results. Setup includes reference-backend factorizations and
Warp allocations/preconditioner construction, not host fixture assembly.

The large isolation replaces cuDSS with a host SuperLU inverse and imports no
Warp/cuDSS (`conformal_cupy_repro --large-host`). Both trials reach the 300-step
budget with failed native stopping and original accuracy gates. It does **not**
reproduce the projected exception: this demonstrates broader sensitivity, not
that the large exception is independent of the supplied inverse. The explicit
host transfers are diagnostic only, never included in performance claims.
[Raw isolation](../../results/conformal-cupy-host-isolation.json).

Validation: **84 focused/related checks pass** on pinned Warp 1.18, including
all 40 new conformal checks. Those **40 also pass on stock Warp 1.15**.
Lint/format checks pass. No installed `warpack/` source or packaging metadata
changes; the previous full-suite result remains 1,034 passed, rather than
claiming an unrun new full-suite total. [Validation record](../../results/conformal-validation.json).
