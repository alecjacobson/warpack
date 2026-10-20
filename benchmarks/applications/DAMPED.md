# Coupled damped and gyroscopic modes (#27, #111)

These fixtures represent two coupled P1 axial FEM fields with fixed endpoints,
SPD stiffness and mass, and positive damping. The variants add nonproportional
damping, a skew gyroscopic term, or an SPD non-diagonal polarization mass.
Spatial mass is lumped in every variant; the coupled-mass case is not a claim
of arbitrary consistent spatial mass support. The benchmark forms the standard
first-order linearization using a known 2x2 mass inverse during fixture setup.
The public solver receives the assembled nonsymmetric sparse matrix.

An independent spatial sine decomposition reduces the reference to 4x4 modal
systems. Small tests additionally compare every eigenvalue against a dense
original generalized pencil, verify positive/symmetric/skew matrix properties,
and check reference left/right biorthogonality. Solver acceptance checks the
original first-order and quadratic equations, the complete requested spectrum,
unit norms, numerical rank and the requested right subspace. Nonorthogonal
nonnormal right eigenvectors are not required to be orthogonal. Reference left
vectors do not imply a public left-eigenvector API.

The native Spectra adapter factors a complex sparse shifted matrix with Eigen
SparseLU and runs complex GenEigsSolver entirely in C++; it has no Python
matvec callback. ARPACK uses an explicit SuperLU inverse. CuPy is recorded as
unsupported because it has no sparse general eigs and this matrix is not
Hermitian. Warp uses the public complex shift-invert solver and its existing
realified cuDSS or pure-Warp GMRES inverse.

Ten local tests pass on pinned Warp (16.36 s) and stock Warp 1.15 (16.03 s),
covering all four fixtures, repeated cuDSS graph replay and a small gyroscopic
GMRES solve. [Focused validation](../../results/performance_followup/damped-coverage-validation.json).
The tests are additional to the preceding 1,213-test full-suite report; this
coverage addition has not been represented as another full upstream run.

## Accepted large recovery profile

The same 65,536-unknown problem now passes every timed sample using an explicit
coordinate/extraction profile: scale 8, one inverse polish, velocity recovery,
two fixed Warp cycles,physical tol 2e-8 and reference iterative tol 1e-12.
The independent physical/quadratic/spectrum/subspace/norm bound remains 1e-8.
All methods recover 20 modes; the original tighter-count profile below remains
rejected. These are adapted reference pipelines, with extraction included.

| Method | Setup(s) | Capture(s) | Median solve(s) | Max physical residual |
|---|---:|---:|---:|---:|
| ARPACK + SuperLU + correction/recovery |0.05049|—|1.119790|3.01e-9|
| Spectra + Eigen SparseLU + correction/recovery |0.06635|—|2.106111|3.07e-9|
| Warp + cuDSS + recovery |0.53466|0.26413|0.218344|2.91e-9|
| CuPy |unsupported|—|—|—|

Warp's warmed solve is about 5.1x faster than ARPACK and 9.6x faster than Spectra
on this profile. Setup/capture/warmup are excluded from solve samples and
recorded separately; this is not a cold-start claim. The problem is two coupled
one-dimensional FEM branches, not the irregular 3D Anvil mesh.
[Method, rejected alternatives and reproduction](../experimental/DAMPED_RECOVERY.md),
[every trial](../../results/damped-large-balanced8.json),
[occupancy](../../results/damped-large-balanced8.occupancy.json).

## Retained large failure

The first large nonproportional problem has 65,536 first-order unknowns,
229,372 nonzeros, k=20, ncv=64, shift=-0.15+6.3i, and requested tol=1e-10.
Warp executes 20 fixed cycles. All three repeated samples of each backend fail
the common physical gates. No timing is accepted as a performance result.

| Method | First physical residual | First quadratic residual | First requested-space error | Count |
|---|---:|---:|---:|---:|
| ARPACK + SuperLU | 6.88e-8 | 1.11e-7 | 3.62e-7 | 20 |
| Spectra + Eigen SparseLU | 8.68e-8 | 1.20e-7 | 3.51e-7 | 20 |
| Warp + cuDSS | 3.63e-6 | 3.55e-6 | 1.49e-10 | 0 physical / 5 transformed |

The monitor observes no competing GPU process; its nonzero child status is a
numerical rejection. [Every trial](../../results/damped-nonproportional-large.json),
[occupancy](../../results/damped-nonproportional-large.occupancy.json).

The independently constructed right reference itself has physical residual
3.43e-9 in FP64 and 3.07e-9 when the stored matrix/vectors are evaluated with
extended-precision accumulation. It passes the common 1e-8 gate, but indicates
that the stricter 1e-10 count needs a precision/conditioning audit. This is
not a proof that 1e-10 is unattainable. Stiffness grows quadratically with mesh
resolution. Exact state-coordinate balancing and sparse inverse accuracy are
next controls; the original physical equations and failed profile remain.
[Reference check](../../results/performance_followup/damped-large-reference-floor.json).
The failure is tracked in [#111](https://github.com/alecjacobson/warpack/issues/111).

## Reproduction

Use the pinned Warp environment and Spectra revision documented in the upstream
harness. Build the native reference once, outside timed setup/solve samples:

```bash
g++ -O3 -DNDEBUG -std=c++17 -shared -fPIC -I/usr/include/eigen3 \
  -Ibuild/upstream-spectra/include benchmarks/applications/damped_spectra.cpp \
  -o build/damped_spectra.so
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  PYTHONPATH="$PWD/build/warp-fsai:$PWD" python -m benchmarks.monitor_gpu \
  --out build/damped-large.occupancy.json --log build/damped-large.log -- \
  python -m benchmarks.applications.damped_compare --nodes 16384 --k 20 \
  --ncv 64 --cycles 20 --out build/damped-large.json
```

Each measured replay is checked; setup, capture, warmup and reference work are
excluded from solve timing and recorded separately. The measurements use the
same L40, driver 570.158.01 and pinned Warp overlay as the other current local
reports, with four OpenBLAS/OpenMP threads and unlocked GPU clocks.


## Physical versus transformed convergence (#112)

The n=1,024 control exposes a benchmark error: all 20 requested modes satisfy
public physical counts and independent equations/subspaces, but two replays
have only 19 internally converged transformed pairs. The internal tolerance is
100 times tighter. The original harness incorrectly rejected these samples.
The corrected harness requires public physical count 20/status 0 and all unchanged
independent physical gates, while retaining internal counts as diagnostics.
A regression uses the actual recorded quality and counts and also checks that
short public counts, failed physical gates and nonzero status remain rejected.
[Retained rejection](../../results/damped-medium-transformed-count-rejection.json),
[occupancy](../../results/damped-medium-transformed-count-rejection.occupancy.json).
This correction does not reclassify the large #111 failure.

The corrected control passes every measured sample. Median solves are scipy: 18.580 ms, spectra: 61.798 ms, warp-cudss: 755.626 ms. Warp's fixed 20-cycle profile is much longer than the reference
iterations; shorter explicit budgets require separate acceptance checks.
[Corrected control](../../results/damped-medium-physical-gates.json),
[occupancy](../../results/damped-medium-physical-gates.occupancy.json).

The [replication and additional large variants](../experimental/DAMPED_RECOVERY.md#replication-and-further-large-variants)
pass another 27 supported-backend samples on nonproportional, gyroscopic and
coupled-mass systems with the same settings. Warp medians are 0.199–0.219 s.
