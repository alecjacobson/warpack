# Anvil CPU factorization audit

[Validation and environment record](../results/anvil_factor_audit_validation.json).

The README's CPU setup costs come from sparse inverse construction, before
ARPACK or Spectra starts iterating. The original log
`build/anvil_benchmarks_direct.log` agrees with the retained
[comparison JSON](../results/anvil_benchmarks.json): SuperLU 449.295386 s and
Eigen simplicial LDLT 799.600810 s. Setup was measured once; the eigensolve
column is a median of three warm trials. They are different timing protocols.

## What the timers include

In `benchmarks/anvil.py`, the CPU matrix is already assembled, shifted,
converted to CSC and sorted before either timer starts. SciPy's timer encloses
only `splu` with `MMD_AT_PLUS_A`, zero diagonal-pivot threshold, and symmetric
mode. It includes ordering, symbolic work and numerical LU factorization.
SciPy's public interface does not expose separate timings for those stages.

The Spectra timer calls `spectra_factor` in `benchmarks/spectra_native.cpp`.
That copies the CSC matrix into Eigen and calls `SimplicialLDLT::compute`.
No Spectra iteration, mesh I/O, assembly, compilation, GPU transfer, or
original-problem validation occurs inside either setup timer.

The input is the same 283,857-DOF Anvil problem, with
`A = M^(-1/2) K M^(-1/2)` and inverse matrix `A + I`. The audit reconstructs
that matrix using the original assembly recipe, then caches it once for all
workers. It has 11,983,107 scalar nonzeros and no explicit zero entries.
The original benchmark did not retain a matrix hash; the new audit records
both the mesh hash and a shared matrix hash, rather than claiming bitwise
identity to an unrecorded original matrix.

## Fresh measurements

On the same Xeon Platinum 8362, Eigen 3.4.0 compiled with GCC 11.4 `-O3
-DNDEBUG` reproduced the original setup cost within 1%:

| Eigen simplicial LDLT phase | Seconds |
|---|---:|
| CSC copy | 0.170 |
| AMD ordering and symbolic analysis | 1.804 |
| Numerical factorization | 790.921 |
| Total | 792.895 |

Numerical factorization consumed 789.943 CPU seconds over 790.921 wall
seconds, consistent with one busy core. It accounts for **99.75%** of setup.
The factor has 371,566,842 strict-lower entries plus 283,857 diagonal values,
31.0 times the number of entries in the **full symmetric** input matrix.
Three independent random RHS checks have relative residual at most 7.28e-11.
[Raw Eigen audit](../results/anvil_factor_eigen_audit.json).

The phase-separated Eigen API re-applies its permutation in `factorize`,
whereas `compute` can retain the temporary ordered matrix from analysis;
this is a diagnostic decomposition, not a bit-for-bit timing boundary match.
The small extra preparation does not explain the hundreds of seconds in
numeric factorization. The original table values remain preserved.

SuperLU with four BLAS threads took **456.356 s**, versus the original
449.295 s (1.6% difference). It spent 677.858 process CPU seconds, so the
factorization did not sustain four-core utilization. L and U each contain
375,039,129 stored entries: 750,078,258 combined, or 62.6 times the full input
nonzero count. The three random-RHS residuals are at most 1.60e-11.
[Raw SuperLU four-thread audit](../results/anvil_factor_superlu4_audit.json).

Both original CPU eigensolves used 65 inverse applications. Fresh warm
single-RHS solves take about 0.62–0.68 s for SuperLU and 0.91–0.95 s for
Eigen. Multiplying by 65 is consistent with most of the original 42.62 s
and 62.42 s solve columns. This is a cross-check, not a new attribution
profile: the RHS vectors differ from the eigensolver's Krylov vectors.

The one-BLAS-thread SuperLU probe was **stopped at the user's request**
after approximately 6 min 20 s of process time, while still factorizing.
It has no completed factorization timing or accuracy result and is not a
performance comparison point. [Cancellation record](../results/anvil_factor_superlu1_audit.json).

## Stronger CPU inverse backend

MKL PARDISO SPD Cholesky through Eigen, with METIS ordering and a four-thread
limit, took **7.242 s** in the initial probe (2.017 s symbolic, 5.053 s
numeric, 0.172 s copy). Its factor contains 237,516,322 entries. This is a
backend and ordering change, not an ARPACK algorithm improvement. It uses
MKL 2020.4.304, Eigen 3.4.0 and the same cached matrix. No production Warpack
code or dependency changes are involved.

The default Eigen/PARDISO wrapper permits two internal refinement steps.
Although the three random RHS residuals were below 7.1e-12, its ARPACK
physical eigenpair residuals reached 6.17e-7, failing the unchanged 1e-7 gate.
All four trials are retained in the
[default-refinement report](../results/anvil_pardiso_arpack_audit.json);
these timings are not accepted CPU baselines.

Disabling PARDISO's internal refinement passes the same original-equation,
orthogonality and mode-agreement checks. The
[initial one-trial probe](../results/anvil_pardiso_noir_audit.json) took
7.122 s setup and 11.300 s solve. The independent
[three-trial ARPACK confirmation](../results/anvil_pardiso_noir_confirmed.json)
took **9.557 s setup and 16.579 s median solve**, with worst timed residual
**4.011e-9**. All warm/timed trials pass; each uses 65 inverse applications.
This setting effect is measured, but its detailed floating-point cause has
not been isolated. It is not a general recommendation to disable refinement
for all matrices. The initial probe is not substituted for a three-trial median.

The [native Spectra confirmation](../results/anvil_pardiso_spectra_confirmed.json)
uses the same PARDISO configuration, with no Python callbacks in its iteration:
**13.472 s setup + 9.937 s median solve**, worst timed original residual
**2.737e-8**. All four trials return all 26 modes and pass validation; each
uses 65 inverse applications. Spectra is pinned to
`db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95`.

| Accepted CPU pipeline | Setup (s), one sample | Solve (s), median of three | Setup + median solve (s) | Worst timed original residual |
|---|---:|---:|---:|---:|
| ARPACK + PARDISO, refinement off | 9.5567 | 16.5791 | 26.1358 | 4.011e-9 |
| Native Spectra + PARDISO, refinement off | 13.4716 | 9.9369 | 23.4085 | 2.737e-8 |

These totals exclude assembly, transfers, compilation, warmup and validation.
They are not directly comparable to the user's complete MATLAB application
time. CPU timing varies: ARPACK trials span 13.077–16.673 s and Spectra trials
8.910–12.466 s; identical factor configurations took 7.1–13.5 s across probes.
CPU affinity was not pinned, and these runs do not establish a precise
ARPACK-versus-Spectra speed ranking. A separate CPU process was observed after
the final timed run; there was no continuous whole-machine contention trace.
The large backend improvement is clear, but smaller timing differences need
controlled repeated measurement. The updated README uses the confirmed medians,
not the faster one-trial ARPACK probe; the GPU rows were not rerun in this audit.

## Interpretation

Sparse factorization creates additional entries (fill-in), so factor work and
storage can be much greater than suggested by the input nonzero count.
Eigen's simplicial implementation performs serial sparse updates; setting
`OPENBLAS_NUM_THREADS=4` does not turn it into a four-core factorization.
[Eigen's solver guide](https://libeigen.gitlab.io/eigen/docs-nightly/group__TopicSparseSystems.html)
recommends this solver for very sparse, not-too-large problems and lists
supernodal and multithreaded alternatives. SciPy calls the sequential SuperLU
library with threaded BLAS, not SuperLU_MT or SuperLU_DIST; these are
[separate SuperLU implementations](https://portal.nersc.gov/project/sparse/superlu/).

Including inverse setup is appropriate for a new FEM pencil: those factors
are needed to compute the requested low modes. Keep it separate to show the
cost for reuse of the same matrix/shift. With changed numerical values,
symbolic analysis may be reusable but numerical factorization is generally
needed again. Neither setup nor warm eigensolve alone is a complete cold
application time; assembly, compilation, capture and validation are excluded.

These rows measure named solver/backend combinations, not an intrinsic
ARPACK-versus-Warp or Spectra-versus-Warp speed difference. In particular,
they should not be presented as the best CPU baseline. Optimized CPU sparse
Cholesky and better ordering are legitimate alternatives for this SPD pencil.

## User-reported MATLAB reference

The user reports **143 seconds total** in MATLAB on a MacBook M4 for the
same mesh and 26 linear-elasticity modes. This is an external, unmatched
reference: MATLAB version, factorization/ordering, thread settings, tolerance,
mode validation, and precise total-time boundaries were not recorded here.
It is not a same-machine benchmark or a numerical acceptance result from
this harness. Nevertheless, it is a useful warning against presenting the
original SuperLU/Eigen setup costs as representative of all CPU pipelines.

## Reproduction

Use the pinned Warp environment from the root README to assemble once:

```bash
mkdir -p build/anvil-factor-audit
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.anvil_factor_audit --backend assemble \
  --out build/anvil-factor-audit/matrix.json
g++ -O3 -DNDEBUG -shared -fPIC -I/usr/include/eigen3 \
  benchmarks/anvil_factor_audit.cpp -o build/anvil-factor-audit/eigen.so
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.anvil_factor_audit --backend eigen \
  --out results/anvil_factor_eigen_audit.json
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.anvil_factor_audit --backend superlu \
  --out results/anvil_factor_superlu4_audit.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m benchmarks.anvil_factor_audit --backend superlu \
  --out results/anvil_factor_superlu1_audit.json
```

The supplementary SPD backend probe requires MKL headers and runtime:

```bash
g++ -O3 -DNDEBUG -DUSE_PARDISO -shared -fPIC \
  -I/usr/include/eigen3 -I/usr/include/mkl \
  benchmarks/anvil_factor_audit.cpp -lmkl_rt -o build/anvil-factor-audit/pardiso.so
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU \
  python -m benchmarks.anvil_factor_audit --backend pardiso \
  --library build/anvil-factor-audit/pardiso.so \
  --pardiso-refinement 0 --arpack-repeats 3 \
  --out results/anvil_pardiso_noir_confirmed.json
```

For the native Spectra confirmation, use the pinned checkout and enable its
adapter when compiling:

```bash
g++ -O3 -DNDEBUG -DUSE_PARDISO -DAUDIT_SPECTRA -shared -fPIC \
  -I/usr/include/eigen3 -I/usr/include/mkl -Ibuild/upstream-spectra/include \
  benchmarks/anvil_factor_audit.cpp -lmkl_rt \
  -o build/anvil-factor-audit/pardiso_spectra.so
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU \
  python -m benchmarks.anvil_factor_audit --backend pardiso \
  --library build/anvil-factor-audit/pardiso_spectra.so \
  --spectra-repeats 3 --pardiso-refinement 0 \
  --out results/anvil_pardiso_spectra_confirmed.json
```

To additionally run one warmup and three ARPACK trials on the same factors,
append `--arpack-repeats 3`. This requires `build/anvil_initial.npy` and
`build/anvil_reference.npz` from the original Anvil comparison. Reference
vectors are used only for final validation; the initial vector, k=26, ncv=64,
tolerance 1e-12 and largest-inverse selection match the original CPU row.

Run workers sequentially. Inverse checks use three fixed random right-hand
sides and report `||Bx-b||/||b||` for `B=A+I`. These checks are separate from
eigenpair acceptance. The original full Anvil comparison remains recorded; the accepted new
CPU eigensolve rows are reported separately from factor-only probes. Factor nonzero counts describe stored entries: SuperLU
stores both L and U, Eigen stores strict lower L plus D. Reported process
peak RSS includes diagnostic factor extraction for SuperLU and is not a
matched backend-only peak-memory comparison.
