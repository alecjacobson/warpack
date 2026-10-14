# Elastic reduced-order mode coverage

This expands class A / [#20](https://github.com/alecjacobson/warpack/issues/20)
with deterministic fixtures shared by regression tests and comparison workers.
It adds no production solver arithmetic. Host reference assembly and validation
are separate from captured device solves.

## Matrices and requested spaces

The tetrahedral fixture is a 3:1:1 beam, with a conforming six-tet subdivision
of each regular-grid cell. Independent strain-displacement assembly constructs
linear isotropic elastic stiffness. Uniform material uses E=10, nu=0.3 and
rho=1. Tests compare this assembly to the production Warp rest tangent and
compare both consistent and lumped mass to independent Warp assembly/row sums.

- **Clamped/lumped** and **clamped/consistent** remove every displacement DOF
  on the x=0 face. Tests expand the output and check the prescribed zero values.
- **Free** retains six analytically constructed translations/rotations, made
  orthonormal in the consistent mass metric. Requested output includes these
  six modes plus the requested nontrivial elastic modes.
- **Disconnected** duplicates the free beam: twelve rigid modes and exact
  twofold copies of every elastic eigenvalue. Independent component spectra and
  block subspaces prevent relying on a single iterative reference to discover
  their multiplicities.
- **Heterogeneous** uses a 20x Young's modulus contrast and a 3x density
  contrast across different material interfaces, with consistent mass and a
  clamped face.
- **Repeated bending** is a clamped Hermite Euler–Bernoulli beam with two
  transverse polarizations. Its elastic eigenvalues have exact multiplicity
  two. Its first discrete frequency is also checked against the cantilever
  continuum value, keeping discretization error separate from eigensolver error.

Small references use dense generalized LAPACK. Larger tetrahedral references
use separately initialized ARPACK with an expanded basis and `1e-12` tolerance.
Reference construction is excluded from every solve timing. A requested cutoff
inside a repeated cluster expands to include the entire cluster; the report
records both requested and actual nontrivial mode counts.

The ROM policy is explicit: solve for the known rigid block plus physical modes,
then retain the physical block. These fixtures do not imply a general adaptive
rigid-mode detector or a weighted constraint/deflation API; the latter is
tracked separately in [#42](https://github.com/alecjacobson/warpack/issues/42).
Hodge/Betti workloads
have the opposite policy: their nullspace is an output and must be retained.

## Acceptance and backends

Every measured trial must pass `1e-8` checks on original-pencil residuals,
requested eigenvalues, M-orthogonality, the known rigid subspace, and the entire
requested invariant subspace. The subspace error is an M-norm projection
remainder, so signs and rotations inside repeated clusters do not matter.
All reference and failed/unsupported configurations remain visible.

`warp` uses generalized restarted Krylov with one-RHS SPD cuDSS solves;
`warp-block` uses block generalized inverse iteration. The iterative profiles
use the same Krylov outer solver and upstream Warp CG (`tol=1e-13`,
`maxiter=10000`):

- `warp-cg`: scalar diagonal preconditioning, retained as the initial control.
- `warp-cg-bj`: upstream `block_jacobi_sequential`, using the natural 3x3
  displacement blocks (2x2 displacement/slope blocks for Hermite bending).
- `warp-cg-fsai`: pinned upstream Warp FSAI, `max_row_size=32`,
  `max_step_size=3`, `apply_lanes=8`, FP32 factors with FP64 CG arithmetic.

No local replacement preconditioner/factorization is introduced. Setup includes
preconditioner construction, recorded separately from capture and replay.
Per-trial counters expose total/max CG iterations and unconverged RHS solves.
Native ARPACK and Spectra use their generalized shift-invert interfaces. CuPy
uses generalized **LOBPCG with a cuDSS preconditioner**, not native `eigsh` on an
incorrect Euclidean `M^-1 K` operator. The separate `cupy-eigsh` path accepts
positive diagonal mass, forms `M^-1/2 K M^-1/2`, and maps its output back before
validation. `cupy-batched` applies one padded k-RHS cuDSS solve per LOBPCG
preconditioner call; `cupy` preserves the original columnwise adapter. Both
restore starting spaces and face the same checks. Tables use the faster accepted
batched path. Native `eigsh` uses no cuDSS and explicitly rejects consistent non-diagonal mass.

The application regression tests request solver tolerance `1e-9` and require
full counts plus independent residuals below `1e-9`. Benchmarks retain the
stricter `1e-11` solver profile and report its actual counts separately from
external `1e-8` numerical acceptance. In the initial repeated-bending test, the
stricter count is 5/6 with a physical residual around 1e-11; the requested
spectrum/subspace already passes. That is truthful nonconvergence at the tighter
tolerance, not a missing repeated mode, and is retained in the diagnostics.

For these PSD pencils, every finite negative shift preserves the requested
low-mode order. `--shift` experiments can improve inner conditioning but reduce
separation in the inverse spectrum; they still face the same original-pencil
and complete-subspace checks. Changed shifts are not eligible for an
identical-configuration timing regression pass. Failed short-budget and shifted
runs remain recorded, including missing rigid multiplicity.

Graph DOT inspection audits nested bodies for HOST callbacks and transfer
kinds; it is not an API trace. Numerical acceptance does not hide the existing
vendor cuDSS HtoD graph nodes tracked in #29.

## Recorded results

On NVIDIA L40, Warp 1.18.0 (pinned FSAI overlay), cuDSS 0.8.0.10,
CuPy 14.2.0, SciPy 1.15.0 and four CPU BLAS threads, the
[composite report](../../results/fem_coverage.json) contains **134 configurations:
118 numerically passed, 11 failed, 5 unsupported**. This combines separately
recorded experiments; it is not a full rerun at one commit. The source manifest
links every original report and its source commit. All trials in accepted rows
pass the independent gates and have no observed competing GPU process.

Best observed accurate median solve/replay times, in milliseconds (three trials;
setup, factorization, capture and warmup excluded):

| Consistent-mass clamped beam | ARPACK | Spectra | CuPy batched LOBPCG + cuDSS | Warp + cuDSS | Warp + FSAI CG* |
|---|---:|---:|---:|---:|---:|
| 10,890 DOFs, 6 modes | 691.441 | 665.685 | 33.655 | **17.481** | 106.137 |
| 41,616 DOFs, 6 modes | 1674.855 | 1553.988 | 48.080 | **38.049** | 248.378 |
| 10,890 DOFs, 20 modes | 704.267 | 642.560 | 327.507 | **66.785** | 417.006 |
| 10,890 DOFs, 40 modes | 1331.936 | 1357.092 | 367.024 | **101.596** | 589.592 |

These are best **tested settings**, not identical-work iteration comparisons or
new defaults. Warp + cuDSS uses sigma=-0.01: ncv=32/one cycle for six modes,
64/two for twenty, and 96/two for forty. All three trials have full requested
counts at solver tolerance 1e-11. CPU rows use ncv=64,20,64,96 respectively;
smaller tested budgets and their timings are retained even when slower.
CuPy LOBPCG uses tolerance 1e-8 for six modes and 1e-10 for twenty/forty.
Native CuPy `eigsh` does not support these consistent-mass pencils directly.

*CG rows pass the independent 1e-8 contract but **do not reach full counts at
solver tolerance 1e-11**. FSAI uses sigma=-100, ncv=64/one cycle for six modes;
-10, ncv=64/two or 96/two for twenty/forty. Reported last-trial counts are
0/6, 0/6, 1/20 and 3/40. The corresponding block-Jacobi times are 263.008,
465.419, 1096.832 and 1539.977 ms (counts 0/6, 0/6, 0/20 and 4/40).
This is an application-tolerance comparison, not a claim of equal strict
convergence. Setup/capture costs and per-trial CG work remain in the raw reports.

For the **lumped-mass** 10,890-DOF beam, six modes take 17.511 ms with
Warp + cuDSS, 35.784 ms with batched CuPy LOBPCG + cuDSS, and 164.263 ms
with native CuPy `eigsh` (no cuDSS). ARPACK and Spectra take 684.710 and
596.722 ms. The diagonal-mass scaling and inverse transformation are explicit.

For the **free** 11,253-DOF beam, retaining six rigid plus six elastic modes
at sigma=-0.01 takes 38.635 ms with one-cycle Warp + cuDSS (11/12 reported
at the tighter tolerance), versus 83.202 ms batched CuPy, 767.901 ms ARPACK
and 712.493 ms Spectra. The five-cycle Warp run takes 163.301 ms and reports
12/12. The fastest accepted FSAI experiment is 406.322 ms at sigma=-1;
block-Jacobi takes 1016.201 ms at that setting. This motivates the separate
mass-weighted known-nullspace projection work in #42.

## Failures and limits

- All six small direct fixtures and thirty small reference/CG configurations
  pass. The native CuPy diagonal-mass addition passes the lumped fixture and
  explicitly marks the other five unsupported. Some small CG profiles fail to
  report full counts at 1e-11 while passing every external gate; counts are kept.
- One-cycle, 40-mode direct Krylov is inaccurate (residual about 9e-3); two
  cycles pass. The 20-mode one-cycle run passes 1e-8 but reports only 19/20;
  the table uses two cycles and full counts.
- Default-tolerance CuPy LOBPCG fails the complete-subspace gate for twenty and
  forty modes despite small residuals. The stricter accepted reruns are shown;
  the failed times are excluded from performance claims.
- Five-cycle, ncv=32 block iteration fails the clamped six-mode gate at both
  large sizes. Ten cycles pass. These failures are retained alongside the
  faster successful Krylov profiles.
- On the free body, sigma=-10/one cycle can miss the sixth rigid mode even with
  a tiny physical residual. The spectrum/subspace checks reject it. Two cycles
  recover all six. At sigma=-100/one cycle all three direct/BJ/FSAI profiles
  fail, recovering only two rigid modes. Stronger shifts are not universal safe
  performance settings; repeated-space budgeting remains part of #31.
- Pure Warp CG graphs have no HOST nodes or host/device transfers. cuDSS
  graphs have no HOST callbacks or DtoH nodes but retain vendor HtoD nodes
  (#29), so their `contract_accepted` graph flag remains false even when the
  numerical result passes. DOT inspection is not a host API synchronization trace.
- These fixtures do not replace Anvil/dragon timings or establish a production
  default for all meshes. The Anvil CG gap (#15) remains open; this report adds
  large consistent-mass and multi-mode evidence for subsequent improvements.

## Regression validation

The final source suite passes **242 tests in 149.24 seconds**. The initial
stricter repeated-bending count failure and subsequent explicit application
profile are recorded in [validation provenance](../../results/fem_validation.json).
The unchanged 10,000-DOF A/B/C direct benchmarks all pass, with replay ratios
1.00027, 0.99992 and 0.99956 against the saved matched-factor baseline; no
accuracy or graph regression was detected. See the
[raw regression report](../../results/fem_existing_regression.json).
No full upstream CuPy/Spectra, complex, or Anvil rerun is implied. This change
adds coverage and reference-adapter improvements, with no production solver
arithmetic changes.

## Reproduction

Use the [pinned environment and native Spectra build](README.md).

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q
python -m benchmarks.applications.fem_compare --out results/fem_small.json
python -m benchmarks.applications.fem_compare --side 10 --cases clamped_consistent \
  --modes 20 --ncv 64 --out results/fem_large_k20.json
```

`--baseline` compares identical fixture hashes and solver parameters on the same
GPU/runtime, rejecting accuracy/graph regressions and accepted uncontended
slowdowns above 1.25. Different cycle budgets or preconditioners are separately
labeled experiments, not identical-configuration regression passes.
