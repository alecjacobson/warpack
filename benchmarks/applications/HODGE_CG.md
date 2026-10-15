# Hodge CG: choose the outer iteration before tuning the inverse

Related: [#40](https://github.com/alecjacobson/warpack/issues/40),
[Hodge coverage](HODGE.md), and the separate Anvil/FEM CG gap
[#15](https://github.com/alecjacobson/warpack/issues/15).

The existing `GeneralizedKrylovSchur` substantially reduces CG work on the tested
Hodge pencils. This follow-up changes benchmark selection and coverage only:
**production solver arithmetic and defaults are unchanged**. CG and FSAI are
upstream Warp implementations, without a local replacement recurrence or
factorization. The new explicit benchmark names are `warp-krylov`,
`warp-cg-krylov` and `warp-cg-fsai-krylov`; existing `warp` / `warp-cg` retain
the block iteration used by the original Hodge report.

## Accepted profiles

Two disconnected triangular tori, identity mass, sigma=-1e-3, k=6, ncv=32.
The requested space contains four harmonic modes and two positive guards.
Independent incidence ranks, coordinate harmonic cochains and Fourier spectra
validate the outputs. No reference nullspace is supplied to the solver.

L40, pinned Warp 1.18.0/FSAI environment; three warm synchronized trials.
Every profile in this table passes physical equations, requested values,
orthogonality, the complete four-dimensional harmonic space, positive guards,
and **6/6 strict counts on every trial**. Times are milliseconds:

| Unknowns | Original block + diagonal CG, 20 cycles | Krylov + diagonal CG | Krylov + FSAI-CG | Strict Warp+cuDSS comparator |
|---|---:|---:|---:|---:|
| 10,086 | 1813.434 | 157.564, 1 cycle | **127.514, 1 cycle** | 29.695, block / 20 cycles |
| 39,366 | 4106.841 | 571.197, 2 cycles | **553.897, 2 cycles** | 23.646, Krylov / 2 cycles |

The best recorded CG profiles reduce solve time by **14.2×** and **7.4×**
against the corresponding original block/diagonal profiles. These are changes
to the outer method and budget, not identical-configuration implementation
speedups. FSAI's small advantage at 39,366 unknowns should not be overstated;
diagonal CG has cheaper setup and is close in replay time.

| Unknowns / CG profile | RHS solves | Accumulated CG iterations | Maximum per RHS | Unconverged inner RHSs |
|---|---:|---:|---:|---:|
| 10,086 block / diagonal | 640 | 75,636 | 207 | 0 |
| 10,086 Krylov / diagonal | 32 | 6,563 | 207 | 0 |
| 10,086 Krylov / FSAI | 32 | 4,001 | 130 | 0 |
| 39,366 block / diagonal | 640 | 154,218 | 399 | 0 |
| 39,366 Krylov / diagonal | 55 | 21,413 | 402 | 0 |
| 39,366 Krylov / FSAI | 55 | 11,464 | 224 | 0 |

The counters above are from the first timed trial; all accepted trials report zero
unconverged inner RHSs. Inner tolerance is 1e-13, maxiter 10,000; outer tolerance 1e-11. FSAI uses
32-entry rows, step size 3, eight application lanes and FP32 factors, with
FP64 products and CG. This is not the Anvil row64/shift+10000 configuration.
No benchmark tolerance is relaxed to obtain the improvement.

Setup / capture at 10,086 unknowns: diagonal Krylov 115.5 / 484.3 ms;
FSAI Krylov 153.8 / 495.3 ms. At 39,366 unknowns: 119.3 / 725.0 ms and
164.7 / 725.1 ms, respectively. Setup, capture, host reference construction and
validation are excluded from solve timing. Every pure-Warp CG graph has no HOST
callbacks or host/device transfers in its nested DOT; this is not an API trace.
The cuDSS comparator still has vendor HtoD metadata copies tracked in #29.
No competing GPU process was observed.

## Native comparisons and remaining gap

Fresh 39,366-unknown native comparisons all pass the same physical/subspace
acceptance, milliseconds:

| Method | Solve |
|---|---:|
| ARPACK | 1861.241 |
| Spectra | 882.240 |
| Native CuPy eigsh SA | 748.546 |
| CuPy LOBPCG + cuDSS preconditioner | 68.352 |
| Pure Warp Krylov + upstream FSAI-CG | 553.897 |
| Warp Krylov + cuDSS | 23.646 |

CuPy LOBPCG uses explicitly requested tolerance 1e-10; native algorithmic
convergence criteria differ. Its six-vector block is not the same work budget
as Krylov ncv32. At 10,086 unknowns, the prior report's accepted native results
remain 210.637 ms ARPACK, 148.599 ms Spectra, 203.885 ms CuPy eigsh and
62.467 ms tighter CuPy LOBPCG. Those reference rows were not rerun here.

Pure Warp now beats the native standard eigsh/CPU profiles on both tested
sizes, but remains slower than the best cuDSS-backed methods. #40 stays open
for that remaining gap. This work does not rerun or improve Anvil: its existing
solver already uses Krylov, and its tuned iterative gap remains under #15.

## Rejected and incomplete profiles

The [composite evidence](../../results/hodge_cg_profiles.json) contains **55
profiles: 43 numerical passes and 12 failures**. Of the numerical passes,
31 Warp profiles have full strict counts, eight Warp profiles are strictly
incomplete, and four are native comparisons without Warpack counters.

- Block budgets 1, 2 and 5 fail. At 10 and 15 cycles, all three tested inverse
  paths meet the external 1e-8 numerical gates but report only 4/6 at 1e-11.
  A larger ncv64 with five block cycles also fails the physical checks.
- At 39,366 unknowns, **one Krylov cycle misses the fourth harmonic mode**
  for all three inverse paths. All three pass after two cycles. This directly
  rules out treating the 10k one-cycle result as a universal budget.
- The 10,086-unknown cuDSS Krylov profiles at one and two cycles meet external
  numerical bounds but report only 4/6; their positive-guard residual is about
  9.35e-11. The table uses the fully converged block comparator instead.
- FSAI does not help the original 20-cycle block route: it takes 2511 ms,
  slower than diagonal CG. A stronger preconditioner alone is not the answer
  for this outer trajectory.
- Benchmark-only initial guesses `x0=(bᵀb)/(bᵀAb) b` and
  `x0=(bᵀAb)/(AbᵀAb) b` reduce that FSAI run to about 2095 ms, still slower
  than the diagonal baseline. The first guess makes diagonal CG slower,
  about 1963 ms. Neither is added to the production API. These are scaled-RHS
  guesses, not recycling across changed matrices or captured replays; they
  do not rule out useful warm starts elsewhere.

All 18 smaller combinations of sphere, disk, annulus, torus, disconnected and
weak-component fixtures with the three Krylov inverse paths pass, including
full strict counts. The disconnected/weak fixtures include non-diagonal SPD
mass. These tests preserve the complete nullspace; they do not use elastic-mode
nullspace removal. Fixed-k success remains distinct from certified Betti counting.

## Default regression and validation

[Fresh main controls](../../results/hodge_cg_regression.json) preserve the
original block defaults, matching matrices, starts, inverse configuration and
budgets. Both accuracy/graph/1.25× timing gates pass: replay ratios 1.00024
(cuDSS) and 0.99929 (diagonal CG). The older Hodge CG result lacks preconditioner
and tolerance metadata, so the first historical comparison correctly marked it
noncomparable; the report does not silently fill in those fields.

Three new captured-replay tests check complete harmonic spaces with cuDSS,
diagonal CG and upstream FSAI using the existing mass-Krylov implementation.
The focused Hodge/transform suite passes **33 tests in 9.30 s**; the final full
suite passes **325 tests in 177.09 s**. Lint, formatting and wheel build pass.
Production source is byte-for-byte unchanged from merge `7aec27f`.
[Validation and source provenance](../../results/hodge_cg_validation.json) records
the measured commits and numerical/default-regression gates.

## Reproduction

Use the pinned environment in [the application protocol](README.md). Keep GPU
workers sequential. Example accepted profiles:

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.applications.hodge_compare --cases large_disconnected \
  --methods warp-cg-krylov,warp-cg-fsai-krylov --steps 1 \
  --out results/hodge_krylov_10k.json
python -m benchmarks.applications.hodge_compare --cases large_disconnected \
  --large-side 81 --methods warp-krylov,warp-cg-krylov,warp-cg-fsai-krylov \
  --steps 2 --out results/hodge_krylov_39k.json
python -m benchmarks.applications.hodge_compare \
  --cases sphere,disk,annulus,torus,disconnected,weak_positive \
  --methods warp-krylov,warp-cg-krylov,warp-cg-fsai-krylov --steps 2 \
  --out results/hodge_krylov_small.json
python -m benchmarks.applications.hodge_cg_probe --variant rayleigh \
  --method warp-cg-fsai --out results/hodge_guess_probe.json
```

The probe uses a benchmark-only reference to the upstream CG operator and
leaves its recurrence/preconditioner unchanged. Rejected budgets, raw trials,
source revisions and strict-count classifications are retained in the report.
