# Hodge nullspaces and topology coverage

This checkpoint expands class D / [#23](https://github.com/alecjacobson/warpack/issues/23)
from one torus into oriented sphere, disk, annulus, torus, disconnected-torus,
and weak-component fixtures. The same fixtures drive correctness tests and
comparisons. Production solver arithmetic is unchanged.

The contract is `K x = lambda M x` with symmetric positive semidefinite K and
positive definite M. These problems **retain** harmonic modes. They differ from
FEM vibration workloads that discard rigid motions.

## Independent references and acceptance

`hodge.py` assembles exact oriented incidence operators B1 and B2 and checks
`B1 B2 = 0`. Small references use independently computed incidence ranks,
`null_space([B1; B2.T])`, and dense generalized eigendecomposition. Their first
Betti numbers are 0, 0, 1, 2, 4, and 3. The disconnected and weak-component
fixtures have non-diagonal SPD mass matrices; the weak annulus stiffness is
scaled by `1e-4`.

The large fixture consists of two periodic triangular tori with 10,086 edge
unknowns. Topological ranks give four harmonic modes; independent coordinate
cochains give their subspace. A Fourier formula for the primal and dual graph
Laplacians supplies the full eigenvalue reference without dense diagonalization.
Small dense tests verify this formula at three resolutions.

The new fixtures request all known harmonic modes and two positive guard modes.
Each measured trial must pass original-pencil residual, requested-spectrum and
mass-orthogonality checks at `1e-8`, recover the independently known nullity, and
have a positive guard above `1e-8`. The M-norm projection error of the returned
nullspace against the independent harmonic basis must also be below `1e-8`.
Tests use tighter bounds and compare the returned subspace with the entire
allowed eigenspace when the positive cutoff splits a repeated cluster.

These are fixture acceptance checks, **not a production Betti-count API**.
Tests reject a complete-count claim when all requested values are zero, or a
weak positive eigenvalue lies below the numerical cutoff, even for exact
eigenpairs. A positive guard alone does not certify completeness in arbitrary
repeated spectra. Device diagnostics and capacity/ambiguity states are tracked
in [#39](https://github.com/alecjacobson/warpack/issues/39).

## Comparison protocol

All workers run sequentially on the same L40. The new fixtures use `ncv=32`
(clamped for small matrices), 20 fixed Warp outer cycles, and three synchronized
solve samples. Setup, reference construction, capture, output conversion and
validation are excluded from solve timing and recorded separately where
applicable. Every timed trial is checked; failed or unsupported rows remain in
the reports. These topology fixtures establish an initial baseline.

- ARPACK uses a prefactored SuperLU shift-invert solve and explicit `ncv`.
- Spectra uses the native generalized shift API and Eigen SparseLU.
- `cupy-eigsh` uses native CuPy `eigsh(..., which="SA")` only for identity mass;
  it explicitly rejects the two non-diagonal-mass fixtures.
- `cupy` uses generalized CuPy LOBPCG with a cuDSS preconditioner. Its sphere and
  disk rows are unsupported because CuPy switches to dense solving below `5*k`.
- `warp` uses block `GeneralizedEigensolver` and SPD cuDSS factors.
- `warp-cg` uses that block solver and pure-Warp CG with upstream diagonal
  preconditioning. This is not the Anvil FSAI configuration. Large comparisons
  record per-trial inner iterations and unconverged right-hand sides.

CuPy eigsh, ARPACK and Spectra request `1e-11`; LOBPCG requests `1e-8` under its
absolute stopping convention. All face the same external acceptance checks.
The harness now passes ARPACK's reported `ncv` explicitly for all modes. Older
application reports used SciPy's default in some paths despite recording a
requested budget. Their files are preserved; the changed implementation label
prevents using those reference timings as identical-configuration baselines.
LOBPCG records its block size rather than claiming an `ncv` budget.

Graph DOT audits inspect nested bodies for host callbacks and transfer kinds.
They are not API traces. Vendor cuDSS HtoD nodes remain tracked in
[#29](https://github.com/alecjacobson/warpack/issues/29); numerical success is
reported separately from the strict device-only graph contract.

## Recorded results (L40)

The [composite](../../results/hodge_comparisons.json) contains **38 passed,
1 failed and 4 unsupported** configurations from four separate runs, not a new
all-cases rerun. Both the failed default LOBPCG run and its tighter-tolerance
follow-up remain. The [validation record](../../results/hodge_validation.json)
records **223 passing local tests**, lint/formatting and wheel checks. A first
suite run caught the guard requirement being applied to the older two-zero-only
fixture; the final explicit per-fixture contract fixes that validation mismatch.

Large two-torus case: n=10,086, k=6, four exact null modes, first positive
reference value 0.00782325135673. All timings below passed every external gate
on all three trials and were uncontended.

| Backend | Median solve | Worst residual | Nullspace projection error |
|---|---:|---:|---:|
| ARPACK | 210.637 ms | 6.63e-14 | 8.28e-14 |
| Spectra | 148.599 ms | 4.10e-12 | 5.64e-14 |
| CuPy native eigsh | 203.885 ms | 3.73e-12 | 4.70e-11 |
| CuPy LOBPCG + cuDSS, tol=1e-10 | 62.467 ms | 7.69e-11 | 8.25e-10 |
| Warp block + cuDSS, 20 cycles | 28.576 ms | 5.55e-13 | 7.44e-14 |
| Warp block + diagonal CG, 20 cycles | 1814.656 ms | 5.16e-13 | 8.44e-14 |

Both Warp paths report 6/6 converged at `1e-11`. CG performs 640 RHS solves per
replay, 75,671–75,688 accumulated iterations, at most 207 per RHS, and zero
unconverged inner solves. The roughly 63.5x CG/cuDSS gap is a measured remaining
optimization target in [#40](https://github.com/alecjacobson/warpack/issues/40),
separate from the tuned Anvil FSAI results. Warp+cuDSS is
about 5.2x faster than Spectra and 2.2x faster than the accepted CuPy LOBPCG
configuration on this fixture; this does not include setup or capture.

Large setup times are 145.7 ms ARPACK, 150.3 ms Spectra, 361.4 ms native CuPy,
374.0 ms tight LOBPCG, 264.1 ms Warp+cuDSS and 111.8 ms Warp+CG. Warp capture is
151.3 ms with cuDSS and 1611.0 ms with CG. These are one-time observations, not
repeated setup benchmarks; warmup and every solve sample are in the raw files.

Default LOBPCG (`tol=1e-8`) takes 49.705 ms but fails the independent nullspace
projection gate at 4.17e-8 despite residuals up to 6.66e-9 and correct eigenvalues.
It is **not an accepted timing**. Explicit `--lobpcg-tol 1e-10` passes without
weakening acceptance. The default remains unchanged because earlier large-bar
experiments showed that tighter LOBPCG stopping can over-iterate other fixtures.

The six small fixtures all pass on applicable backends; CPU solvers win these
small comparisons. At 20 cycles the disconnected non-diagonal-mass case passes
the external `1e-8` gate but reports only **4/6 at `1e-11`** for both Warp paths
(residual about 8.26e-10). The 100-cycle regression test requires and obtains
6/6. The short-budget timing is not labeled fully converged at the tighter
solver tolerance.

All audited Warp graphs have zero HOST callbacks and zero DtoH copies. CG
has only DtoD copies; cuDSS has 40 vendor HtoD nodes per 20-cycle graph. Thus
CG passes the strict device-only graph check and cuDSS does not, independently
of numerical accuracy.

Source/data provenance:

- [Small comparisons](../../results/hodge_small.json): source recorded as dirty
  parent `18b8d75`, implemented by `64be4fe`.
- [Native CuPy small comparisons](../../results/hodge_cupy_eigsh_small.json):
  dirty `64be4fe`, implemented by `c6ebe65`. **Metadata erratum:** these native
  calls use `1e-11`, although that file's tolerance field incorrectly says
  `1e-8`; the harness field is fixed, the original report is preserved.
- [Large comparisons](../../results/hodge_large.json): numerical/harness source
  `7b77d82`; pending documentation/results explain its dirty flag.
- [Tight LOBPCG follow-up](../../results/hodge_lobpcg_tight.json): source `405707c`.
- [Existing torus regression](../../results/hodge_existing_regression.json):
  unchanged legacy matrices, k=2 and 100 cycles, compared against the saved
  matching checkpoint rather than claiming the new fixtures have a past baseline.
  Both accuracy and timing gates pass, with time ratios 0.99985 and 1.00023.

## Reproduction

Use the pinned environment and native Spectra build in the
[application protocol](README.md).

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q
python -m benchmarks.applications.hodge_compare --out results/hodge_comparisons.json
python -m benchmarks.applications.hodge_compare --cases large_disconnected --methods cupy \
  --lobpcg-tol 1e-10 --out results/hodge_lobpcg_tight.json
python -m benchmarks.applications.hodge_compare --out results/hodge_followup.json \
  --baseline results/hodge_comparisons.json
```

The baseline gate requires matching GPU/runtime, matrix hashes and solver
configuration, accepted accuracy and uncontended timings; a slowdown above
1.25 or a numerical/graph regression fails. Comparison workers have an explicit
180-second wall limit. Timeouts are preserved rather than converted into wins.

## Device evidence follow-up

[NumericalNullspaceDiagnostics](NULLSPACE.md) now evaluates physical and metric
evidence after the solve inside one graph. It reports weak positives, capacity
and invalid states explicitly, with completeness always unverified. The
10,086-unknown saved solver regression passes. An extreme weak_scale=1e-10
case exposes a separate pre-existing non-finite block-solver failure (#46),
retained as an expected-failure regression; the earlier weak_scale=1e-4
coverage remains passing.

The [Hodge CG follow-up](HODGE_CG.md) compares the existing mass-Krylov outer solver at larger sizes, retains failed short budgets, and reduces the tested iterative cost without changing production defaults.
