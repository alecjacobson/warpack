# Starting subspaces for related FEM solves

`SymmetricEigensolver` and `GeneralizedEigensolver` accept caller-owned device
starting blocks. This implements part of [#59](https://github.com/alecjacobson/warpack/issues/59)
and [#17](https://github.com/alecjacobson/warpack/issues/17). It does **not** add
LOBPCG or close the fresh-Anvil iterative performance gap in
[#15](https://github.com/alecjacobson/warpack/issues/15).

## API and replay

```python
starts = wp.clone(previous.eigenvectors)  # (k, n), FP64 device storage
solver = GeneralizedEigensolver(
    stiffness,
    mass,
    current_inverse,
    k=k,
    ncv=ncv,
    initial_vectors=starts,
    tol=1e-8,
)
graph = solver.capture(8, adaptive=False)
wp.capture_launch(graph)

# Reuse the same graph and pointer with updated starting data.
wp.copy(starts, solver.eigenvectors)
wp.capture_launch(graph)
```

There must be between one and `ncv` independent rows. Initialization copies
those rows, seeds any remaining rows randomly, then orthogonalizes in the
appropriate metric. The caller's block is never modified. `solver.q` on these
block solvers holds the full current `ncv`-row iteration subspace; a separate
`wp.clone(previous.q)` can preserve more information than only the requested
modes. This feature does not change the single-vector Krylov interfaces.

All numerical initialization work is in Warp on device. Structural shape,
dtype and device checks occur at construction. Invalid/dependent numerical
blocks are not automatically repaired; inspect device `status` after a solve.
The symmetric Cholesky diagnostic now also marks nonfinite pivots as failures.
An invariant starting space can exclude wanted modes, so residual convergence
alone is not proof of requested-spectrum completeness.

Updating starts does not update a factorization. Supply an inverse matching the
current stiffness/shift/mass, and respect that inverse's own value-update and
capture requirements. The benchmark below creates new cuDSS factors for the
changed matrix; it does not claim captured refactorization.

## Related-problem comparison

The [harness](fem_starts.py) independently assembles a clamped tetrahedral beam,
computes its modes with Warpack, then increases stiffness by 2% in the right
half of the beam. Mass stays fixed. Starting vectors come exclusively from the
first Warpack solve. Independent SciPy ARPACK references are used only to check
the source and changed problems, including the requested subspace.

The larger fixtures have **10,890 DOFs**, six requested modes, `ncv=16`, and
shift `-0.01`. Three replay trials are checked at every budget. Acceptance
requires original residual, spectrum error, mass orthogonality and requested
subspace error below `1e-8`, zero solver status and all six device-converged
pairs at the same `1e-8` tolerance. Smaller budgets that fail remain in the
JSON. The full previous subspace includes ten additional approximate directions.

For both mass fixtures, the first accepted budgets among 4, 6, 8 and 12 steps
were:

| Starting block | Accepted steps | Consistent-mass worst residual | Lumped-mass worst residual |
|---|---:|---:|---:|
| Fresh random | 8 | 1.75e-10 | 2.17e-10 |
| Previous six modes + random padding | 6 | 2.14e-10 | 2.31e-10 |
| Previous full 16-dimensional subspace | 4 | 2.92e-9 | 4.40e-9 |

These are convergence/work-budget comparisons, **not speedup claims**. Other
GPU jobs were present, so every timing is marked ineligible. The existing
fresh-solve performance results and Anvil README table are unchanged. A dedicated
GPU run is still required for performance regression acceptance.

The CuPy comparison uses LOBPCG with the same normalized initial block and
the same batched cuDSS factorization, padded to width 16. It passes independent
accuracy checks for all three starts. Consistent-mass residual-history lengths
were 46 for fresh starts, 43–45 for previous modes and 31–32 for the full previous
space. Those lengths include history bookkeeping and must not be equated with
Warp inverse-iteration counts. CuPy computes all 16 modes, while acceptance
checks the first six. Its absolute stopping tolerance is `1e-10`; original
problem acceptance is shared. This is a reference adapter supplied with cuDSS,
not a native CuPy cuDSS backend.

Results: [consistent mass](../../results/fem_block_starts_consistent_large.json),
[lumped mass](../../results/fem_block_starts_lumped_large.json).
Earlier 2,646-DOF exploratory results are retained for
[consistent](../../results/fem_block_starts_consistent.json) and
[lumped](../../results/fem_block_starts_lumped.json) mass. Those used the stricter
internal `1e-11` tolerance; their original physical acceptance and stricter
device-count acceptance are recorded separately.

## Anvil preconditioned-block screen

Before implementing another eigensolver, a
[reference-only screen](anvil_lobpcg_screen.py) applies upstream Warp FSAI
directly as a CuPy LOBPCG preconditioner. It uses mass-normalized stiffness,
FSAI on `A+10000 I`, 64-entry rows, FP32 factor storage, FP64 vectors/products,
and six analytic rigid constraints. The first 20 elastic modes are checked
against the existing independent Anvil reference.

| Block width / maximum steps | Worst original residual | Accepted |
|---|---:|---|
| 20 / 200 | 1.21e-2 | No |
| 20 / 400 | 2.92e-4 | No |
| 32 / 300, retain first 20 | 3.16e-9 | Yes |

The accepted run also has maximum relative eigenvalue error `6.88e-14`,
orthogonality error `8.15e-13` and minimum elastic MAC above `0.99999999999999`.
It used 6,449 preconditioner-vector applications. The random seed is fixed, but
changing block width changes the random columns; this is a feasibility screen,
not an isolated causal comparison of width alone.

All three runs were contended and include cold work, so their times are not
performance evidence. A shift-100/width-24 screen failed during FSAI setup with
CUDA out of memory while unrelated jobs were active; no convergence conclusion
is drawn from it. [All results and the rejected setup](../../results/anvil_lobpcg_screen.json)
are preserved. No CPU Anvil factorization was launched.

The wider-block result warrants a pure-Warp implementation experiment. Stable
handling of dependent search directions, repeated modes and near-zero residuals
must precede performance claims. The basis-selection and stopping analysis in
[Duersch et al., 2018](https://arxiv.org/abs/1704.07458) is a relevant starting
point. Oversampling also enlarges projected eigensolves, so fewer preconditioner
applications alone do not establish an improvement over current FSAI-CG.

## Reproduction and regression coverage

Use the [pinned upstream Warp environment](README.md). Run GPU jobs sequentially.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_block_starts.py tests/test_metric_qr.py
python -m benchmarks.applications.fem_starts --side 10 --cupy \
  --budgets 4,6,8,12 --out results/fem_block_starts_consistent_large.json
python -m benchmarks.applications.fem_starts --case clamped_lumped --side 10 \
  --cupy --budgets 4,6,8,12 --out results/fem_block_starts_lumped_large.json
python -m benchmarks.applications.anvil_lobpcg_screen --block-size 32 --steps 300 \
  --out build/anvil_lobpcg_repeat.json
```

Focused tests exercise current device block contents after graph capture,
changed matrix values, standard and non-diagonal-mass problems, repeated modes,
partial-block random padding, nonmutation of caller storage, adaptive replay,
invalid block metadata and dependent/nonfinite block diagnostics. Test reference
eigenvectors are permitted to establish exact starting-span behavior; benchmark
starts are always computed by the solver.

Validation: **359 full-suite tests passed** on the pinned Warp 1.18 environment;
**26 focused block-start/metric-QR tests passed** on minimum-supported Warp
1.15. Lint and a clean staged wheel build passed; the wheel contains only the
18 core modules. See the [validation record](../../results/block_starts_validation.json).
