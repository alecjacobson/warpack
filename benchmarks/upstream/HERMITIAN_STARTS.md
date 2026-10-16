# Complex Hermitian starting vectors, 2026-10-16

`HermitianEigensolver` now accepts an optional `initial_vector`: a finite,
nonzero `wp.vec2d` array of shape `(n,)` on the operator device. Each entry holds
the real and imaginary parts. The solver reads the current contents during
initialization, including graph replay, and never modifies the input array.
This extends the standard Hermitian API; complex shift-invert starts and
generalized complex mass are still unsupported.

```python
# A is a HermitianCSR; start is a device wp.vec2d array of shape (A.complex_n,).
solver = HermitianEigensolver(A, k=6, which="LA", initial_vector=start)
graph = solver.capture(100)
wp.capture_launch(graph)
# Update start in place with Warp operations, then replay the same graph.
```

The implementation converts the complex vector to the realified layout with
Warp kernels before each Krylov initialization. It scales by the largest finite
component before normalization, avoiding overflow/underflow when computing
squared norms. A nonzero start in the matrix's nullspace remains valid; the
existing on-device Krylov breakdown recovery explores independent directions.

Wrong dtype, shape or device raises `ValueError` at construction. Zero or
nonfinite contents set `solver.real.status` to **1 on device**. Outputs from that
replay must not be used. A valid replacement start clears the status on the next
replay. Invalid starts use a finite internal placeholder solely to avoid
propagating NaNs; this is not a successful solve of the requested input.

Numerical decisions and normalization stay on device. The existing setup warmup
synchronization is unchanged; no host readback or per-replay synchronization is
added. `ncv` still counts real basis vectors and must exceed `2*k`. A supplied
start does not certify convergence or discovery of the full requested spectrum.

## Focused correctness validation

No performance benchmark or large FEM run was made. The Anvil timing table is
unchanged. Tests ran sequentially on an NVIDIA L40, driver 570.158.01, with the
pinned Warp 1.18.0+cu12 environment.

- **101 local tests pass** across Hermitian, multiplicity, symmetric solver,
  Arnoldi, Schur and upstream-report contracts. This is not a full-suite rerun.
- All **33 Hermitian tests also pass on stock Warp 1.15**. Repository lint,
  changed-file formatting and a clean wheel build pass. The wheel contains 18
  core Python modules and excludes benchmarks and examples.
- The **33 Hermitian tests** include eight new parameter cases: changed complex
  starts for LM/LA/SA on a unitarily rotated matrix; exact nullspace starts;
  scaling by `1e-250` and `1e250`; input immutability; zero/NaN/Inf diagnostics
  followed by valid replay; and dtype, length, dimensionality and device checks.
  Initialization is checked against an independent normalized complex vector,
  and solves against dense eigenvalues, original residuals and conjugate
  orthogonality. Three graphs with supplied starts contain no host nodes or
  HtoD/DtoH transfers.
- **15 newly enabled CuPy cases pass**, with **60 upstream fixture skips** in
  the selected methods. These cover Gaussian-rotated repeated spectra,
  nullspace starts, dominant eigendirections absent from the start, and a
  two-dimensional exact-eigenvector start. Assertions and upstream inputs are
  unchanged. [Raw report](../../results/hermitian_starts_cupy.json).

The adapter now forwards complex starts for ordinary Hermitian LM/LA/SA solves.
The **48 complex SM start cases remain skipped**, with the corrected reason
that complex shift-invert lacks a starting-vector API. This resolves 15 of the
63 historical complex-start skips in [#17](https://github.com/alecjacobson/warpack/issues/17);
it does not close the broader API tracker or claim full upstream conformance.
The [earlier breakdown report](HERMITIAN.md) remains the record for #3.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_hermitian.py tests/test_multiplicity.py \
  tests/test_solver.py tests/test_arnoldi.py tests/test_schur.py \
  tests/test_upstream_contracts.py
WARPACK_UPSTREAM_REPORT=/tmp/hermitian-starts-cupy.json \
  python -m pytest -q -p benchmarks.upstream.cupy_plugin \
  build/test_upstream_cupy_linalg.py \
  -k 'complex128 and (test_clustered_large_k_gaussian or test_null_space_start or test_dominant_eigenvalue_hidden_from_v0 or test_n2_exact_v0)'
```
