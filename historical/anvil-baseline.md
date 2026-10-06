# Anvil baseline comparison

Archived measurements and experiment notes from the README before its documentation cleanup. Results, test counts, and environment descriptions refer to the runs recorded here. Run commands from the repository root.

[Current README](../README.md) · [Historical index](README.md)

The same **283,857-DOF** free-body anvil problem is solved for **six rigid plus 20 elastic modes**, using a 64-vector subspace, the same starting vector restored every trial, and `(M⁻¹/² H M⁻¹/² + I)⁻¹` for all inverse-based rows. These are median wall times from three synchronized trials after warmup, on an NVIDIA L40 or Intel Xeon Platinum 8362 (four BLAS threads). Mesh I/O/assembly, input transfers, JIT/warmup/graph capture, and final residual evaluation are excluded from solve time and reported separately where measured. Setup is the sparse factorization or preconditioner setup, not an entire cold application start.

| Method | Setup (s) | Solve (s) | Max elastic residual | Max residual incl. rigid | Accepted |
|---|---:|---:|---:|---:|---|
| ARPACK + SuperLU | 449.2954 | 42.6216 | 8.651e-10 | 4.519e-09 | yes |
| Spectra + Eigen LDLT | 799.6008 | 62.4199 | 2.167e-09 | 3.078e-08 | yes |
| CuPy eigsh + cuDSS | 2.1534 | 0.4587 | 1.498e-09 | 2.852e-09 | yes |
| Warp + cuDSS | 2.1534 | 0.4670 | 1.523e-09 | 2.595e-09 | yes |
| Warp + upstream block-Jacobi PCG | 0.0091 | 67.8428 | 8.687e-08 | 1.835e-07 | **no** |

Residuals use `||Av−λv||₂ / max(1,|λ|)` on the original mass-normalized operator, with unit-norm vectors. PCG reports the worst residual across all three timed replays; acceptance requires all 26 residuals below 1e-7, orthogonality error below 1e-8, and matching elastic mode shapes. Across accepted references the maximum relative elastic-frequency difference from Warp+cuDSS is **6.550e-14**. This is a CPU/GPU comparison with different inverse backends, not a claim about eigensolver iteration alone.

CuPy's inverse-based row uses **the same cuDSS factorization** as Warp. Native CuPy `eigsh(A, which="SA")`, with no inverse or preconditioner, was also run with a 2,000-iteration budget: **0.9428 s**, maximum original residual **1.592e+02**, **not converged**. That bounded attempt is not a competitive accepted-solve timing. ARPACK uses SuperLU with `MMD_AT_PLUS_A`, zero diagonal-pivot threshold, and symmetric mode for the SPD shifted matrix. Spectra runs entirely in C++ with Eigen's AMD-ordered sparse LDLᵀ, with no Python matvec callback. These CPU numbers are specific to SuperLU and Eigen’s simplicial LDLᵀ; they do not benchmark other CPU factorization packages.

The pure-Warp row calls **upstream `warp.optim.linear.preconditioner(..., ptype="block_jacobi_sequential")`** and upstream CG. There is no local block-Jacobi factorization or application kernel. Blocks are the full 3×3 nodal diagonals. Inner CG uses relative tolerance 1e-13, a 10,000-iteration limit, and device-side stopping. For the first timed eigensolve, the recorded CG counters are **251,160 total inner iterations**, **4,811 maximum per RHS**, **0 unconverged RHSs**, and **64 inverse applications**. Host enqueue takes **1.332 ms**, while device execution takes **67.8417 s**. CUDA graph inspection finds 259 device-to-device copies and no host-copy or host-callback nodes. No other GPU process was observed in one-second occupancy samples during the three timed replays. The [raw report](../results/anvil_benchmarks.json) includes graph node types, all timing samples, original residuals, frequencies, orthogonality, and mode/subspace agreement.

PCG matches the elastic frequencies to 3.397e-14 relative error and has minimum elastic MAC 1.000000000000, but it **does not pass** the common all-26-modes residual cutoff. Its recursive inner stopping criterion reports convergence, so this case illustrates why the original-problem residual must be checked. The earlier 1e-12 inner-tolerance attempt is retained under `previous_trials`; tightening to 1e-13 did not eliminate this accuracy limit.

For driver compatibility, all GPU rows use the official **Warp 1.18.0+cu12 runtime**, with the **entire unmodified linear-solver module from upstream main revision `5a0c33d17d847d0be07ae1003263b572aae0729a`**. This is an isolated compatibility environment, not a stock 1.18 installation or the CUDA-13 nightly runtime. The source hash is recorded in the report. The current nightly requires a newer driver than this machine provides. All **104 tests pass** in this environment; see [validation](../results/anvil_benchmark_validation.json).

Reproduce the exact environment without changing the default Warp installation:

```bash
python -m pip install --no-deps --target build/warp-upstream \
  'https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
curl --fail --location \
  https://raw.githubusercontent.com/NVIDIA/warp/5a0c33d17d847d0be07ae1003263b572aae0729a/warp/_src/optim/linear.py \
  --output build/warp-upstream/warp/_src/optim/linear.py
# Use the pinned Spectra checkout described in Validation and benchmarking in the main README.
g++ -O3 -DNDEBUG -shared -fPIC -I/usr/include/eigen3 -Ibuild/spectra/include \
  benchmarks/spectra_native.cpp -o build/spectra_native.so
PYTHONPATH="$PWD/build/warp-upstream:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.anvil --part direct
PYTHONPATH="$PWD/build/warp-upstream:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.anvil --part pcg --pcg-tol 1e-13
```
