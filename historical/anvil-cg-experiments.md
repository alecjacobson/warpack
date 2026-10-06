# Anvil CG tuning and experiments

Archived measurements and experiment notes from the README before its documentation cleanup. Results, test counts, and environment descriptions refer to the runs recorded here. Run commands from the repository root.

[Current README](../README.md) · [Historical index](README.md)

The anvil's pure-Warp solve is now **7.06× faster**, with a smaller original-problem residual. The same L40, mesh, material, mass matrix, and 20 elastic modes are used. Times are medians of three fresh solves after warmup; each replay starts from the same saved random vector and clears any experimental solve history. Factor/preconditioner setup, assembly/transfers, JIT/capture, and final residual validation are outside solve timing. The six rigid motions are now constructed analytically in Warp and projected out of the inverse operator; all 20 elastic modes are still computed numerically and validated, together with the six analytic rigid modes, against the original 26-mode reference.

| Method | Factor/preconditioner setup (s) | Solve (s) | Max original residual, all 26 | Passes 1e-7 cutoff |
|---|---:|---:|---:|---|
| Original block-Jacobi CG | 0.009 | 67.843 | 1.83e-07 | no |
| Retuned block-Jacobi CG | 0.008 | 16.953 | 1.09e-08 | yes |
| FSAI CG, 32-entry rows | 1.143 | 10.442 | 4.11e-09 | yes |
| FSAI CG, 64-entry rows | 5.048 | 9.608 | 3.04e-09 | yes |
| Warp + cuDSS recheck | 2.206 | 0.467 | 2.61e-09 | yes |

Rigid-basis setup adds **0.0155 s** once. The revised CG runs use inverse shift **+10,000**, a **120-vector** Krylov space, and inner relative tolerance **1e-12**. These are settings tuned for the normalized anvil, not universal defaults. The shift belongs to the inverse iteration; physical eigenvalues and residuals use the original unshifted operator. All three revised configurations pass the shared residual, orthogonality, and mode-shape checks. Their largest relative elastic-frequency difference from the reference is **6.36e-14**. The reference eigensolutions are used only for independent validation, never as initial guesses or recycling vectors.

FSAI is the unmodified implementation from [Warp PR #16](https://github.com/alecjacobson/warp/pull/16), pinned to `04427b997fe7ff07bc6ed1dde5ca9813496c6b55`. Both FSAI cases use `max_step_size=3`, `apply_lanes=8`, and **FP32 factor storage with FP64 products and CG arithmetic**. The 32-entry factor is a useful balance for one-off solves: about 1.14 s setup and 10.44 s solve. The 64-entry factor saves about 0.83 s per solve but adds about 3.90 s of setup. FSAI construction synchronizes and is outside graph capture; applying the preconditioner and the complete eigensolve remain graph capturable. No local block-Jacobi or FSAI factorization kernels were added.

The first measured solve takes **61,531 CG iterations** with the retuned block Jacobi, **22,825** with 32-entry FSAI, and **18,592** with 64-entry FSAI, across 120 inverse applications. The fastest graph's median host enqueue is **1.311 ms**, while its CUDA-event interval is **9.6080 s**. Its graph contains only device-to-device copies, with no host-copy or host-callback nodes. One-second occupancy samples found no other GPU processes during the timed CG replays. Standalone sparse-product timings are included as diagnostics; cached-RHS microbenchmarks are not an exact attribution of full CG runtime.

Other experiments:

- **Warm starts:** reusing the previous solution, scaling it by an energy projection, and recycling 16 energy-orthogonal solution directions all took about **12.6 s**, essentially the same as zero starts, in the 16-entry FSAI screen. History was cleared before every full eigensolve; these timings do not reuse a previously solved answer. The inner right-hand sides are Krylov directions, so a previous solution is a weak predictor here.
- **Inner tolerance:** relaxing to 1e-11 gave borderline or failed original residuals near 1e-7. The reported configurations keep 1e-12.
- **Device convergence checks:** benchmark-only batching of 1/4/8/16 CG steps produced less than 1% improvement for FSAI in a single-RHS screen. The reported implementation retains upstream's normal device checks.
- **Next experiments:** a block eigensolver that reuses previous elastic subspaces for changing Hessians, and a stronger global/coarse-space preconditioner. These remain hypotheses, not measured speedups. Warm-starting a sequence of changed problems is different from the fresh-solve benchmark above.

The new APIs are `OrthogonalComplementOperator`, `fem.rigid_body_basis`, prebuilt Warp preconditioners in `CGInverse`, and an optional device `initial_vector` in `KrylovSchur`. For a connected free body, the core setup is:

```python
import warp as wp
from warp.optim.linear import FSAI
from warpack import CGInverse, EigenpairEvaluation, KrylovSchur
from warpack import OrthogonalComplementOperator, SparseOperator, rigid_body_basis
from warpack.fem import mass_normalized

# positions is a device vec3d array; H and lumped nodal mass come from assemble_rest.
a = mass_normalized(H, mass)
shifted = mass_normalized(H, mass, shift=10000.0)
pre = FSAI(shifted, max_row_size=32, max_step_size=3,
           apply_lanes=8, factor_dtype=wp.float32)
inverse = CGInverse(shifted, preconditioner=pre, tol=1e-12, maxiter=10000)
rigid = rigid_body_basis(positions, mass)
solver = KrylovSchur(OrthogonalComplementOperator(inverse, rigid),
                     20, ncv=120, which="LA", tol=1e-12)
evaluation = EigenpairEvaluation(SparseOperator(a), solver.eigenvectors, tol=1e-7)
graph = solver.capture(1, adaptive=False, finalizer=evaluation.run)
wp.capture_launch(graph)
# evaluation.values/residuals and solver.eigenvectors stay on device.
```

**111 tests pass** with the PR modules. The stock Warp 1.15 compatibility checks also pass (six tests; four upstream block-Jacobi cases skipped). Lint, formatting, and wheel building pass. See [final timings and per-replay accuracy](../results/anvil_cg_optimized.json), [cuDSS recheck](../results/anvil_cudss_recheck.json), [exploratory results including failed configurations](../results/anvil_cg_search.json), and [validation/provenance](../results/anvil_cg_validation.json).

All measurements use the same official Warp 1.18.0+cu12 native runtime as the previous comparison, with the three unmodified PR Python modules overlaid for this driver. Reproduce without changing the default installation:

```bash
python -m pip install --no-deps --target build/warp-fsai \
  'https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
for f in warp/_src/optim/linear.py warp/_src/optim/fsai.py warp/optim/linear.py; do
  curl --fail --location \
    "https://raw.githubusercontent.com/alecjacobson/warp/04427b997fe7ff07bc6ed1dde5ca9813496c6b55/$f" \
    --output "build/warp-fsai/$f"
done
# The script creates a cuDSS validation reference if one is not cached in build/.
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.cg_experiments --mode eigen --shifts 10000 \
  --preconditioners bj fsai32f fsai64f --ncv 120 --tol 1e-12 \
  --project-rigid --repeats 3 --profile --out results/anvil_cg_optimized.json
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.cg_experiments --mode eigen --shifts 1 \
  --preconditioners cudss --ncv 64 --repeats 3 --out results/anvil_cudss_recheck.json
```
