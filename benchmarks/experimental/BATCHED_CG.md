# Batched and mixed-precision inner CG for FEM blocks

Follow-up on 2026-10-19 to [#15](https://github.com/alecjacobson/warpack/issues/15),
[#59](https://github.com/alecjacobson/warpack/issues/59), and setup-accounting
[#62](https://github.com/alecjacobson/warpack/issues/62). The block solver and
these adapters remain experimental, outside the installed `warpack` package.

The new adapter calls upstream Warp's batched `linear.cg` with one independent
coefficient/residual/tolerance sequence per RHS (`batch_offsets`). It does not
implement a replacement CG algorithm or share a single CG coefficient across
the block. FSAI factors still come from the pinned upstream implementation.
Their sparse application batches four RHSs per tile. All numerical work is Warp.

Optional FP32 inner CG casts the shifted operator and CG workspaces at setup,
then casts each residual in and correction out on device. Stiffness/mass products
in the outer iteration, Rayleigh–Ritz, metric orthogonalization and final
original-pencil residuals remain FP64. This uses low-accuracy solves as an
**eigenvector preconditioner**; it does not establish FP32 accuracy for a
shift-invert eigenproblem. Mixed precision is explicit, never a production default.

## Uncontended Anvil comparison

NVIDIA L40, 283,857 DOFs, 20 computed elastic modes plus six analytic rigid
motions. Every accepted row passes the independent spectrum, original residual,
orthogonality and mode-shape checks. All block rows also require 20/20 device
counts at tolerance 1e-8 and returned-pair residual below 1e-8. Medians of three
synchronized warm replays, with unchanged starting vectors:

| Configuration | Solve (s) | Worst original residual | Inner conditional graph nodes |
|---|---:|---:|---:|
| Production FSAI64 + FP64 CG, ncv120 | 9.5782 | 3.305e-9 | — |
| Block + serial FP64 CG10, FSAI32, 40 steps | 7.9033 | 7.734e-9 | 960 |
| Block + batched FP64 CG10, same settings | 7.3614 | 8.150e-9 | 40 |
| Block + batched FP32 CG10, same settings | **4.8388** | **4.262e-9** | 40 |
| Production Warp + cuDSS, ncv64 | 0.4671 | 2.827e-9 | — |

The FP32 block profile is 1.98× faster than the fresh production CG baseline,
but still 10.36× slower than cuDSS. These are different eigensolver/preconditioner
configurations meeting the same external accuracy gates, not a change to the
production API or a claim that the iterative gap is closed. The original README
CuPy+supplied-cuDSS comparison was not rerun in this campaign.

All block profiles use width 24, shift 10000, inner relative tolerance 0.01 with
`atol=0`, maximum ten inner steps and Gram chunk 4096. FP32 FSAI factor storage
is shared by all three block profiles. Each performs 9,600 scheduled vector
iterations; batched accounting includes masked RHSs and is not a sum of their
individual first-convergence counts. Truncated inner solves are expected here.

Profiling a continued single step reduces inner preconditioning from 101.35 ms
(batched FP64) to 37.56 ms (FP32), while total step time falls from 183.43 ms to 119.85 ms.
FP64 outer products/orthogonalization now dominate. Profile categories overlap;
these are diagnostics, not additive full-solve timings. Kernel-node count drops
from 32,565 (serial CG) to 9,685 (batched FP64). Graph audits find only device-to-device
copies, with no HOST callback or host-transfer nodes in the pure-Warp graphs.
This is a DOT audit, not a CUDA API trace.

A stronger FSAI64/32-outer-step screen took 4.267 s but returned only 17/20 converged
pairs and residual 9.226e-8. It is retained as **rejected**, not a faster accepted
profile. No acceptance bound was relaxed.

## Coverage beyond Anvil

Three replays on each independent tetrahedral fixture pass all
20 modes, original-pencil residuals, requested spectra and complete invariant
subspaces, M-orthogonality and device status/count checks. The coupled consistent
mass is used directly; it is not replaced with a diagonal approximation.

| Clamped beam | DOFs | Fixed outer budget | Median replay (s) |
|---|---:|---:|---:|
| Consistent mass, side 10 | 10,890 | 40 | 0.6794 |
| Lumped mass, side 20 | 79,380 | 60 | 2.5522 |

Here each benchmark replay restores the same starting block; changed RHS data
are exercised separately by the unit regressions. These conservative budgets
oversolve the beams and are accuracy coverage, not claims of improved FEM
performance outside Anvil. Their graph audits contain only device-to-device
copies and no HOST callback nodes. Independent CPU references are outside the
timing intervals.

Twenty-two new adapter regressions cover scalar and 3×3 sparse products, seven RHSs
including zero and scales from 1e-8 to 1e8, in-place output, changed-input captured
replay, independent stopping, FP32/FP64 factors, upstream block Jacobi and coupled
mass eigenproblems with FP32 inner preconditioning and FP64 output accuracy.
Padded sparse row storage is rejected in both working precisions; callers must
compress it first, matching the production sparse operator contract.

## Setup boundaries and reproduction

The runner now separates `preconditioner_setup_seconds`, `solver_setup_seconds`
and their sum `total_setup_seconds`. Solver setup includes operators, CG state,
block workspaces, device starting-vector upload and the optional FP32 sparse
matrix copy. Mesh assembly, host start generation and reference construction are
outside those intervals. Capture/JIT warmup, first graph replay and timed replays
are separate. JIT work is included in whichever setup/capture phase encounters
it; setup is a single process-local observation, not a warmed median.
Historical `setup_seconds` retains its meaning as preconditioner-only setup.
Older records lack the workspace measurement and must not be interpreted as
end-to-end setup. The unnecessary serial wrapper in the batched branch is removed.

For the accepted FP32 Anvil run, preconditioner setup is 1.05995 s, solver setup
0.79914 s, total 1.85909 s, and capture 2.75564 s. Those values include compilation
encountered in that process and should not be compared directly with warmed
historical factor-only setup rows.

All raw reports, rejected screens and per-run occupancy records are retained in
[`results/performance_followup/`](../../results/performance_followup/).
GPU process sampling observed no competing compute process during these runs.
Sampling does not reserve the GPU and can miss brief jobs. A reusable monitor
now follows benchmark worker descendants as well as the direct child.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu --out build/anvil.occupancy.json --log build/anvil.log -- \
  python -m benchmarks.experimental.compare_block --case anvil --width 24 \
  --steps 40 --shift 10000 --fsai-width 32 --inner-cg 10 --batched-cg --inner-fp32 \
  --inner-tol .01 --gram-chunk 4096 --batched-products --repeats 3 --profile \
  --out build/anvil-fp32.json
```

Omit `--inner-fp32` for batched FP64, then omit `--batched-cg` for serial FP64.
The mesh and cached reference are the same local inputs as the main Anvil
showcase; no new large CPU factorization is run.

Validation: the full local suite passes **472 tests** in 211.28 s. A final adapter
rerun passes **22 tests** in 2.08 s, including two added compact-storage rejection
checks. Lint, formatting and isolated wheel build pass; the wheel contains only
the 18 core modules. [Validation provenance](../../results/performance_followup/validation.json).
This does not rerun the full upstream CuPy/Spectra conformance suites.
