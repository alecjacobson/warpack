# Adaptive outer graphs with upstream CG

Latest: [symmetric products and fused projection updates](OUTER_PRODUCTS.md) reduce the accepted Anvil block replay to 3.9028 s, with matched-preconditioner and related-matrix checks.

Latest: the [matched-CG and related-FEM report](MATCHED_CG.md) removes CuPy active-block padding overhead, records the remaining Anvil gap, and compares computed warm starts.

2026-10-19 follow-up to [#64](https://github.com/alecjacobson/warpack/issues/64),
[#15](https://github.com/alecjacobson/warpack/issues/15),
[#40](https://github.com/alecjacobson/warpack/issues/40), and
[#59](https://github.com/alecjacobson/warpack/issues/59).

## Capture compatibility

The pinned Warp 1.18 CG state preallocates conditional-loop control. It can run
inside a device-controlled outer eigensolver loop, including batched upstream CG.
Stock Warp 1.15 allocates that storage inside each solve. If the old guard is
manually bypassed, CUDA rejects nested capture with an unsupported memory
allocation. This is a storage-lifetime restriction, not a prohibition on nesting
conditional loops in general.

Production `CGInverse(..., adaptive_capture=True)` explicitly opts into this
capability. The default remains fixed outer capture. A conservative capability
check requires upstream's preallocated `_cur_iter_and_condition`; no public
upstream capability flag exists. Missing storage rejects the opt-in before
capture, including on stock 1.15. A future upstream layout change may therefore
require updating this check. Upstream CG itself is unmodified.

The experimental batched adapter enables adaptive capture on compatible states.
Its driver accepts `--fixed-outer` to reproduce the older fixed-budget results.
cuDSS still uses fixed outer graphs. These changes do not remove vendor HtoD
metadata copies or change the public API's numerical defaults.

Regressions cover nested production inverse iteration with changed starting
vectors and batched FP32 preconditioning on torus, disconnected and weakly
connected Hodge fixtures. They check every replay's physical residual, requested
spectrum, complete nullspace, orthogonality, status, counts and early stopping.
Graph DOT inspection finds no HOST callbacks or host-transfer nodes; this is not
a CUDA API trace. Stock-version tests preserve default behavior and verify the
explicit rejection.

## Anvil

Same 283,857-DOF problem, 20 elastic modes and six excluded analytic rigid modes.
NVIDIA L40, pinned upstream FSAI32, width 24, maximum ten FP32 inner CG steps,
relative tolerance .01 and `atol=0`, FP64 outer arithmetic, shift 10000, Gram
chunk 4096, unchanged accuracy gates and device tolerance 1e-8.

| Configuration | Median warm replay (s) | Outer steps | Worst original residual |
|---|---:|---:|---:|
| Production FSAI64 + FP64 CG | 9.5782 | See baseline | 3.305e-9 |
| Experimental fixed block, batched FP32 CG | 4.8388 | 40 | 4.262e-9 |
| Experimental adaptive block, same settings | **4.4616** | 36–38 | 1.801e-8 |
| Production cuDSS | 0.4671 | See baseline | 2.827e-9 |

All three adaptive replays return 20/20 converged pairs and status zero, pass
independent spectrum/mode-shape/M-orthogonality checks, and have returned-pair
residual below 1e-8 (worst 9.005e-9). The independent original residual uses a
different normalization, as in the earlier runs. Adaptive stopping avoids some
oversolving; it is not a claim of identical final error. Floating-point variation
changes the stopping step across restored-start replays. Scheduled inner vector
iterations are 8,640–9,120, including masked RHSs.

The adaptive graph has two CONDITIONAL nodes and 361 KERNEL nodes, versus 40
inner conditional nodes in the fixed graph. Total preconditioner/workspace setup
is 1.12974 s and capture is 1.01845 s. Compiled kernels were cached; these are
single observations, not cold-start or factor-only comparisons. See the
[setup accounting](BATCHED_CG.md#setup-boundaries-and-reproduction).

## Large disconnected Hodge problem

Two disconnected triangular tori, side 81: 39,366 edge unknowns, four exact
harmonic modes and two positive guards. Identity mass is represented implicitly
in the experimental driver, without changing the pencil. Scalar sparse block
products now batch four RHSs per tile, sharing the existing FSAI product kernel.
A Gram chunk of 256 avoids the oversized small-Gram launch used for Anvil.

Fresh controls and accepted experimental runs, medians of three synchronized
warm solves on the same GPU/CPU environment:

| Method | Solve (ms) |
|---|---:|
| SciPy ARPACK | 1414.95 |
| Spectra | 866.55 |
| CuPy native eigsh, SA | 733.08 |
| CuPy LOBPCG + supplied cuDSS preconditioner | 67.68 |
| Production Warp Krylov + cuDSS | 23.64 |
| Production Warp Krylov + FSAI CG | 553.84 |
| Experimental adaptive block, width 8, FP32 CG | 84.90 |
| Experimental adaptive block, width 6, FP32 CG | **79.08** |

These compare accepted configurations, not identical algorithms or inner
preconditioners. Native LOBPCG and the final block run use six vectors; Krylov
references use ncv32. The block profile uses FSAI32, shift .001, inner CG10,
relative tolerance .01, `atol=0`, and outer tolerance 1e-11. Independent gates
check the Fourier spectrum, physical residuals, orthogonality and the complete
four-dimensional harmonic subspace. All three final replays pass, stop at 24
outer steps, and return six converged pairs with status zero. Worst residual is
4.761e-12, harmonic-subspace error 1.650e-10, and the first positive eigenvalue
is 0.00200537614371825. The graph has two conditional nodes, 133 kernel nodes,
and only device-to-device copies. Total setup is .07349 s; capture is .20476 s.

The 79.08 ms profile is 7.00× faster than production CG, but remains 3.34× slower
than cuDSS and 1.17× slower than CuPy LOBPCG with cuDSS. The earlier 80-step,
width-12 screen took 847 ms; reducing to width 8 and 40 fixed steps took 290 ms.
Both were accurate but oversolved. Their single-trial results remain diagnostic
screens, not accepted timing comparisons. Width 8 adaptive takes 23 steps; width
6 takes 24, with less block work. This does not establish universal defaults.

## Reproduction and records

All reports and occupancy records are in
[`results/performance_followup/`](../../results/performance_followup/):
`anvil-block-adaptive.json`, `hodge-block-width6.json`,
`hodge-block-adaptive.json`, `hodge-large-before.json`, and
`hodge-references.json`. The fixed Anvil baselines are in the
[batched-CG report](BATCHED_CG.md). Occupancy sampling detected no competing GPU
compute process; sampling cannot exclude brief jobs or reserve the GPU.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu --out build/hodge.occupancy.json --log build/hodge.log -- \
  python -m benchmarks.experimental.compare_block --case hodge_large_disconnected \
  --side 81 --width 6 --steps 100 --shift .001 --tol 1e-11 \
  --fsai-width 32 --inner-cg 10 --batched-cg --inner-fp32 --inner-tol .01 \
  --gram-chunk 256 --batched-products --repeats 3 --out build/hodge-adaptive.json
python -m benchmarks.monitor_gpu --out build/anvil.occupancy.json --log build/anvil.log -- \
  python -m benchmarks.experimental.compare_block --case anvil \
  --width 24 --steps 100 --shift 10000 --tol 1e-8 \
  --fsai-width 32 --inner-cg 10 --batched-cg --inner-fp32 --inner-tol .01 \
  --gram-chunk 4096 --batched-products --repeats 3 --out build/anvil-adaptive.json
```

The block solver remains outside the installed package. Public promotion still
requires broader coverage, a settled API, and comparisons using the same
preconditioner with CuPy; the reported CuPy Hodge preconditioner uses cuDSS.
Production changes here only expose compatible adaptive capture explicitly.

Validation: **500 local tests pass** in 195.17 s; **13 stock-Warp 1.15 tests
pass** in 9.09 s. Lint, formatting and clean-source wheel build pass. The wheel
contains only the 18 core modules. [Validation provenance](../../results/performance_followup/adaptive-validation.json).
