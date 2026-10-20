# Truncated block preconditioning for conformal modes

The public `BlockLOBPCG` can solve the [exact conformal pencil](../applications/CONFORMAL.md)
with short CG applications used as preconditioning. The experimental adapter
keeps the singular boundary mass, translation quotient, flat-patch zero modes
and original-equation validation. No installed solver source or default changes.

With `A=K-sigma M` SPD and `sigma<0`, it solves
`sigma M x = gamma A x`, selecting the smallest algebraic values
`gamma=sigma/(lambda-sigma)` in `[-1,0)`. The signed scaling fits the public
smallest-mode block interface. Recovery uses separate device storage for
`mu=gamma/sigma`, preserving the core Ritz values and convergence count, then
normalizes in the original boundary mass and recovers `lambda=sigma+1/mu`.
The full original CSR equations and Euclidean eigenspace are still checked.

Outer arithmetic is FP64. Each preconditioner application uses unmodified
upstream Warp batched CG, an upstream FSAI32 factor stored in FP32, and at most
25 FP32 CG iterations with relative tolerance 0.01 and zero absolute tolerance.
These are **truncated preconditioner applications, not accurate inverse solves**.
Their unfinished RHS counts are reported but are not an inverse-convergence
claim. Both physical and reciprocal outer counts must still equal the requested
number of modes. The explicit shifted matrix still constructs the FSAI factor
and supplies the inner CG products; outer boundary/shifted products are implicit.

## Large accepted comparison

L40, pinned Warp 1.18/FSAI revision from the conformal report, 32,766 reduced
unknowns, 20 modes, shift `-0.001`. The new block configuration uses width 24,
an outer budget of 200 and normalized outer tolerance `1e-10`. Setup and capture
are single measurements; solve is the median of three synchronized trials after
warmup. Every worker runs sequentially with no sampled competing GPU process.

| Method | Setup (s) | Capture (s) | Solve median (s) |
|---|---:|---:|---:|
| Corrected CG, explicit products, preceding profile | 1.4927 | 5.9511 | 12.686540 |
| Corrected CG, implicit products, preceding profile | 1.5210 | 4.3992 | 6.801015 |
| cuDSS, implicit products, preceding profile | 0.4925 | 0.7455 | 0.383698 |
| Block + truncated CG, new profile | 1.5154 | 0.4949 | **0.518807** |

This is a **13.1x** solve improvement over the implicit accurate-CG route and
**24.5x** over its explicit predecessor; cuDSS remains about **1.35x faster**.
The methods have different inner/outer stopping formulas, but all pass the
same original-pencil residual, finite spectrum, complete full-vector subspace,
boundary orthogonality and gauge gates at `1e-8`. The new trials have maximum
original residual `3.43e-12`, full-vector subspace error `3.35e-11`, both counts
20/20 and status 0. They take 22–23 outer steps and 13,200–13,800 scheduled CG
vector iterations, versus 135,836 accurate inverse iterations in the preceding
implicit route. Scheduled batched iterations include masked RHS work and are
not independent first-convergence counts. All 528–552 RHS applications in this
block profile reach the truncated limit without meeting the inner tolerance.

The captured graph contains 859 kernel, 8 conditional, 59 memset and 19 device-to-device
copy nodes, counting conditional bodies. There are no HOST, HtoD or DtoH nodes.
Setup remains host orchestrated. cuDSS's separate vendor-transfer limitation is #29.

A same-configuration three-trial repeat gives **0.521445 s**, ratio **1.005085**,
passing every gate. The shared recovery refactor also passes unchanged-configuration
controls: cuDSS **0.383651 s**, ratio 0.999880; accurate CG **6.799384 s**,
ratio 0.999760. Runtime implementation fingerprints match the recorded baselines.

## Retained screens and reference limitations

- At outer tolerance `1e-10`, CG budgets 10/25/50 take 0.7875/0.5198/0.6345 s in
  one-trial screens; all pass. Increasing inner work is not uniformly better.
- The stricter outer tolerance `1e-12` passes all three trials at 0.6103 s median.
- The large curved **two-mode** case passes with width 8 in 99 outer steps,
 19,800 scheduled CG iterations and 0.7260 s in a single diagnostic trial.
 Its full-vector subspace error is `5.47e-10`. This is coverage, not a repeated
 timing claim or a universal recommendation for the narrower block.
- Native CuPy LOBPCG with the same FSAI32/FP32-CG25 construction, independently
  set up in its worker, passes the small 286-unknown, six-mode control. Exact
  active-width captures avoid padding its shrinking residual blocks. Their
  workspace setup grows with `sum(1..k)` and is included in reference setup.
- On the large 20-mode case, that CuPy variant exhausts its 300-step budget:
  original residual `5.00e-7`, subspace error `2.42e-7`, native history maximum
  `4.03e-7`, freshly evaluated native residual `1.15e-5` against requested
  `1e-10`. Its 3.8724 s trial is **rejected**, not a valid speed comparison.
  This run does not reproduce the projected-factor exception of #99; it adds
  evidence of large-case convergence sensitivity with a different preconditioner.
  No CuPy source is modified, and no cuDSS is used in this variant.

## Reproduction and evidence

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m benchmarks.monitor_gpu \
  --out build/conformal-block.occupancy.json --log build/conformal-block.log -- \
  python -m benchmarks.applications.conformal_compare \
  --side 128 --modes 20 --steps 200 --shift -0.001 --block-width 24 \
  --inner-cg 25 --block-tol 1e-10 --repeats 3 --methods warp-block-cg \
  --out build/conformal-block.json
```

Add `--baseline results/conformal-block-best.json` for same-configuration
regression. The adapter requires the pinned upstream FSAI/batched-CG environment;
stock Warp receives an explicit unsupported-runtime error. Use `--methods cupy-cg
--cupy-tol 1e-10 --diagnostic` to retain the rejected large CuPy comparison.

- [Accepted repeated profile](../../results/conformal-block-best.json),
  [occupancy](../../results/conformal-block-best.occupancy.json).
- [Repeat](../../results/conformal-block-best-regression.json),
  [unchanged inverse controls](../../results/conformal-products-block-control.json).
- [CG10](../../results/conformal-block-cg10.json), [CG25](../../results/conformal-block-cg25.json),
  [CG50](../../results/conformal-block-cg50.json), [stricter outer tolerance](../../results/conformal-block-32766.json).
- [Two-mode coverage](../../results/conformal-block-map.json),
  [rejected large CuPy](../../results/conformal-cupy-cg-screen.json),
  [small CuPy control](../../results/conformal-cupy-cg-small.json).

The tests cover flat/curved patches, FP32/FP64 preconditioning, two/six requested
modes, changed starts, repeat recovery, non-finite failure followed by same-graph
recovery, insufficient budgets, original metric/completeness gates, native CuPy's
shrinking active block and incompatible timing baselines. Application-specific
assembly, gauge and recovery stay outside `warpack`; batched CG composition
also remains experimental.

[Focused validation](../../results/performance_followup/conformal-block-validation.json):
**89 passed** on the pinned runtime; **78 passed, 11 skipped** on stock Warp 1.15.
The skipped cases require upstream FSAI/batched CG; the explicit rejection
guard passes on both runtimes. Lint, formatting and whitespace checks pass.
