# Block eigensolver experiments and historical evaluation

Latest performance: [device-selected active CG batches](ACTIVE_CG.md) reduce the
Anvil median to 3.3398 s, versus 3.1462 s for the matched CuPy comparison, with
extra preallocated memory and fresh residual-based row reactivation.

Current API: [`warpack.BlockLOBPCG`](../upstream/BLOCK_API.md). Only the numerical
iteration and required dense block kernels moved into the installed package.
The batched CG and sparse-product experiments, FEM construction, reference
comparisons and profiling tools remain here. Compatibility imports preserve
older benchmark commands. Existing Krylov defaults are unchanged.

The latest [coefficient-refresh report](BATCHED_REFRESH.md) records Anvil at
3.7232 s versus 3.0469 s for CuPy with the same supplied CG. The
[matched-CG report](MATCHED_CG.md) explains active-width adapter memory/setup
costs and computed warm starts. New API and exploration validation is in the
[public block report](BLOCK_API.md).

The original evaluation below is historical, including its earlier limitations
and contended timings. Current `BlockLOBPCG` accepts 1..width independent FP64
starting rows, fills missing rows randomly, optionally perturbs supplied rows,
and optionally normalizes invariant constraints in SPD mass. It does not
support arbitrary KKT constraints, singular mass or complex arithmetic.
All numerical iteration and replay decisions use Warp. Preconditioners use
upstream Warp FSAI, block Jacobi or CG; optional supplied cuDSS inverses retain
their separate fixed-capture and host-metadata-transfer limitations.

## Method and accuracy regressions

Each iteration uses `[X, preconditioner(R), P]`. It preserves the current X
block with two-pass metric Cholesky QR, projects the new/history directions
against X and any invariant constraints, and applies rank-revealing metric
whitening twice to those search directions. Inactive directions receive a
finite bound outside the requested projected spectrum. Rayleigh--Ritz selects
the smallest modes, followed by fresh products and original-equation residuals.
All workspaces are allocated before capture. Direct preconditioners support a
device-controlled outer loop. The batched CG adapter also supports adaptive
outer capture when the upstream state preallocates conditional-loop storage.
The original serial-CG measurements below use fixed outer graphs; production
`CGInverse` retains that default and provides an explicit `adaptive_capture=True`
opt-in for compatible states. Stock Warp 1.15 rejects that opt-in.

A rejected formulation whitened the entire `[X,W,P]` together. With a 24-wide
block on a 100-dimensional fixture, accurate eigenvectors degraded from about
1e-9 residual to 1e-5 when residual/history directions became dependent.
Preserving X fixes this regression: continued iterations remain below 1e-10.
The [rejected log](../../results/block_lobpcg/prototype-smoke72.log) and
[corrected log](../../results/block_lobpcg/prototype-smoke72-v1.log) are retained.

Independent tests cover SPD and indefinite stiffness, repeated eigenvalues,
identity and coupled SPD mass, rigid constraints, zero/dependent search blocks,
invalid starts, continued iteration after convergence, changed-pencil graph
replay, and truncated CG. They check requested spectra and invariant spaces,
original residuals, mass orthogonality, device status and convergence counts.
Product tests include partial tiles, changed input scales, scalar and 3x3 BSR
matrices, and a partial group of batched right-hand sides.

The experiment exposed a production API omission:
[#60](https://github.com/alecjacobson/warpack/issues/60). Upstream CG treats `tol`
as both relative and absolute tolerance when `atol` is omitted. For residual
preconditioning with loose `tol=0.01`, this returns zero corrections once the
RHS norm falls below 0.01. `CGInverse` now accepts `atol=0.0` to request relative
stopping, subject to upstream dtype safety floors. Its default is unchanged.
The [pre-fix failure](../../results/block_lobpcg/cg-absolute-floor-failure.log)
is retained alongside the scale-dependent CG regression tests.
The prototype deliberately permits truncated inner solves; their unconverged
counts are reported, and only independent outer accuracy certifies the modes.

## Anvil convergence and profiling

The matrix has 283,857 DOFs. We request 20 elastic modes with width 24 and
exclude six analytic rigid motions; validation includes all 26 modes against
the existing independent reference. The preconditioner uses `A + 10000 I`.
FSAI uses upstream FP32 factors and FP64 products. All runs below used the
pinned Warp 1.18 FSAI environment documented in the main README.

**Every timing in the original experiments below is ineligible for a performance claim:** other
GPU jobs were running. These are numerical/work-budget screens, not replacements
for the uncontended Anvil baseline. Setup and capture are recorded separately.

| Preconditioning | Block steps | Inner CG iterations | Maximum original residual | Accepted |
|---|---:|---:|---:|---|
| Direct FSAI64, original products | 322 (adaptive) | — | 1.97e-8 | yes |
| Direct FSAI32, tuned products | 409 (adaptive) | — | 1.98e-8 | yes |
| FSAI32 + up to 20 CG steps | 30 (fixed) | 14,394 | 2.17e-9 | yes |
| FSAI32 + up to 10 CG steps | 40 (fixed) | 9,600 | 5.63e-9 | yes |
| FSAI32 + up to 5 CG steps | 60 (fixed) | 7,200 | 5.87e-8 | **no** |

The CG rows use a larger Gram chunk (4096) and experimental batched 3x3 BSR
products. They are not an isolated preconditioner comparison with the first
row. The direct FSAI32 row uses the same products, starting block, shift and
preconditioner as the CG rows, providing a matched convergence control: it
requires 409 outer steps, compared with the accepted 40-step CG10 budget.
All accepted rows have zero device status and all 20 pairs converged.
Spectrum errors are below 1.1e-13, mass-orthogonality errors below 3e-15,
and minimum elastic MAC is effectively one. The 10-step CG row exhausts its
inner budget on every RHS: an accurate inverse application is unnecessary for
this preconditioning role. The current production Anvil inverse iteration
uses 18,592 inner CG iterations, but the block algorithm has additional costs;
iteration counts alone do not establish faster execution.

The five-step CG screen converges only 16 of 20 requested pairs. Its returned
eigenpair residual is 2.94e-8 under the solver's relative normalization, above
the 1e-8 requirement, despite accurate eigenvalues and a physical residual
below the older Anvil 1e-7 threshold. It is rejected rather than weakening the
acceptance criterion to favor a smaller CG budget. The harness now also checks
the returned eigenvalues directly, in addition to the shared Anvil validator's
independently recomputed Rayleigh quotients.

Graph DOT inspection finds only device-to-device copies and no host callback
nodes in all completed Anvil configurations. The fixed CG10 graph contains
960 inner conditional nodes and 32,565 kernel nodes, including their bodies;
this graph size is another reason to investigate a less expensive stopping
strategy. DOT inspection is not a CUDA API trace and excludes setup/validation.

Single-step profiling identifies Gram products, basis rotations and stiffness
products as the main costs, larger than FSAI application. In these contended
samples, increasing the Gram chunk and using upstream tiled sparse products
reduced the observed step from 318 to 259 ms. The profiles overlap categories,
use only two initial iterations, and are **not accepted full-solve benchmarks**.
Larger dense rotation tiles were all slower in the tested samples; the original
rotation tile remains the default. The 72x72 projected problem uses an existing
fused Jacobi kernel only when sufficient shared memory is available, with the
production dispatch otherwise unchanged. Wider blocks still need efficient
projected eigensolves and cost/accuracy validation.

Artifacts: [direct FSAI](../../results/block_lobpcg/anvil-prototype24.json),
[matched FSAI32 control](../../results/block_lobpcg/anvil-fsai32-block500.json),
[CG20](../../results/block_lobpcg/anvil-cg20-block30.json),
[CG10](../../results/block_lobpcg/anvil-cg10-block40.json),
[rejected CG5](../../results/block_lobpcg/anvil-cg5-block60.json),
[original profile](../../results/block_lobpcg/anvil-prototype-profile.json),
[product profile](../../results/block_lobpcg/anvil-profile-products.json),
[rejected rotation tuning](../../results/block_lobpcg/rotation-tiles.json).

## CuPy reference and unresolved observations

A [consistent-mass FEM comparison](../../results/block_lobpcg/fem-prototype32.json)
uses 2,646 DOFs, six requested modes, width 12 and identical upstream FSAI32
preconditioning. Both Warp and CuPy LOBPCG pass independent accuracy checks.
Warp uses 31 iterations (worst residual 6.46e-9); CuPy's residual history has
93 entries (worst residual 1.89e-11). History length is not a Warp iteration
count. CuPy converges the entire oversampled block while Warp stops on the
first k, and their native stopping criteria differ. The shared original-pencil
acceptance checks are the comparison gate; elapsed times are contended.
CuPy is only a benchmark adapter, not part of production computation.

An FSAI64 FEM setup ran out of GPU memory while unrelated jobs occupied much
of the device; its [failure log](../../results/block_lobpcg/fem-prototype.log)
is retained. FSAI32 was used for the completed comparison.

Three Gram-product tests once returned doubled results immediately after
unsynchronized eager warmup and graph capture. Isolated and repeated runs
initially passed. A [follow-up investigation](FOLLOWUP.md) now reproduces the
failure without Gram arithmetic, including a CuPy-only control with explicit
graph upload. Tests complete warmup before capture, matching the solver protocol;
this contains the observed failure but is **not an upstream runtime fix**. The
[failure log](../../results/block_lobpcg/experimental-tests.log) is preserved and
[#61](https://github.com/alecjacobson/warpack/issues/61) tracks reproduction and
root-cause investigation. Experimental product dispatch will remain outside
production pending that work.

The same [follow-up](FOLLOWUP.md) validates computed starts after a regional
FEM stiffness change, including a retained profile that failed the independent
requested-subspace check. It demonstrates a warm-start iteration reduction
within the prototype without claiming an Anvil speedup.

## Reproduction and next gates

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_lobpcg_experimental.py tests/test_block_products_experimental.py tests/test_cg_tolerances.py
python -m benchmarks.experimental.compare_block --case clamped_consistent --side 6 --modes 6 --width 12 --fsai-width 32 --cupy --out build/block-fem.json
python -m benchmarks.experimental.compare_block --case anvil --width 24 --steps 40 --shift 10000 --fsai-width 32 --inner-cg 10 --inner-tol .01 --gram-chunk 4096 --batched-products --repeats 1 --out build/block-anvil.json
```

Anvil needs the existing local mesh and `build/anvil_reference.npz`; this
harness does not launch a new CPU sparse factorization. Run on an otherwise
idle GPU before drawing timing conclusions. Remaining gates include wider
projected eigensolves, an efficient stopping strategy with truncated CG,
matched large-case CuPy comparisons, related-FEM starts, and reproducible
end-to-end performance against the existing CG and cuDSS paths. Neither #15
nor #59 is closed by this prototype.
