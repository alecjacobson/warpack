# Device-selected active CG batches

The block solver can now avoid preconditioning residual rows that already meet
its physical tolerance. `DeviceBucketedCG` remains an experimental adapter around
**upstream Warp CG**, outside the installed numerical package. It packs rows,
selects preallocated batch sizes using CUDA conditional nodes, and unpacks the
corrections. It implements no separate CG algorithm or factorization.

The active set is rebuilt from `BlockLOBPCG.errors` after each fresh original
operator/mass application. A row can reactivate after a Ritz update; there is no
permanent historical lock. Omitted corrections are zero, while the corresponding
vectors remain in the outer Ritz basis. Every final original-pencil residual,
requested spectrum/subspace, orthogonality and status/count check is retained.
This differs from CuPy's permanent soft-lock mask implicated in
[#69](https://github.com/alecjacobson/warpack/issues/69).

The inner CG solves remain deliberately truncated; nonzero unconverged-RHS
counters are reported rather than relabeled successful linear solves. Acceptance
uses the fresh outer eigenpair checks.

## Capture and workspace contract

The adapter needs the pinned upstream batched-CG implementation with preallocated
loop state. It rejects older states. Shapes, pointers and sparse topology stay
fixed; FP32 working coefficients refresh from the current FP64 source inside the
selected solve. Factors remain caller-managed. The current stream's capture
state selects implementation plumbing, never a numerical host decision.
Outside capture, warmup runs the same packed solve at full width, avoiding
`capture_if`'s uncaptured host condition readback. During replay, selection is
entirely on device. All bucket states are compiled and synchronized during
setup, before nesting them in the outer graph.

Anvil uses widths 4, 8, 12, 16, 20 and 24. A selected bucket can contain up to
three padded rows; the report counts that work rather than claiming exact active
width. Original and working sparse values and FSAI factors are shared; packed
buffers and CG workspaces are distinct. This adds memory and setup to save
replay work. Setup/capture free-memory decrease is a pool/runtime observation,
not a peak-memory measurement or exact live allocation count.

Sixteen independent regressions cover FP32/FP64 inner arithmetic, scalar/3x3
sparse storage, in-place application, empty/full/changing active sets, row
reactivation, changed coefficient replay and nested conditional graphs. The
outer tests include SPD/indefinite symmetric stiffness with an SPD shifted
preconditioner, coupled mass, repeated modes, rigid exclusions and computed warm
starts across mode crossings. Explicit starting-block perturbation is retained
for crossing tests; residuals alone still cannot certify spectral completeness.

The focused CG suite passes **49 tests in 3.07 s**, including the 16 new
cases; lint and formatting pass. The installed numerical package is unchanged.
The latest full local suite remains the prior 593-test run; this is not a new
full or upstream rerun. [Validation and hashes](../../results/performance_followup/active-device-cg-validation.json).

## Anvil comparison

The first complete three-trial run gives **3.3398 s Warp** versus **3.1462 s
CuPy with the same FSAI32/FP32 CG15**, about a 6.2% gap. All 20 elastic modes pass
every independent gate; both backends also pass their own outer stopping checks.
Warp's maximum paired residual is **9.8015e-9** and original residual
**1.9603e-8**. Warp uses normalized tolerance `1e-8`, width24; CuPy uses absolute
`1e-5`, width20. Shared physical gates, rather than identical stopping formulas,
define acceptance. No claim of universally faster or identical algorithms.

Warp takes 28–29 outer steps, **8,580–8,820 scheduled vector CG iterations** and
564–572 truly active residual rows over the whole solve. Selected buckets account
for 572–588 RHS slots; the difference is within-bucket padding. Before this
change, the fresh [public-API control](BLOCK_API.md) took 3.9814 s with 29–31
steps and 10,440–11,160 scheduled CG iterations. The improvement combines less
inner work and a different outer convergence trajectory; it is not a perfectly
isolated kernel speedup. The earlier single-trial screen at 3.3245 s versus
CuPy 3.3507 s is retained and is not used to claim parity.

Setup is **1.2002 s**, capture **.5735 s** and setup/capture free-memory decrease
**4.308 GB**, versus **3.301 GB** in the preceding non-bucketed control. The
captured graph contains 13 conditional nodes, 536 kernels and 37 device copies;
only the selected bucket body executes. The DOT audit finds no host callback or
host/device transfer. CuPy's separate active-width adapter shares the factors;
its extra memory/setup is recorded separately and is not a native CuPy factor
construction or an independent end-to-end setup measurement.

[Three trials](../../results/performance_followup/anvil-active-device-cg15.json),
[single-trial screen](../../results/performance_followup/anvil-active-device-cg15-screen.json).
The adjacent occupancy files contain no observed competitors. GPU jobs ran
sequentially; occupancy sampling can miss very short competing jobs.

The [FSAI64/CG10](../../results/performance_followup/anvil-fsai64-cg10-screen.json)
and [FSAI64/CG15](../../results/performance_followup/anvil-fsai64-cg15-screen.json)
controls pass accuracy but do not improve the earlier non-bucketed setting:
4.0666/4.3658 s respectively, with about 4.9 s setup. These are single-trial
screens, not accepted median speedups, and no default factor width changed.

## Other matrix classes

The 39,366-DOF Hodge run passes all four harmonic modes, two positive guards,
original equations, orthogonality, topology/subspace and strict counts. Warp
is **72.96 ms**, CuPy **119.81 ms** with the same CG10, versus the preceding
76.27 ms Warp control. CuPy uses explicit `5e-11` absolute tolerance and meets it
on freshly returned vectors. All checks pass in all three trials.
[Raw Hodge report](../../results/performance_followup/hodge-active-device-cg10.json).

For a 2% regional stiffness change in the 10,890-DOF consistent-mass beam,
three-trial medians are:

| Start | Warp active CG10 (s) | CuPy same CG10 (s) |
|---|---:|---:|
| Random | .29045 | .42344 |
| Previous 20 modes + random guards | .22116 | .32308 |
| Previous full 24-row block | .17262 | .31180 |
| Previous full block with explicit 1e-3 exploration | .18691 | — |

All shared gates and full-block native CuPy tolerances pass. The exploration
row is an additional Warp policy measurement, not a matched-start CuPy
comparison. Source modes are computed in Warp; reference vectors never enter
starts. [Raw consistent-mass report](../../results/performance_followup/related-consistent-active-device.json).

The 79,380-DOF lumped-mass case also passes every shared/native gate:

| Start | Warp active CG10 (s) | CuPy same CG10 (s) |
|---|---:|---:|
| Random | .72633 | 1.15223 |
| Previous 20 modes + random guards | .49900 | .74877 |
| Previous full 24-row block | .44540 | .66858 |
| Previous full block with explicit 1e-3 exploration | .47987 | — |

The full/perturbed warm blocks take 16/17 outer steps. All trials pass; the
exploration row retains its separate-policy interpretation.
[Raw large related-FEM report](../../results/performance_followup/related-lumped-active-device.json).

An additional [CG18 screen](../../results/performance_followup/anvil-active-device-cg18-screen.json)
is slower: Warp 3.5951 s, CuPy 3.3591 s, with all accuracy checks passing.
It does not replace the three-trial CG15 profile. The final adapter queries the
current stream's capture state specifically; earlier screens used the device's
capture-state property. These coincide in the recorded single-stream runs, but
the narrower query also avoids uncaptured condition readback when a different
stream is being captured.

## Reproduce

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_active_cg_experimental.py
python -m benchmarks.experimental.compare_block --case anvil \
  --width 24 --steps 200 --shift 10000 --fsai-width 32 --inner-cg 15 \
  --inner-tol .01 --inner-fp32 --batched-cg --active-cg-bucket-size 4 \
  --batched-products --gram-chunk 4096 --symmetric-products \
  --fused-updates --paired-rotation --cupy --cupy-width 20 --repeats 3 \
  --out build/anvil-active.json
```

Related-FEM commands in [BLOCK_API.md](BLOCK_API.md) accept the same
`--active-cg-bucket-size 4` option. The public block solver and its defaults are
unchanged; these allocation/dispatch policies remain experimental. The remaining
Anvil performance gaps are tracked in #67/#15, and the Hodge direct-solve gap in
#40. cuDSS's host metadata transfers (#29) are separate.
