# Supported block starts and mode-crossing exploration

`warpack.BlockLOBPCG` now contains the validated FP64 numerical iteration. Its
benchmark compatibility module only reexports the implementation. Tuning
switches remain opt-in, and existing Krylov solvers retain their defaults.
Batched inner CG and batched sparse-product experiments remain outside the
installed package. The public API also accepts production `CGInverse`, cuDSS
or upstream FP64 Warp preconditioners; their capture requirements still apply.
See the [API example](../../examples/block_reuse.py) and
[upstream coverage](../upstream/BLOCK_API.md).

## Warm starts and crossings

A small residual only proves that a vector approximates an eigenvector. It
does not show that an arbitrary supplied block contains the requested low
spectrum. For a diagonal matrix with old eigenvalues 1..64, raising its first
eight eigenvalues by 100 makes 9..12 the requested lowest four. Reusing the
exact old eight-vector block makes both Warp and CuPy return 101..104 with
zero residuals. Reusing a *computed* old block also fools CuPy in the recorded
control. Warp's correction step can escape through roundoff, but applications
must not rely on that contamination as an exploration policy.

`initial_perturbation=1e-3` explicitly adds seeded random exploration to the
supplied rows, relative to their Euclidean norms, before projection and metric
orthogonalization. It runs on device during every initialization/replay and
leaves input arrays untouched. The default is zero: callers can retain exact
starts when they know the target subspace. Perturbation improves exploration;
it is not a mathematical completeness certificate, especially for weakly
coupled components, extremely small noise or an inadequate iteration budget.

Regressions cover exact and computed old blocks, ordinary/coupled SPD mass,
simple/repeated low modes, unchanged sparse topology with changing values,
partial starts, normalized invariant constraints, and invalid input status.
The independent requested spectrum/subspace checks are separate from residual
counts. The upstream Fiedler case and native controls are recorded in the
[crossing reproducer](../../results/performance_followup/block-api-crossings.json).
Original failures are retained in the
[exact-start](../../results/performance_followup/block-crossing-exact-start.json)
and [computed-start](../../results/performance_followup/block-crossing-full-start.json)
records. Tracking: [#71](https://github.com/alecjacobson/warpack/issues/71),
[#72](https://github.com/alecjacobson/warpack/issues/72).

The full local suite passes **593 tests in 243.80 s** in the pinned FSAI
environment. **90 focused block/API/product tests pass in 47.45 s on stock
Warp 1.15.0**, including all 19 new public API tests. Lint and formatting pass.
[Validation and source hashes](../../results/performance_followup/block-api-validation.json).

## Capture, packaging and API boundaries

Initialization, seeded exploration, metric normalization, rank decisions,
residuals and stopping stay on device. Changing sparse values, starts and
constraint-basis values is supported with stable storage and topology;
preconditioner-factor updates remain the caller's responsibility. Normalized
constraint bases use private device storage. Operators/preconditioners must
be graph capturable, and the outer graph follows their adaptive/fixed policy.
The [clean-source wheel check](../../results/performance_followup/block-api-wheel.json)
contains 20 `warpack` modules, imports the new API away from the source tree,
and contains no examples, benchmarks, FEM/skinning assembly or experimental CG.

## Related FEM comparisons

A 2% stiffness increase in the right half of a tetrahedral beam, k=20,
width=24, FSAI32 with upstream FP32 CG10 and FP64 outer iteration. Three
uncontended L40 trials per row; both unperturbed backends use the same supplied
start and preconditioner. CPU references only validate outputs. All shared
physical, requested-spectrum/subspace and orthogonality gates pass, as do
Warp status/counts and CuPy's fresh native absolute tolerance for all 24 rows.

| Start | Warp consistent, 10,890 DOFs (s) | CuPy same CG (s) | Warp lumped, 79,380 DOFs (s) | CuPy same CG (s) |
|---|---:|---:|---:|---:|
| Random | .31130 | .41400 | .82778 | 1.15730 |
| Previous 20 modes + random guards | .25688 | .32545 | .61249 | .75111 |
| Previous full 24-row block | .17823 | .28266 | .55014 | .73844 |
| Previous full block, explicit 1e-3 perturbation | .20553 | — | .58125 | — |

The last row measures Warp's optional exploration policy, not a matched-start
CuPy comparison. It takes 15 steps versus 13 without perturbation on consistent
mass, and 19 versus 18 on the larger lumped case. It retains most of the
warm-start benefit. These modest related changes do not create a mode crossing;
the separate analytic/coupled regressions test that failure mode directly.
New source blocks can differ after floating-point product changes, so these
figures do not isolate the performance effect of moving code into the package.

Setup is recorded separately: host FEM/reference construction and initial
uploads are outside replay; source solves, factor/CG setup, operator/workspace
construction, capture and first replay are separate fields. CuPy's changing
active-width adapter preallocates additional CG graphs/workspaces; its recorded
free-memory decrease is not peak memory or independent end-to-end setup.
[Consistent data](../../results/performance_followup/block-api-related-consistent.json),
[lumped data](../../results/performance_followup/block-api-related-lumped.json).
The adjacent occupancy files record no competing GPU processes; sampling cannot
exclude arbitrarily short competing jobs.

## Large Anvil and Hodge revalidation

The promoted code passes the 283,857-DOF Anvil comparison in all three fresh
trials. Warp takes **3.9814 s median**, versus **3.0982 s** for CuPy with the
same FSAI32/CG15. Warp uses 24 rows and 29–31 correction steps; CuPy uses 20.
Worst paired residual is **9.9853e-9**, worst original residual **1.9971e-8**;
all status/count, requested-spectrum/subspace and native CuPy tolerance gates
pass. The prior accepted 3.7232 s Warp median is retained in
[BATCHED_REFRESH.md](BATCHED_REFRESH.md). These data do not establish a speedup
from promotion; small floating-point differences can change the step count.

Setup totals **1.1250 s**, capture **.6085 s** (JIT charged where encountered).
Device free memory decreases by **3.301 GB** between prepared fixtures and
completed setup/capture, including pools/runtime allocations: this is not peak
memory or a live-workspace-only measurement. CuPy adapter memory is separately
reported and shares the Warp factors. The graph has two conditional nodes,
340 kernels and 26 device-to-device copies, with no host nodes or host/device
copies in the DOT audit. [Raw Anvil report](../../results/performance_followup/block-api-anvil.json).

The 39,366-DOF disconnected Hodge case also passes all shared and native
stopping gates: **76.27 ms Warp** versus **119.21 ms CuPy**. Four harmonic modes
and two guard rows are checked against independent topology/subspace evidence.
The CuPy tolerance is explicitly tightened to `5e-11` to avoid its known
soft-locking tolerance miss (#69); that stricter native criterion passes too.
The graph audit shows only device kernels/memsets/copies and two conditional
nodes. [Raw Hodge report](../../results/performance_followup/block-api-hodge.json).

## Reproduce

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_block_api.py
python -m examples.block_reuse
python -m benchmarks.experimental.mode_crossings --out build/crossings.json
python -m benchmarks.experimental.fem_starts \
  --case clamped_consistent --side 10 --modes 20 --width 24 --steps 200 \
  --fsai-width 32 --tol 1e-9 --repeats 3 --inner-cg 10 --inner-fp32 \
  --inner-tol .01 --batched-products --gram-chunk 1024 \
  --symmetric-products --fused-updates --paired-rotation \
  --initial-perturbation .001 --cupy --cupy-tol 1e-9 --out build/related.json
```

For the larger beam use `--case clamped_lumped --side 20`. The source solves
and independent CPU references are necessary preparation, not timed replay.
