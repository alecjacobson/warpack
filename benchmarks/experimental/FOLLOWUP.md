# Related FEM starts and graph-upload investigation

Follow-up on 2026-10-16 to [#59](https://github.com/alecjacobson/warpack/issues/59)
and [#61](https://github.com/alecjacobson/warpack/issues/61). These are experimental
results, not a new public block solver or an Anvil speedup claim.

## Computed starts after a material change

The [driver](fem_starts.py) independently assembles a clamped tetrahedral beam,
computes its first 20 modes with the pure-Warp block prototype, and increases
stiffness by 2% in the right half of the beam. The changed solve uses an upstream
FSAI32 preconditioner constructed for the changed pencil, shared by all starting
strategies. No reference eigenvectors enter the computation. The block has 24
rows, so the previous full block includes four additional approximate directions.

At 10,890 DOFs, with device residual tolerance 1e-9 and three replays per start:

| Starting block | Consistent mass: iterations | Lumped mass: iterations |
|---|---:|---:|
| Fresh random | 81 | 82–83 |
| Previous 20 modes + four random directions | 59 | 63–64 |
| Previous full 24-row block | 46 | 51 |

The larger **79,380-DOF** lumped-mass beam also passes all 20-mode checks:
fresh starts require **216–221** iterations, previous modes plus random padding
require **154–155**, and the previous full block requires **110** in every
trial. Original residuals and requested-subspace errors stay below 1.1e-9.
The [large-case record](../../results/block_followup/starts-lumped-large.json)
retains every replay and the separately validated source solve.

All 27 changed-problem trials across these three fixtures pass the same independent original-residual,
requested-spectrum, mass-orthogonality and requested-subspace checks, as well
as zero device status and all 20 device-converged pairs. This demonstrates a
warm-start work reduction **within this prototype**, not superiority over the
existing solvers. The source solve and both preconditioner setups are reported
separately. They must be accounted for when the previous solution is not already
available from an application.

An initial consistent-mass run used device tolerance 1e-8. The previous-full-block
start stopped at 38 steps, with all pairs reported converged, but its requested
subspace error was 1.07–1.08e-8 and failed the independent 1e-8 cutoff. That run
is retained as rejected. The table above uses a uniformly tighter stopping
tolerance for **every** starting strategy; the acceptance cutoff was not relaxed.

Artifacts: [consistent mass](../../results/block_followup/starts-consistent-strict.json),
[lumped mass](../../results/block_followup/starts-lumped-strict.json),
[rejected earlier profile](../../results/block_followup/starts-consistent-tol1e-8.json).
All timings are ineligible because other GPU jobs were present. A fresh Anvil
baseline timing attempt began on an idle GPU, but unrelated work started during
setup; the [process monitor](../../results/block_followup/baseline.occupancy.json)
caught this and our benchmark was stopped. The README timing table is unchanged.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.experimental.fem_starts --case clamped_consistent --side 10 --modes 20 --width 24 --tol 1e-9 --out build/starts-consistent.json
python -m benchmarks.experimental.fem_starts --case clamped_lumped --side 10 --modes 20 --width 24 --tol 1e-9 --out build/starts-lumped.json
python -m benchmarks.experimental.fem_starts --case clamped_lumped --side 20 --modes 20 --width 24 --tol 1e-9 --out build/starts-lumped-large.json
```

## The doubled Gram result now has a small reproducer

A stress run exercised wide projected eigensolves and conditional graphs before
queued Gram warmups. Among 360 checks, 60 failed: exactly the first replay after
unsynchronized warmup in each of 20 trials for each of three Gram chunk sizes.
Later replays and all synchronized-warmup checks passed.
[Raw stress results](../../results/block_followup/gram-stress.json).

The [reduced diagnostic](capture_ordering.py) does not use eigensolver or Gram
code. It repeatedly zeros one FP64 scalar and adds one to it from a million
threads, then captures and replays the same operations. Expected output is
exactly 1,000,000. Without finishing eager warmup first, the first replay instead
returns approximately 1,999,600; later replays return 1,000,000. This reproduces
on both the pinned Warp 1.18 environment and stock Warp 1.15, including an
explicit Warp stream. Replacing the memset with a zeroing kernel avoids the
failure. Synchronizing after eager warmup also avoids it.

An initial CuPy control passed. Adding an explicit graph upload made that
control fail too. The [CuPy-only reproducer](cupy_graph_upload.py) imports no
Warp and confirms the distinction: all three unsynchronized, explicit-upload
trials fail; all no-upload or synchronized controls pass. It runs on CuPy 14.2.0,
CUDA runtime 12.9, driver API 12.8, NVIDIA L40 / driver 570.158.01.
This narrows the trigger to the upload/memset/pending-work interaction on this
CUDA stack, rather than a Warp-only error or an error in our Gram arithmetic.
It does **not** yet identify a specific CUDA implementation defect or establish
which other driver/toolkit combinations are affected.

The inspected Warp runtime explicitly uploads the executable graph when it is
first created. CuPy's [Graph.upload implementation](https://github.com/cupy/cupy/blob/v14.2.0/cupy/cuda/graph.pyx)
forwards to the CUDA runtime. NVIDIA's [graph API documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html)
specifies that upload does not execute the graph and is ordered after earlier
work in the upload stream. The observed failure warrants upstream investigation.

Warpack's symmetric, Arnoldi and experimental block capture methods already
synchronize warmup before capture. Their derived solver paths use those methods.
That protection stays in place. No per-replay or per-iteration synchronization
has been added, and no upstream runtime files were modified. #61 remains open
for upstream diagnosis and a durable fix. The minimized Warp and CuPy reproducers
are filed as [NVIDIA/warp #2096](https://github.com/NVIDIA/warp/issues/2096).

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" python -m benchmarks.experimental.capture_ordering --out build/ordering-118.json
python -m benchmarks.experimental.capture_ordering --out build/ordering-115.json
python -m benchmarks.experimental.cupy_graph_upload > build/cupy-upload.json
```

Artifacts: [Warp 1.18](../../results/block_followup/ordering-118.json),
[Warp 1.15](../../results/block_followup/ordering-115.json),
[explicit stream](../../results/block_followup/ordering-explicit.json),
[CuPy-only upload controls](../../results/block_followup/cupy-only-upload.json).
These diagnostics deliberately preserve failing outcomes; they are not passing
production regression tests.

## Benchmark housekeeping

The block-comparison driver now performs a full graph warmup before timed trials.
It also deletes large temporary DOT dumps after extracting the graph audit;
`--keep-graph` retains them when needed. Numerical reports and graph node counts
remain in the JSON. These changes do not alter solver arithmetic.
