# Coarse-to-fine FEM starts

Measured 2026-10-22, L40 / Xeon Platinum 8362, four CPU threads pinned to cores
8–11; see [environment](environment.json). Uses the pinned Warp/FSAI environment
from the [installation guide](../../docs/installation.md#fast-block-solves).

A computed coarse basis reduces fine-grid iterations modestly in these cases,
but building and solving the coarse problem costs more than the saved work.
This is a baseline under [#134](https://github.com/alecjacobson/warpack/issues/134),
not a recommendation to enable hierarchy starts by default. It uses the
existing block solver with a starting subspace; it does not implement multilevel
Lanczos or a multilevel preconditioner.

## Problem and method

Locally generated 3:1:1 clamped linear-elastic beams use nested six-tet cube
splits: side 5 gives 1,620 free displacement DOFs; side 15 gives 34,560. Both
stiffness and mass are real symmetric positive definite after clamping x=0.
The mass is either P1 consistent or row-sum lumped. Young's modulus is 10,
Poisson ratio 0.3, density 1. No external mesh or reference eigenvectors seed
an iteration.

`BlockLOBPCG` requests the first 20 or 50 modes with widths 24 or 56. It uses
upstream FSAI (row cap 32, FP32 factors) and FP32 batched CG (25-step cap,
relative tolerance 0.01) for K+0.01M; outer arithmetic is FP64. Both coarse and
fine solves request tolerance 1e-11 and at most 200 adaptive iterations, using
the existing fast block-product/rotation settings.

The transfer kernel uses four barycentric weights in a coarse tetrahedron,
repeats the interpolation for xyz components, and inserts zero motion at the
clamped boundary. Tests compare PᵀK_fP and PᵀM_fP against independently assembled
coarse matrices (the mass identity applies to consistent mass), and reproduce
affine fields. This is P1 tetrahedral interpolation, not trilinear interpolation.
The fine solver orthonormalizes transferred vectors in the **fine** mass metric.
It retains 4/6 seeded random padding rows; no completeness claim follows from
coarse residual convergence alone.

The public call for transfer and refinement in one graph is:

```python
# coarse_solver.eigenvectors contains a computed coarse solution.
# transfer.apply writes into fine_solver.initial_vectors, with fixed buffers.
graph = fine_solver.capture(200, initializer=transfer.apply)
wp.capture_launch(graph)
```

`BeamTransfer` and procedural assembly are checkout-only benchmark helpers in
[`fem_hierarchy.py`](../../benchmarks/applications/fem_hierarchy.py). The generic
initializer hook is library code. It warms eagerly and records device operations
before initialization, rather than running a Python callback during replay.

## Accepted warm timings

Each cell is a median of three repeats after one warmup. Every coarse row
recomputes its source basis from seeded random starts. Transfer is included in
fine refinement through the initializer graph. Random and retained rows include
a device copy into the starting buffer. Components are individually medianed;
the reported total is the median of each trial's sum, so rounded columns need
not add exactly. Setup is reported separately below.

| Mass / modes | Random fine solve (s) | Coarse source (s) | Transfer + fine refinement (s) | Coarse workflow total (s) | Fine iterations, random → coarse |
|---|---:|---:|---:|---:|---|
| consistent-20 | 0.3371 | 0.1263 | 0.3061 | 0.4332 | 26–27 → 25–25 |
| lumped-20 | 0.3413 | 0.1474 | 0.2854 | 0.4344 | 26–27 → 23–24 |
| consistent-50 | 1.1073 | 0.5149 | 0.9892 | 1.5011 | 25–25 → 23–23 |
| lumped-50 | 1.0880 | 0.5491 | 0.9778 | 1.5300 | 25–25 → 23–23 |

Retaining the already-computed fine basis of the **unchanged** problem takes
about 0.014 s for 20 modes and 0.052 s for 50 modes, in one outer iteration. This
is a replay sanity check and optimistic reuse bound; it requires a prior fine
solve and is not a first-use speedup. Changed-matrix reuse is measured in the
[shape-family report](../fem_shapes/README.md).

## Setup and first-use cost

These are observed synchronized component costs, not medians of independent
cold processes. Fine setup includes device conversion, FSAI/CG/solver allocation,
kernel warmup and ordinary graph capture. The hierarchy adds coarse assembly,
coarse solver setup and transfer-map construction/upload plus capture of the
combined transfer/refinement graph. The benchmark keeps both ordinary and
combined fine graphs for comparison; a standalone hierarchy path could omit
the ordinary graph's capture, so these totals are conservative for that path.
CUDA context/JIT reuse within the worker can affect individual setup components.

| Mass / modes | Fine assembly (s) | Fine solver setup + capture (s) | Additional coarse assembly + setup (s) | Transfer setup + combined capture (s) | Random setup + first solve (s) | Hierarchy setup + first workflow (s) |
|---|---:|---:|---:|---:|---:|---:|
| consistent-20 | 0.6804 | 1.1751 | 0.2845 | 0.6308 | 2.1996 | 3.2214 |
| lumped-20 | 0.7181 | 1.5803 | 0.3139 | 0.1454 | 2.6613 | 3.1967 |
| consistent-50 | 0.6516 | 3.6053 | 0.9290 | 0.7955 | 5.4210 | 7.5681 |
| lumped-50 | 0.6701 | 2.1025 | 0.9314 | 0.7820 | 3.9090 | 6.1002 |

Independent SciPy reference solves, host accuracy checks, graph auditing and
JSON/log output are validation overhead outside these costs. Reference time is
recorded separately in each raw JSON. Runtime phases use synchronization for
measurement; no host numerical decisions or synchronization are inside the
captured solvers. Coarse solve and combined transfer/refinement can be launched
sequentially without a host readback; reference validation is benchmark-only.

All 64 coarse/fine results in these four accepted workers pass native status
and full counts, original residuals, M-orthogonality, requested eigenvalues and
complete requested invariant-subspace checks. Maximum normalized residual is
9.453e-12; maximum M-subspace error is
2.266e-11. Independent ARPACK
shift-invert references use tolerance 1e-12, expanded search spaces and a guard
mode; requests that split a boundary cluster are rejected. Graph DOT audits of
coarse, ordinary fine and combined transfer/refinement graphs show no HOST,
HtoD or DtoH nodes. DOT inspection is not an API trace.

Peak all-device occupancy in these workers is 2,692 MiB for 20 modes and
4,032 MiB for 50 modes; the occupancy logs include CUDA context and both solvers,
not just factor storage. These are sampled process-level totals, not allocation
or peak-memory guarantees. Two lumped-50 attempts overlapped an unrelated GPU
process and are retained in [contended/](contended/); their timings are excluded.

## Reproduction and regression gates

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_fem_hierarchy.py tests/test_fem_hierarchy_report.py tests/test_block_finalizer.py
python -m benchmarks.monitor_gpu --out build/hierarchy.occupancy.json \
  --log build/hierarchy.log -- taskset -c 8-11 \
  python -m benchmarks.applications.fem_hierarchy --mass consistent \
  --coarse 5 --fine 15 --modes 20 --width 24 --out build/hierarchy.json
# Record matching hardware/pinned environment metadata beside the raw report.
cp results/fem_hierarchy/environment.json build/environment.json
python -m benchmarks.applications.fem_hierarchy_report build/hierarchy.json \
  --baseline results/fem_hierarchy/consistent-20.summary.json
```

Use width 56 for 50 modes and `--mass lumped` for the other configurations. Run
GPU jobs serially. The reporter rejects incomplete or contended workers, stale
accepted flags hiding native/physical failures, invalid timings and host graph
nodes. The baseline gate rejects a >25% warm-workflow slowdown and requires
matching source hashes, matrices, configuration and recorded environment;
update hardware metadata truthfully when reproducing elsewhere. Setup is not
regression-gated because it is a single observation per worker.

The next experiments under #134 should separate accurate-coarse-solve overhead
from interpolation quality: looser source tolerances, transferring a full search
width, and symmetric coarse correction during refinement. Coarse bases that are
already available may amortize their cost. None of those possibilities is
established by this baseline. Large 100-mode, free/disconnected, contrasting
material and surface-grid hierarchy coverage remains open.
