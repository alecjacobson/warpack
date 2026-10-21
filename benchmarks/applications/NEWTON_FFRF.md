# Newton FFRF sparse eigen comparison

This benchmark extracts the sparse modal eigenproblem from
[Newton's `exp/ffrf-sparse-eigen` branch](https://github.com/skulkar22/newton/tree/360a9b29a469af04eae6909f6d1709460a8dc35e).
It generates its own spring geometry and matrices. No upstream meshes,
matrix datasets, or simulation assets are downloaded or checked in.

## What the branch solves

[`ModalGeneratorFEM._solve_arpack`](https://github.com/skulkar22/newton/blob/360a9b29a469af04eae6909f6d1709460a8dc35e/newton/_src/sim/modal.py#L2293)
solves the real symmetric generalized problem `K phi = lambda M phi` with SPD
mass. For a free body it deflates six analytic, mass-orthonormal rigid motions.
Its default shift is `-1e-6 * ||K||_1 / ||M||_1`. It factors `K-sigma*M` with
SuperLU (`COLAMD`, symmetric mode, zero diagonal-pivot threshold), then calls
ARPACK shift-invert with `which="LM"`, `tol=0`, and SciPy's default `ncv`.

[`lowest_generalized_eigenpairs`](https://github.com/skulkar22/newton/blob/360a9b29a469af04eae6909f6d1709460a8dc35e/newton/_src/sim/modal_sparse.py#L631)
requests two guard pairs, checks residuals, and performs another sparse
factorization at a gap above the wanted spectrum. Its pivot inertia counts
the eigenvalues below that gap. Missing copies trigger additional deflated
searches; unavailable inertia emits a warning. This second factorization is
real application work and is reported separately.

The branch also has a
[Craig–Bampton generator](https://github.com/skulkar22/newton/blob/360a9b29a469af04eae6909f6d1709460a8dc35e/newton/_src/sim/modal_craig_bampton_fem.py#L562).
After constraint elimination, its interior eigenproblem is
`K_ii phi = lambda M_ii phi`, with fixed interfaces and default `sigma=0`.
The same stiffness factor also serves static constraint-mode solves.

## Recreated problems

[`newton_ffrf.py`](newton_ffrf.py) constructs a `40 x 20 x 12` grid of nodes,
spacing 0.01 m, with axial springs along all neighboring edges, face diagonals,
and body diagonals. Spring stiffness is `2e9 / length`; each node has mass
0.008 kg. A sparse bar-extension matrix `D` independently assembles
`K = D.T diag(stiffness) D`. Mass is `0.008 I`.

* **Free-free:** 28,800 DOFs, six excluded rigid modes, 12 requested elastic
  modes. This reproduces the model specification and size of Newton's
  [`test_fem_arpack_large_sparse_model`](https://github.com/skulkar22/newton/blob/360a9b29a469af04eae6909f6d1709460a8dc35e/newton/tests/test_modal_fem_sparse.py#L394).
* **Fixed-interface:** the same generated model with both end faces fixed,
  giving 27,360 interior DOFs and 12 requested modes. This isolates the
  Craig–Bampton interior eigenproblem.

These are three-displacement-DOF, lumped-mass models. They do **not** reproduce
the branch's external HMG six-DOF shell dataset, consistent shell mass,
RBE2/RBE3/MPC transformations, static constraint modes, modal recovery, or the
subsequent FFRF dynamics. Timings are for the eigensolver stage, not complete
`ModalGeneratorFEM.build()` or `ModalGeneratorCraigBamptonFEM.reduce()` calls.

The CPU benchmark imports the **unchanged** upstream `modal_sparse.py` from the
pinned commit, after checking its SHA-256. No locally rewritten ARPACK loop
stands in for Newton. The optional downloader fetches that source file only.

## Warpack compositions

Both paths exploit the exact constant diagonal mass: work with
`A = K / (2e9 / 0.01)`, then recover physical eigenvalues and mass-normalized
vectors on device. Scaling, shifting, rigid-basis construction, eigen
iterations, and output recovery use Warp kernels.

* **Warpack + cuDSS:** SPD factorization at Newton's shift;
  `OrthogonalComplementOperator` excludes rigid motions for the free body;
  `KrylovSchur` uses `ncv=48`, `tol=1e-12`, and two captured restart cycles.
* **Pure Warp:** public `BlockLOBPCG`, width 16, FP64 outer arithmetic,
  `tol=1e-11`, image reuse and search/preconditioner buckets of four.
  `BatchedCGPreconditioner` uses FP32 CG, relative tolerance `1e-3`, at most
  25 iterations, column layout, and upstream Warp FSAI with row width 32.
  The preconditioner factors `A + 0.01 I`; the outer problem remains `A`.
  Device-controlled stopping has a 200-iteration ceiling.

All replays restart from the solver's default deterministic seed. No previous
eigenvectors or reference eigenvectors are used as warm starts. Different
algorithms use different initial spaces and stopping conventions; acceptance
uses the same physical accuracy gates.

## Timing and accuracy protocol

See [recorded results](../../results/newton_ffrf/README.md). Each backend/case
has three independent process setups. Each GPU setup has one untimed warmup
followed by three timed replays. CPU runs rebuild the factor for each trial.
GPU process occupancy is sampled throughout every worker; competing-process
runs are ineligible. Workers run serially with four CPU cores and four BLAS
threads. CUDA/JIT disk caches already exist.

Setup includes upload and device normalization, factor/preconditioner creation,
workspaces, and graph capture for Warp. CPU setup includes Newton's shifted
factor construction and pivot diagnostics. Common procedural assembly, CPU
analytic rigid-basis construction, imports/runtime initialization, output
downloads, and benchmark validation are outside solver timing. The GPU's
first graph launch is reported separately; add it to setup for first-use cost.
Newton's residual checks remain inside its solve time; its inertia check is
shown separately. Thus the reported CPU solve is a conservative comparison
with GPU replay plus out-of-band validation.

Every trial must pass original-equation relative residual, relative eigenvalue
error, mass orthogonality and rigid overlap below `1e-8`; complete requested
subspace error must be below `1e-6`. The reference comes from Newton's checked
solve, with a positive spectral gap after mode 12. Small independent dense
LAPACK tests verify both generated models and both Warp solver compositions.
Tests also reject exact eigenpairs from the wrong requested spectrum.

Warp's successful convergence count is **not** an inertia certificate.
Benchmark comparison establishes coverage for these fixtures; the production
Warp solvers do not perform Newton's second-factorization completeness check.
This optional capability is tracked in [#133](https://github.com/alecjacobson/warpack/issues/133).
GPU graphs have no host callback nodes. The pure-Warp graph contains only
device-to-device copies; cuDSS retains its known captured host-to-device
metadata copies ([#29](https://github.com/alecjacobson/warpack/issues/29)).
Graph inspection is not a CUDA API trace.

## Reproduction

Install the [pinned fast-block environment](../../docs/installation.md#fast-block-solves)
and cuDSS extra. From the repository root:

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest tests/test_newton_ffrf.py -q
for boundary in free fixed; do
  for backend in newton cudss warp; do
    python -m benchmarks.monitor_gpu \
      --out "build/newton-$boundary-$backend.occupancy.json" \
      --log "build/newton-$boundary-$backend.log" -- \
      python -m benchmarks.applications.newton_ffrf \
        --download-source --boundary "$boundary" --backend "$backend" \
        --reference "build/newton-$boundary-reference.npz" \
        --out "build/newton-$boundary-$backend.json"
  done
done
```

Run `newton` first to create the reference file. References include a matrix
hash and mismatched files are rejected. Change `--shape nx ny nz` for a
different resolution. Choose a mode count with a gap at its boundary; this
benchmark rejects counts that split a repeated cluster. For the recorded
table, use `taskset -c 8-11`, three separate workers per row, `--repeats 1`
for Newton and `--repeats 3` for each GPU worker.

`python -m benchmarks.applications.newton_ffrf_report results/newton_ffrf`
rebuilds the summary from the recorded workers. It rejects failed, contended,
mixed-matrix and mixed-source batches. `--baseline path/to/summary.json`
checks warm-solve regressions above 25% (override with `--regression-limit`).
Baseline comparison requires the same benchmark definition, matrix, solver
settings, runtime, CPU affinity and hardware/overlay provenance in each
folder's `environment.json`; it does not compare different machines.
