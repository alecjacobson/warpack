# v0.2.0 release validation and review

Date: 2026-10-20. NVIDIA L40 48 GB, driver 570.158.01, Python 3.10,
Warp 1.18.0+cu12 plus checksum-pinned upstream FSAI/batched CG revision
`04427b997fe7ff07bc6ed1dde5ca9813496c6b55`. Four BLAS/OpenMP threads.

## Anvil promotion regression

All profiles use the same 283,857-DOF Anvil, width-24 Warp block, FSAI32,
FP32-CG25, shift 100, column layouts, image reuse, bucket width four and full
history. CuPy uses width 20 and the same captured Warp preconditioner. Three
synchronized replay trials after warmup, with immutable original starts.
No unrelated GPU processes were sampled. Setup/capture are separate.

| Profile | Warp median (s) | CuPy median (s) | Whole comparison accepted? |
|---|---:|---:|---|
| Frozen `3198340`, CuPy absolute tolerance 1e-5 | 1.8255 | 2.1552 | No: one CuPy returned-pair residual 1.102e-8 exceeds 1e-8 |
| Frozen `3198340`, CuPy absolute tolerance 1e-6 | 1.8455 | 2.3980 | Yes, all six outputs |
| Public API `db5826f`, CuPy absolute tolerance 1e-6 | 1.8669 | 2.3881 | Yes, all six outputs |
| Standalone installed-API example | 1.8888 | — | All three device count/residual checks pass |

The first failed aggregate is retained in `release-control.json`; its occupancy
wrapper correctly marks it ineligible because the comparison process exits
nonzero. All its Warp outputs pass. We tightened the **reference's** stopping
rule rather than weakening the common physical gate. This adds a CuPy iteration
cost, so do not compare the two CuPy tolerances as an implementation speedup.
The public/frozen difference is 1.2% in this run, not evidence of a speedup.
The earlier 1.7784-second observation remains historical. Promotion preserves
the algorithm and supported settings, not that exact wall time on every run.

Each accepted benchmark output checks all 20 elastic modes plus six analytic
rigid modes against the cached independently computed reference: original
residual <=1e-7, orthogonality <=1e-8, reference eigenvalue and mode-shape gates,
and returned eigenpair residual <1e-8. Warp additionally requires status zero
and 20 converged pairs. CuPy additionally meets its fresh full-block absolute
tolerance. Public Warp's worst original residual is 1.957e-8, returned normalized
residual 9.785e-9 and returned eigenvalue relative error 9.42e-14.
Reference loading and host validation are outside solve timing.

Public setup is 2.3932 s (1.0734 s factor plus 1.3199 s solver/workspace including
encountered JIT), capture 0.9274 s. The standalone example's warmed-JIT setup is
1.2103 s, capture 0.9232 s. Memory decrease through setup/capture is 4.454 GB,
not peak memory. CuPy shares matrix/factors and records an extra 0.1511 s for
its inner adapter setup/capture; that is not independent end-to-end setup.

The public graph has 2,002 KERNEL, 30 CONDITIONAL, 183 MEMSET and 110 MEMCPY nodes.
All copies are DtoD, and there are no HOST nodes, including conditional bodies.
This is graph DOT inspection, not a complete CUDA API trace. Setup is allowed
to synchronize; optional cuDSS vendor metadata is a separate known issue.

Raw reports include every trial, settings, source hashes and occupancy:
[public](release-public.json), [public occupancy](release-public.occupancy.json),
[frozen tight](release-control-tight.json), [frozen tight occupancy](release-control-tight.occupancy.json),
[rejected original control](release-control.json), [original occupancy](release-control.occupancy.json),
[standalone example](release-example.json), [example occupancy](release-example.occupancy.json).

Reproduce the comparison with the pinned environment from the release guide:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH="$PWD/build/warp-fsai:$PWD" \
 python -m benchmarks.monitor_gpu --out build/release.occupancy.json \
 --log build/release.log -- python -m benchmarks.experimental.compare_block \
 --public-api --case anvil --width 24 --steps 200 --shift 100 --fsai-width 32 \
 --fsai-layout columns --matrix-layout columns --repeats 3 --tol 1e-8 \
 --cupy --cupy-width 20 --cupy-tol 1e-6 --gram-chunk 4096 --symmetric-products \
 --fused-updates --paired-rotation --reuse-x-image --batched-products \
 --inner-cg 25 --batched-cg --active-cg-bucket-size 4 --inner-fp32 --inner-tol .01 \
 --active-search-bucket-size 4 --active-search-keep-history --out build/release.json
```

The cached `build/anvil_reference.npz` is needed only by the comparison harness;
the standalone example needs neither it nor cuDSS. Reference generation is
covered by the existing benchmark documentation. Never include reference setup
in eigensolver replay times.

### Independent CuPy setup follow-up (2026-10-21)

The README now charges setup to each solver independently, including recorded
graph capture: Warp block **3.3206 s** (2.3932 + 0.9274 from the release run),
and CuPy LOBPCG **1.4052 s** from a new standalone setup measurement. CuPy builds
fresh FSAI32 factors and base CG state in 1.0725 s; matrix/start/constraint upload,
active-width workspaces, the LinearOperator and inner graph capture take another
0.3327 s. No Warp eigensolver is constructed or run first. The setup uses the
release settings, including a width-24 base CG and active widths 1–20 for CuPy.

This excludes imports/runtime startup, mesh assembly/mass normalization, cached
reference loading, host random-start generation and outer eigensolve warmup.
Encountered JIT is included with existing disk caches. These are separately
recorded observations, not a simultaneous cold-start comparison. One validation
trial after warmup passes all physical and native stopping gates, with maximum
original residual 1.6835e-9 and returned-pair residual 8.4173e-10. The README's
release solve timings and residuals remain unchanged. No unrelated GPU process
was sampled.

[Raw setup and validation](cupy-independent-setup.json) ·
[Occupancy](cupy-independent-setup.occupancy.json)

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH="$PWD/build/warp-fsai:$PWD" \
 python -m benchmarks.monitor_gpu --out build/cupy-setup.occupancy.json \
 --log build/cupy-setup.log -- python -m benchmarks.experimental.anvil_cupy_setup \
 --out build/cupy-setup.json
```

## Test and package validation

- Full local suite: **1,449 passed, 1 skipped in 547.63 s**. This run precedes
  the final setup-only shape/device and capability guards. The skip is the
  stock-runtime negative control in the pinned environment.
- Final promoted-path regression subset: **197 passed, 1 skipped** after those
  guards. It exercises public composition, current-input replay, repeated
  subspaces, ownership, FSAI/CG, scalar and block Jacobi, layouts, active rows,
  invalid metadata and graph inspection.
- Stock Warp 1.15 subset: **202 passed, 22 skipped in 8.64 s**, including cached
  images, device-selected search, starts, finalizers and projected solvers.
  Unsupported fast-CG and upstream-preconditioner cases are skipped; the explicit
  rejection test for incomplete batched-CG support passes.
- The dependency installer downloaded all three upstream files and verified
  their pinned SHA256 hashes against the tested overlay.
- Wheel/source-distribution construction, core-module hashes and outside-checkout
  replay smoke results are recorded in [validation.json](validation.json), [pinned wheel](wheel-pinned.json)
  and [stock wheel](wheel-stock.json) reports.
  The clean smoke forbids benchmark, example, CuPy, SciPy and optional cuDSS
  imports, imports all 33 core modules and tests three changed-value replays.
- Python lint, formatting and local documentation links pass. Hosted CI covers
  package construction and static checks; local GPU results supply numerical
  validation. Complete external CuPy/Spectra inventories were **not** rerun for
  this release; their separately dated reports remain in the README.

## Code review

Self-review by the implementing agent; no independent reviewer claimed.

| Area / finding | Resolution and evidence |
|---|---|
| Fast path depended on checkout-only numerical subclasses | Generic code moved to installed modules; explicit `BlockLOBPCG` options replace kernel/pointer interception. Compatibility wrappers retain existing benchmark recipes. |
| Eager block iteration count stayed zero (#125) | Fill device counter with executed fixed budget; regression alternates eager 2, graph 3, eager 1. |
| Search branches could retain their parent through an ownership cycle | Weak owner reference; test releases the solver without a garbage-collection call. |
| Batched preconditioner width mismatch | Reject during setup with and without device buckets. Invalid budgets/tolerances also tested. |
| Current-value replay and image reuse | Repeated/coupled SPD generalized tests check independent eigenvalues, full repeated subspaces, B-orthogonality, changed matrices/constraints and unchanged supplied starts. Existing FP32 matrix refresh and RHS reactivation tests run against installed code. |
| Host synchronization | Reviewed capture branches and graph DOT, including nested bodies. Eager path selects full preallocated width without reading device flags. Setup synchronization is documented. |
| Experimental compatibility import removed during lint cleanup | Full-suite collection exposed it; restored explicit old-name alias before final validation. |
| Upstream preconditioner shape/device mismatch (#126) | Reject before CG state construction; invalid metadata callbacks never execute. Upstream scalar diagonal operators expose `(n,)` shape and remain supported; the existing CSR/block-Jacobi tests caught and verify this distinction. This final setup-only guard does not change timed numerical kernels. |
| Incomplete stock batched-CG API (#127) | Check both `batch_offsets` and `max_batch_length`; stock 1.15 exposes the former only. Clear setup rejection replaces a late TypeError, verified on stock and pinned environments. |
| Numerical dependency boundary | Promoted modules import Warp and standard-library code only. Upstream CG/FSAI performs its own arithmetic; optional sparse direct calls remain cuDSS. FEM and CuPy adapters remain outside core. |
| Reproducibility | Fixed upstream wheel/revision, checked file hashes, executable example, raw rejected/accepted profiles retained. |

The review found no unresolved blocker within this release's declared contracts.
It does not certify global spectral completeness, every external runtime version,
or the deferred issue inventory. See the release guide's explicit limitations.
