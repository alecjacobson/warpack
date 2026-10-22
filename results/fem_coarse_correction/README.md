# Coarse correction during FEM refinement

This increment of [#134](https://github.com/alecjacobson/warpack/issues/134)
implements and tests two symmetric coarse corrections, both directly as outer
preconditioners and inside upstream Warp CG. **It does not improve the current
FP32 bucketed FEM path on the measured beam.** No default or public solver API
changes are made. The implementation stays in `benchmarks/experimental/`.

Measured 2026-10-22 on an NVIDIA L40 / Xeon Platinum 8362, four CPU threads
pinned to cores 8–11. Uses the current pinned Warp 1.18.0/FSAI overlay; each raw
record contains its verified dependency manifest and exact source hashes.

## Problem and construction

A procedural clamped P1 tetrahedral beam has 34,560 fine DOFs, consistent SPD
mass, 20 requested lowest modes and a 24-vector outer block. The source beam
has 1,620 DOFs. Warp computes all 24 source vectors at relative tolerance
`1e-3`; no reference vectors seed either solve. Barycentric P1 prolongation
produces a coarse space on the fine mesh. The fine eigensolver retains its
`1e-11` stopping tolerance and independent physical spectrum/subspace checks.

For shifted SPD stiffness `H = K + 0.01 M`, row basis `Z`, and a fixed-linear
FSAI application `S`, the experiment forms

```
E = Z H Z.T
Q = Z.T E^-1 Z
additive: B = S + Q
balanced: B = Q + (I - Q H) S (I - H Q)
```

Both preconditioners are SPD in exact arithmetic when the supplied inputs meet
the stated assumptions. Warp Cholesky factors the small `E`; only rectangular
basis/image arrays and small projected matrices are stored, never dense fine
inverses. The coarse basis and factors stay fixed throughout each CG solve.
Singular/numerically dependent coarse spaces set a device error flag and poison
the output, rather than silently regularizing the problem. Update and apply
support fixed-address graph replay. The implementation is a rank-24 Galerkin
correction, not the multiscale Lanczos recurrence or full-grid multigrid.

FSAI factors use upstream Warp FP32 storage. Correction arithmetic and the
new inner-CG adapter use FP64. The existing production-style baseline uses
FP32 inner arithmetic and residual-selected CG buckets. A separate FP64,
full-width baseline isolates the comparison with the new inner correction.

## Uncontended comparison

Each row is a fresh worker with one warmup and three measured replays. The
warm workflow includes recomputing the coarse modes when needed, transfer,
coarse-factor refresh and fine refinement. Setup includes host assembly,
device conversion, FSAI/solver allocation, capture, and the initial coarse
solve needed to construct a valid correction before fine capture. Independent
reference computation and output validation are excluded and separately
recorded. These setup values are single observed constructions, not repeated
cold-process medians.

| Method | Setup + capture (s) | Fine replay median (s) | Complete warm workflow median (s) | Observed first-use component sum (s) | Setup/capture memory decrease (MiB) |
|---|---:|---:|---:|---:|---:|
| Existing FP32 CG, active buckets | 1.5190 | 0.3351 | 0.3351 | 1.8548 | 1322 |
| FP64 CG, full width | 1.4783 | 0.5163 | 0.5163 | 2.0023 | 1226 |
| Additive correction inside FP64 CG | 2.1507 | 0.5291 | 0.5992 | 2.7588 | 1346 |

First-use sums are setup plus the observed warmup workflow, not independent
end-to-end cold-process measurements. Memory is the device/context free-memory
decrease after setup/capture, **not peak memory or live factor storage**.

The additive inner correction reduces scheduled vector iterations from
13,800–14,400 to 8,016–8,352, and no RHS misses the inner relative tolerance
within its 25-step cap. Nevertheless its additional work outweighs that saving:
complete warm time is about 16% slower than FP64 CG and 79% slower than the
current FP32 bucketed path. The latter schedules 6,388–6,608 vector iterations;
counts include padding and masked RHSs, so they are not sums of each RHS's
first-convergence iteration. All three produce 20 accepted modes.

The raw records and gated summaries are in [qualified/](qualified/). Shared
fine-matrix, source and dependency identities are checked in
[validation.json](validation.json). The table uses only its three
`timing_eligible` workers.

## Accuracy and excluded timing trials

All nine final profiles pass one warmup plus three measured solves: 36 fine
results with native status zero/full count, fresh original residuals below
`1e-11`, and maximum mass-weighted requested-subspace error `1.01e-11`.
Coarse results separately satisfy their explicit approximate source contract.
All audited coarse/fine graphs have no HOST, HtoD or DtoH nodes.

The final balanced-inner, coarse-start, additive-outer, balanced-outer and
balanced-with-start workers encountered other GPU processes. They establish
accuracy, not timing. The FSAI-only worker had a clean GPU inventory but is
conservatively excluded because a short separate dense-reference check ran on
the same CPU affinity; [its exclusion record](qualified/fsai.exclusion.json)
is enforced by the report CLI. Raw occupancy records are preserved unchanged.

Additional diagnostics are retained:

- `screening/`: small 2,646-DOF, six-mode screening; contended, not timing evidence.
- `large-screening/`: one measured replay per outer-only profile. These show
  reduced iteration counts versus FSAI alone, but do not meet the three-repeat
  timing gate.
- `inner-screening/`: initial CG adapter omitted `atol=0`. Upstream's default
  `atol=tol` caused outer stagnation as RHS norms shrank. Those failures are
  retained. A small-RHS regression now checks scale `1e-12`; the existing
  public `BatchedCGPreconditioner` already set the correct absolute tolerance.
- `pre_stats_fix/`: preliminary attempts before correcting aggregate CG
  metadata; do not use the baseline's inner-work counts.

[#143](https://github.com/alecjacobson/warpack/issues/143) fixes the same metadata
bug in the existing shape and hierarchy drivers: read the actual outer solver
preconditioner's aggregate, not the original full-width state's last call.
Those historical reports are annotated; their measured timings and physical
accuracy are unaffected. The final `qualified/` records use the corrected
aggregate counters.

## Validation and reproduction

The focused suite passed **67 tests** ([log](tests.log)), including independent
dense identities, symmetry/positivity, changed-operator/basis replay, aliased
buffers, rank failure/recovery, tiny RHSs, lumped/consistent generalized FEM,
multiple CG buckets, and reporting rejection of hidden costs, bad spectra,
partial repeats, contention, host copies and a 30% injected slowdown.
The full library suite was not rerun: no installed `warpack/` code changed.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu \
  --out build/coarse-additive.occupancy.json --log build/coarse-additive.log -- \
  taskset -c 8-11 python -m benchmarks.applications.fem_coarse_correction \
    --coarse 5 --fine 15 --modes 20 --width 24 \
    --profile additive-inner-cg --repeats 3 --out build/coarse-additive.json
python -m benchmarks.applications.fem_coarse_correction_report build/coarse-additive.json
```

Install the **current** pinned overlay, not the historical v0.2.0 profile.
Other profiles are `cg`, `cg64`, `start-cg`, `fsai`, `additive`, `balanced`,
`balanced-start`, and `balanced-inner-cg`. `--inner-cg` adjusts the experimental
FP64 cap; the existing bucketed baseline remains fixed at 25.

The reporter accepts `--baseline path/to/prior.summary.json`; matrices,
configuration, numerical source, dependencies and environment must match.
A warm-workflow slowdown above 25% fails. Rejection is not a speedup claim.

Remaining #134 work: broader mode counts, geometry/material/nullspace cases,
more economical coarse applications, full coarse-grid/multilevel correction,
and hierarchy reuse across changing systems. These negative timings do not
establish that multilevel methods cannot help those workloads.
