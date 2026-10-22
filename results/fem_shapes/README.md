# Geometry changes, mass updates and captured FEM reuse

**Source-provenance scope ([#145](https://github.com/alecjacobson/warpack/issues/145)):**
the archived source fingerprint covers only the driver. It does not certify
unchanged helper/core code. New workers fingerprint transitive local imports
and all Warpack source files. Historical raw timings and accuracy are preserved;
new regression baselines must use the complete manifest.

**Inner-work metadata correction ([#143](https://github.com/alecjacobson/warpack/issues/143)):**
the archived `cg_stats` arrays read the last full-width CG bucket, not aggregate
work across all buckets. Do not use those arrays as total inner-iteration counts.
Measured times, outer iterations and independent accuracy checks are unaffected.
The current driver reads the outer solver's aggregate; historical raw files are preserved.

This increment of [#128](https://github.com/alecjacobson/warpack/issues/128)
qualifies changes to both stiffness **and mass**, upstream FSAI refits, and warm
starts. All eight large configurations pass 20 modes at every geometry state:
96 timed solves and 32 warmups. Worst fresh normalized original residual is
below `1e-11`; worst mass-weighted requested-subspace error is `1.08e-11`.

The independently generated clamped tet beam has 34,560 free DOFs. A parameter
sequence `0, +0.12, -0.12, 0` stretches the three axes by
`(1+0.1*t, 1+t, 1-0.6*t)` and twists cross sections by `0.2*t*x/3` radians.
Connectivity and retained DOF identities stay fixed; element orientations are
checked before assembly. The geometry changes the dominant transverse bending
direction. Both consistent and lumped mass are tested. Small independent-bar
tests separately exercise **exact** mode crossings and repeated clusters; we
do not claim the coupled tet eigenvalues cross exactly.

The shape-family coverage was motivated by
[Shape Space Spectra (SIGGRAPH 2025)](https://www.dgp.toronto.edu/projects/sss/).
This is a conventional discrete FEM qualification fixture, not a reproduction of
that paper's neural method. No external meshes or datasets are used.

## Measured comparison

All numerical solver/preconditioner work here is Warp. No cuDSS or CuPy is used
in these timed runs. Independent SciPy ARPACK references use the original
pencil, expanded search spaces and `tol=1e-12`; small fixtures use dense LAPACK.
Validation checks residuals, full requested spectra/subspaces, mass
orthogonality, native status and the complete convergence count.

| Mass / start / factor policy | Solver setup + capture (s) | Source update + solve (s) | Per-change update (ms) | Per-change solve (ms) | Three-change sequence (s) |
|---|---:|---:|---:|---:|---:|
| Consistent / cold / refit | 1.284 | 0.374 | 35.48 | 342.38 | 1.1471 |
| Consistent / previous / refit | 1.187 | 0.383 | 35.07 | 299.95 | 1.0261 |
| Consistent / perturbed / refit | 1.298 | 0.383 | 36.17 | 315.62 | 1.0594 |
| Consistent / previous / retain | 1.241 | 0.380 | 32.91 | 303.19 | 1.0324 |
| Lumped / cold / refit | 1.204 | 0.378 | 32.10 | 344.89 | 1.1315 |
| Lumped / previous / refit | 1.144 | 0.379 | 31.30 | 312.88 | 1.0303 |
| Lumped / perturbed / refit | 1.247 | 0.375 | 32.21 | 320.30 | 1.0450 |
| Lumped / previous / retain | 1.215 | 0.374 | 27.64 | 314.16 | 1.0238 |

Computed previous modes reduce the **update-plus-solve sequence** by 10.6%
with consistent mass and 8.9% with lumped mass, comparing refit against refit.
This is a smaller benefit than the previous 2% material-only experiment.
Perturbation costs some convergence speed here, but provides exploration that
matters in the exact invariant-space crossing test below. Retaining the initial
FSAI and refitting it are nearly tied in these sequences; neither is promoted
as a universal default.

The first column is one observed constructor/capture per worker, including
initial FSAI construction, allocations, transfers and capture preparation.
It is **not** a median over independently repeated setups. Source/update/solve
columns are medians of three full sequences after one discarded warmup. Each
sequence starts from random vectors on the original shape; the next three
states use the declared start policy. Source modes are computed by Warpack,
never supplied by the reference solver. Width is 24 for 20 requested modes:
previous-start runs reuse 20 modes and seed four additional rows randomly.
Perturbed starts additionally use `initial_perturbation=1e-3`.

Update time includes host extraction/upload of new K, M and `K+.01*M` values,
synchronization and optional `FSAI.update()`. It is not a device-kernel-only
measurement. Solve includes the starting-vector copy and graph replay. The
sequence column is the median of each trial's **sum** of the three changed-state
update/solve times, rather than a sum of pooled medians. Initial setup and the
source solve remain separate. Host mesh assembly and independent reference work
are excluded and recorded as `assembly_reference_seconds` in each raw result.
No end-to-end mesh-to-ROM speedup or cross-library speedup is claimed here.

Hardware: NVIDIA L40 48GB, Xeon Platinum 8362, driver 570.158.01; affinity 8–11;
OpenBLAS/OpenMP threads 4. Warp 1.18.0 with FSAI overlay
`04427b997fe7ff07bc6ed1dde5ca9813496c6b55`. All eight occupancy records report
`timing_eligible=true`; CPU references and GPU runs were executed serially.

## Replay and factor contract

The benchmark reserves the union sparse pattern before capture, including
entries that are zero in one shape and nonzero in another. It updates values
at fixed addresses. Outer products use the current K/M and inner FP32 CG
refreshes from the changed `K+.01*M`. FSAI uses `max_row_size=32`, FP32 factor
storage and 25 truncated CG iterations (`tol=.01`). The outer tolerance is
`1e-11`. Search/CG buckets have size four; cached images and the supported
symmetric-product/update options are enabled.

Refitting calls upstream `FSAI(..., reuse_pattern=True)` / `factor.update()`
outside capture. Factor supports and their storage stay fixed. Retained-factor
runs keep the initial positive preconditioner while changing the CG matrix.
Both policies replay one captured solver graph for every shape. DOT audits
contain no HOST, HtoD or DtoH nodes. Host assembly, uploads and factor updates
are explicit setup work outside that graph.

This is an initial refit qualification. More extreme shapes/material changes
can make the retained supports ineffective or make a refit fail. A new adaptive
factor construction and graph may then be necessary. This experiment does not
qualify arbitrary topology changes, much wider mode counts or smaller GPUs.

## Exact crossings and optional recovery

`tests/test_fem_shapes.py` also generates two independent consistent-mass P1
bars. Exchanging their lengths changes which member supplies the four lowest
modes. A full computed invariant starting block, with roundoff leakage into
the other independent member removed, returns four converged but wrong modes
without exploration. This is the documented warm-start limitation, not a new
claim that residual convergence implies completeness.

The optional certificate from #133 rejects the wrong subspace. A perturbed
start recovers all four correct modes and passes the certificate. At equal
member lengths, tests include both copies of the repeated eigenvalues, rotate
the computed basis within each cluster, and accept it by invariant-subspace
comparison rather than individual eigenvector identity.

[#137](https://github.com/alecjacobson/warpack/issues/137) adds the opt-in
[`certified_recovery` example](../../examples/certified_recovery.py) and
[usage/limitations](../../docs/completeness.md#opt-in-recovery-after-a-rejected-warm-start).
Its finite retry list stops on certified/native success, refuses invalid or
unavailable diagnostics, and reports exhaustion explicitly. It reuses one
inertia count for a fixed pencil and gap; no additional factorization is hidden
in a retry. The **policy** reads statuses on the host and is not capturable;
each candidate solve and verification remains a captured device computation.
This example is separate from the timed pure-Warp runs above and from solver
core. General automatic gap discovery is not implemented.

## Reproduction and regression gates

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu \
  --out results/fem_shapes/consistent-previous-refit.occupancy.json \
  --log results/fem_shapes/consistent-previous-refit.log -- \
  taskset -c 8-11 python -m benchmarks.applications.fem_shapes \
  --side 15 --modes 20 --width 24 --mass consistent \
  --strategy previous --factor refit \
  --out results/fem_shapes/consistent-previous-refit.json
```

Run serially for each mass (`consistent`, `lumped`) and policy
(`cold/refit`, `previous/refit`, `perturbed/refit`, `previous/retain`). Each raw
JSON retains all trials, warmups, native/physical checks, graph audits, source
and matrix hashes. The report command requires all eight accepted, uncontended
configurations and matching matrices/source/runtime:

```bash
python -m benchmarks.applications.fem_shapes_report results/fem_shapes
# Compare a fresh results directory with this archived baseline:
python -m benchmarks.applications.fem_shapes_report build/new-shape-results \
  --baseline results/fem_shapes/summary.json
```

The default regression limit is 1.25 times the update-plus-solve sequence time.
Hardware/overlay provenance in `environment.json` must also match. Report tests
reject contention, hidden native-count failures, mismatched matrices, nonfinite
timing data and an injected 30% slowdown. A successful self-baseline check only
validates the reporting/gate path; it is not another performance experiment.

Validation logs cover 79 distinct pinned-runtime tests: an initial 75-test
focused run, a 20-test reuse/recovery subset after three additions, and a final
10-test recovery review including a bad-nullspace regression.
These include eight shape/refit/start combinations, exact crossing and
rotated-cluster tests, bounded recovery/early rejection, reporting gates,
existing material replay/factor-application regressions and the completeness
suite. Stock Warp records 64 passes and 13 expected skips for unavailable
FSAI/batched-CG support before the final additional FSAI-only cluster test.
A final stock subset records 9 passes and 12 expected skips, including the
additional nullspace-policy case. The recovery policy's non-FSAI
rejection/nullspace cases pass on stock Warp.
Lint and formatting checks pass. This is not a full upstream conformance rerun.
