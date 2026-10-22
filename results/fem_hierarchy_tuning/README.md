# Coarse accuracy and full-width transfer

Follow-up to [the initial hierarchy comparison](../fem_hierarchy/README.md),
2026-10-22, same L40 and pinned Warp/FSAI environment. Each profile runs in a
fresh monitored worker, with one warmup and three accepted repeats. All seven
workers are uncontended. Mesh: procedural clamped P1 beam, 34,560 fine DOFs,
1,620 coarse DOFs, consistent mass. Outer fine tolerance remains 1e-11.

The coarse tolerance is an input-quality choice, not a relaxation of the fine
answer. `partial` transfers k computed coarse modes; `full` transfers the entire
fine search width (24 for k=20, 56 for k=50). The coarse search has 4/6 additional
random rows. Full fine starts have no random padding; every trial still faces
an independent full requested-spectrum and M-subspace reference check.

| Modes / profile | Coarse solve or input copy (s) | Fine refine, including transfer (s) | Total warm (s) | Observed assembly + setup + capture (s) |
|---|---:|---:|---:|---:|
| 20 / random | 0.0000 | 0.3368 | 0.3368 | 2.5582 |
| 20 / partial-strict | 0.1304 | 0.3068 | 0.4372 | 2.1113 |
| 20 / partial-loose | 0.0474 | 0.3066 | 0.3540 | 2.0663 |
| 20 / full-strict | 0.1859 | 0.2671 | 0.4529 | 2.1800 |
| 20 / full-loose | 0.0700 | 0.2662 | 0.3364 | 2.2111 |
| 50 / random | 0.0001 | 1.0828 | 1.0828 | 2.6861 |
| 50 / full-loose | 0.2081 | 0.9415 | 1.1474 | 4.6129 |

Strict coarse solves request 1e-11; loose coarse solves request 1e-3. Native
coarse counts/status, original coarse residuals and M-orthogonality must pass
that declared source tolerance. Coarse spectrum/subspace errors and the stricter
`source_quality.accepted` flag remain recorded; a loose source is not labeled
an accurate eigenbasis. Every final fine result must pass the original 1e-8
physical/spectrum/subspace gates and the native fine 1e-11 stopping test.

Full loose starts bring the 20-mode warm workflow to near parity with random
starts, while reducing the fine refinement portion by about 21%. This does not
establish a general hierarchy speedup or improve the Anvil recipe. The 50-mode
comparison is a separate measured control. No multilevel preconditioner or
multiscale Lanczos recurrence is implemented here.

All assembly, coarse/fine factor construction, transfers and one fine graph
capture are charged in setup; this driver avoids the initial report's duplicate
fine capture. Setup values are single observations, not cold-start medians, and
include JIT/context effects. Do not infer a stable first-use speedup from their
variation. Raw reports additionally record setup plus first-workflow components
and independent reference time. CPU reference solves, host accuracy checks and
logging are outside solver timing. Graph audits show no HOST/HtoD/DtoH nodes.

The original report remains tied to its source at b316ce2; its recorded source
hashes should not be silently replaced by this updated benchmark utility.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu --out build/full-loose.occupancy.json \
  --log build/full-loose.log -- taskset -c 8-11 \
  python -m benchmarks.applications.fem_hierarchy_tuning \
  --profile full-loose --out build/full-loose.json
# Record truthful hardware metadata next to the report, then:
python -m benchmarks.applications.rom_followup_report build/full-loose.json \
  --kind hierarchy --baseline results/fem_hierarchy_tuning/full-loose.summary.json
```

The reporter rejects contaminated/incomplete runs, stale acceptance flags,
missing coarse counts, bad physical checks, NaNs and host graph operations.
Its 25% warm-workflow gate requires matching source, matrices, configuration and
environment. Setup is not gated. Tests cover looser-source refinement, full and
partial starts, changed replay, and injected accuracy/timing failures.

[#134](https://github.com/alecjacobson/warpack/issues/134) remains open for
symmetric multilevel correction, broader geometry/material/nullspace coverage
and larger mode counts. These measurements argue for reusing coarse information
during refinement rather than only spending more on a starting basis.
