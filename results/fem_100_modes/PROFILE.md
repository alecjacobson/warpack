# Active-search profile of the 100-mode FEM solve

Follow-up to [#147](https://github.com/alecjacobson/warpack/issues/147), measured
2026-10-23 on L40 with the same clamped consistent-mass beam, 34,560 variables,
100 requested modes, width 104, FSAI 32/FP32-CG 25, cached operator images, full
history and four-row search/CG buckets. The configuration and seeded host start
are read from the earlier accepted comparison. No solver arithmetic or stopping
rule is changed by the profiler.

The diagnostic restores an uninstrumented fixed prefix for each sample, reads
the active branch on the host, then profiles that selected step in a separate
graph. This host selection is **profiling-only**, not part of normal solve
replay. The ordinary complete device-controlled solve passes native 100-pair,
original residual, metric orthogonality, requested spectrum and full-subspace
checks both before and after instrumentation.

Three samples per checkpoint, median milliseconds:

| Completed prefix steps | Selected-step total | Preconditioner | X orthogonalization | Cholesky factors | Triangular applications | Metric eigensolves | Ritz eigensolve |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 154.471 | 26.065 | 21.546 | 14.663 | 2.554 | 14.230 | 24.583 |
| 10 | 124.142 | 7.820 | 19.913 | 13.423 | 2.374 | 19.445 | 17.957 |
| 20 | 85.953 | 2.394 | 19.913 | 13.423 | 2.374 | 15.463 | 11.944 |
| 30 | 75.319 | 1.222 | 19.910 | 13.424 | 2.373 | 11.562 | 10.784 |

These nested categories overlap and **must not be added**. Cholesky and
triangular operations are included in X orthogonalization. Metric eigensolve
events sum both whitening passes; the raw `metric_sweeps` field describes only
the last one. Prefix replays can take slightly different trajectories, so each
sample records its actual active count, selected bucket and projected dimensions.
These are diagnostic step costs, not a replacement for adaptive whole-solve
benchmarks or a claim about CuPy's bottleneck.

The full X block remains 104 rows even when few residual rows need further CG.
Its two serial Cholesky factorizations account for a persistent 13.4 ms per late
step. [#154](https://github.com/alecjacobson/warpack/issues/154) tracks cooperative
factorization without changing the triangular solves or eigensolver policy.
The projected metric/Ritz solves and basis products also remain significant;
CG is a small fraction of late steps.

The accepted detailed run is [`profile-detail/active-a3.json`](profile-detail/active-a3.json)
with its [uncontended occupancy record](profile-detail/active-a3.occupancy.json).
Attempts a1/a2 are contended and excluded. The earlier basic profiler's
[`profile/active-a2.json`](profile/active-a2.json) is an accepted diagnostic
without the Cholesky/triangular split; a1 there is contended. All attempts are
retained. Source/dependency identities and independent before/after checks are
in each raw file. The profiler's preservation test also verifies wrapper
restoration and valid event measurements.

```bash
python -m benchmarks.monitor_gpu --out profile.occupancy.json --log profile.log -- \
  python -m benchmarks.experimental.profile_active_block \
  --config results/cg_partitioned/final/exact-a1.json \
  --checkpoints 1,10,20,30 --repeats 3 --out profile.json
python -m pytest -q tests/test_active_block_profile.py
```

Use the pinned FSAI environment and four CPU threads/cores as in the main
comparison; validation and host instrumentation are outside the measured CUDA
event intervals. The ordinary captured solve's graph contains no HOST, HtoD or
DtoH nodes.
