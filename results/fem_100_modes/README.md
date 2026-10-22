# Wide FEM mode extraction

Follow-up for #128; performance investigation #147 and adapter-memory issue
#146. Procedural P1 elastic beams use independently assembled stiffness/mass
and an expanded ARPACK shift-invert reference. No external meshes are copied.
Both native stopping and original requested-spectrum/subspace checks are
required. The pinned upstream Warp FSAI/CG overlay is recorded in every worker.

## Qualified replay comparison

L40 / driver 570.158.01, Xeon Platinum 8362, four CPU threads on cores 8–11.
For the 34,560-DOF clamped beam with consistent mass, request 100 modes with a
104-vector starting block. FSAI32 and FP32 CG capped at 25 iterations (relative
tolerance 0.01, absolute zero) precondition both outer methods. Inner
preconditioner nonconvergence is allowed; final physical accuracy is checked.

| Method | Replay median (s) | Individual replays (s) | Worst original relative residual | Worst requested subspace error |
|---|---:|---|---:|---:|
| Public Warp BlockLOBPCG | 3.7120 | 3.7120, 3.7863, 3.6414 | 9.35e-12 | 1.50e-11 |
| CuPy LOBPCG + Warp CG adapter | 3.0425 | 4.0081, 3.0425, 2.8874 | 7.33e-11 | 8.10e-12 |

[Raw worker](final/clamped_consistent-a2.json),
[occupancy record](final/clamped_consistent-a2.occupancy.json),
[strict summary](final/clamped_consistent-a2.summary.json).
This pair was uncontended. The CuPy trials vary substantially. Warp stops on
100 normalized residuals at 1e-11 (34–36 outer steps); CuPy converges all 104
absolute residuals at 1e-10 (history lengths 263/186/172). Thus the 1.22× median
replay difference is a measured workflow gap, not an identical-stopping
algorithmic comparison. Width and stopping controls remain under #147.
The worker records a warmup duration but does not retain warmup physical checks;
only the three measured solves have complete saved validation.

Setup is **not included in replay**. Warp preconditioner plus solver/upload
setup is 0.4327 s and capture is 4.2094 s; warmup adds 3.7800 s. These are single
stage observations, with cached kernels, not repeated cold-process setup
medians. The CuPy matrix/start/adapter preparation takes an additional 0.7857 s
and reuses the already constructed FSAI/base CG. That excludes construction of
those supplied objects and is not standalone CuPy setup. We therefore make no
end-to-end setup-speed claim from this worker. Source/reference assembly is
also outside these reported GPU setup stages.

The exact-width CuPy adapter reserves states for every active width 1..104:
5,460 RHS workspace slots. That is quadratic storage, tracked in #146; it avoids
charging zero-padded work to CuPy. The observed device-free-memory decrease is
4.50 GiB for Warp setup/capture and an additional 9.72 GiB for the CuPy adapter.
These context-level deltas include runtime allocations and are not peak or
isolated solver-memory measurements.

## Accuracy coverage and exclusions

All three Warp replays pass 100 modes for consistent and lumped clamped mass.
The free beam passes 106 pairs: six known rigid modes and 100 elastic modes,
with worst residual 7.73e-12 and requested subspace error 1.11e-11 in
[free-a1](final/free-a1.json). CuPy also passes this free-beam worker. The
lumped/free runs seen so far have overlapping external GPU work and are not
promoted as timings.

The third lumped worker is rejected: one CuPy replay reports its last history
residual below 1e-10, but a fresh residual for its final padding vector is
1.0002837e-10. The first 100 physical modes pass; the entire 104-vector native
criterion does not. This small boundary miss is preserved, not rounded into
acceptance. Related native soft-locking/history behavior is tracked in #69.
Earlier accepted but contended attempts and the one-repeat screening are kept.

The fast-block regression suite additionally tests 30 elastic modes for all
five tetrahedral families: clamped consistent, clamped lumped, free,
disconnected and heterogeneous. Together with existing 3/6-mode cases and
Hermite bending, all 17 tests pass across two replays per case. High-mode strict
Hermite qualification remains separate under #130. This does not complete #128:
wider disconnected/heterogeneous comparisons and additional hardware remain.

## Reproduction and regression checks

Use the exact command stored in the occupancy JSON. It includes all bucket,
layout, CG, projected-product and stopping options; shortening it can change
the algorithm being timed. Generate a summary with:

```bash
python -m benchmarks.applications.fem_wide_report \
  results/fem_100_modes/final/clamped_consistent-a2.json
```

The reporter rejects contention, missing repeats, incomplete boundary clusters,
wrong rigid counts, native failures (including padding residuals), physical
accuracy failures, non-device graphs and invalid timings. Twenty-eight CPU
reporter tests cover these gates and incompatible/performance-regressed
baselines. Regression comparisons require identical source, dependency,
configuration and environment records. Historical worker source hashes use a
manual list; the incomplete transitive scope is tracked separately in #149.
