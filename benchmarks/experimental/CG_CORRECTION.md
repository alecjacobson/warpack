# Fresh-residual correction for connection CG

Follow-up to [#95](https://github.com/alecjacobson/warpack/issues/95) and the
[connection comparison](../applications/CONNECTION.md). The first prototype
wraps unmodified upstream Warp CG with one additional solve of the freshly
computed original residual. Both states share the same upstream FSAI32
preconditioner (FP32 factor, FP64 CG). Initial relative tolerance is1e-11,
absolute tolerance0; correction relative tolerance is0.01. The final original
linear residual is audited at1e-11. Recursive and fresh statistics count
original RHS calls and total initial+correction iterations; the constituent
states' counters are also retained.

All correction products, updates, stopping and audits execute in Warp. This
is an experimental adapter; the prototype does not modify installed core CG.
It supports one contiguous FP64 RHS, including in-place calls, and rejects
output aliases of its correction workspaces before mutation.

## Evidence

L40 / driver570.158.01 / pinned Warp1.18, consistent complex mass, k20, complex
ncv64, σ−0.1, three outer cycles, three warm synchronized samples. All original
physical accuracy/orthogonality/spectrum/subspace gates remain unchanged at1e-8;
all20 complex and40 real physical convergence counts are required at1e-10.

| Profile | Complex n | Fresh inner misses | Complex / real counts | Solve median (s) | Accepted |
|---|---:|---:|---:|---:|---|
| No correction, inner1e-11 | 4096 | 2 / 298 | 10 / 20 | 0.8821 | No |
| One correction, inner1e-11 | 4096 | 0 / 298 | 20 / 40 | 1.0133 | Yes |
| One correction, inner1e-13 | 4096 | 6 / 298 | 20 / 40 | 1.0967 | No |
| One correction, inner1e-11 | 16384 | 0 / 298 | 20 / 40 | 2.5857 | Yes |

The matched n4096 control establishes that relaxing the inner request alone
does not restore strict stopping. Correction costs about14.9% and restores all
counts and fresh checks. This is an accuracy improvement with measured cost,
not a speedup over the rejected control. The tighter1e-13 experiment still
fails: its maximum fresh relative residual is1.14e-12. Correction does not
guarantee an arbitrarily strict floating-point residual, and that old request
has not been relabeled as passing.

At n16384, maximum original eigenpair residual is1.65e-11, fresh relative linear
residual4.54e-12, and all requested-space gates pass. Iterations are50492 initial
+7331 correction =57823 total. Setup0.1825s and capture/JIT/warmup4.0387s are
separate from replay. A matching regression replay gives2.5904s (ratio1.001815),
passes all checks, and stays below the1.25 performance gate. Relative to the
previous accepted cuDSS0.3296s, this still leaves about7.8× replay cost, tracked
in [#97](https://github.com/alecjacobson/warpack/issues/97).

All profiles retain their raw samples and stopping evidence:
[matched control](../../results/connection-correction-control.json),
[n4096 correction](../../results/connection-correction-screen.json),
[rejected tighter request](../../results/connection-correction-tight.json),
[n16384](../../results/connection-correction-16384.json),
[baseline replay](../../results/connection-correction-regression.json).
Each has a corresponding `.occupancy.json` record. Accepted runs were
uncontended; failed runs remain rejected. Representative graphs contain no
HOST, HtoD or DtoH nodes.

The prototype and connection suite pass50 tests on pinned Warp1.18 in36.61s:
scaling, original-RHS preservation, in-place aliases, stage counters, zero and
nonfinite input/recovery, graph transfer audits, and repeated complex pencils
with changed starts. [Checkpoint validation](../../results/corrected-cg-prototype-validation.json).

```bash
python -m benchmarks.monitor_gpu --out build/correction.occupancy.json \
  --log build/correction.log -- python -m benchmarks.applications.connection_compare \
  --side 128 --modes 20 --ncv 64 --steps 3 --repeats 3 --cg-tol 1e-11 \
  --methods warp-cg-corrected --out build/correction.json
```

The pinned upstream Warp FSAI environment is required for this comparison.
`--correction-steps 0` is the explicit control; `--correction-tol` controls the
relative tolerance for the residual solves. Baseline gates include both values.
