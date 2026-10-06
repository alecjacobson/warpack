# Pre-optimization performance and synchronization audit

Archived measurements and experiment notes from the README before its documentation cleanup. Results, test counts, and environment descriptions refer to the runs recorded here. Run commands from the repository root.

[Current README](../README.md) · [Historical index](README.md)

These historical measurements motivated the v0.1.1 changes. The [audit scripts](../benchmarks/audit_profile.py) separate host enqueue time from CUDA-event elapsed time and inspect captured graph nodes. This is a source/graph/event audit, **not** a full Nsight CUDA API trace. Nsight Systems was unavailable in the test environment.

The pure-Warp 125,000-variable solve took **587.27 ms on device**, **587.36 ms wall time**, and **0.36 ms to enqueue**. The graph contains only kernels, memsets, device-to-device copies, and a device conditional loop. There are no host callbacks or host/device copies. Explicit setup synchronization and final benchmark completion waits are outside the numerical iteration. The representative restart costs are:

| Stage | GPU time per restart (ms) |
|---|---:|
| Expand basis and reorthogonalize | 3.56 |
| Form projected matrix | 1.34 |
| Solve 48×48 projected eigenproblem | 2.81 |
| Rotate basis and cached operator products | 1.02 |
| Copy results and compute residuals | 0.17 |

These are separate instrumented stage measurements, not an exact decomposition of an entire converged run. The optimization targets identified at that point were a cheaper Warp projected eigensolve, incremental Lanczos/arrowhead projection instead of rebuilding the full Gram matrix, more efficient orthogonalization reductions, and device-conditional breakdown recovery. Ten fixed Jacobi sweeps and thousands of shared-memory barriers are expensive even for a small projected problem. Reducing accuracy alone does not close the gap: Warp at tolerance 1e-10 took 553 ms with a maximum absolute residual of 4.18e-10. The fixed-seed CuPy run at tolerance 1e-9 took 439 ms with residual 9.95e-10. [CuPy 14.2.0's small projected eigensolve](https://github.com/cupy/cupy/blob/v14.2.0/cupyx/scipy/sparse/linalg/_eigen.py#L300) uses CPU NumPy; reproducing that choice would violate this project's device-computation requirement.

**The optional cuDSS path has a separate caveat.** Its dragon graph contains **172 pageable host-to-device metadata copies of 8 bytes each**, two per inverse application. `capture_launch` itself takes about **614 ms**, overlapping a **933 ms** GPU solve. This is blocking host submission, despite no Python-level readback. Replacing those sources with pinned memory or device memory in a diagnostic graph experiment left total solve time essentially unchanged (933–934 ms), and the launch call still took about 600 ms. The copy nodes therefore do not explain the elapsed-time gap by themselves. The experiment is not a production workaround: the metadata belongs to cuDSS internals. A CUDA API timeline is still needed to distinguish driver submission/backpressure from other backend serialization. Host enqueue time must **not** be added to device time because they overlap.

The original dragon run also did avoidable work: **two** 48-vector restart cycles produced a maximum original-problem residual of **2.22e-8 in 0.717 s**, versus 0.933 s for three cycles, reducing inverse applications from 86 to 67. The same-factor CuPy comparison with 64 vectors took 0.591 s with 64 inverse applications and residual 7.18e-8. CuPy with 48 vectors took 0.644 s but had residual 2.58e-7, failing the showcase's 1e-7 threshold. This is evidence for better stopping/locking and equivalent original-problem accuracy checks; it is not a blanket recommendation to use two cycles on other matrices.

The dragon numbers above **include cuDSS**. Eliminating that optional backend as well needs a competitive Warp preconditioner for the inverse/lowest-mode problem; the existing basic CG inverse is not a demonstrated replacement at this scale.

Reproduce after generating the benchmark inputs:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m benchmarks.audit_profile
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m benchmarks.audit_dragon
# Diagnostic graph-copy substitution also requires cuda-bindings:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m benchmarks.audit_cudss_copies
```

Raw follow-up results: [pure Warp](../results/audit_profile.json), [shared-factor dragon comparison](../results/audit_dragon.json), [cuDSS copy experiment](../results/audit_cudss_copies.json). [Original release measurements](v0.1.0-benchmarks.md) remain unchanged.
