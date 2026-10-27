# Library boundaries and execution contracts

`warpack/` contains reusable solvers, operators, device kernels and the optional
cuDSS integration. Public entry points are exported by `warpack.__init__`.
Underscore modules and kernel implementation modules are internal; their tuning
switches are not a stable public API. Numerical work uses Warp, with cuDSS used
only for sparse factorization/solves and explicitly requested factor metadata.

Application assembly belongs in `examples/` or `benchmarks/applications/`.
Independent NumPy/SciPy/CuPy/Spectra references, timing instrumentation and
experimental algorithm variants belong in `benchmarks/`. Regression tests live
in `tests/`; they may reuse benchmark fixtures, but runtime modules must not
import either directory. The wheel contains only `warpack/` and package metadata.
`scripts/audit_release_wheel.py` checks the exact module inventory and bytes;
`scripts/verify_release_wheel.py` exercises an installed wheel while rejecting
imports of tests, scripts, examples, benchmarks and optional competitor libraries.

## Arrays and ownership

Real solver vector blocks use FP64 rows of shape `(width, n)` on the operator's
device. Sparse operators accept strided views; input/output storage must not
overlap. cuDSS requires contiguous FP64 rows with the width fixed at setup and
supports in-place application. Construct, apply and capture a cuDSS inverse on
its construction stream; the native handle and workspace are stream-bound.
Application from another active stream is rejected before dispatch. This does
not prevent caller-ordered graph replay on another stream after capture;
concurrent use of the same solver scratch is unsupported. Validation checks array metadata without reading
device values. Complex CSR operators use `wp.vec2d(real, imaginary)`.

Solvers own mutable workspaces. Retain the solver, input buffers, graph and
inverse factors until execution completes. Shared-width CG variants also share
resources and must execute serially. Change values in existing device buffers
for replay; rebuild for topology or shape changes. Factor/preconditioner validity
remains the caller's responsibility.

## Graph setup and synchronization

`capture()` is standalone setup: it warms kernels, completes pending eager work
and records a graph. Calling it from an active capture raises before warmup.
For composition, construct and warm the solver first, then record `solve()` or
its `initialize()` / `step()` / `finalize()` methods in the caller's capture.
Method availability and fixed/adaptive policies depend on the solver class.
Integer iteration budgets are validated before initialization; `iterations`
remains a device output. A graph must be launched before its results are valid.

Standalone graph recording is centralized in `warpack/_capture.py`. Projected
Jacobi/Schur solvers record into an existing capture, or reuse an eager child
graph prepared during setup. Keep the setup synchronization: driver 570 has a
tracked graph-upload failure with pending eager writes (#61). This workaround
is not a synchronization between iterations or replay launches.

On the affected driver stack, also complete the graph's first launch during
setup, before enqueueing changed inputs or preceding simulation work:

```python
graph = solver.capture(100)
wp.capture_launch(graph)
wp.synchronize_device(solver.device)
# Subsequent input updates and graph launches can stay on the ordered stream.
```

Warp instantiates/uploads a graph lazily on its first launch. Synchronizing only
before recording does not protect a later first launch from pending uploads:
the #198 diagnostic observed a mixture of old and new input values at the first
graph node, and a kernel-only copy reproducer also fails without an eigensolver.
This first solve is setup/first-use work and must be included when reporting
end-to-end cost; repeated replay has no additional host barrier. The workaround
does not establish a fixed driver-version boundary. See #61 for upstream status.

Other explicit synchronization boundaries are:

| Operation | Reason |
|---|---|
| Sparse/KKT setup | Finalize sparse nonzero counts in cuDSS construction and saddle-point assembly |
| Active-CG bucket construction | Complete eager compilation/warmup of the allocated states |
| `independent_constraints()` | Read rank/validity integers to allocate the result's CSR shape |
| `CuDSSInverse.inertia()` | Explicit host query of native factor metadata |
| Caller `.numpy()` or synchronization | Retrieve results for inspection |

The last two are never called by solver replay. Pure-Warp replay has no host
convergence decision. Vendor cuDSS 0.8 can still record small HtoD metadata copies;
this is distinct from a device-to-host convergence readback and remains tracked
in #29/#189. Graph-DOT inspection checks recorded node/transfer kinds; it is not
a complete CUDA API trace.

Run accuracy checks on original equations, the requested spectrum and the
appropriate orthogonality metric. Include changed-input replay, rejected-input
state preservation and graph inspection. Compare performance in isolated workers
with identical inputs/settings and retain all timing samples, setup costs,
dependency identities and failures. A passing focused rerun does not erase an
unexplained full-suite failure.
