# Library API, capture and separation review

2026-10-27, supported main baseline `a1b739a`; L40, driver 570.158.01.
This review excludes the held wide-projection candidate in PR #201. It does not
resolve the intermittent mass-QR issue #198 or certify other GPUs/drivers.

## Changes

- [#202](https://github.com/alecjacobson/warpack/issues/202): centralized the five
  standalone graph-recording sites in a private setup helper. Public capture
  methods reject an already active capture before warmup. Symmetric integer
  budgets now fail before initialization, and eager solves replace stale
  device iteration counts from previous replay.
- [#203](https://github.com/alecjacobson/warpack/issues/203): sparse operators
  validate row shapes, FP64 dtype and device before dispatch. cuDSS additionally
  requires packed contiguous storage and its configured RHS width. Checks use
  host metadata only; valid strided Warp products remain supported.
- [#204](https://github.com/alecjacobson/warpack/issues/204): moved graph-DOT
  inspection into a standard-library-only benchmark utility. Migrated consumers
  and retained the old import as a compatibility alias. CuPy imports in legacy
  profiling, Anvil and upstream comparison drivers now occur only when those
  comparisons run, rather than while ordinary Warp tests import helpers.
- [#205](https://github.com/alecjacobson/warpack/issues/205): cuDSS application
  rejects a different active stream before any work. The pinned native handle
  and workspace are bound at construction. A non-default-stream regression
  covers changed RHSs, captured eigensolves and caller-ordered graph replay.

Public constructors, method names and numerical kernels are unchanged. New
helpers are private. The one added eager device write updates the public
iteration-count output; it does not change eigenpair arithmetic. Setup and
warm replay are measured separately below.

## Review conclusions

The runtime package imports only its own modules, Warp, standard-library tools
and the optional cuDSS binding. FEM/skinning/geometry assembly, independent
references, experimental comparisons, profiling and test handlers remain
outside `warpack/`. A static dependency guard and installed-wheel import blocker
protect that boundary. No application data or comparison dependency was moved
into the wheel.

The replay paths inspected contain no convergence readback. CG/GMRES adapters
request upstream `check_every=0`; the pinned upstream implementation records its
loop condition on device. Active-row selection and fallback decisions are
also device operations. Python validation inspects shapes/types/devices/stream
handles and executes during eager calls or graph recording, not graph replay.

The setup device synchronization remains deliberate: pending eager writes can
trigger the driver graph-upload failure tracked in #61. Sparse nonzero-count
finalization, active-CG warmup, rank-revealing constraint setup and explicitly
requested inertia queries are other setup/diagnostic synchronization points.
The review does not claim that eliminating all setup synchronization is safe.
cuDSS 0.8 vendor HtoD metadata copies remain tracked in #29/#189; they are not
host convergence checks. DOT inspection checks recorded node types and copies,
not every CUDA API call.

[Library execution contracts](../../docs/library_contracts.md) document array
layout, ownership, stream restrictions and composing solves inside caller graphs.

## Regression evidence

Validation and timing results are being collected; this report is not yet a
completed qualification. Raw failures and interrupted attempts will be retained
alongside final passing checks. No README timing row has been replaced.
