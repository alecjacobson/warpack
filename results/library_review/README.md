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

The initial complete regression failed; qualification remains held while the
first-launch containment is checked. Completed timings and both failing and
passing controls are below. No README timing row has been replaced.

### Completed checks (source `127fd97`)

- Pinned Warp 1.18/FSAI full suite: **3,511 passed, 11 skipped, 8 failed** in
  1,867.66 s. All eight failures are the shared mass-QR fixture tracked in #198.
  This is a failed full regression, not a passing qualification. No new API,
  stream or separation test failed. An earlier deliberately interrupted run
  (1,677 passes, 30 skips) is retained separately and is not counted as complete.
- Stock Warp 1.15 focused regression: **232 passed** (173.86 s).
- Installed-wheel cuDSS capture/pivot/stream checks: **26 passed** (66.30 s).
- Installed-wheel pure-Warp smoke passed with both stock and pinned Warp while
  imports of comparison libraries and source-tree helpers were blocked. The
  audit verified all 45 installed modules byte-for-byte against source;
  wheel SHA256 `55206be388eb30b0c4fcd53fe694e4d6a0e7a70924d75af12284b8ea5be17585`.
- All 3,528 tests collected with CuPy blocked before the two final stream tests
  were added; final pinned collection contains 3,530 tests. Hosted package/lint
  checks pass. No numerical tolerance or existing skip condition was relaxed.

Unchanged main `a1b739a` also fails the focused 195-test stage group: one run
returned 194 passes and a subspace error of 0.2273; a later fresh process again
returned 194 passes and a subspace error of 0.6098. Both outputs in the latter
case are mass-orthonormal, but the first has relative distance 0.08163 from the
input row-space. The second output is correct to 9.52e-16. Actual device mass
values, columns and offsets exactly match the host reference. This establishes
an existing correctness failure on main, not its root cause or equivalence to
all eight full-suite status failures. Passing intervening 195-test runs do not
resolve it. State snapshots are collected only after the original assertion
fails; the diagnostic plugin adds no pre-assertion synchronization.

### Matched FEM timing regression

Procedural clamped consistent-mass elasticity, 34,560 DOFs, 1,413,566 stiffness
nonzeros; identical input fingerprints/settings on main and this branch. Three
fresh workers per version, alternating order, five timed replays per worker.
Every worker builds its own GPU factors/preconditioner; only the independently
computed CPU reference is cached. GPU occupancy monitoring accepts all workers.
The existing kernel cache is shared, so setup is not a cold-JIT comparison.

| Solver / requested modes | Main replay (s) | Review replay (s) | Change | Main setup + capture (s) | Review setup + capture (s) |
|---|---:|---:|---:|---:|---:|
| Pure Warp BlockLOBPCG + FSAI/CG / 100 | 1.99860 | 1.99323 | -0.27% | 1.87818 | 1.86746 |
| Warp generalized Krylov + cuDSS / 20 | 0.410379 | 0.410599 | +0.054% | 1.20965 | 1.19760 |

Replay entries are medians of 15 samples; setup + capture is the median of
three worker sums. Raw records also separate setup, recording and first launch.
The two rows request different mode counts and are **not** a direct backend
speed comparison. All 60 timed solves pass physical residual, orthogonality and
requested-spectrum/subspace checks. Review maximum relative residuals are
9.08e-12 (100-mode block) and 8.51e-13 (20-mode direct). The predeclared limits of
10% replay regression and 25% setup-plus-capture regression both pass.

Recorded graph counts match main exactly. Pure Warp records 323 device-to-device
copies, no host copies and no host nodes. cuDSS records 702 host-to-device
metadata copies and 1,078 device-to-device copies, with no host nodes or
device-to-host copies. Its strict device-only contract consequently remains
false (#29/#189); this review does not disguise vendor copies as a passing
contract. Timing/accuracy of these cases does not override the full-suite hold.

[Raw evidence](evidence/) includes complete failed logs/JUnit, passing checks,
post-failure states and every timing sample with source/dependency/occupancy
records. [The manifest](evidence/manifest.json) hashes decompressed artifacts;
[the timing summary](evidence/timing-summary.json) contains aggregate values.
No Anvil README timing row has been replaced by these different FEM cases.

### Mass-QR first-launch diagnosis and containment

A subsequent unchanged-main run reproduced the wrong-subspace result again
(194 passes, one failure). DOT exports show the expected serial node chain and
separate solver allocations. Instrumenting the start of the QR graph with a
device-to-device snapshot then reproduced a **status failure** in the unchanged
main `BlockLOBPCG` path at width 120, on the first normal input:

- The graph execution counter is one.
- The first 14,080 values of its 44,040-value input match the newly assigned
  source exactly. The remaining 29,960 values match the previous orthonormalized
  warmup output to 1e-14. This is input-ordering corruption before QR arithmetic.
- The original host source and all sparse mass arrays are correct. The second
  solver has not yet launched its graph.

The [input snapshot](evidence/state-probed-fixed-2-baseline/) and the uninstrumented
[graph snapshots](evidence/state-dot-1-baseline/) are retained. Instrumentation
changes the program, so its failure membership is not substituted for the full
regression. An initial instrumentation attempt allocated in conditional capture
and failed for that reason (10 failures); its log is also retained. The corrected
instrumentation allocates only during eager setup.

The eigensolver-free `copy-upload.py` sequence reproduces stale input values in
310/1,500 trials (294 at size 126,276 and 16 at size 1,048,576). Each graph merely
zeros a status word and copies input through a kernel. Synchronization after
input upload passes 1,500/1,500; completing one graph launch during setup also
passes 1,500/1,500. The latter adds no subsequent replay barrier. An isolated
single-size reduction passed all 300 trials for each Warp/CuPy mode, so those
controls do **not** establish a Warp-free reproduction of this particular input
failure. An earlier first-node snapshot probe (600 trials) and direct fixture
stress (400 trials) also passed. Keep this sensitivity explicit. The older #61
CUDA-only upload/memset reproducer is related evidence, not proof of an identical
implementation defect.

CUDA Compute Sanitizer 12.9.79 memcheck reports zero errors on two selected
original QR fixtures (both pass under instrumentation). This does not rule out
all races or establish a driver fix. NVIDIA's redist SHA256 was checked before
using the tool; it is not a project/runtime dependency.

The shared fixture now completes its first graph launch at setup, before its
changed-input uploads. All original six input/mass variants, status assertions,
image equations, orthogonality and requested-subspace checks remain. The
195-test stage group passes with this containment; the complete regression is
in progress. The installed numerical code and matched timing configurations
have not changed. Public documentation now distinguishes completion before
capture from completion of first launch; neither step is a per-iteration barrier.
