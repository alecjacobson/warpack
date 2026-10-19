# Pure-Warp indefinite inverse for FEM frequency bands

`warpack.GMRESInverse` supplies a reusable real FP64 inverse of a caller-provided
sparse matrix, including indefinite `K-sigma M`. It calls unmodified upstream
Warp GMRES. It adds device staging for multiple/in-place RHSs, fresh original
linear residual checks, and device statistics. It implements no separate GMRES
algorithm or factorization. Scalar CSR and 3x3 BSR are supported.

The public inverse can be supplied to `GeneralizedKrylovSchur`. The outer
problem still requires symmetric stiffness and SPD mass. A nonsymmetric
linear inverse being supported does not make a nonsymmetric stiffness valid
for that symmetric eigensolver. Complex shift-invert retains its existing
separate realification adapter.

## Capture and accuracy contract

Every RHS starts from zero. A supplied preconditioner must be a **fixed linear
right preconditioner**, not a variable inner CG solve. The FEM measurements
build upstream FP64 FSAI from SPD `K+M`; they never apply an SPD factorization
or CG to indefinite `K-sigma M`. Factor construction remains setup work.

Initialization, GMRES iterations, stopping and fresh residual diagnostics all
run in Warp on device. Matrix values may change in place with fixed sparse
topology and buffer addresses. Preconditioner updates are caller-managed;
replacing its buffers requires rebuilding the corresponding captured graph.
There is no cached copy of the source matrix to become stale on replay.

The default selects fixed outer capture for stock Warp 1.15 compatibility.
`adaptive_capture=True` requires upstream preallocated loop control, as in the
pinned unmodified Warp 1.18 environment. Unsupported states reject that opt-in.
The nested graphs are audited for HOST callbacks, HtoD and DtoH transfers.
The separate cuDSS comparison retains its vendor scalar HtoD copies (#29).

`stats` stores total inner iterations, maximum iterations per RHS, failed RHSs
and RHS count. Failure means the **fresh original linear residual** exceeds
the upstream effective tolerance, including its dtype safety floor. GMRES
checks at restart boundaries and may exceed `maxiter` by up to `restart-1`;
the counters report the actual work. `residual_maxima` stores maximum absolute
and relative fresh residuals. Both arrays accumulate until `reset_stats()`.
Inner failures are retained even when the outer eigenpairs pass.

Small residuals alone do not establish a complete requested eigenspace. Every
application trial additionally checks the nearest physical eigenvalue set,
the complete requested subspace, mass orthogonality, status and full count.
Rejected trials are preserved, without choosing favorable retries or relaxing
tolerances. All initial vectors are independent of reference eigenvectors.

## Validation

The full pinned-environment suite passes **664 tests in 339.53 s**. All **35
new GMRES tests also pass on stock Warp 1.15 in 431.71 s**; constructing the
larger fixed outer graphs dominates that compatibility run. Coverage includes
scalar/3x3 storage, symmetric-indefinite/nonsymmetric matrices, fixed right
preconditioning, multiple/zero/in-place RHSs, changed values on the same graph,
visible exhausted solves, and all five coupled FEM fixtures at ordinary/near
shifts. Pinned runs exercise nested adaptive graphs; stock rejects that opt-in
and uses fixed outer capture. Independent original equations and complete
requested sets remain mandatory.

[Validation and source hashes](../../results/performance_followup/gmres-validation.json).
[Clean wheel check](../../results/performance_followup/gmres-wheel.json): 21 core
modules, isolated installed `GMRESInverse` import, no benchmark/application
modules. This is a full local rerun, not a new complete CuPy/Spectra upstream
conformance run.

## Recorded controls

For the 10,890-DOF consistent-mass tetrahedral beam, six modes around an interior
shift, ncv48 and five outer cycles, FSAI64/GMRES128 gives a three-trial median
of **3.8204 s**, compared with **.12509 s** for matched cuDSS factors.
All physical/requested-space and status/count gates pass in every trial.
GMRES performs **52,224 inner iterations over 204 RHSs**, at most 256 per RHS,
with **zero failed inner solves** at relative tolerance `1e-12`.

The earlier GMRES64/maxiter512 screen takes **5.5775 s**, with 95,552 inner
iterations and seven failed inner solves. The outer pairs still pass all
gates, but that does not turn the inner failures into successes. It is a
single-trial control, not a median speedup comparison.

At 41,616 DOFs, GMRES128/maxiter768 is insufficient: all three recorded
trials have requested-subspace error **2.5428e-8**, above the unchanged `1e-8`
gate, despite eigenvalue/residual/full-count checks passing. There are 18
failed inner solves, with maximum relative fresh linear residual **7.3128e-11**.
These approximately 26-second results are rejected, not accepted timings.
The report retains all trials and the matching cuDSS controls.

Increasing the restart to 256 resolves this accuracy limitation. With
maxiter1024 and the same `1e-12` inner tolerance, all three large-case trials
pass at **19.3223 s**, versus **.27735 s** for cuDSS. The requested-subspace
error is **1.4946e-11** and maximum original residual **6.7949e-12**. Each
trial performs **55,296 inner iterations**, at most 512 per RHS, with **zero
failed inner solves**. The preceding tighter `1e-13`/maxiter1536 screen also
passes outer checks but takes 28.6925 s and records 17 inner failures; it is
retained as an accuracy/cost control, not the chosen repeated profile.

The 10,890-DOF near-eigenvalue shift (`fraction=.001` within the reference gap)
passes every outer gate at **4.9394 s**, versus **.12442 s** for cuDSS. However,
GMRES128/maxiter768 reports **seven failed inner solves out of 204**, with
maximum relative fresh linear residual **1.0960e-10**. Its physical eigenpair
residual is **7.0267e-12** and requested-subspace error **1.4251e-13**. This is
an accepted outer eigenproblem result with explicitly incomplete inner
convergence, not evidence that the linear tolerance was achieved.

The large run's setup/capture device free-memory decrease is **14.594 GB**:
**14.575 GB** occurs during setup and **.0189 GB** during capture. These include
runtime allocations and pools, not exact live or peak workspace. Upstream FSAI
documents quadratic per-thread setup workspace as row support grows; the
measurements do not attribute all runtime allocations to a particular kernel.
The optional indefinite backend is currently much slower than cuDSS and is
not a replacement for the flagship low-mode CG configuration.

Raw reports: [GMRES64 screen](../../results/performance_followup/interior-gmres64-screen.json),
[GMRES128 screen](../../results/performance_followup/interior-gmres128-screen.json),
[GMRES128 repeated comparison](../../results/performance_followup/interior-gmres128.json),
[larger rejected profile](../../results/performance_followup/interior-gmres128-large.json),
[GMRES256 accuracy control](../../results/performance_followup/interior-gmres256-large-screen.json),
[accepted larger comparison](../../results/performance_followup/interior-gmres256-large.json),
[near-eigenvalue shift](../../results/performance_followup/interior-gmres128-near.json).
Adjacent occupancy files record sequential runs and no observed competing GPU
processes; clocks were not locked. CPU references, matrix assembly/transfers,
setup, capture, warmup and validation are excluded from replay timing.
The final driver adds an explicit `--band-index` argument (default 6); all
recorded runs use that same sixth positive spectral gap. Earlier driver hashes
also differ in progress output and setup-memory instrumentation; the numerical
inverse implementation is unchanged across these recorded comparisons.

## Reproduce

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_gmres_inverse.py
python -m benchmarks.experimental.interior_gmres \
  --case clamped_consistent --side 10 --restart 128 --maxiter 768 \
  --steps 5 --repeats 3 --out build/interior-gmres.json
```

`--fraction .001` moves the shift close to a reference eigenvalue within its
enclosing gap. `--case` also accepts lumped mass, heterogeneous stiffness,
disconnected repeated modes and repeated bending. The reference construction
is shared with the existing interior-FEM comparisons.
