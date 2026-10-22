# Optional completeness certification on Newton FEM modes

This measures the new `LowestModeCertificate` stage from #133, separately from
the eigensolve. The independently generated spring lattices and unchanged Newton
CPU reference are described in [the Newton comparison](../newton_ffrf/README.md).
There are 12 elastic modes, plus six supplied rigid modes for the free case.
No Newton mesh or matrix data are copied.

Every case passes the certificate and the existing independent original-equation,
requested-spectrum, mass-orthogonality and invariant-subspace checks against the
cached Newton reference. The reference's separate CPU inertia check previously
certified those modes; the new cuDSS counts are 18 (free) and 12 (fixed), with
INFO=0 and pivot counter=0. This is a numerical check under symmetry/SPD-mass
preconditions, not an interval-arithmetic proof.

| Case / eigenbasis source | DOFs | Certificate setup (s) | Refactor/count refresh (s) | Device checks (ms) | Max certificate residual |
|---|---:|---:|---:|---:|---:|
| Free / Warp + cuDSS | 28,800 | 0.5058 | 0.5052 | 0.4228 | 4.15e-14 |
| Free / pure Warp FSAI/CG | 28,800 | 0.5015 | 0.5058 | 0.4246 | 1.08e-10 |
| Fixed / Warp + cuDSS | 27,360 | 0.4835 | 0.4812 | 0.2370 | 5.31e-14 |
| Fixed / pure Warp FSAI/CG | 27,360 | 0.4833 | 0.4810 | 0.2382 | 4.69e-11 |

All four **certificates use cuDSS**, regardless of which backend computed the
modes. Pure Warp alone can run the residual/basis checks but cannot certify the
count. Setup includes device diagnostic buffers, shifted-matrix construction,
a fresh factorization, host metadata query and factor release. Refresh is an
alternative cost after changing matrix values, not an additional required stage
on first use. Device checks use the cached count and recompute all vector checks.
The checks' graphs contain only kernels, memset and device-to-device copies; no
HOST, HtoD or DtoH nodes. The factor/query is explicitly outside capture.

Hardware: NVIDIA L40 48GB, Xeon Platinum 8362, driver 570.158.01. Warp 1.18.0
with the pinned FSAI overlay, cuDSS 0.8.0.10. Times are medians of three trials
in each worker after one discarded warmup, with fresh factors each time. Device
check time is wall time averaged over 100 queued graph replays per trial; it is
not isolated single-launch latency. CPU affinity is 8–11, BLAS/OpenMP threads 4.
All published workers have `timing_eligible=true` in their occupancy records.
One contended free/pure-Warp attempt was excluded and replaced. Compilation,
mesh assembly, eigensolve and host reference validation are outside these
certificate timings. The raw JSON retains the cold warmup separately.

The physical pencil has large coefficients. For certification we divide K,
eigenvalues and the gap bound by `SPRING/SPACING = 2e11`, retaining physical M
and eigenvectors. This preserves the problem and makes the absolute residual
floor meaningful for rigid modes. The table uses that scaled residual; each
worker also records the original physical eigenpair checks in `quality`.

## Reproduction

Generate the references with the [Newton commands](../../benchmarks/applications/NEWTON_FFRF.md)
first. The benchmark verifies their matrix hashes and certified CPU reports.
For each boundary `free`/`fixed` and eigenbasis backend `cudss`/`warp`:

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python -m benchmarks.monitor_gpu \
  --out results/completeness/free-cudss.occupancy.json \
  --log results/completeness/free-cudss.log -- \
  taskset -c 8-11 python -m benchmarks.applications.completeness \
  --boundary free --backend cudss \
  --reference build/newton-free-reference.npz \
  --cpu-report results/newton_ffrf/free-newton-1.json \
  --out results/completeness/free-cudss.json
```

The bound is the midpoint between the last requested value and the next guard
in the certified reference. Bound discovery is not implemented or timed here;
production callers must supply an appropriate gap, possibly by solving for extra
guard modes. This benchmark does not retime Newton or alter the original
setup/solve performance table.

## Regression coverage

`tests/test_completeness.py` covers complete and deliberately incomplete repeated
spectra, wrong exact spectra, duplicate vectors, a split boundary cluster,
perturbed pivots, unavailable/invalid metadata, non-diagonal SPD masses,
verified rigid deflation, corrupted values/vectors, conservative whole-subspace
gap checks, failed refresh invalidation and changed counts on the same graph.
Small dense and sparse congruence problems have independently known spectra.

The pinned-runtime regression log contains 103 passing existing/new tests across
completeness, Newton, cuDSS capture, numerical nullspaces, mass projection,
finalization and general solver regression. The subsequently added changed-count
capture test passes separately (`regression-refresh.log`). All 19 completeness
tests also pass on stock Warp 1.15.0 (`regression-stock.log`). No CPU numerical
backend was added to the production library.

Package/lint checks also pass. The built wheel imports every core module while
blocking SciPy, CuPy, application/benchmark packages and `warp_cudss`, and passes
three changed-matrix replays. Thus the new optional metadata API does not make
cuDSS a mandatory import for pure-Warp users (`wheel-smoke.json`).
