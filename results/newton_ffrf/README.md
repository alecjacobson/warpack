# Newton FFRF comparison — 2026-10-21

[Problem, implementation, scope and reproduction](../../benchmarks/applications/NEWTON_FFRF.md).
Newton source pinned to `360a9b29a469af04eae6909f6d1709460a8dc35e`.
Only upstream source code was downloaded. All geometry and matrices were
generated locally; no upstream dataset or mesh was copied.

NVIDIA L40; Intel Xeon Platinum 8362; four CPU cores/BLAS threads.
Warp 1.18.0 with the documented checksum-pinned FSAI overlay; FP64 outer
eigensolvers, FP32 FSAI/CG preconditioning. See [environment.json](environment.json).

## Timings

Seconds, medians of three independent setups and nine warm GPU solves
(three CPU solves). Twelve elastic modes in each case.

| Problem | Solver | Setup | Warm solve | Inertia certificate | First-use total |
|---|---|---:|---:|---:|---:|
| Free-free, 28,800 DOFs | Newton ARPACK + SuperLU | 17.7310 | 4.2114 | 18.5932 | 40.6331 |
| Free-free, 28,800 DOFs | Warpack + cuDSS | 0.9931 | 0.0834 | — | 1.0912 |
| Free-free, 28,800 DOFs | Pure Warp FSAI/CG + block LOBPCG | 0.6072 | 0.2308 | — | 0.8540 |
| Fixed-interface, 27,360 DOFs | Newton ARPACK + SuperLU | 15.5864 | 3.3086 | 15.5368 | 34.4517 |
| Fixed-interface, 27,360 DOFs | Warpack + cuDSS | 0.9445 | 0.0793 | — | 1.0379 |
| Fixed-interface, 27,360 DOFs | Pure Warp FSAI/CG + block LOBPCG | 0.5765 | 0.1931 | — | 0.7752 |

First-use total is the median of each worker's actual setup plus its first
solve, including Newton's inertia certificate. It is not the sum of column
medians. GPU first-launch cost is included there; warm solves exclude it.
All times exclude common geometry/matrix assembly and external validation.
This is a cached-JIT measurement, not cold installation/startup time.

The certificate is a service difference in these recorded runs, so the total
column is **not** an equal-service speedup claim. Warp results here are independently
checked against Newton's certified reference and do not generate their own inertia
certificate. The subsequently added optional [certificate stage](../completeness/README.md)
is measured separately; its cost is not included in this table.
Even excluding Newton's certificate, its setup plus solve takes about 21.94 s
free-free and 18.89 s fixed-interface. These compare with first-use Warp totals
of 0.78–1.09 s. Static constraint solves, constraint elimination, modal recovery
and full FFRF simulation are outside this comparison.

For repeated solves, cuDSS is 50.5x/41.7x faster than Newton's solve stage on
free/fixed cases; pure Warp is 18.2x/17.1x faster. Pure Warp has the lowest
first-use cost because its preconditioner setup is cheaper. This does not
establish the same performance for the external six-DOF shell panel or a
consistent-mass model.

Setup ranges are preserved in [summary.json](summary.json). The largest
spread between independent warm-solve batch medians is 4.1% for Newton and
0.31% for Warp. All 18 occupancy records are timing-eligible. No rejected
worker is omitted from this recorded batch.

## Accuracy and coverage

Worst values over measured solves; all eigenvalues and vectors are recovered
in physical units and tested against the original `K phi = lambda M phi`.

| Problem | Solver | Relative residual | Relative eigenvalue error | Mass orthogonality | Requested subspace error |
|---|---|---:|---:|---:|---:|
| Free | Newton | 5.861e-13 | Reference | 2.221e-15 | Reference |
| Free | Warpack + cuDSS | 4.168e-13 | 1.589e-13 | 1.422e-14 | 7.675e-14 |
| Free | Pure Warp | 1.231e-10 | 2.610e-14 | 1.111e-15 | 5.370e-11 |
| Fixed | Newton | 6.596e-13 | Reference | 2.221e-15 | Reference |
| Fixed | Warpack + cuDSS | 4.124e-13 | 4.441e-14 | 1.333e-14 | 3.642e-14 |
| Fixed | Pure Warp | 5.229e-11 | 2.276e-14 | 6.662e-16 | 2.052e-11 |

All 42 measured solves pass, plus all 12 GPU warmups. Every GPU solve reports
status zero and all 12 modes converged. Pure Warp takes 21 outer iterations
free-free and 19 fixed-interface; cuDSS uses the declared two-cycle budget.
Newton performs one successful inertia check per trial, without warnings.
Six analytic rigid modes are excluded in the free case, with overlap below
`5e-15` across all backends. Requested mode 12 is separated from mode 13.

Small regression tests compare against independent dense LAPACK and check
both graph replay and rejection of a wrong requested spectrum whose
eigenpairs have tiny residuals. See [regression.log](regression.log).
The new tests and related block/finalizer/mass-projection regressions pass
**35/35** in the pinned environment. Stock Warp 1.15 passes **5**, with the
two FSAI-dependent cases skipped ([stock log](stock-regression.log)). An
[independent assembly audit](assembly_audit.json) against Newton's procedural
fixture agrees to `3.834e-16` relative maximum stiffness-entry error and has
identical mass entries.

Graph DOT audits find no host callbacks. Pure-Warp graphs have no HtoD
transfers. cuDSS graphs contain the existing vendor metadata HtoD nodes;
these remain tracked by [#29](https://github.com/alecjacobson/warpack/issues/29).
Optional production completeness certification from
[#133](https://github.com/alecjacobson/warpack/issues/133) is documented in the
[API guide](../../docs/completeness.md).

Raw JSON, logs and occupancy records are stored alongside this report. The
JSON records matrix/source hashes, exact native settings, individual
timings and acceptance checks. Large generated matrices and reference
vectors stay in ignored `build/` and are reproducible from the driver.
