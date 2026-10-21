# Fresh Anvil comparison — 2026-10-21

| Solver | Setup (s) | Warm solve (s) | Max original residual, all 26 modes |
|---|---:|---:|---:|
| SciPy ARPACK + MKL PARDISO | 7.0520 | 11.8392 | 3.976e-9 |
| Spectra + MKL PARDISO | 7.0642 | 6.9934 | 3.227e-9 |
| CuPy eigsh + cuDSS | 2.1151 | 0.4593 | 2.841e-9 |
| Warp + cuDSS | 2.3480 | 0.4663 | 2.863e-9 |
| CuPy LOBPCG + Warp FSAI/CG | 1.3652 | 2.3754 | 1.929e-9 |
| **Warp block + FSAI/CG** | **2.1296** | **1.8843** | **1.914e-8** |

All six rows were refreshed in this sweep. Each has three independent setup
constructions in fresh processes, each followed by one outer warmup and five
timed solves from the same initial data. The table reports median setup over
three samples, median solve over all 15 samples, and the worst original residual
across those 15 solves. No factors or CG state are reused between rows or setup
repetitions. Jobs run serially; all 18 occupancy records are eligible and show
no sampled unrelated GPU process.

## Timing stability

| Solver | Setup range (s) | Solve range, all 15 (s) | Spread of three batch medians |
|---|---:|---:|---:|
| SciPy ARPACK + MKL PARDISO | 7.0518–7.0537 | 11.4300–12.0248 | 1.91% |
| Spectra + MKL PARDISO | 7.0549–7.0898 | 6.7625–7.2726 | 1.67% |
| CuPy eigsh + cuDSS | 2.1151–2.1210 | 0.4586–0.4602 | 0.10% |
| Warp + cuDSS | 2.3478–2.3530 | 0.4658–0.4667 | 0.06% |
| CuPy LOBPCG + Warp FSAI/CG | 1.3556–1.4075 | 2.3281–2.5823 | 1.37% |
| Warp block + FSAI/CG | 2.1281–2.1486 | 1.8765–1.9516 | 0.32% |

Batch spread is `(max(batch medians) − min(batch medians)) / median(batch medians)`.
All six are below 2%; the two direct GPU rows are below 0.1%. The block solvers
have variable adaptive iteration counts, so their individual solve times vary
more than the direct inverse methods. All samples are retained, including the
slower ones; no fastest-run selection or outlier removal is used. The first
ARPACK run overlapped an unrelated GPU job and was rejected by occupancy
monitoring, then repeated; both rejected files remain in `rejected/`.

## Scope and correctness

Setup begins with assembled matrices available. It includes factor or
preconditioner construction, backend-specific workspaces/operator/start upload,
and graph capture where used. CPU iteration workspaces are allocated inside the
native solve calls and charged there. Matrix assembly and mass normalization,
imports/runtime initialization, reference loading, host random-start generation,
outer solve warmup and final validation are excluded. Existing compilation
caches are populated; encountered module loads are included. This measures
solver preparation and repeat solves, not a cold application launch.

The physical problem is unchanged: 283,857 DOFs, six rigid plus 20 elastic
Anvil modes. The direct methods use shift 1, k=26, ncv=64, tolerance 1e-12 and
the same saved starting vector. CPU rows use MKL PARDISO with refinement off;
Spectra's iteration is native C++ with no Python matvec callback. The block
methods use shift 100, FSAI32, FP32 CG capped at 25 steps and FP64 outer work.
Warp uses width 24, normalized tolerance 1e-8, cached operator images, search/CG
buckets of four and full search history. CuPy uses width 20, absolute tolerance
1e-6 and separately allocated active-width Warp CG graphs. The benchmark supplies
CuPy's inverse/preconditioner; these are not built-in CuPy backends.

Every timed solve passes the existing original-equation residual (<1e-7),
orthogonality (<1e-8) and elastic-mode matching (MAC >0.99999) gates against the
cached reference. Block rows additionally check returned eigenvalue error and
normalized returned-pair residual (<1e-8); Warp requires status zero and all
requested pairs converged, and CuPy LOBPCG must meet its native absolute
stopping tolerance across its full block. The block rows append six analytic
rigid modes for the common all-26 validation. GPU assembly can vary at roundoff
level between processes; CPU workers use the same freshly assembled cached
matrix. This refresh includes the [cuDSS capture-preparation fix for #132](../cudss_setup_132/README.md).
Its final direct-Warp measurements follow the last starting-vector preservation
guard; preliminary measurements remain in that issue report. Numerical iteration
and the other backend implementations are unchanged.

## Environment and reproduction

NVIDIA L40, driver 570.158.01; Intel Xeon Platinum 8362, affinity CPUs 8–11
(four distinct physical cores on the GPU-local NUMA node). Four OpenBLAS,
OpenMP and MKL threads. MKL 2020.4.304, SciPy 1.15.0, CuPy 14.2.0;
Warp 1.18.0+cu12 with the pinned upstream FSAI/batched-CG overlay.
Source, matrix, reference and runtime-library hashes are in
[environment.json](environment.json). No cold compilation is benchmarked.

Use the [pinned Warp overlay](../../docs/installation.md#fast-block-solves)
and [PARDISO/Spectra adapter build](../../benchmarks/ANVIL_FACTORIZATION.md#reproduction).
The cached reference and initial vector are needed for validation; do not
regenerate them with a slow factorization as part of this timing run. In this
environment the matching MKL runtime was restored under `build/mkl-runtime`;
an installed matching runtime may instead be on the system library path.

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export LD_LIBRARY_PATH="$PWD/build/mkl-runtime/usr/lib/x86_64-linux-gnu"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
mkdir -p build/anvil-factor-audit
taskset -c 8-11 python -m benchmarks.anvil_factor_audit --backend assemble \
  --out build/anvil-factor-audit/fresh-matrix.json
taskset -c 8-11 python -m benchmarks.anvil_showcase \
  --out build/anvil-showcase-fresh --rounds 3 --repeats 5
```

`--resume` skips completed, accepted measurements with eligible occupancy and
the requested sample count. `--summary-only` regenerates the aggregate without
running solvers. The [summary](summary.json) contains exact medians and ranges.

## Raw reports

Each link contains setup costs, all timed samples and numerical validation;
its adjacent `.occupancy.json` records the execution command and GPU sampling.

- SciPy ARPACK + MKL PARDISO: [run 1](arpack-1.json), [run 2](arpack-2.json), [run 3](arpack-3.json)
- Spectra + MKL PARDISO: [run 1](spectra-1.json), [run 2](spectra-2.json), [run 3](spectra-3.json)
- CuPy eigsh + cuDSS: [run 1](cupy-direct-1.json), [run 2](cupy-direct-2.json), [run 3](cupy-direct-3.json)
- Warp + cuDSS: [run 1](warp-direct-1.json), [run 2](warp-direct-2.json), [run 3](warp-direct-3.json)
- CuPy LOBPCG + Warp FSAI/CG: [run 1](cupy-block-1.json), [run 2](cupy-block-2.json), [run 3](cupy-block-3.json)
- Warp block + FSAI/CG: [run 1](warp-block-1.json), [run 2](warp-block-2.json), [run 3](warp-block-3.json)
