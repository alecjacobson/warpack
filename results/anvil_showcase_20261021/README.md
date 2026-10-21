# Fresh Anvil comparison — 2026-10-21

| Solver | Setup (s) | Warm solve (s) | Max original residual, all 26 modes |
|---|---:|---:|---:|
| SciPy ARPACK + MKL PARDISO | 7.1506 | 11.5887 | 3.995e-9 |
| Spectra + MKL PARDISO | 7.0687 | 7.0833 | 3.538e-9 |
| CuPy eigsh + cuDSS | 2.2091 | 0.4594 | 2.847e-9 |
| Warp + cuDSS | 2.8123 | 0.4660 | 2.767e-9 |
| CuPy LOBPCG + Warp FSAI/CG | 1.3895 | 2.3757 | 1.747e-9 |
| **Warp block + FSAI/CG** | **2.1506** | **1.9400** | **1.934e-8** |

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
| SciPy ARPACK + MKL PARDISO | 7.0465–7.3056 | 11.4017–11.9705 | 2.09% |
| Spectra + MKL PARDISO | 7.0354–7.1091 | 6.8888–7.4206 | 2.96% |
| CuPy eigsh + cuDSS | 2.1221–2.2228 | 0.4588–0.4598 | 0.08% |
| Warp + cuDSS | 2.8080–2.8844 | 0.4656–0.4664 | 0.07% |
| CuPy LOBPCG + Warp FSAI/CG | 1.3520–1.4034 | 2.3227–2.5867 | 3.83% |
| Warp block + FSAI/CG | 2.1362–2.2183 | 1.8718–1.9495 | 2.84% |

Batch spread is `(max(batch medians) − min(batch medians)) / median(batch medians)`.
All six are below 4%; the two direct GPU rows are below 0.1%. The block solvers
have variable adaptive iteration counts, so their individual solve times vary
more than the direct inverse methods. All samples are retained, including the
slower ones; no fastest-run selection or outlier removal is used.

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
matrix. This refresh changes measurements, not core solver algorithms.

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
