# Fresh Anvil comparison — 2026-10-23

| Solver | Setup (s) | Warm solve (s) | Max original residual, all 26 modes |
|---|---:|---:|---:|
| SciPy ARPACK + MKL PARDISO | 7.0288 | 12.0792 | 4.040e-09 |
| Spectra + MKL PARDISO | 7.0222 | 6.9091 | 9.068e-09 |
| CuPy eigsh + cuDSS | 2.1396 | 0.4589 | 2.856e-09 |
| Warp + cuDSS | 2.3825 | 0.4655 | 2.693e-09 |
| CuPy LOBPCG + Warp FSAI/CG | 0.9617 | 2.3365 | 1.728e-09 |
| **Warp block + FSAI/CG** | **1.3984** | **1.5432** | **1.943e-08** |

All six rows were refreshed on numerical source `8a8c769`. Each has three independently constructed setups in fresh processes, each followed by one outer warmup and five measured solves from identical starts. Setup is the median of three constructions; solve is the median of all 15 timed samples; residual is the worst over those 15. Factors/preconditioners and solver state are never borrowed between rows or repetitions. All **90 timed solves pass** and all 18 external occupancy records are clean. No attempt was discarded or repeated. The three first captured Warp-block outputs also pass their independent physical/native checks.

## Timing stability

| Solver | Setup range s | Solve range s | Spread of three batch medians |
|---|---:|---:|---:|
| SciPy ARPACK + MKL PARDISO | 6.9904–7.1362 | 11.7323–12.3902 | 1.99% |
| Spectra + MKL PARDISO | 6.9622–7.0361 | 6.6929–7.3029 | 1.56% |
| CuPy eigsh + cuDSS | 2.1250–2.1975 | 0.4584–0.4593 | 0.09% |
| Warp + cuDSS | 2.3604–2.3834 | 0.4650–0.4661 | 0.06% |
| CuPy LOBPCG + Warp FSAI/CG | 0.9410–1.0204 | 2.3091–2.5069 | 0.32% |
| Warp block + FSAI/CG | 1.3972–1.3993 | 1.5338–1.5934 | 0.40% |

Spread is `(max(batch medians) - min(batch medians)) / median(batch medians)`. All solve batch spreads are below 2%; setup variability is shown separately. Adaptive block iterations can vary between individual samples. No fastest-run selection or outlier removal is used.

## Scope and accuracy

Setup starts with assembled matrices available and includes factor/preconditioner construction, backend-specific workspaces/operator/start upload and graph capture where applicable. CPU iteration workspaces are allocated inside native solve calls and charged there. Assembly, mass normalization, imports/runtime initialization, reference loading, host-start generation, outer warmup and final validation are excluded. Existing compilation caches are populated; encountered module loads are included. This measures solver preparation and repeated solves, not cold process launch.

The common problem has 283,857 DOFs and six rigid plus 20 elastic Anvil modes. Unlike older sweeps that reassembled between processes, every worker here verifies the exact same stored stiffness/mass/normalized-matrix/rigid arrays, mesh/material/assembly identity and reference. The [shared-input protocol](../anvil_inputs_142/README.md) defines these fingerprints. Factors and workspaces are independently constructed and fully charged to each row.

Direct methods use shift 1, k=26, ncv=64, tolerance 1e-12 and the same saved start. CPU references use MKL PARDISO with refinement off; Spectra iteration is native C++, without Python matvec callbacks. Block methods use shift 100, FSAI32 and FP32 CG capped at 25 steps with FP64 outer work. Warp uses width 24, normalized tolerance 1e-8, cached operator images, search/CG buckets of four and full history. Its installed Gram optimization is enabled by chunk 4096. CuPy uses width 20, absolute tolerance 1e-6 and separately allocated active-width Warp CG graphs. Those inverse/preconditioner adapters are supplied by this benchmark; they are not built-in CuPy backends.

Every timed solve passes original-equation residual (<1e-7), orthogonality (<1e-8) and elastic-mode matching (MAC>0.99999) gates. Block rows additionally check eigenvalue error and normalized returned-pair residual (<1e-8). Warp requires status zero/full requested count; CuPy LOBPCG must meet its native absolute tolerance over its full returned block. Block validation appends six analytic rigid modes for common all 26 checks. The source/dependency and exact input hashes match across all repetitions and the common physical inputs match across methods.

## Environment and reproduction

NVIDIA L40, driver 570.158.01; Xeon Platinum 8362; affinity CPUs 8–11; four OpenBLAS/OpenMP/MKL threads; MKL 2020.4.304; SciPy 1.15.0; CuPy 14.2.0; pinned Warp 1.18 FSAI/batched-CG overlay. [Environment/source/runtime identities](environment.json), [exact summary](summary.json) and every adjacent log/occupancy file are retained. Use the pinned overlay and [CPU adapter prerequisites](../../benchmarks/ANVIL_FACTORIZATION.md#reproduction), and the existing verified inputs/reference/start. Do not regenerate the reference inside timing.

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export WARPACK_ANVIL_INPUTS=/path/to/verified-anvil-inputs.npz
# Set PYTHONPATH to the pinned Warp overlay and checkout, and configure MKL.
taskset -c 8-11 python -m benchmarks.anvil_showcase --out NEW_DIRECTORY --rounds 3 --repeats 5
```

Run with an idle GPU and preserve any failed or contaminated attempts. The reported sweep serialized each worker under the shared GPU lock and checked external occupancy continuously. CPU affinity in the summary is generated with the same 8–11 affinity as all worker commands. All numerical code remained unchanged throughout the sweep.

## Raw reports

- SciPy ARPACK + MKL PARDISO: [run 1](arpack-1.json), [run 2](arpack-2.json), [run 3](arpack-3.json).
- Spectra + MKL PARDISO: [run 1](spectra-1.json), [run 2](spectra-2.json), [run 3](spectra-3.json).
- CuPy eigsh + cuDSS: [run 1](cupy-direct-1.json), [run 2](cupy-direct-2.json), [run 3](cupy-direct-3.json).
- Warp + cuDSS: [run 1](warp-direct-1.json), [run 2](warp-direct-2.json), [run 3](warp-direct-3.json).
- CuPy LOBPCG + Warp FSAI/CG: [run 1](cupy-block-1.json), [run 2](cupy-block-2.json), [run 3](cupy-block-3.json).
- Warp block + FSAI/CG: [run 1](warp-block-1.json), [run 2](warp-block-2.json), [run 3](warp-block-3.json).
