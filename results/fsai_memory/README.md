# FSAI setup memory and a lower-memory FEM profile

[Issue #131](https://github.com/alecjacobson/warpack/issues/131) tracks the
upstream setup footprint. The numerical implementation remains unmodified
upstream Warp FSAI/CG plus Warpack. These measurements use NVIDIA L40 48 GB,
driver 570.158.01, the pinned Warp 1.18.0+cu12/FSAI environment, and serial
process-occupancy monitoring. No competing processes or sampling errors were
observed.

## Isolated setup

Each fresh process constructs only a 96-row SPD tridiagonal and an upstream
FSAI factor. No eigensolver is created. Setup is FP64; stored factors are FP32.

| Maximum factor row size | Device free-memory decrease | Increase in live Warp pool | Build-kernel local bytes/thread |
|---|---:|---:|---:|
| [16](detailed16.json) | 360 MiB | 24,200 bytes | 2,536 |
| [32](detailed32.json) | 1,730 MiB | 42,760 bytes | 9,128 |
| [64](detailed64.json) | 13,844 MiB | 42,760 bytes | 67,368 |

Releasing the factor restores the Warp pool's live-byte counter to its initial
level, but does not restore the lost free device memory while the CUDA context
remains alive. Pool high-water stays below 223 KB. This is not explained by a
large retained factor or cached Warp-pool allocations. The width dependence and
compiler-reported local storage are consistent with CUDA backing for per-thread
local memory; the precise driver allocation policy/generated-code contribution
still needs investigation. These observations are not proof of a factor-buffer
leak, nor a guaranteed footprint on other GPUs.

The drop is **real device/context usage** during setup even when the input is
tiny. A pool-only workspace counter would miss it. The `compare_block` harness
now records device-free and Warp-pool counters before setup, after factor
construction, and after capture. Pool high-water covers the whole process;
it is not a separately reset setup peak.

Reproduce an isolated width in a fresh process:

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.monitor_gpu --out build/fsai64.occupancy.json \
  --log build/fsai64.log -- python -m benchmarks.experimental.fsai_memory \
  --width 64 --out build/fsai64.json
```

## Matched-accuracy FEM comparison

The public block path solves 20 lowest modes of 10,890-DOF clamped tetrahedral
pencils with consistent mass, width 24, and shifted matrix `K + .01 M`.
One fixture is homogeneous; the other has 20:1 Young's-modulus and 3:1 density
contrast. Only FSAI row size changes within each pair. Five repeats use the same
restored random starts; all physical spectrum/subspace/residual/orthogonality,
Warp count, and applicable CuPy native stopping gates pass.

| Material | FSAI row size | Setup s | Warp median ms | CuPy median ms | Setup/capture device-memory decrease |
|---|---:|---:|---:|---:|---:|
| Homogeneous | [64](fem_consistent_w64.json) | 0.8519 | 226.25 | not rerun | 13,974 MiB |
| Homogeneous | [32](fem_consistent_w32.json) | 0.2410 | 226.70 | not rerun | 1,860 MiB |
| Heterogeneous | [64](fem_heterogeneous_w64.json) | 0.8455 | 214.12 | 200.07 | 13,974 MiB |
| Heterogeneous | [32](fem_heterogeneous_w32.json) | 0.2360 | 207.24 | 210.56 | 1,860 MiB |

On these two fixtures, FSAI32 reduces the measured device-memory increase by
about **7.5×** and setup time by about **3.5×**, with similar or slightly faster
Warp replay. The heterogeneous FSAI32 comparison is near parity with CuPy;
the small timing difference is not evidence of a universal speed advantage.
CuPy is native LOBPCG supplied with the same Warp preconditioner, at absolute
`1e-10` over its entire width-24 block. Warp requests normalized `1e-11` on its
first 20 modes. Shared original-problem gates remain `1e-8`.

Setup includes preconditioner construction, solver/operator/CG allocation and
upload, including JIT encountered. Assembly/reference generation, host random
starts, capture, warmup and replay are separate. The memory difference includes
all device/context allocations; it is not peak memory or live solver workspace.
CuPy adapter allocation is excluded. Exact commands, source hashes, every trial
and graph/occupancy records accompany each JSON. The same device-only graph
checks pass in both configurations.

This qualifies FSAI32 as a useful lower-memory choice for these FEM fixtures;
it does not change global defaults or establish that width 32 is optimal for
all meshes. **The released Anvil recipe already uses FSAI32** and is unchanged.
The FSAI64 warm-start results remain valid, with their setup footprint explicit.
Upstream local-storage reduction and smaller-memory GPU qualification remain
open under #131.
