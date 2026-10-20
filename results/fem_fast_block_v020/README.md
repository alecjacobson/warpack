# Fast public block solver: FEM qualification, 2026-10-20

[Protocol, commands and limitations](../../benchmarks/applications/FAST_BLOCK.md) · [Tracker #128](https://github.com/alecjacobson/warpack/issues/128)

GPU: NVIDIA L40 48 GB, driver 570.158.01. Pinned upstream Warp 1.18.0+cu12 / FSAI overlay, CuPy 14.2.0, NumPy 2.2.6 and SciPy 1.15.0. Installed public solver numerics are unchanged from v0.2.0; this increment changes benchmark coverage and error reporting. Each numerical source hash is recorded in its JSON.

All five tetrahedral families pass Warp original-equation, spectrum, mass orthogonality, complete requested-space/nullspace and stopping-count gates at both 6 and 20 requested elastic modes. Free cases include another six rigid modes, disconnected cases twelve. Bending passes at the practical `1e-8` Warp tolerance; the separate `1e-11` run is rejected at 4/6 counted pairs. The largest fixture here is 22,506 DOFs, below Anvil scale.

## Warm replay results

Milliseconds are medians of three restored-start runs. CuPy uses native LOBPCG and the same supplied Warp FSAI/CG preconditioner. Its complete starting block must meet its absolute stopping bound; Warp stops on the first k using normalized residuals. Both face the same independent physical gates. Different stopping rules and counts of padding modes remain part of this comparison.

| Fixture / requested elastic modes | DOFs | Returned k | Warp ms | CuPy ms | Worst Warp normalized residual | Comparison |
|---|---:|---:|---:|---:|---:|---|
| [clamped consistent / 20](clamped_consistent_s10_k20.json) | 10,890 | 20 | 226.57 | 423.70 | 6.517e-12 | accepted |
| [clamped consistent / 6](clamped_consistent_s10_k6.json) | 10,890 | 6 | 49.58 | 129.15 | 2.081e-12 | accepted |
| [clamped lumped / 20](clamped_lumped_s10_k20.json) | 10,890 | 20 | 230.91 | 481.26 | 9.339e-12 | accepted |
| [clamped lumped / 6](clamped_lumped_s10_k6.json) | 10,890 | 6 | 49.58 | 141.74 | 1.580e-12 | accepted |
| [disconnected / 20](disconnected_s10_k20.json) | 22,506 | 32 | 410.84 | rejected | 9.231e-12 | rejected reference |
| [disconnected / 6](disconnected_s10_k6.json) | 22,506 | 18 | 142.51 | rejected | 6.885e-12 | rejected reference |
| [free / 20](free_s10_k20.json) | 11,253 | 26 | 331.06 | 524.33 | 8.200e-12 | accepted |
| [free / 6](free_s10_k6.json) | 11,253 | 12 | 90.31 | 145.85 | 2.333e-12 | accepted |
| [heterogeneous / 20](heterogeneous_s10_k20.json) | 10,890 | 20 | 209.68 | 198.35 | 8.433e-12 | accepted |
| [heterogeneous / 6](heterogeneous_s10_k6.json) | 10,890 | 6 | 51.31 | 147.70 | 5.916e-12 | accepted |
| [repeated bending / 6](repeated_bending_s6_k6_practical.json) | 96 | 6 | 22.95 | rejected | 7.920e-09 | rejected reference |

Eight tetrahedral comparisons are accepted and uncontended. Warp is faster in seven; CuPy is about 6% faster on the 20-mode heterogeneous case. These results establish useful configurations, not a universal speed advantage or tuned optimal settings. Disconnected and bending comparisons do not establish speedups: their Warp times above are diagnostic, with rejected references. No reference failure has been replaced by a successful rerun or counted as a win.

## Setup and memory

Warp setup includes FSAI construction and solver/operator/CG workspaces, upload and FP32 copies, including JIT if encountered. Mesh assembly, independent reference generation, host random starts, capture and warmup are excluded. Each JSON records capture and warmup separately. CuPy adapter setup is additional to shared factors; it is not an independent end-to-end setup measurement.

| Fixture / elastic modes | Setup s | Capture s | Device free-memory decrease MiB |
|---|---:|---:|---:|
| clamped_consistent / 20 | 0.8556 | 0.5191 | 13974.0 |
| clamped_consistent / 6 | 0.8342 | 0.4058 | 13898.0 |
| clamped_lumped / 20 | 0.8577 | 0.5412 | 13974.0 |
| clamped_lumped / 6 | 0.8393 | 0.4652 | 13898.0 |
| disconnected / 20 | 1.4857 | 0.7428 | 14250.0 |
| disconnected / 6 | 0.9157 | 0.5369 | 14070.0 |
| free / 20 | 0.9155 | 0.6249 | 14050.0 |
| free / 6 | 0.8830 | 0.4464 | 13930.0 |
| heterogeneous / 20 | 0.8361 | 0.5049 | 13974.0 |
| heterogeneous / 6 | 0.8471 | 0.4337 | 13898.0 |
| repeated_bending / 6 | 0.1086 | 0.4785 | 13864.0 |

Memory is a device free-memory difference including allocator pools/runtime/JIT, excluding original fixture storage and the CuPy adapter. It is **not peak memory or live solver workspace**; the roughly 14 GiB tetrahedral deltas include real CUDA context usage. A [subsequent isolated study](../fsai_memory/README.md) attributes the dominant drop to upstream FSAI setup outside the Warp pool, consistent with its per-thread local storage. It is not merely cached factor storage; smaller-GPU qualification remains open.

## Failures and validation

- Disconnected CuPy: all requested physical pairs pass, but padding modes miss the declared `1e-10` bound in some trials at both mode counts. Full results remain in the JSONs; tracked under [#69](https://github.com/alecjacobson/warpack/issues/69).
- Bending CuPy: residual orthogonalization returns `None` and native LOBPCG raises `AttributeError`; both strict and practical reference profiles fail. Tracked under [#90](https://github.com/alecjacobson/warpack/issues/90).
- Strict bending Warp: see the [original log](repeated_bending_s6_k6.log) and [structured rerun](repeated_bending_s6_k6_strict_recorded.json). The shared physical gates pass but only 4/6 satisfy the requested `1e-11` stopping tolerance. [#130](https://github.com/alecjacobson/warpack/issues/130) remains open.
- The structured rerun verifies the harness fix for [#129](https://github.com/alecjacobson/warpack/issues/129): a reference exception now leaves completed trial evidence plus an explicit rejected error row, and the command exits unsuccessfully.

All recorded graph audits contain only device-to-device copies and no host nodes. This is DOT inspection, not a complete CUDA API trace. Occupancy records contain no observed competing GPU processes or monitoring errors; their `timing_eligible` flag remains false for rejected runs because the command failed.

Focused validation: **53 passed, 1 skipped**, including the twelve new two-replay FEM cases, release API regressions, benchmark acceptance/error handling, and independent FEM fixture checks. This is not a new full-library/upstream conformance run. Stock Warp: **5 passed, 12 skipped**, with the fast-path cases correctly skipped because upstream batched CG/FSAI is unavailable.

Pending: broader mesh scale/aspect ratio and contrast, projected rigid-mode variants, warm-start/repeated-assembly studies, GPU portability, peak memory, and robust settings for stricter ill-conditioned problems. The pinned upstream overlay remains required.
