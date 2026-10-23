# Installed paired Ritz-rotation tiles (#186)

All **64 fresh installed-wheel first/later outputs pass**. On the L40, warm medians improve 3.0–4.3% across the wide FEM controls (3.8% with compact graphs, 4.1% with retained history). Anvil improves only 0.9% in this final control; the earlier 1.6% and 4.2% observations remain retained as variability, not a stable large Anvil speedup. #147 remains open.

## Implementation and review

With `paired_rotation=True`, the installed specialization requires CUDA Ada SM89, n>=32768 and block width24..256. It uses 8×32 output tiles, the original inner width8 and 128 threads. Other configurations retain the original four-row kernel. All arithmetic remains FP64 in Warp. Separate bounded X/U views prevent partial inner tiles crossing their boundary; separate accumulators preserve Xnew=Cx*X+Cu*U and Pnew=Cu*U. No global workspace, host readback, new numerical backend or stopping/rank policy is introduced.

Review checked both outputs, zero/partial/strided views, unchanged input and padding, changed replay/scales, static architecture/size guards, copied search-branch configuration, original-kernel controls and profiler attribution. The promoted kernel body matches the qualified prototype AST after substituting its constants. The original kernel is retained. The benchmark-only prototype disables the installed specialization so it cannot silently test a different kernel. The installed comparison requires the paired path explicitly.

## Tests and isolated package

- Complete fresh-checkout run: **3,358 passed, 37 skipped in 1124.40s**. This checkout lacked the optional historical FSAI source fixture, causing26 additional skips.

- After copying the exact checksum-verified archived fixture, both affected modules pass **36 tests in6.38s**, covering all26 omitted cases plus10 already covered. Thus combined coverage is **3,384 distinct passes and11 remaining expected skips**, not3,394 passes or a rewritten full-run log. [Coverage accounting](validation/coverage.json), [initial full log](validation/full-tests.log), [supplement](validation/historical-tests.log) and its JUnit case identities retain the distinction.

- Focused installed/product/scope/changed-pencil/profiler/driver regressions: **111 pass in63.32s**. Stock Warp1.15: **104 pass,7 expected upstream-FSAI skips in51.53s**.

- The test wheel contains **43 byte-identical core modules**, SHA256 `5f50baac32474aceca1e8f421ddeafbc2897db9935cd03c67a065de351e9cc47`. Its isolated public, projected and wide checks pass, while rejecting benchmark/example/SciPy/CuPy/cuDSS imports. The new n=32768,width104,k20 smoke exercises the actual default rotation together with Gram/QR specializations on nine changed K/M replays with constraints, repeated modes, history and compact graphs. Its largest absolute residual is4.294e-10 and mass-orthogonality error3.168e-14; DOT checks require the new kernel and reject HOST/HtoD/DtoH nodes. This is a qualification wheel, not a newly published release.

- Repository Ruff lint and modified-file formatting pass. An initial comparison watcher used an unsupported Python3.10 launcher flag and exited before any worker/measurement; the [incident note](validation/initial-launcher-note.txt) preserves that tooling failure and its logging limitation. The corrected launcher asserts the loaded installed package path and all43 module hashes before timing.

## Complete installed controls

L40 / four pinned Xeon8362 cores / pinned Warp1.18 FSAI/batched-CG overlay. These workers load the audited wheel, not checkout core modules. Each owns its preconditioner and solver; only verified independent CPU references are cached outside timing. All pairwise input/source/dependency/reference/wheel identities match, and occupancy records are clean. Graphs contain no HOST or non-device-to-device copies. Every first and measured output passes original residual, requested spectrum/subspace, mass orthogonality, expected nullity and native status/count/stopping.

Each paired cell is **original / default**, seconds. Warm medians use three later replays. Setup/capture are single cached-compilation observations, not setup stability distributions. First-use compute includes setup, capture/eager preparation and the first captured solve. Assembly, imports, references and validation are outside these timers.

| Problem | Setup | Capture | First-use compute | Warm median | Change |
|---|---:|---:|---:|---:|---:|
| 200 | 0.6372 / 0.6289 | 3.9149 / 3.8884 | 11.4547 / 11.1115 | 6.3763 / 6.1036 | -4.28% |
| 100 | 0.4263 / 0.4368 | 1.4709 / 1.4919 | 4.0579 / 4.0352 | 2.0553 / 1.9882 | -3.26% |
| anvil | 0.7852 / 0.7870 | 0.6620 / 0.6591 | 2.9998 / 3.0377 | 1.5545 / 1.5412 | -0.86% |
| 200-compact | 0.6541 / 0.6445 | 1.6521 / 1.6229 | 8.8354 / 8.5038 | 6.4544 / 6.2067 | -3.84% |
| 100-history | 0.4323 / 0.4272 | 1.6849 / 1.6709 | 4.5848 / 4.4446 | 2.2490 / 2.1571 | -4.09% |
| free | 0.4625 / 0.4492 | 1.6289 / 1.6518 | 4.1873 / 4.1309 | 1.9562 / 1.8983 | -2.96% |
| disconnected | 0.5048 / 0.5293 | 1.9066 / 1.8784 | 6.1222 / 6.0047 | 3.5817 / 3.4462 | -3.78% |
| heterogeneous | 0.4319 / 0.4319 | 1.4607 / 1.4717 | 4.2859 / 4.2410 | 2.2820 / 2.2106 | -3.13% |

[Raw workers](production/) include actual installed-package identities and launcher hashes. [Qualified summary](validation/installed-qualified.json) retains every sample, iteration count and error maximum. The [pilot](WHOLE.md) and [reversed/broader controls](CONTROLS.md) retain all112 prototype outputs, including slower alternatives. No accuracy threshold or failed result was replaced.

Reproduce with `python -m benchmarks.experimental.compare_paired_rotation_whole --rotation original --reference-cache CACHE --public-api --paired-rotation ...`; use `--rotation default` for the installed policy. Exact arguments and monitored commands are in each raw worker/occupancy record.
