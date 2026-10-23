# Reversed-order and broader rotation controls (#186)

All 64 first/later outputs pass unchanged physical/requested-spectrum, mass-orthogonality, nullity and native status/count/stopping gates. Input/source/dependency/reference identities match within every pair; external occupancy records are clean; captured graphs contain no HOST or non-device-to-device copies. Each candidate runs before its fresh original control, reversing the pilot order. The library is still unchanged at this checkpoint.

L40, four pinned Xeon 8362 cores, pinned Warp 1.18 overlay. All workers own their factors/preconditioner/solver. Only independently verified CPU references are reused outside timing. Warm medians use three later replays. Setup/capture are single cached-compilation observations; first-use compute also includes the first captured solve. Assembly, imports, references and validation are excluded.

Each paired cell is **original / primary r8k8**, seconds. All individual samples and iteration counts remain in `controls-qualified.json`.

| Problem | Setup | Capture | First-use compute | Warm median | Change |
|---|---:|---:|---:|---:|---:|
| 200 | 0.6358 / 0.6365 | 3.8639 / 3.8806 | 11.3799 / 11.0888 | 6.4067 / 6.1163 | -4.53% |
| 100 | 0.4317 / 0.4387 | 1.4361 / 1.4979 | 4.0429 / 4.0536 | 2.0523 / 1.9889 | -3.09% |
| anvil | 0.7911 / 0.7893 | 0.6217 / 0.6847 | 2.9873 / 3.0164 | 1.6152 / 1.5891 | -1.61% |
| 200-compact | 0.6410 / 0.6552 | 1.6086 / 1.6181 | 8.7579 / 8.5336 | 6.4914 / 6.1735 | -4.90% |
| 100-history | 0.4347 / 0.4331 | 1.6527 / 1.7031 | 4.4459 / 4.4471 | 2.2442 / 2.1850 | -2.64% |
| free | 0.4464 / 0.4426 | 1.5471 / 1.6227 | 4.0911 / 4.0993 | 1.9621 / 1.9007 | -3.13% |
| disconnected | 0.5274 / 0.5174 | 1.8596 / 1.9093 | 6.1384 / 6.0088 | 3.5937 / 3.4279 | -4.61% |
| heterogeneous | 0.4370 / 0.4319 | 1.4329 / 1.4738 | 4.2704 / 4.2558 | 2.2739 / 2.2165 | -2.52% |

The primary improves these warm medians by 1.6–4.9%. Anvil improved 4.2% in the pilot and 1.6% here, with retained 16/17-step variation; neither result is discarded. Compact graphs, retained history, free-body nullspaces, disconnected components and heterogeneous FEM all pass. Combined with the pilot, 112 complete outputs have passed. These controls support installed qualification of the primary, not a claim that #147 is solved.

The production candidate is being prepared separately with a static Ada SM89 / n>=32768 / block-width24..256 guard. Other configurations retain the original kernel. It preserves FP64, inner width8, separate bounded X/U views, both outputs, and all accuracy policies, with no new scratch or host numerical decisions. Full-suite, clean-wheel and fresh installed-versus-original comparisons remain required before merge.

Raw evidence: [controls](controls/); [qualified summary](controls-qualified.json). The earlier [pilot](WHOLE.md) retains every tile alternative.
