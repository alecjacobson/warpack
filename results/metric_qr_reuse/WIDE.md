# Wide-block metric-checked Ritz reuse (#183)

All **64 first/later complete outputs pass** the unchanged native/physical gates. These eight pairs use candidate threshold 1e-13 only at widths 96–256, followed by the original full QR control. The 200-mode pair reverses the pilot order. This remains a benchmark prototype; the installed solver is unchanged.

| Problem | Original setup / capture / first-use s | Candidate setup / capture / first-use s | Original warm s | Candidate warm s | Change |
|---|---:|---:|---:|---:|---:|
| 200-reverse | 0.6421 / 3.8240 / 11.9040 | 0.6208 / 3.8352 / 11.2499 | 6.9488 | 6.3992 | -7.91% |
| 100 | 0.4337 / 1.4415 / 4.2003 | 0.4367 / 1.4184 / 4.0274 | 2.2058 | 2.0452 | -7.28% |
| 200-compact | 0.6572 / 1.5868 / 9.3129 | 0.6538 / 1.6156 / 8.8195 | 7.0344 | 6.4494 | -8.32% |
| 100-history | 0.4339 / 1.6471 / 4.5099 | 0.4345 / 1.6470 / 4.5028 | 2.2743 | 2.2393 | -1.54% |
| anvil | 0.7963 / 0.6185 / 3.0365 | 0.8206 / 0.6144 / 3.0063 | 1.5630 | 1.6125 | +3.17% |
| free | 0.4746 / 1.5431 / 4.2287 | 0.4513 / 1.5612 / 4.1249 | 2.0748 | 1.9624 | -5.42% |
| disconnected | 0.5165 / 1.8464 / 6.3302 | 0.5219 / 1.8523 / 6.1131 | 3.7938 | 3.5782 | -5.68% |
| heterogeneous | 0.4461 / 1.4206 / 4.4245 | 0.4263 / 1.4031 / 4.2215 | 2.4346 | 2.2780 | -6.43% |

First-use compute includes setup + capture/eager preparation + the first captured solve. Warm medians use three validated replays per worker. Each worker constructs its own solver/preconditioner; verified CPU references alone are cached outside the backend timers. Setup/capture are single observations with cached compilation, not stability distributions. All input/source/dependency/reference identities match within pairs, and all occupancy records are clean. Graphs contain no HOST or host/device-copy nodes. See [every sample and quality maximum](wide/qualified.json), with raw worker/occupancy/log files adjacent.

The retained-history case gains less and changes outer iteration counts. Those samples are included without selection. Anvil has no candidate control arrays and its complete graph summary is exactly equal to the original control; the measured difference there is run variability, not an algorithmic improvement. Free/disconnected cases retain the independently checked rigid nullspaces.

Twenty-four focused tests pass on pinned Warp and stock Warp 1.15 ([pinned](wide-pinned-tests.log), [stock](wide-stock-tests.log)). Initial scope-boundary test fixtures incorrectly supplied n < 3*width and were correctly rejected by the existing constructor; their [pinned](wide-initial-test-fixture.log) and [stock](wide-initial-stock-fixture.log) failures are retained. Only the test dimensions were corrected.

Decision: proceed to a guarded installed implementation and independent installed/full-suite/wheel qualification. Keep the 1e-13 computed-metric bound; retain complete original QR for unsupported configurations and strict tolerances. No native stopping, rank or physical acceptance threshold is relaxed. The bound covers the computed Gram defect, not exact-arithmetic mass products.
