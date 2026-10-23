# Metric-checked Ritz-basis reuse pilot (#183)

All **32 first/later complete outputs pass** across the predeclared two problems and four policies. The candidate remains benchmark-only. It checks fresh MX and G=X^T MX on device, reuses X/cached AX only when the computed metric is sufficiently close to identity, and otherwise runs the unchanged complete two-pass QR. The computed-product bound does not certify exact-arithmetic mass products; original physical gates remain essential.

| Problem | Defect threshold (0 = full QR) | Setup s | Capture s | First-use compute s | Warm median s | Warm change |
|---|---:|---:|---:|---:|---:|---:|
| 200 | 0e+00 | 0.6252 | 3.8055 | 11.7975 | 6.9558 | +0.00% |
| 200 | 1e-14 | 0.6236 | 3.8243 | 11.6297 | 6.7115 | -3.51% |
| 200 | 1e-13 | 0.6377 | 3.8712 | 11.4218 | 6.4079 | -7.88% |
| 200 | 1e-12 | 0.6284 | 3.8356 | 11.2661 | 6.3353 | -8.92% |
| anvil | 0e+00 | 0.8170 | 0.6139 | 2.9900 | 1.5584 | +0.00% |
| anvil | 1e-14 | 0.7942 | 0.6215 | 2.9891 | 1.5762 | +1.14% |
| anvil | 1e-13 | 0.7998 | 0.6139 | 2.9503 | 1.5827 | +1.56% |
| anvil | 1e-12 | 0.7957 | 0.6161 | 3.0041 | 1.5359 | -1.45% |

First-use compute is setup + capture/eager preparation + first captured solve. Each warm median uses three validated replays; setup/capture are single cached-compilation observations. Every worker constructs its own solver and preconditioner; only verified CPU references are cached outside backend timers. All input/source/dependency/reference identities match within each problem and occupancy is clean. Graph audits contain no HOST or host/device-copy nodes.

The 200-mode 1e-13 profile saves about 7.9%, reusing the basis on 53–54 of 59 QR calls in the measured replays. The stricter 1e-14 threshold saves 3.5%; 1e-12 saves 8.9%. Every threshold passed this screen, but a looser threshold is not automatically preferable. Anvil does not show a consistent gain: 1e-14 and 1e-13 are slower in the warm median, while 1e-12 is modestly faster. Iteration counts and all samples remain in [qualified.json](pilot/qualified.json).

**Next decision:** advance the 1e-13 profile only as a wide-block candidate, retain the full-QR path for Anvil, and require reverse-order and broader FEM controls. Promotion requires broader accuracy and complete-workflow qualification. All thresholds and slower results remain archived.

Nineteen focused tests pass on both pinned Warp and stock Warp 1.15 ([pinned](pinned-tests.log), [stock](stock-tests.log)). They cover computed row/column defect bounds, nonfinite detection, exact reuse, fallback on changed/random/scaled mass metrics, zero/negative/nonfinite rejection followed by healthy recovery, cached AX and fresh MX, coupled/ill-conditioned mass, changed constraints/clusters, both history settings, and retention of original QR for custom/wrapped mass operators. Conditional fallback is restricted to owned Warp sparse/identity mass operators. Eager execution retains original QR. The prototype shares 24 bytes of diagnostic/control device values across mutually exclusive branches and adds no dense workspace.

Reproduce with `python -m benchmarks.experimental.compare_metric_qr --metric-defect 0|1e-14|1e-13|1e-12 --reference-cache CACHE --public-api ...`. Exact remaining arguments, first and per-replay counters, physical qualities, source hashes and adjacent occupancy/logs are in [pilot](pilot/). Counters are read during post-solve validation, outside timing.
