# Complete Gram tile prototype controls

All **88 complete first/later outputs pass** in 11 paired comparisons. These qualify the independent prototype at commit `5f1d649`, before installed dispatch is introduced. Each pair has identical input, source, dependency and verified CPU-reference identities; external GPU occupancy is clean. Every first solve and all three measured replays pass original equations, requested spectrum/subspace, mass orthogonality, expected nullities and native stopping. Graphs contain no HOST or host/device-copy nodes.

L40 / Xeon 8362, four pinned CPU cores, checksum-pinned Warp1.18 FSAI/CG overlay. The only numerical change is the Gram output tile: installed8×8 versus prototype16×16, both128threads and reduction chunk4096. The candidate uses a rolled reduction loop to avoid compiler shared-memory expansion at short chunks. Both full QR passes, rank thresholds, installed whitening/register-triangular kernels, FSAI/CG settings and physical gates are unchanged. Only verified CPU references are reused outside timers; each worker constructs its own solver and preconditioner.

Each cell is **installed / prototype**, in seconds. Setup excludes capture; **first-use compute = setup + capture/eager preparation + first captured solve**. Warm compute is the median of three later replays. Setup/capture are single observations with cached compilation, not established setup speedups. Mesh assembly, imports, references and host validation are outside these timings.

| Case | Setup | Capture | First-use compute | Warm compute | Warm reduction |
|---|---:|---:|---:|---:|---:|
| whole/200 | 0.6578 / 0.6670 | 3.9508 / 3.8563 | 13.2127 / 11.9690 | 8.1113 / 6.9298 | 14.57% |
| whole/100 | 0.4347 / 0.4343 | 1.4330 / 1.4261 | 4.4484 / 4.1792 | 2.4396 / 2.2093 | 9.44% |
| whole/anvil | 0.7991 / 0.7987 | 0.6099 / 0.6151 | 3.1125 / 2.9727 | 1.6300 / 1.5624 | 4.14% |
| families/free | 0.4478 / 0.4464 | 1.5599 / 1.5388 | 4.4828 / 4.2036 | 2.3398 / 2.0759 | 11.28% |
| families/disconnected | 0.5119 / 0.5564 | 1.9173 / 1.8650 | 6.9958 / 6.3326 | 4.3920 / 3.7927 | 13.64% |
| families/heterogeneous | 0.4296 / 0.4364 | 1.4255 / 1.4098 | 4.6517 / 4.3960 | 2.7033 / 2.4364 | 9.88% |
| families/200-reverse | 0.6321 / 0.6546 | 3.9029 / 3.9224 | 13.1793 / 12.0065 | 8.1077 / 6.9887 | 13.80% |
| families/200-compact | 0.6357 / 0.6596 | 1.6772 / 1.5971 | 10.5693 / 9.3991 | 8.2136 / 7.0250 | 14.47% |
| midwidth/48 | 0.2846 / 0.2835 | 0.6853 / 0.7267 | 1.8712 / 1.8809 | 0.8695 / 0.8394 | 3.46% |
| midwidth/60 | 0.3055 / 0.3127 | 0.8779 / 0.8639 | 2.2484 / 2.1775 | 1.0207 / 0.9539 | 6.55% |
| midwidth/anvil-reverse | 0.7812 / 0.7809 | 0.6109 / 0.6168 | 3.0257 / 2.9516 | 1.6189 / 1.5451 | 4.56% |

The first three pairs run installed then prototype. All family and midwidth pairs run prototype first, including explicit reverse-order200 and Anvil controls.200-mode savings repeat at13.8–14.6%; Anvil at4.1–4.6%. Small one-iteration fluctuations remain in some cases; the100-mode,48/60-mode, free and disconnected controls have identical iteration counts. Every individual time and count is retained. There is no claim that this closes the remaining CuPy gap (#147).

Raw reports/logs/occupancy are in [whole](whole/), [families](families/) and [midwidth](midwidth/), with paired worst-quality/phase summaries in their `qualified.json` files. No complete comparison was rejected or discarded. The failed initial kernel and all slower component variants remain in the [component report](README.md). The upcoming installed dispatch still requires separate source, regression, package and complete-workflow validation.
