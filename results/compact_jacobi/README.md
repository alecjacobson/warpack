# Compact projected Jacobi graph experiment (#164)

This benchmark-only experiment keeps the installed FP64 rotation order and bounded convergence test, but records groups of rounds inside a device loop. It explores graph construction cost versus warm replay overhead. Installed dispatch is unchanged.

## Component controls

Each row uses the same indefinite symmetric matrix for all backends, one warmup and three measured replays. All full-spectrum, original-equation and orthogonality checks pass, with no host nodes or non-device transfers in the DOT graph audit. External occupancy is clean. Setup/module loading, eager graph preparation, outer capture, first captured replay and warmed replay are recorded separately. These are cached-compilation, fixed-order controls; they are not a cold-install comparison.

| Dimension | Outer capture ms, baseline / compact16 | Replay median ms, baseline / compact16 | Kernel nodes, baseline / compact16 |
|---:|---:|---:|---:|
| 48 | 3.429 / 2.044 | 1.644 / 1.787 | 99 / 70 |
| 104 | 5.835 / 1.622 | 4.156 / 4.489 | 211 / 54 |
| 208 | 11.259 / 1.945 | 12.338 / 12.181 | 419 / 70 |
| 312 | 16.743 / 1.635 | 21.967 / 22.956 | 627 / 54 |
| 416 | 22.308 / 2.394 | 37.526 / 39.781 | 835 / 70 |
| 612 | 37.228 / 1.531 | 82.386 / 85.202 | 1227 / 46 |

[Initial two-round control](component-a1.json) adds approximately15–40% replay overhead. The [grouped controls](component-chunks-a1.json) retain that variant alongside8/16/32-round loops; no rejected or slower variant is removed from the evidence. Compact16 generally trades a smaller replay penalty for substantially smaller capture cost. Component timings alone do not justify changing the default solver.

## Complete controls

All six complete workers pass unchanged accuracy gates and external occupancy checks. Each pair has identical source/dependency manifests and exact matrix/reference/start identities. Three measured solves per worker, plus one discarded warmup.

| Problem | Setup s, default / compact16 | Capture s, default / compact16 | Replay median s, default / compact16 | Compute cost through first validated result s |
|---|---:|---:|---:|---:|
| [anvil](full/anvil-compact16-a1.json) | 0.7765 / 0.7741 | 0.8225 / 0.7994 | 1.6129 / 1.6169 | 4.8815 / 4.8721 |
| [100](full/100-compact16-a1.json) | 0.4181 / 0.4382 | 3.9151 / 2.5450 | 2.9739 / 3.0861 | 10.4917 / 9.0130 |
| [200](full/200-compact16-a1.json) | 0.6345 / 0.6259 | 18.0919 / 12.6345 | 10.6451 / 10.7765 | 40.4520 / 34.6692 |

The complete driver validates every measured replay but does not independently check its discarded warmup. The last column therefore includes setup + capture + warmup + the first measured solve, and excludes host validation, mesh assembly and reference loading. It is an accounting of measured compute phases, not total process wall time or proof that the warmup was the first accurate solve. Setup/capture are single measurements per fresh worker with cached compilation.

The wide fixtures reduce this accounted cost by about14%, while median warmed replay increases3.8% (100 modes) and1.2% (200 modes). Anvil is essentially unchanged. This supports an opt-in first-use tradeoff, not a replacement for the replay-focused default or a CuPy comparison. The production integration will be qualified separately, guarded to wide projected dimensions so Anvil keeps its existing path.

## Validation

Expanded tests exercise changed graph replay, strided storage padding, random/clustered and indefinite matrices, extreme scales, nonfinite bounded termination and recovery, and exact installed matrix/eigenvector/sweep outputs. Grouped16/32-round bodies cover the same changed-input cases as the initial2-round version. Latest expanded results: **74passed pinned and74passed stock**. Earlier40-case runs also passed.
