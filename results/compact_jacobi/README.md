# Compact projected Jacobi graph experiment (#164)

The initial benchmark experiment keeps the installed FP64 rotation order and bounded convergence test, but records groups of rounds inside a device loop. It explores graph construction cost versus warm replay overhead. The production integration below adds an explicit option; the default remains unchanged.

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

## Production integration

`BlockLOBPCG(..., compact_projected_graphs=True)` selects 16-round compact graphs for existing even global projected dimensions 96–1024, including active-search branches. Default is False. Small/odd projected problems retain their existing dispatch; Anvil has no eligible dimensions. Only graph organization changes: FP64 rotation order, bounded sweeps, rank decisions, preconditioning and outer stopping are unchanged.

Fresh public-API controls use the independently validated first-output policy from #167. All six workers pass first-output plus three subsequent replay checks (**24 complete solves**), with clean external occupancy and exact paired inputs, sources and dependencies. Graph DOT audits contain no HOST nodes and only device-to-device copies.

| Problem | Setup s, default / enabled | Capture s, default / enabled | First-solve compute s, default / enabled | Warm replay median s, default / enabled |
|---|---:|---:|---:|---:|
| [anvil](production/anvil-enabled-a1.json) | 0.7850 / 0.8086 | 0.8183 / 0.8183 | 3.2727 / 3.2507 | 1.6050 / 1.6134 |
| [100](production/100-enabled-a1.json) | 0.3765 / 0.4413 | 3.8931 / 2.4968 | 7.4548 / 6.1213 | 3.0242 / 3.0277 |
| [200](production/200-enabled-a1.json) | 0.6175 / 0.6084 | 18.2689 / 12.6694 | 29.9574 / 24.1223 | 10.5005 / 10.5873 |

First-solve compute is setup + capture (including eager preparation) + the first captured solve. Its original equations, spectrum/subspace and native count/stopping are checked immediately afterward, outside timing. It excludes mesh assembly, reference construction/loading, input fingerprinting, host validation and initial imports, and uses cached compilation where available. It is **not total process wall time**. Setup/capture are single observations per fresh worker.

This reduces first-solve compute by **17.9% for 100 modes and 19.5% for 200 modes**. Warm replay is roughly unchanged in this set (+0.1%/+0.8%), while the preceding prototype controls show +3.8%/+1.2%; retain the explicit tradeoff rather than infer universal replay parity. Anvil executes the same graph (2,002 kernel nodes in both controls). Wide graph kernel nodes fall 34,471→7,450 and 123,646→14,290. No CuPy ratio or replacement of the README six-backend table is claimed.

Validation: **122 passed / 1 skip** in the affected pinned suite; **91 passed / 40 upstream-feature skips** on stock Warp (also including the eight benchmark acceptance tests). An additional 1024-dimensional original-equation/full-spectrum/orthogonality case passes on each runtime. The focused tests cover exact installed round outputs, extreme scales, changed replay, strided padding, bounded nonfinite termination, public default/small/odd routing, coupled mass, constraints and repeated modes. The preceding full-suite baseline is separately recorded; it was not rerun for this optional path.

First-output acceptance tests reject a failed first result even when later replays pass, and leave legacy reports uncertified for first use instead of inventing historical warmup evidence. Test logs and wheel evidence are retained alongside this report. The isolated wheel contains 38 core modules matching source byte-for-byte, excludes experiments/examples/tests, and passes changed-input replay while forbidding benchmark, example, SciPy, CuPy and optional cuDSS imports.

Review: each inner iteration performs a fixed even group of disjoint rotations before checking the device counter. Only one thread updates that counter after the second out-of-place rotation; other threads do not read it in that kernel. Tail rounds retain the original buffer order. Every sweep resets counter/condition state, and every solver/branch owns its counters. Setup may load modules and synchronize as before; replay introduces no host condition reads or allocation. Default dispatch returns before constructing any compact state.
