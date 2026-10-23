# Whole-solver pilot: projection subtraction (#192)

Every first output and all three measured outputs pass unchanged physical and native convergence gates. Source hashes, exact inputs, dependency identity, independently cached reference identity and external occupancy are verified. Captured graphs have no HOST callbacks or host/device copies. All controls own their factors/preconditioners; only independently verified CPU references are cached outside timing.

Median warm seconds:

| Case | original | r4 | r8 | r16 | 8-row change vs original |
|---|---:|---:|---:|---:|---:|
| 200 | 6.1038 | 6.1431 | 5.7146 | 6.1446 | -6.38% |
| 100 | 1.9700 | 1.9828 | 1.8808 | 1.9904 | -4.53% |
| anvil | 1.5314 | 1.5909 | 1.4906 | 1.5437 | -2.66% |

48 checked outputs across 12 fresh workers. Setup, capture, first-compute times, every warm sample and iteration count are retained in [whole/qualified.json](whole/qualified.json). Setup observations here are individual worker measurements, not stable setup distributions. Four-row generated controls and slower sixteen-row variants remain in the pilot.

The broader controls run the eight-row candidate before the original, reversing the pilot order, and include compact projected graphs, retained history, free-body modes, disconnected components and heterogeneous elasticity. Atomic reduction and convergence-boundary variation can change iteration counts; timing differences include those changes. No installed dispatch has changed. Promotion still requires a guarded implementation, full regression and exact installed-wheel qualification.
