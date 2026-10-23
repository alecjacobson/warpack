# Reversed-order and broader FEM controls: projection subtraction (#192)

Every first output and all three measured outputs pass unchanged physical and native convergence gates. Source hashes, exact inputs, dependency identity, independently cached reference identity and external occupancy are verified. Captured graphs have no HOST callbacks or host/device copies. All controls own their factors/preconditioners; only independently verified CPU references are cached outside timing.

Median warm seconds:

| Case | original | r8 | 8-row change vs original |
|---|---:|---:|---:|
| 200 | 6.1139 | 5.7424 | -6.08% |
| 100 | 1.9860 | 1.8715 | -5.76% |
| anvil | 1.5448 | 1.4823 | -4.05% |
| 200-compact | 6.1875 | 5.7914 | -6.40% |
| 100-history | 2.1432 | 2.0164 | -5.92% |
| free | 1.8981 | 1.7842 | -6.00% |
| disconnected | 3.4342 | 3.1733 | -7.60% |
| heterogeneous | 2.2133 | 2.1120 | -4.58% |

64 checked outputs across 16 fresh workers. Setup, capture, first-compute times, every warm sample and iteration count are retained in [controls/qualified.json](controls/qualified.json). Setup observations here are individual worker measurements, not stable setup distributions. Four-row generated controls and slower sixteen-row variants remain in the pilot.

The broader controls run the eight-row candidate before the original, reversing the pilot order, and include compact projected graphs, retained history, free-body modes, disconnected components and heterogeneous elasticity. Atomic reduction and convergence-boundary variation can change iteration counts; timing differences include those changes. No installed dispatch has changed. Promotion still requires a guarded implementation, full regression and exact installed-wheel qualification.
