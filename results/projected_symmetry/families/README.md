# Broader FEM qualification after projected symmetry recovery

These default-policy workers supplement the paired production controls with
large repeated/null-mode and heterogeneous-material cases. Each solves for 100
positive modes; free/disconnected cases also include their rigid nullspaces in
the returned and natively converged count. The history policy remains retained
for this regression. All numerical work in the eigensolve/preconditioner is Warp.

| Case | DOFs | Returned pairs | Known nullity | Block width | Warm median s | Worst original residual, first + replays |
|---|---:|---:|---:|---:|---:|---:|
| free | 35,328 | 106 | 6 | 112 | 2.9956 | 7.023e-12 |
| disconnected | 70,656 | 112 | 12 | 120 | 5.3649 | 8.177e-12 |
| heterogeneous | 34,560 | 100 | 0 | 104 | 3.3924 | 7.573e-12 |

**All 12 complete outputs pass**: the first captured solve and three subsequent
replays per case. Native stopping/count/status, original equations, spectrum,
mass orthogonality, known nullspace and requested invariant-subspace checks are
enforced. Source identities match the installed production change; upstream
dependency manifests verify and all external occupancy records are clean.
Graph DOT inspection finds device-to-device transfers and no host nodes.

These are additional current accuracy/performance observations, not a paired
speed comparison against older source or CuPy. Raw reports retain complete
arguments, input/reference/start identities, all phases and samples. Setup
and first-compute scope follows the [production controls](../production/README.md).
Together with those 32 paired-control outputs, this qualifies 44 complete
first/later solves after the fix.
