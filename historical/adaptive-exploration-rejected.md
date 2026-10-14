# Rejected early-exploration candidate, 2026-10-14

Related: [#31](https://github.com/alecjacobson/warpack/issues/31).
This experiment was rejected on accuracy and is **not production code**.
Its [patch](adaptive-exploration-rejected.patch) is retained for reproduction.
The baseline numerical implementation is `e0a4f58`; default fixed-budget
cuDSS trajectories were unaffected by the proposed adaptive change.

The candidate set a device flag once all k requested residuals passed while
the independent-chain guard was still unsatisfied. The next basis expansion
consumed that flag and started a fresh orthogonal direction immediately,
incrementing the existing recovery counter. This avoided waiting for natural
roundoff breakdown and introduced no host convergence checks.

It failed the existing captured cycle test at **k=3, ncv=6**:

| Requested largest eigenvalues | Returned |
|---|---|
| 1.95105652, 1.95105652, 2 | 1.80901699, 1.95105652, 2 |

The eigenvalue error is **0.14203952**: the same missing multiplicity that #1
fixed. The isolated checkout's suite reported 179 passed and one failed
(including an additional loose-tolerance repeated-eigenspace test that passed).
The failing test is already permanent on main:
`tests/test_multiplicity.py::test_spectra_cycle_requested_multiplicities[adaptive-3-6]`.
No performance result from this candidate is accepted; the queued benchmark
stage stopped after the accuracy failure.

The evidence rules out treating a freshly seeded direction as sufficient
exploration under this policy. A replacement needs to preserve and develop
that direction through restart, or establish another meaningful completeness
check. Merely disabling the guard or exempting this fixture is not a fix.

To reproduce, apply the saved patch in an isolated checkout of `e0a4f58`, use
the pinned Warp environment, and run the single test above. Do not apply it to
a production checkout. Current production behavior retains the validated
multiplicity fix; #31 remains open.

A separate unchanged-production diagnostic extends the tall correlated POD
fixture to 10,000 by 48 snapshots (`k=4`, `ncv=32`). Its three accurate,
uncontended graph replays have median 0.119218 s; see the
[raw record](../results/refinement_pod_large.json). This is a baseline only,
not a speedup from the rejected candidate or a comparison with another library.
