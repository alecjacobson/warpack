# Certified second metric whitening (#162)

This benchmark-only subclass preserves the first canonical rank-revealing
whitening and both orthogonalization passes. In the second whitening, it
computes a fresh symmetric search Gram matrix and checks a conservative
certificate entirely on device. Production dispatch is unchanged.

For the previously active rows, let δ bound the maximum absolute row sum of
`G_active − I`. Require δ ≤ 0.25 and `1 − δ > rank_tol * (1 + δ)`. Gershgorin
bounds then put every active eigenvalue in `[1 − δ, 1 + δ]`, above the existing
relative rank threshold. Every inactive Gram row/column must be exactly zero;
all entries and flags must be valid. Thus this branch changes no retained rank.
It applies Cholesky QR on the active subspace, filling inactive diagonal entries
with one only in separate factorization scratch. Inactive vectors remain zero.

An unsuccessful certificate uses the existing canonical eigensolve. A separate
factorization status also selects this fallback if Cholesky fails, preserving
the original Gram matrix and the solver's prior status. Three device graph
conditions select factorization, successful application and fallback. Eager
setup retains the canonical implementation. All numerical work remains Warp;
extra factor/matrix/work arrays are allocated for each search branch at setup.

Initial nine focused tests pass: changing ranks, zero/partially inactive search,
nearly dependent directions, incorrect inactive flags, scaled rows, stringent
rank thresholds, nonfinite recovery, deliberately failed factorization and
complete clamped-consistent/free/heterogeneous FEM replays. Extended changed
coupled-mass, normalized-constraint and repeated-cluster checks and stock-runtime
validation are queued. Complete default/candidate 100-mode, Anvil and 200-mode
controls are running; no performance improvement is claimed yet.
