`spectra_generalized_shift_5.npz` contains the real A and B arrays exported by
`benchmarks/upstream/prepare_spectra.py` for `SymGEigsShift_5`, from Spectra
revision `db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95` (MPL-2.0).
The exporter and upstream reference use seed 123. This is generated numerical
data, not copied implementation source. The case has n=100, k=10, ncv=20 and
sigma=1.2345. Issue #7 records the original block-iteration convergence failure.
The regression independently computes the dense generalized spectrum and
compares original-equation residuals, B-orthogonality and the invariant subspace.

`spectra_null_start.npz` preserves A and the explicit initial vector from
`Example4_1` at the same Spectra revision and exporter. It is generated numerical
data for the nullspace-start regression in issue #8 (n=100, k=3, ncv=6, LA).
