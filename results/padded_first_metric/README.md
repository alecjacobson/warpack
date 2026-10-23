# Exact-padding first-metric certificate (#171)

This experiment extends the [full-rank-only first-whitening certificate](../first_metric_certificate/README.md)
to normalized search metrics with **exact zero padding**. Production dispatch
is unchanged. Component and small complete-solve tests pass; large whole-solve
cost/accuracy qualification is still required. There is no speed claim here.

## What is certified

Require a finite symmetric normalized Gram G, each active diagonal in [0.5,2],
and exactly zero rows/columns wherever the diagonal is zero. Build a separate
auxiliary matrix Gplus = G + I on those zero directions. The original G remains
unchanged for fallback; the auxiliary entries are never physical basis vectors.

The existing conservative factor/inverse/congruence certificate proves
lambda_min(Gplus) > rank_tol * upper(Gplus), with upper(Gplus) bounding its
largest eigenvalue. Its congruence-error bound includes product, reduction and
roundoff margins. Since Gplus is the direct sum of the original active block
and an identity, this also certifies every active eigenvalue of G above the
unchanged relative rank threshold. The identity directions can make the test
more conservative, never justify retaining a smaller physical eigenvalue.

Additionally verify that the computed inverse factor preserves exact block
separation: inactive rows/columns are identity, with zero active/inactive
couplings. Rotate the normalized physical basis, keep the **original active
mask/count**, and leave padded output rows zero. The existing second projection
and whitening remain. Zero, dependent, uncertain or invalid cases use canonical
rank revelation on the original normalized Gram; failure in the attempted
factorization does not contaminate the outer status.

The wrapper shares the existing three dense scratch matrices with second
whitening. It also reuses the existing transform/eigenvector scratch. Factor
workspace becomes the first congruence-product output only after factor and
inverse formation, with ordered launches. No new dense allocation, host rank
decision, readback or vendor call is added to the tested block composition.

## Validation

L40, 2026-10-23, pinned Warp/FSAI and stock Warp:

- **72 pinned tests pass** across three runs:46 first/component tests,21 safety
  and coupled-mass tests, and5 additional conservative-bound tests.
- The same combined tests on stock Warp: **49 passed,23 expected dependency
  skips** for upstream FSAI/batched-CG features.
- Full-to-partial-to-zero-to-full captured replay, permuted padding, true active
  dependency and near-rank cutoffs, nonfinite/asymmetric/nonzero padding entries,
  extreme physical row scaling, strided inputs/scratch and sentinel preservation.
- Rotated active spectra spanning condition ranges1 through1e14; every accepted
  certificate checked against independent eigenvalues and extended-precision
  congruence error. Rejection is allowed; false rank certification is not.
- Forced factor failure matches the canonical output bit-for-bit, retaining
  original Gram/status. Explicit checks verify temporal scratch aliasing across
  active-search branches and only three shared dense buffers.
- Complete small clamped/free/disconnected/heterogeneous FEM solves, both history
  policies, expected rigid nullity, original equations and requested subspaces;
  coupled mass, constraints, clustered modes and changed operators also pass.
- Captured paths have no host nodes or non-device-to-device transfers.

[Transcripts and exact source identities](validation/) retain the evidence.
The full-rank-only wrapper was refactored into small overrideable preparation,
attempt and application methods; its existing tests remain included. Existing
historical timing reports continue to describe their original committed source,
not this new padded variant. The benchmark driver explicitly selects
`--first-whitening padded`; `certified` retains the earlier full-rank policy.
