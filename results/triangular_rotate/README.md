# Retaining both X-orthogonalization passes (#173)

The first experiment computes the small inverse triangular factor once per
pass, then applies it to X and cached AX with the existing dense tiled rotation.
Both full CholeskyQR passes, initial scaling, image transformations and final
mass application remain. It reuses existing gx/bx scratch; no dense buffers are
added. Production dispatch is unchanged.

**This generic dense-rotation version loses:** about 50% slower for the isolated
stage, so it is not promoted or sent to expensive whole-solve comparisons.
The inverse is triangular; generic dense rotation does unnecessary arithmetic
on its zero upper triangle. Exploiting that structure is a separate possible
follow-up, not a result established by these measurements.

L40, pinned Warp/FSAI, four Xeon 8362 cores, 2026-10-23; n=34,560, synthetic
coupled tridiagonal SPD mass. Default block-product settings, not the tuned FEM
recipe. Each captured stage restores the same device X/AX inputs and performs
the complete two-pass orthogonalization. Five warm replays after a separately
checked first replay, cached kernels, fixed baseline-then-candidate order;
continuous occupancy clean.

| Width | Input | Sequential triangular stage ms | Inverse + dense rotation stage ms |
|---|---|---:|---:|
|104|Random basis|6.5389|9.8340|
|104|Already mass orthonormal|6.5373|9.8451|
|204|Random basis|26.1577|39.2410|
|204|Already mass orthonormal|26.1744|39.2313|

All **48 first/measured stage outputs pass** status, fresh M-orthogonality and
independent AX/MX checks. Maximum Frobenius orthogonality error is below 6.1e-15.
Graph audits show no host nodes or non-device-to-device transfers. This is a
component result, not an eigensolve or FEM speed claim; allocation, compilation,
capture and host validation are outside stage timing.

Eight pinned tests additionally cover widths 8/52/104/204, with and without
cached images, changed mass, physical row scaling by 1e±60, correlated inputs,
and a mass matrix equilibrated with diagonal factors spanning 1e±6. Both
methods retain the complete physical subspace and cached-image consistency.
[Transcript](validation/tests.log); adjacent reports retain exact source/input
identities, commands, samples and occupancy. The candidate remains outside core.
