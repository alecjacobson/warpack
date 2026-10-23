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

## Existing triangular kernel: thread grouping

A second screen changes only the original kernel's block size. Twelve tests
verify bit-identical triangular outputs at widths24/104/204 for32/64/128/512
threads versus the256-thread baseline, including strided storage, scale1e±80,
zero/changed replay and independent triangular equations.

All120 first/measured full-stage outputs pass original M-orthogonality/AX/MX
checks; occupancy is clean. Same fixture/protocol as above. Warm stage medians
in milliseconds:

| Width/input | 256 threads | 32 | 64 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
|104/random|6.535|6.443|6.511|6.496|7.006|
|104/orthonormal|6.538|6.443|6.517|6.499|7.011|
|204/random|26.074|26.031|26.075|26.103|26.991|
|204/orthonormal|26.106|26.018|26.094|26.123|27.023|

The widest problem shows no substantive gain. The small104-width improvement
is about0.09ms for this stage, not a demonstrated whole-solve improvement.
512threads loses. This does not justify a production policy or expensive
whole-FEM tuning runs. [Raw controls](block-size-a1.json) and
[exact-output tests](validation/block-size-tests.log) retain the experiment.

## Skipping the inverse's zero upper triangle

The next variant skips coefficient tiles strictly above the inverse's lower
triangle. It preserves both orthogonalization passes and reuses the same scratch.
Twenty-one tests pass, including odd/partial tiles, strides/sentinels, exact
agreement with the dense product, coupled/scaled/changed mass and cached images.
All72 first/measured stage outputs pass; continuous occupancy is clean.

| Width/input | Original triangular ms | Dense inverse application ms | Lower-triangular inverse application ms |
|---|---:|---:|---:|
|104/random|6.539|9.838|7.614|
|104/orthonormal|6.534|9.837|7.617|
|204/random|26.095|39.717|29.905|
|204/orthonormal|26.064|39.733|29.907|

Avoiding zero work helps but still leaves the full stage roughly15–17% slower
than the original solve. This version is also **not promoted**; no whole-FEM
speedup is claimed or assumed. [Raw controls](lower-a1.json) and
[test transcript](validation/lower-tests.log) preserve the result. The initial
generic variant now calls a small overrideable application method; its algorithm
is unchanged, and the original committed source is retained in32af963.
