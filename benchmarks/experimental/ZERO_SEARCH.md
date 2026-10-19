# Rejected zero-search-row optimization

An optional prototype skipped Gram/rotation work for rows already set to zero
by rank-revealing metric whitening. It used the existing device rank, preserved
both orthogonalization passes and all fresh physical checks, and introduced no
new truncation threshold. It was tested together with active CG packing.

The numerical checks passed: **80 focused tests in 47.26 s**, including partial
tiles, zero/full/changing ranks, ordinary/coupled mass, invalid constraints,
repeated modes and captured mode crossings. All three Anvil trials also pass
shared physical, requested-space, orthogonality, status/count and native CuPy
stopping gates.

However, the Anvil median is **3.3388 s**, effectively unchanged from the
preceding active-CG **3.3398 s**. CuPy is **3.1839 s** in this run. Warp takes
28–29 outer steps and ends with 29 retained search directions out of 48. This
does not establish a meaningful end-to-end gain or justify another core API
option. The code was removed; the accepted active-CG implementation is unchanged.

[Three-trial report](../../results/performance_followup/anvil-prefix-active.json),
[single-trial/profile screen](../../results/performance_followup/anvil-prefix-active-screen.json),
[validation and source hashes](../../results/performance_followup/zero-search-validation.json).
The instrumented late-step profile is 71.60 ms, continued from the final solve;
its overlapping categories are not additive or a controlled isolated speedup.
The new rotation kernel is included in `whiten`, not the standalone rotation
category. Adjacent occupancy files record no observed competing GPU jobs.

The complete [prototype patch](../../results/performance_followup/zero-search-prototype.patch)
applies to commit `fbac5b3` in an isolated checkout. It restores the
`--skip-zero-search` option, kernel and integration tests. Use the command in
[ACTIVE_CG.md](ACTIVE_CG.md) with that flag for reproduction. The production
package and supported API do not include this rejected option.
