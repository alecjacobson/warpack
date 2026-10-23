# Fresh Ritz operator images (#179)

**Decision: retain the installed cached-image transformations.** Recomputing `A X` after the two complete mass-QR passes gives less than 1% in these wide sparse cases and slows Anvil in both orders. This does not resolve the wide CuPy gap (#147). The candidate remains benchmark-only; it can also be more expensive for implicit operators.

Measured on L40 / Xeon 8362 with the pinned Warp 1.18 FSAI/CG overlay and four CPU threads. Each worker constructs its own solver and preconditioner. Only verified CPU reference arrays are cached outside backend timers. Both backends retain installed register-triangular dispatch, certified whitening, two QR passes, rank handling, constraints and all original stopping/accuracy gates.

Each cell below is **transform / refresh**, in seconds. Setup excludes capture and first compute, which are separate columns. Later compute is the median of three fully checked replays. Allocation/module preparation is outside replay timing; compilation was cached.

| Case | Setup | Capture | First compute | Later compute | Later change |
|---|---:|---:|---:|---:|---:|
| pilot/100 | 0.4600 / 0.4325 | 1.4570 / 1.4263 | 4.5001 / 4.4108 | 2.4426 / 2.4371 | -0.23% |
| pilot/200 | 0.6467 / 0.6335 | 3.9681 / 3.9174 | 13.2409 / 13.0980 | 8.1351 / 8.0633 | -0.88% |
| pilot/disconnected | 0.5135 / 0.5563 | 1.9208 / 1.9152 | 7.0183 / 7.0455 | 4.4052 / 4.3748 | -0.69% |
| pilot/anvil | 0.8086 / 0.7940 | 0.6146 / 0.6179 | 3.0718 / 3.1204 | 1.6864 / 1.7161 | +1.76% |
| recheck/200 | 0.6407 / 0.6432 | 3.9534 / 3.9950 | 13.1861 / 13.1339 | 8.1190 / 8.0671 | -0.64% |
| recheck/anvil | 0.8016 / 0.8148 | 0.6279 / 0.6086 | 3.0643 / 3.0806 | 1.6338 / 1.6467 | +0.79% |

The pilot runs transform first; the recheck reverses that order. In the reverse-order Anvil pair, both have iteration counts 16, 16, 17, and refreshing still costs 0.79% more. The 200-mode gain repeats at only 0.64–0.88%. The 100-mode pilot and Anvil pilot also vary by one outer iteration; these small differences are not evidence for a general default change.

All **48 complete outputs** (12 first solves plus 36 timed replays) pass native stopping, original residuals, eigenvalue and requested-subspace checks, mass orthogonality and expected nullities. Each worker passes external GPU-occupancy checks. Paired input, dependency, source and reference-cache identities match. Graph audits contain no HOST node or host/device copy. Largest wide original normalized residual is below 1e-11; requested-subspace error is below 5.8e-11. Anvil retains its existing physical gates, with absolute residual below 2e-8. These are different residual definitions and are not combined into one accuracy number.

The candidate passes 12 focused tests on pinned Warp and 12 on stock Warp 1.15, including coupled/scaled/changed masses, cached images, changed constraints, repeated modes, both history settings and full two-pass QR. No installed source changed.

Raw reports, execution logs and occupancy samples are in [pilot](pilot/) and [recheck](recheck/); [qualified.json](qualified.json) retains phase costs, every measured time, iteration counts and worst checked quality. Reproduce with `python -m benchmarks.experimental.compare_ritz_image_refresh --ritz-images transform|refresh --reference-cache CACHE --public-api ...`; exact remaining arguments and source hashes are in each JSON. All completed results are retained.
