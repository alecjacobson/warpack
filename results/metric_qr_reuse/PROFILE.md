# Profile after metric-checked Ritz reuse

The complete uninstrumented solves before and after profiling pass the original physical/native gates. All eight selected-step samples contain finite nonnegative timings, source/dependency checks pass, and occupancy is clean. The numerical source is `d5f5b1e`. This is an instrumented stage diagnostic, not a replacement for complete-workflow timings.

| Completed iterations | Active rows | Selected step ms | CG ms | Ritz QR ms | Search orthogonalization ms | Ritz eigensolve ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 204 | 359.112 | 56.687 | 3.424 | 72.296 | 74.125 |
| 20 | 60 | 79.852 | 4.636 | 2.556 | 18.374 | 20.016 |
| 40 | 9 | 34.891 | 1.641 | 2.557 | 4.567 | 8.292 |
| 50 | 5 | 30.253 | 1.233 | 2.565 | 3.168 | 6.792 |

At checkpoint 50, Ritz normalization takes 2.565 ms, versus 13.900 ms in the [prior profile](../gram_tiles/PROFILE.md). Its fresh computed-metric check is included. The selected step is 30.253 ms versus 41.630 ms previously. Complete fresh workflow controls independently show the 200-mode warm solve at 6.4222 s versus 6.9630 s; stage timings alone are not used to claim that gain.

Nested spans overlap and must not be added. Conditional helpers are timed at their boundary, suppressing event nodes inside child graphs. Consequently the generic Gram/mass child totals exclude work inside the new QR helper and are not directly comparable to previous aggregate Gram/mass totals. The paired Ritz rotation also bypasses the module-level rotation wrapper and is not yet separately attributed; the profile is not an exhaustive decomposition of the selected-step total. That instrumentation gap is part of follow-up #185.

[Raw profile](profile/200-a1.json), [occupancy](profile/200-a1.occupancy.json), [worker log](profile/200-a1.log).
