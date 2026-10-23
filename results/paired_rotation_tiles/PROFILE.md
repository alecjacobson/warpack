# Active-step profile after paired rotation (#186, #147)

Complete uninstrumented solves before and after profiling pass original/native gates. Eight selected-step samples (two at each checkpoint) have finite nonnegative event times, matching source/dependency evidence and clean occupancy. The full graph has no host callbacks or host/device transfers. Numerical source is `0c5ecfc`.

Median milliseconds below are **nested diagnostics**: Gram/QR/whitening/stage spans overlap and must not be added. Branch selection is outside the timed selected-step graph; conditional helpers are timed at their boundaries. These are not complete solve timings or an isolated A/B stage experiment.

| Checkpoint | Selected step | Preconditioner | Ritz QR | Search projection (two passes) | Whitening (two passes) | Projected eigensolve | Paired rotation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 362.511 | 56.570 | 21.042 | 67.900 | 71.408 | 74.156 | 9.274 |
| 20 | 75.708 | 4.639 | 2.555 | 18.366 | 6.919 | 20.012 | 4.396 |
| 40 | 30.985 | 1.643 | 2.560 | 4.565 | 1.939 | 7.153 | 2.965 |
| 50 | 27.572 | 1.230 | 2.558 | 3.171 | 1.349 | 6.771 | 2.857 |

Paired rotation is about9.27ms at the first checkpoint and2.86ms at50, compared with18.13/5.58ms in the preceding profile. QR and projected-solve times also vary with the iterative state; for example QR is about21ms at checkpoint1 here and2.56ms later. Subtracting old/new parent spans would therefore misattribute the gain. Use the [whole-solver qualification](VALIDATION.md) for performance conclusions.

The largest early costs now include search projection, whitening and the projected eigensolve. The projection parent still includes both Gram formation and basis subtraction, so the subtract kernel needs separate attribution before selecting the next tile optimization. [Raw profile and graph audit](profile/200-a1.json).
