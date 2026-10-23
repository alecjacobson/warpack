# Current 200-mode profile and Gram instrumentation fix (#182)

The corrected profile passes full uninstrumented solves before and after instrumentation, all eight selected-step replays, source/dependency identity and clean occupancy. These are diagnostic CUDA-event spans around a selected branch after a fixed prefix; host selection is outside the production solver. **Nested spans overlap and must not be added.** Use [complete controls](VALIDATION.md) for actual adaptive solve timings.

| Checkpoint | Active rows | Selected step ms | CG ms | Gram ms | X orthogonalization ms | Search orthogonalization ms | Ritz eigensolve ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 204 | 364.081 | 56.500 | 37.627 | 17.839 | 65.595 | 74.344 |
| 20 | 60 | 91.192 | 4.623 | 12.847 | 13.894 | 18.367 | 20.032 |
| 40 | 9 | 45.678 | 1.645 | 7.538 | 13.892 | 4.569 | 7.709 |
| 50 | 5 | 41.630 | 1.231 | 6.714 | 13.900 | 3.178 | 6.804 |

The Gram span falls from roughly 82.7 to 37.6 ms at the first checkpoint and 26.9 to 12.8 ms at checkpoint 20 relative to the [preceding profile](../register_triangular/PROFILE.md). Remaining costs include the full-width Ritz-basis QR even late in convergence: at checkpoint 50 its 13.9 ms span includes 5.41 ms triangular applications and 2.21 ms Cholesky. The early projected eigensolve also remains substantial. This identifies further investigation targets, not a new accepted optimization.

The initial profile [200-a1.json](profile/200-a1.json) omitted every Gram span because it wrapped `warpack.block.gram`, which the specialized dispatch bypasses. That profile is retained as **incomplete instrumentation**, with its [exact profiler source](profile/incomplete-instrumentation.py.txt). Its physical solve gates passed, but it cannot support Gram-cost conclusions. The complete uninstrumented timings were unaffected.

The fix wraps the branch `gram` method, covering both original and specialized dispatch. Existing suspension of external event nodes inside conditional whitening bodies remains. Six focused profile tests pass ([log](validation/profile-tests.log)), requiring positive Gram spans on original and specialized paths and restoring wrappers. The corrected [200-a2.json](profile/200-a2.json), adjacent log and occupancy record retain all measurements. No numerical core, tolerance or iteration policy changes in this profiling fix.
