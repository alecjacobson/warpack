# Legacy block profiler conditional-capture failure (#185)

The legacy `compare_block.profile_step` diagnostic on numerical source `d5f5b1e` fails after an accepted uninstrumented solve. The original CLI already rejected active-search `--profile`; the first reproduction below calls the helper directly. A separate static reproduction confirms a failure on the path the CLI permits. The [isolated reproducer](reproducer.py.txt) uses the existing side-3 consistent-mass fixture, width 52, six modes and active search. The [failure log](initial-failure.log) records `Conditional body graph contains an unsupported operation (event record)` inside the active-search branch. No solver arithmetic is changed by the reproducer.

The legacy profiler wraps shared operator calls globally, then captures `solver.step()` while incorrectly assuming no conditional bodies. Those wrappers are reached inside active-search children; installed whitening/QR conditional helpers add another nesting hazard. Its module-level Gram wrapper also misses installed instance dispatch. The ordinary solve succeeds before this instrumentation failure. Source and exact control are preserved here; the fix must cover static and active search, nested helpers, actual Gram/paired-rotation dispatch and wrapper restoration.

The [static reproducer](static-reproducer.py.txt) extracts the exact old profiler function AST from `d5f5b1e` and applies it to a public width-52 solver without active search. Its accepted full solve is followed by the same unsupported-event error inside first-whitening fallback ([log](static-initial-failure.log)). Thus the bug is not limited to bypassing the old CLI guard.

The fixed helper uses one conditional-safe instrumentation implementation for static and selected active branches, wrapping instance Gram calls and the actual paired Ritz-rotation kernel. For active search it selects outside timing before each continued diagnostic step and captures the chosen branch directly. Inner conditional helpers suspend nested event recording. All wrappers, including the temporary tiled-launch interceptor, are restored in `finally`. These diagnostic graphs omit outer conditional-dispatch overhead and are not ordinary whole-solve timings.

Fifteen focused tests pass ([log](fixed-tests.log)); stock Warp 1.15 passes eight, with seven existing FSAI-required tests skipped ([log](stock-tests.log)). Tests check complete original equations/spectrum/subspaces/native counts before and after, positive Gram/QR/paired-rotation spans, static/active scopes and restoration after refused capture. The initial test-fixture attempt asked an active-only construction helper for a static solver and was correctly rejected in three tests ([retained log](initial-test-fixture.log)); static regressions now construct independent public diagonal pencils directly. No numerical tolerance was relaxed.

The first active CLI validation attempt ([log](cli-a1.log)) is separately retained: it exited at the existing argument guard before any solve. Enabling that previously unsupported CLI combination needs the guarded route and a fresh successful CLI validation; it is not reported as a numerical failure.

## Wide profile with paired rotation attributed

The [fresh profile](200-a2.json) passes complete uninstrumented solves before/after, source/dependency checks and clean occupancy. Every selected-step timing is finite. The numerical core remains `d5f5b1e`; no solver arithmetic changed.

| Checkpoint | Selected step ms | Ritz QR ms | Paired Ritz rotation ms | Ritz eigensolve ms |
|---:|---:|---:|---:|---:|
| 1 | 349.431 | 2.965 | 18.127 | 74.174 |
| 20 | 79.941 | 2.560 | 8.532 | 20.092 |
| 40 | 41.802 | 9.468 | 5.795 | 8.290 |
| 50 | 29.135 | 2.561 | 5.580 | 5.708 |

Nested spans overlap and must not be added. These two-sample checkpoint medians are diagnostics, not complete-workflow speed claims. QR may still take its full fallback, as its variable checkpoint-40 span illustrates. The paired rotation costs about 5.6 ms at checkpoint 50 and 18.1 ms at checkpoint 1; it is a concrete remaining kernel target alongside projected eigensolves. All first-profile omissions and the failed reproductions remain archived.

## CLI completion

The active-search CLI combination is now enabled through the tested selected-branch route, while still requiring image reuse. [The fresh CLI run](cli-a2.json) passes its first captured solve, three measured solves, all five continued diagnostic steps and final post-profile physical/native checks. Sources/dependencies and occupancy are verified. The original full graph has no HOST or host/device-copy nodes. The earlier argument-guard rejection stays archived separately.

All seventeen profile/CLI regressions pass ([final log](final-tests.log)). Stock Warp previously passed eight numerical profiling tests with seven FSAI skips; the two added CLI argument-contract checks also pass separately ([log](cli-guard-tests.log)). No core numerical code changed. This completes #185; the remaining paired-rotation kernel work is tracked in #186.
