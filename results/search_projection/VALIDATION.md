# Projection promotion held: mass-QR stage regression

**Do not treat this candidate as qualified or promoted.** The full regression
returned 13 failures; [#198](https://github.com/alecjacobson/warpack/issues/198)
tracks the diagnosis. The installed-wheel smoke and whole-solver timing workers
were stopped before they started. #192 remains open, and the README performance
table has not been replaced with candidate results.

The proposed core change uses eight-row output tiles for FP64 projection
subtraction on CUDA Ada SM89, n>=32768 and block width24..256, when fused updates
are enabled. It preserves both projection passes, the inner width 8, output
columns 32, 128-thread launch, original stopping/rank policies and fallback kernel.
It adds no workspace, host numeric reads or numerical backend. The code review
and focused tests are complete; full and installed qualification are not.

| Check | Observation |
|---|---|
| Focused candidate checks | 101 passed, 72.20 s |
| Focused stock Warp 1.15 | 94 passed, 7 expected FSAI skips,66.21 s |
| Full pinned Warp suite | **13 failed, 3,455 passed, 11 skipped**,1220.97 s |
| Same four affected modules, uninstrumented candidate | **12 failed, 48 passed**,17.52 s |
| Same group, clean main fc5d678 | 60 passed, 13.78 s |
| Candidate with Python exception tracing | 60 passed, 17.52 s |
| Candidate with added failure context at existing assertion | 60 passed, 12.82 s |
| Three alternating fresh workers per version | All 360 results pass |
| Candidate after deliberate NaN allocator poisoning | All 96 stage fixtures pass |

Every failure is at the device-status check in the shared captured mass-QR stage
fixture, used by register/shared/inverse triangular and fresh Ritz-image tests.
It captures `orthogonalize_x()` directly at n=3*width+7, with unfused updates;
it does not call the new projection subtraction kernel. Failed membership varies
between runs. Passing later or instrumented controls do not establish the cause
or erase the failures. No accuracy gate has been relaxed. A relationship to the
known graph-upload issue #61 has not been established.

The complete initial logs/JUnit, exact test/benchmark/core source fingerprints,
failure-context scripts, passing controls and poisoned-allocation diagnostic are
retained in [validation-held/](validation-held/). The optional historical FSAI
fixture was present and checksum-verified before this full run. No omitted-fixture
supplement is needed to explain these 11 skips.

A clean test wheel was built and all 43 core modules match source byte for byte
(SHA256 303ffff0f409ef6e8d56c03ee8ffd3bb02abe7650b43c4b4d29d5432d802ac31).
That byte audit is not an installed replay result: the replay workers did not run.
It is not a published release. Earlier successful [component](SCREEN.md),
[whole-problem](WHOLE.md) and [reversed/broader](CONTROLS.md) prototype results
remain available as separate evidence; they do not override this promotion gate.
