# Isolated projection subtraction profile (#192)

The numerical core remains unchanged. The profiler now measures `subtract_projection` separately, respecting conditional-child event suspension and restoring every wrapped callable. Complete uninstrumented 200-mode FEM solves before and after profiling pass the physical and native convergence gates. Source hashes, actual dependency identity, finite event samples, clean external GPU occupancy, and the full graph audit pass.

Pinned Warp: 17 profile regressions pass; stock Warp 1.15: 10 pass, 7 expected FSAI skips. Tests exercise static and active-search solves and verify that the new span is present and positive.

Mean milliseconds from two samples per checkpoint:

| Checkpoint | Selected step | Search projection, both passes | Subtraction kernels | Paired Ritz rotation |
|---|---:|---:|---:|---:|
| 1 | 355.120 | 69.166 | 45.729 | 9.272 |
| 20 | 75.747 | 18.388 | 11.787 | 4.396 |
| 40 | 32.136 | 4.579 | 2.406 | 2.965 |
| 50 | 27.518 | 3.182 | 1.637 | 2.856 |

The subtraction span totals all directly recorded subtract kernels in the selected step; conditional QR helpers remain timed at their boundaries. Enclosing stage times overlap these spans and must not be added. At the early and middle checkpoints, subtraction accounts for much of the two-pass search-projection cost. This supports testing wider output tiles without changing either projection pass, Gram products, QR, rank selection, or stopping gates. It does not establish a whole-solver speedup.

[Raw profile and full-graph audit](profile/200-a1.json). GPU: L40, driver 570.158.01, pinned Warp/FSAI environment, four CPU threads on cores 8–11.
