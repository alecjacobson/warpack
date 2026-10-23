# Global-workspace projected Cholesky prototype (#155)

The [qualified production factor](../parallel_cholesky/README.md) uses shared
memory for CUDA widths 48–128 when capacity permits. Larger blocks retain the
serial fallback. This experiment investigates a cooperative factor that stores
its lower triangle in a transposed device workspace, so capacity no longer
scales with shared memory. **Production dispatch is unchanged.**

The prototype computes independent column entries in parallel with the serial
kernel's subtraction order, pivot-status flag and 1e-30 diagonal containment.
A transposed workspace coalesces accesses across rows. It costs `8 * width²`
additional device bytes, allocated during solver construction and shared with
active-search branches. The final factor retains the existing layout for
triangular application. Replay allocates nothing and makes no host numerical
decision. The implementation is entirely Warp.

The implementation uses two shared-tile collectives per column: one broadcasts
the diagonal and the second completes all global workspace writes before the
next dependent column. Inspection of the tested Warp 1.15 and pinned 1.18
`warp/native/tile.h` confirms `tile_scatter_masked` ends in `WP_TILE_SYNC`, which
maps to `__syncthreads` on CUDA. That barrier orders earlier global and shared
accesses within a block, as specified by the
[CUDA synchronization documentation](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#synchronization-functions).
Both collectives are reached by every thread, including inactive rows. This
implementation dependency must be rechecked when changing the Warp dependency.

## Correctness evidence

The pinned overlay passes 27 tests: dimensions 1, 7, 48, 104, 128, 204, 256 and
512; scaled SPD matrices; strided input/output/workspace; three changed-input
replays; independent reconstruction and SciPy/serial factors; early/late invalid
pivots, sticky status and recovery; and complete coupled/free/heterogeneous FEM
replays with the active branches sharing setup workspace. Stock Warp 1.15
passes 24 direct tests and skips three upstream-FSAI-dependent FEM cases.

These tests do not establish a speedup. Isolated serial/packed/global factors
and then full larger-mode FEM controls must qualify before any production
change. All failed or contended measurements will be retained.

The first [isolated run](isolated/factor-a1.json) passes every factor equation
and graph-residency check through width 512, but its
[occupancy record](isolated/factor-a1.occupancy.json) shows contention. Its times
are excluded from performance conclusions; uncontended qualification is pending.

The subsequent [uncontended isolated control](isolated/factor-a2.json) passes
all reconstruction/status and device-only graph checks. Medians below are
microseconds per factor. Each backend has three measured captured batches;
sizes below 256 use ten factors per batch, and sizes 256/512 use one long factor
per batch. Actual counts are recorded per row. Setup and capture are excluded.

| Width | Serial (µs) | Packed shared (µs) | Global workspace (µs) |
| ---: | ---: | ---: | ---: |
| 24 | 163.136 | 35.526 | 25.008 |
| 48 | 878.683 | 79.754 | 60.018 |
| 104 | 6708.715 | 235.126 | 197.733 |
| 128 | 11849.260 | 326.024 | 285.381 |
| 204 | 60386.988 | Not supported | 1107.102 |
| 256 | 125150.505 | Not supported | 1905.691 |
| 512 | 1031782.117 | Not supported | 9139.322 |

At width 204, the global prototype reduces the isolated factor time by about
55×. That is not a whole-eigensolve result. The small absolute differences
against the packed implementation at widths 104/128 do not by themselves
justify replacing the qualified existing path. Larger-mode full-solver controls
remain necessary. The additional workspace is `8 * width²` bytes, reaching
2 MiB at width 512. [External occupancy](isolated/factor-a2.occupancy.json)
reports no competing GPU process throughout this worker.

## Drivers

- `benchmarks.experimental.global_cholesky_compare`: isolated factor controls,
  reconstruction/status checks, graph audit and explicit working-memory bytes.
- `benchmarks.experimental.compare_global_cholesky`: unchanged block comparison
  with `--cholesky-backend serial`, `global` or `default`, plus `--public-api`.
  `default` keeps the current production packed/serial dispatch.
- `tests/test_global_cholesky.py`: direct and complete FEM correctness tests.

Use the pinned dependency overlay and serial GPU occupancy monitor described in
[development](../../docs/development.md). Tests alone support no performance
claim for L40 or other hardware.
