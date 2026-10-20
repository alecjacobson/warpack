# Cooperative complex projected Schur solve

Experimental checkpoint for [#14](https://github.com/alecjacobson/warpack/issues/14).
The prototype uses pure-Warp tiled Householder reduction, Givens QR and right
back substitution. Shifts, deflation tolerance and bounded QR iteration policy
follow the serial reference. It preallocates all workspaces and supports nested
device-conditional capture. It contains no vendor dense eigensolver or host
convergence test. Sizes 1–64 are supported by this prototype.

Twenty tests pass on pinned Warp 1.18 (2.72 s) and stock Warp 1.15 (22.44 s,
including compilation). Tests cover scaled real/complex matrices, changed
replay, identity/zero/repeated/defective spectra, bounded QR failure, nonfinite
recovery, and saved 50-by-50 projections inside an outer conditional loop.
Independent SciPy references and residual/Schur-similarity/unitarity checks are
outside timed solves. Captured graphs have no host nodes or host transfers.

| Projection | Serial median ms | Tiled median ms |
|---|---:|---:|
| Saved direct, n50 | 199.707 | 17.990 |
| Saved shifted, n50 | 199.635 | 17.990 |
| Random complex n6 | 0.633 | 0.539 |
| Random complex n16 | 8.474 | 2.515 |
| Random complex n32 | 60.773 | 8.495 |
| Random complex n50 | 222.827 | 21.755 |
| Random complex n64 | 458.646 | 31.850 |

These are **component timings**, not whole sparse solves. Every case passes
independent spectrum, backward-error, normalization and Schur checks. Each
method uses the same input, one warmup, three timed batches of three replays,
resetting its projection each replay. Timing excludes setup/capture/reference
work. GPU process monitoring observed no competing work. Numerical gates and
raw samples are in [results](../../results/schur-components.json) and
[occupancy](../../results/schur-components.occupancy.json). General integration,
full strict-count checks and end-to-end timings are still required.

```bash
python -m pytest -q tests/test_tiled_schur.py
python -m benchmarks.experimental.compare_schur --out build/schur-components.json
```
