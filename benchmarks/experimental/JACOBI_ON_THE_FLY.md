# Rejected projected-Jacobi launch fusion

This experiment follows the [100-mode profile](../../results/fem_100_modes/PROFILE.md)
under [#147](https://github.com/alecjacobson/warpack/issues/147). It combines
angle computation with the two-sided Jacobi update in one Warp kernel per
round. A scalar variant recomputes angles per output; tiled variants share row
and column angles within 16×16 or 32×32 output tiles. The existing 30-sweep limit,
2e-15 off-diagonal stopping criterion, pair order and original-equation checks
are unchanged. No production dispatch was modified.

Fresh L40 measurements on 2026-10-23, three warm batches of five complete
captured diagonalizations, median milliseconds:

| Dimension | Production default | Scalar fused | 16×16 tiled | 32×32 tiled |
|---|---:|---:|---:|---:|
| 32 | 0.846 | 2.270 | 1.587 | 2.068 |
| 48 | 2.476 | 3.991 | 2.825 | 3.776 |
| 64 | 3.580 | 5.731 | 4.101 | 5.591 |
| 72 | 4.095 | 6.629 | 4.666 | 6.346 |
| 104 | 6.156 | 10.017 | 6.950 | 9.407 |
| 128 | 7.684 | 12.683 | 8.663 | 11.683 |
| 208 | 17.400 | 39.335 | 18.158 | 22.312 |
| 312 | 35.516 | 94.443 | 34.868 | 37.715 |

All 32 rows pass complete spectrum, original equation and orthogonality checks
at 1e-11 after warmup and every measured batch. Input is restored before every
solve; setup/capture/reference work is excluded. Occupancy is uncontended.
Dimension 32 uses the existing shared-memory production solver; the larger Ada
dimensions use the existing global-memory path. The prototype is slower almost
everywhere, with only a small 312-dimensional difference, so it is not promoted.
Reducing launches alone does not compensate for repeated angle arithmetic.

[Raw timings and validation](../../results/projected_on_the_fly/isolated-a1.json)
· [occupancy](../../results/projected_on_the_fly/isolated-a1.occupancy.json).
These isolated dense results are not an end-to-end eigensolver speed claim.
33 additional tests compare independent plane rotations (including partial
tiles), complete repeated/indefinite spectra, orthogonality and changed-input
graph replay. The prototype stays outside the installed library.

```bash
python -m benchmarks.monitor_gpu --out jacobi.occupancy.json --log jacobi.log -- \
  python -m benchmarks.experimental.jacobi_on_the_fly_compare --out jacobi.json
python -m pytest -q tests/test_jacobi_on_the_fly.py
```
