# Replayable device starts for partial SVD

`PartialSVD(..., initial_vector=start)` accepts a one-dimensional device FP64
array of length `min(matrix.shape)`. For tall/square matrices it supplies a
right-side start; for wide matrices it supplies a left-side start. Initialization
lifts it into the corresponding half of the symmetric augmented operator,
leaving the other half zero. The augmented formulation and original two-sided
SVD equations are unchanged; normal equations are not introduced.

```python
start = wp.ones(min(A.shape), dtype=wp.float64, device=A.device)
solver = PartialSVD(A, k=4, initial_vector=start)
graph = solver.capture(100)
wp.capture_launch(graph)

# Reuse a computed mode as a new start, entirely on device.
wp.copy(start, solver.v[0] if A.shape[0] >= A.shape[1] else solver.u[0])
wp.capture_launch(graph)
```

The source array is never used as scratch. Each replay reads its current values,
so the same captured graph supports changed starts and changed matrix values
at fixed topology. Tests include a previously unrequested singular mode becoming
the largest while reusing a computed old mode. Reference vectors are used only
for validation, not to seed this reuse test.

The input is scaled by its maximum absolute entry before the ordinary norm
calculation, avoiding overflow/underflow for finite starts tested from 1e-320
to 1e300. Zero/nonfinite runtime starts set a nonzero `status` and clear both
accepted-triplet counts. A finite safe work vector keeps intermediate arithmetic
well-defined; those outputs remain rejected. A later valid start recovers on
the same graph. Shape, dtype and device mismatches raise at construction.

`solver.status` aliases the existing core status. Require status zero and
`count_left == count_right == k`, and inspect the original left/right residuals.
All these arrays are device-resident; a caller reading `.numpy()` explicitly
synchronizes outside the solve. Optional starts add one augmented-size vector
buffer and two scalar validation arrays. The default path allocates none of
that state and launches no start-validation kernels.

## Correctness and upstream coverage

The 63 SVD/normal-SVD/replay tests pass on pinned Warp1.18 and stock Warp1.15.
The new tests cover both matrix orientations, repeated singular values, a
requested zero singular triplet, an exact null-space start, changed matrices,
computed-start reuse across a spectral crossing, extreme input scales,
invalid-start recovery and unchanged source buffers. Both physical equations,
both orthogonalities, requested values and strict counts are independently
checked. Captured start graphs have no HOST, HtoD or DtoH nodes.

All three original pinned CuPy `TestSvdsV0` cases now pass: repeatability and
input preservation, parity with SciPy, and the smaller-side length for wide
matrices. The adapter materializes the supplied boundary array before capture
and forwards it to this API. CuPy RNG/internal-generator parity remains
unsupported. The separate complete upstream and full local candidate runs are
recorded once finished; focused passes do not stand in for a full-suite result.

## Default-path performance control

Matched five-trial warm replay controls on the L40 use a 100,000-by-48 snapshot
matrix and its transpose, k4, ncv32 and a maximum of 30 adaptive cycles. Every
trial passes original paired equations, both complete requested subspaces,
orthogonality, requested spectrum, status and both full counts. Every solve
uses five outer cycles and 132 augmented products.

| Matrix | Existing default, ms | Candidate default, ms |
|---|---:|---:|
| Tall | 35.95195 | 35.94555 |
| Wide | 35.76192 | 35.76829 |

The observed difference is below .02% in both directions; no speedup is claimed.
The candidate's default graph retains 35 DtoD copies and no host callbacks or
host transfers, matching the control. Setup/capture and independent reference
construction are excluded from replay times and recorded separately. Both
occupancy records have no sampled competing process or monitoring error.
These controls check the unchanged default path; they do not claim that a
particular supplied start always accelerates convergence.

## Reproduction and records

```bash
python -m pytest -q tests/test_svd.py tests/test_svd_replay.py \
  tests/test_normal_svd.py tests/test_svd_starts.py
WARPACK_UPSTREAM_REPORT=build/cupy-svd-starts.json \
  python -m pytest -q -p benchmarks.upstream.cupy_plugin \
  build/test_upstream_cupy_linalg.py -k TestSvdsV0
python -m benchmarks.monitor_gpu \
  --out build/pod.occupancy.json --log build/pod.log -- \
  python -m benchmarks.applications.pod_compare --cases tall,wide \
  --sizes 100000 --methods warp --modes 4 --ncv 32 --steps 30 --repeats 5 \
  --out build/pod.json
```

Use the pinned environment from the main README; GPU numerical jobs run
sequentially. Evidence in [`results/performance_followup/`](../../results/performance_followup/):
`svd-start-validation.json`, `cupy-svd-starts.json`,
`svd-start-default-control.json`, `svd-start-default-candidate.json` and their
occupancy records. The validation file fingerprints the candidate source even
though its first timing report records the preceding base commit.
