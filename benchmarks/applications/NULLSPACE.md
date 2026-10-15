# Device numerical-nullspace diagnostics

`NumericalNullspaceDiagnostics` implements the evidence API in
[#39](https://github.com/alecjacobson/warpack/issues/39). It evaluates a supplied
Ritz space of a real symmetric PSD problem or SPD-mass generalized pencil.
All arithmetic, classification and output flags stay in Warp and can run after
the eigensolve inside the same CUDA graph.

**It does not certify total nullity or a Betti number.** Even exact independent
returned pairs, including a positive guard, can omit another zero direction.
An adversarial regression returns one exact zero and one exact positive pair
from a matrix with two zeros. The completeness-unverified bit remains set.
An optional independently known nullity checks agreement with that external
information; it does not turn the floating eigensolve into a topological proof.

This keeps Hodge harmonic modes as outputs. The separate
[mass-weighted FEM projection](MASS_PROJECTION.md) intentionally removes a
known rigid span; that policy is not inherited by this diagnostic.

## Evidence and units

The diagnostic recomputes Rayleigh values, original-pencil residuals and the
mass Gram matrix from the supplied vectors. Inputs must be FP64, independent
and M-orthonormal. An entrywise Gram tolerance smaller than 1/capacity preserves
a positive independence margin. Invalid or duplicate bases suppress numerical
counts, rather than counting the same null direction multiple times.

For SPD M, the caller supplies a valid positive lower spectral bound `m_lower`,
either as a scalar or a one-element FP64 device array. Device bounds are checked
and consumed on device; they can be updated in place under replay without a
host numerical readback. Invalid values suppress counts and set INVALID_DATA.
Mathematical validity of a positive bound for general M is still a caller
assumption, not something the diagnostic can prove from a small Ritz space.
The residual-based eigenvalue error estimate is

```text
||Kx - lambda Mx|| / sqrt(m_lower * x.T M x).
```

In exact arithmetic this bounds distance to some generalized eigenvalue; it
does not identify its index or multiplicity. Floating estimates are not rigorous
interval-arithmetic certificates. Identity mass uses `m_lower=1`. The Hodge
fixtures with M=T.T T and T=I+0.05*shift have the independent structural bound
lambda_min(M)>=0.95^2; their comparisons conservatively use 0.9. No mass spectrum
is computed on the host inside the production diagnostic.

The zero and positive thresholds are in **eigenvalue units**, so choose them
for the matrix scaling. Defaults are 1e-8 and 1e-7, leaving an explicit gray
band. The physical residual tolerance separately uses
`||Kx-lambda Mx|| / max(1, ||Kx||+|lambda| ||Mx||)`, as in the generalized solver.

- `numerical_count`: returned residual intervals wholly inside the zero band.
  This is a count within the supplied space, not the total matrix nullity.
- `weak_positive_count`: near-zero intervals whose lower endpoint is positive.
  These are explicitly distinguished from topological-zero claims and keep
  the gap unresolved.
- `ambiguous_count`: other unclassified, unconverged, negative or invalid rows.
- `positive_guard`: the smallest lower endpoint above the positive threshold;
  infinity if absent, NaN for invalid input/metric independence.
- `values`, `residuals`, `error_bounds`, `classification`, and
  `orthogonality_error`: device evidence behind those decisions.
- `suggested_capacity`: a device growth hint, bounded by matrix dimension,
  when all returned modes are near zero or known nullity fills the space.
  It does not resize storage or run an adaptive eigensolve.

## Explicit states

`flags` is a device bitmask; class constants expose these values:

| Flag | Meaning |
|---|---|
| `COMPLETENESS_UNVERIFIED` (1) | Always set; the returned floating space cannot certify total nullity |
| `INVALID_BASIS` (2) | Metric independence/normalization failed |
| `INVALID_DATA` (4) | Non-finite data, invalid metric norms or invalid bound |
| `UNCONVERGED` (8) | At least one original-pencil residual exceeds tolerance |
| `NEGATIVE_SPECTRUM` (16) | A residual interval contradicts PSD beyond the zero band |
| `CAPACITY_EXHAUSTED` (32) | More room is needed for the known/observed near-zero space and a guard |
| `GAP_UNRESOLVED` (64) | Missing guard, ambiguous rows or weak positives within the zero band |
| `EXPECTED_COUNT_MISMATCH` (128) | The numerical count differs from optional external nullity |
| `AT_MAX_CAPACITY` (256) | The supplied capacity already equals matrix dimension |

All flags can be inspected by subsequent Warp kernels without a host decision.
Classification codes are 0 near-zero, 1 positive guard, 2 ambiguous, 3
unconverged, 4 invalid, 5 negative spectrum and 6 weak positive within the zero
band. Near-zero count includes codes 0 and 6, with code 6 also reported separately.

## Capture and lifecycle

```python
import warp as wp
from warpack import NumericalNullspaceDiagnostics

# stiffness and mass implement the real FP64 sparse-operator interface.
# solver already owns its preallocated (k, n) eigenvector array.
diagnostic = NumericalNullspaceDiagnostics(
    stiffness,
    solver.eigenvectors,
    mass,
    mass_lower_bound=0.9,  # Valid structural bound for this particular fixture.
    zero_threshold=1e-8,
    positive_threshold=1e-7,
)
graph = solver.capture(100, finalizer=diagnostic.run)
wp.capture_launch(graph)
# diagnostic.flags/counts/guard remain on device.
```

The existing `capture(finalizer=...)` hook places diagnostics after the solver
in one graph. It avoids attempting to nest a cuDSS executable graph inside a
second stream capture. The diagnostic can also be captured by itself with
`ScopedCapture`. Construction preallocates scratch; replay recomputes evidence
after in-place vector/operator changes. Replacing arrays or changing scalar
configuration requires recapture. The device mass-bound array is the exception:
its value can change in place without recapture. The scratch is not shared concurrently across
streams. Known topology is optional input, never inferred from the requested k.

## Validation and performance

The comparison uses independent incidence ranks, Fourier spectra/harmonics and
dense generalized references on sphere, disk, annulus, torus, disconnected and
weak-component fixtures. Regressions include omitted repeated directions,
exhausted/full capacity, weak positives below the cutoff, duplicate/non-finite
vectors, non-diagonal mass, scaling units, and changed data under graph replay.
The production diagnostics also run as the finalizer of an actual captured
Hodge eigensolve.

The original extreme weak_scale=1e-10 run exposed a block-solver failure:
its shift is about -1e-11 and the standalone solver returns non-finite pairs,
status 1 and zero converged modes. [#46](https://github.com/alecjacobson/warpack/issues/46)
tracks that failure separately; the subsequent [metric-QR recovery](WEAK_BLOCK.md) fixes it. The diagnostic correctly reports invalid data
and basis, five ambiguous rows, zero accepted numerical modes, and flags 199.
This is **a rejected eigensolve**, not a successful nullity estimate or timing
win. The original strict expected-failure regression is now a passing multi-scale
regression; the failed reports below preserve the pre-fix evidence. The independent-reference weak
fixture separately verifies that finite weak positive modes remain distinguishable.

At the diagnostic implementation commit, the suite had **282 passing tests and one strict expected failure (#46)**
in 161.62 seconds. The 26 focused diagnostics tests pass, including device-bound
updates and invalid-bound recovery. Lint, formatting and wheel build pass. See
[validation provenance](../../results/nullspace_diagnostics_validation.json).

The [seven ordinary comparison fixtures](../../results/nullspace_diagnostics.json)
pass both the solver numerical gate and the diagnostic behavior contract. The
[extreme weak fixture](../../results/nullspace_diagnostics_weak.json) rejects the
solver output while passing the diagnostic's invalid-data behavior contract.
Every report explicitly leaves completeness unverified.

On the **10,086-unknown two-torus** case, three-trial median solver replay is
28.637 ms and solver-plus-diagnostic is 28.658 ms. Standalone captured diagnostics
take 0.060 ms; the CPU evidence calculation takes 1.832 ms with input vectors
already on the host. The observed increment is small enough that its precise
percentage should not be overinterpreted from three samples. The unchanged
solver passes the saved Hodge accuracy/graph/timing gate, with replay ratio
**1.00213**. Four returned near-zero modes agree with independent topology, and
the positive lower-endpoint guard is about 0.007823; this agreement still does
not certify nullity from floating eigenpairs alone.

Earlier reports remain linked in the provenance manifest, including the initial
CPU reference timing that omitted Gram independence. They are not substituted
for the final measurements. No full upstream, complex or Anvil rerun is implied.
 Host reference timings assume CPU-resident vectors
and cover Rayleigh/residual/metric evidence; they exclude the discrete classifier
and transfers and are not an end-to-end solver comparison. Pure diagnostic graphs
have no HOST/HtoD/DtoH nodes. Combined cuDSS graphs retain vendor HtoD nodes
tracked in #29. DOT inspection is not a host-synchronization API trace.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_nullspace_diagnostics.py tests/test_hodge.py
python -m benchmarks.applications.nullspace_compare --baseline results/hodge_large.json
python -m benchmarks.applications.nullspace_compare --cases weak_positive \
  --weak-scale 1e-10 --out results/nullspace_diagnostics_weak.json
```

The weak run can pass the **diagnostic behavior** contract while its explicitly
separate `solver_accepted` field is false. No report marks completeness certified.
