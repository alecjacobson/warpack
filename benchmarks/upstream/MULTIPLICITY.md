# Krylov restart and multiplicity regressions

Implementation: `84df32b`, with final SM-retention correction `7bcc019`.
Local verification: **147 passed, 1 strict expected
failure**, plus lint, formatting and wheel-build checks. The expected failure
remains the damped-chain residual in #27.

This change fixes the three cycle-Laplacian cases in [#1](https://github.com/alecjacobson/warpack/issues/1)
and the preserved explicit-nullspace-start case in [#8](https://github.com/alecjacobson/warpack/issues/8).
The original requested spectra, starts, subspace sizes and assertion bounds are
unchanged. These direct real symmetric cases do not use cuDSS.

## Diagnosis and implementation

`Example1_0` could stop with small residuals after finding distinct eigenvalues
but missing a requested repeated eigenvalue. The adaptive exploration guard
previously applied only after a breakdown, allowing this first-chain exit.
Other small-subspace cases stalled because recovery reused the same random
direction at each restart and retained too many vectors to expand effectively.

Recovery now uses a fresh device-generated direction, and compact subspaces
reserve more expansion columns for extremal targets. Interior `SM` selection
preserves the previous larger retained space. After recovery, retained Ritz residuals need
not be collinear. Restart therefore expands the requested vector with the
largest relative residual, instead of an arbitrary last retained vector.
The basis permutation preserves the ordered output vectors and projected
diagonal. All decisions, reductions and recovery remain Warp kernels.

Adaptive replay requires at least k seeded chains, including the initial
chain, unless the basis spans the full problem. This conservative guard is
not a proof of global spectrum completeness. `converged` still counts pairs
passing the residual test; a fixed or exhausted iteration budget does not
certify that the requested spectrum has been completely explored. Improving
termination reporting remains part of [#13](https://github.com/alecjacobson/warpack/issues/13).

## Numerical evidence

| Exact Spectra case | Maximum absolute residual | Requested-spectrum error | Reported converged |
|---|---:|---:|---:|
| Example1_0 | 5.78e-14 | 9.99e-15 | 3/3 |
| Example1_2 | 5.77e-15 | 1.55e-14 | 5/5 |
| Example1_4 | 4.77e-15 | 5.11e-15 | 6/6 |
| Example4_1 | 3.37e-13 | 3.64e-13 | 3/3 |

Permanent regressions check the spectrum, original-equation residuals,
orthogonality and containment in the requested invariant subspace. They cover
eager, fixed-graph and adaptive-graph execution; repeated replay; changed device
starts; rotated clusters with multiplicities 2, 4 and 8; compact and wider
subspaces; a full-space projection; and the exact explicit-nullspace start.
When k cuts through a repeated eigenvalue, subspace validation allows any
orthonormal selection inside the full boundary eigenspace.

The affected Spectra families contain **102 cases: 77 pass and 25 fail**,
compared with 65 passes and 37 failures in the published baseline. All 65
previous passes remain passes. The twelve improvements include three of the
four SVD cases, one ordinary symmetric case and three shifted symmetric cases.
The remaining SVD case and complex Hermitian failures stay open. This focused
run does not replace the complete 360-case upstream inventory.

`SVD_1` now also meets all original numerical bounds (singular-value error
5.55e-12, left/right vector errors 1.04e-10 and 4.22e-11), but reports 4/5
converged and therefore remains a failure under the unchanged adapter rules.

Comparison also checks numerical bounds separately from outcome labels. An
initial attempt worsened `SymEigsShift_17` from a count-only failure (physical
residual 5.11e-12) to a numerical failure (1.59e-7), despite its unchanged
`failed` label. Preserving the previous retention policy for interior `SM`
targets restores its physical residual to 5.69e-12. All six transformed-SM
cases were rerun, and the rejected attempts remain in their JSON histories.
There are no numerical-pass to numerical-failure transitions in the final
102-case comparison. A permanent captured test also checks this case's
requested physical spectrum, residual and orthogonality; its stricter inverse
count still does not satisfy the upstream adapter, so #13 remains open.

The complete adapted CuPy eigensolver/SVD suite retains exactly the same
outcome for every one of its 3,984 parameter cases: 673 ordinary passes,
31 ordinary failures, 3,172 skips, 72 expected failures and 36 unexpected
passes under upstream markers. The 31 ordinary failures remain the 24 real
near-singular SM cases in [#2](https://github.com/alecjacobson/warpack/issues/2)
and seven complex Hermitian cases in [#3](https://github.com/alecjacobson/warpack/issues/3).
Neither unsupported skips nor inherited expected-failure markers count as
new coverage. This rerun uses Warpack through the adapter, not native CuPy.

For `Example1Shift_3` and `_5`, physical residuals are now below 2.5e-16 and
spectrum errors below 8e-16, but inverse-operator convergence counts remain
4/5 and 5/6. They are still recorded as failures under the unchanged adapter
rules; [#5](https://github.com/alecjacobson/warpack/issues/5) and #13 remain open.
No failed baseline timing is used to claim a speedup.

## Large-problem performance checks

Same NVIDIA L40, Warp 1.18.0+cu12, cuDSS 0.8.0.10, four BLAS/OpenMP threads,
matrices, starts and solver settings as the application-coverage baseline.
Values are medians of three warm synchronized solves; assembly, factorization,
capture and host validation are outside timing. All three trials pass the
existing accuracy gates. No competing GPU process was observed.

| Workload | Previous median | Current median | Change |
|---|---:|---:|---:|
| 10,000-DOF FEM, low modes, cuDSS | 20.48 ms | 20.79 ms | +1.51% |
| 10,000-DOF FEM, interior modes, cuDSS | 21.13 ms | 21.23 ms | +0.44% |
| 10,000-DOF FEM, negative modes, cuDSS | 20.50 ms | 20.79 ms | +1.41% |
| 283,857-DOF Anvil, cuDSS | 0.466930 s | 0.467120 s | +0.041% |
| 283,857-DOF Anvil, FSAI64f-CG | 9.586478 s | 9.579671 s | −0.071% |

These show no material regression in the primary FEM workloads. cuDSS's
captured scalar HtoD copies remain [#29](https://github.com/alecjacobson/warpack/issues/29);
the graph audit distinguishes those vendor transfers from host callbacks and
device-to-host copies. This change adds no host convergence decisions.

The separate diagonal-preconditioned CG comparisons for application classes
A/C/D also retain accuracy and their device-only graph contract, with timing
changes below 0.03%. Those controlled fixtures are distinct from Anvil's tuned
FSAI64f configuration.

## Adaptive performance tradeoff

All nine application classes were rerun at requested sizes 80 and 1024.
Numerical outcomes remain 16 passes and the two known damped-dynamics failures;
there are no accuracy or graph-contract regressions. Classes A–G have timing
changes between −0.54% and +2.02%.

The POD/SVD fixtures fail the unchanged 25% performance regression gate:

| Snapshot matrix | Previous median | Current median | Ratio |
|---|---:|---:|---:|
| 80 × 40 | 3.625 ms | 8.429 ms | 2.326× |
| 1024 × 48 | 9.276 ms | 21.560 ms | 2.324× |

Both solves pass the same singular-spectrum, two-sided residual and
orthogonality checks on all three trials. Their captured graphs contain no
host callback or host transfer. The application comparison command correctly
exits unsuccessfully for these timing regressions. The stronger exploration
and restart policy has a throughput cost on these adaptive paths; the stable
FEM timings above do not imply unchanged performance for every problem class.
The measured regression and acceptance criteria for recovering throughput are
tracked in [#31](https://github.com/alecjacobson/warpack/issues/31).

All twenty original Spectra symmetric benchmark datasets were also rerun with
Warp and CuPy: all forty backend/dataset results pass. These are small dense
diagnostics, with direct Warp iteration and no cuDSS. Each cell below is the
median of ten dataset medians, each based on ten warm synchronized solves:

| Dimension | Previous Warp | Current Warp | Current CuPy | Warp ratio |
|---|---:|---:|---:|---:|
| 100 | 7.100 ms | 18.432 ms | 37.520 ms | 2.596× |
| 1000 | 22.424 ms | 42.883 ms | 63.506 ms | 1.912× |

Maximum final Warp residuals are 2.40e-13 and 6.28e-13 respectively, much below
the requested 1e-10 internal tolerance. More efficient independent-direction
exploration after the residual test passes is a follow-up optimization; simply
restoring the old early exit would reintroduce the multiplicity failure.
Benchmark output validation checks the final replay, unlike the application
runner's per-trial checks. These historical timing comparisons are diagnostics;
the application runner supplies the explicit contention and regression gates.

## Reproduction and evidence

Use the [pinned upstream sources and fixture preparation](README.md) and
[application protocol](../applications/README.md). For the focused run:

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_multiplicity.py
python -m benchmarks.upstream.bounded spectra \
  --families Example1,Example1Shift,Example4,SymEigs,SymEigsShift,HermEigs,SVD \
  --fresh --timeout 90 --out results/multiplicity_spectra.json
WARPACK_UPSTREAM_REPORT=results/multiplicity_cupy.json \
  python -m pytest -q -p benchmarks.upstream.cupy_plugin build/test_upstream_cupy_linalg.py
python -m benchmarks.applications.compare --classes ABC --sizes 10000 \
  --methods warp --steps 5 --baseline results/application_coverage_large.json \
  --out results/multiplicity_large_regression.json
python -m benchmarks.applications.compare --sizes 80,1024 --methods warp \
  --baseline results/application_coverage.json --out results/multiplicity_applications.json
python -m benchmarks.applications.compare --classes ACD --sizes 1024 --methods warp-cg \
  --baseline results/application_inverse_comparison.json \
  --out results/multiplicity_inverse_regression.json
python -m benchmarks.upstream.bounded benchmarks --families benchmark_sym \
  --fresh --timeout 180 --out results/multiplicity_benchmarks.json
```

- [Affected Spectra cases](../../results/multiplicity_spectra.json)
- [Validation summary, transitions and source hashes](../../results/multiplicity_validation.json)
- [Complete adapted CuPy cases](../../results/multiplicity_cupy.json)
- [10,000-DOF FEM regression](../../results/multiplicity_large_regression.json)
- [All application classes](../../results/multiplicity_applications.json)
- [Pure-Warp CG inverse regression](../../results/multiplicity_inverse_regression.json)
- [Twenty symmetric benchmark datasets, Warp and CuPy](../../results/multiplicity_benchmarks.json)
- [Anvil cuDSS](../../results/multiplicity_anvil_cudss.json)
  and [Anvil FSAI-CG](../../results/multiplicity_anvil_cg.json)

Spectra workers record one timed replay per case; their timings are diagnostic,
not robust performance comparisons. Application reports record three trials
and separate numerical acceptance from the stricter zero-host-transfer graph
contract. Reports collected before committing name the parent revision
`8c1dbec` with a dirty working tree; the implementation is the source change
published with this report.

The final SM-only retention correction was followed by the six-case targeted
rerun and the full local suite. It does not change the LM/LA/SA paths used in the
CuPy adapter, POD benchmarks or application/Anvil performance reports; those
earlier measured samples remain applicable. No performance gain is claimed
from the targeted single-sample SM rerun.
