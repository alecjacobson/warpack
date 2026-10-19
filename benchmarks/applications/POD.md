# Large snapshot POD: Warp row-product improvements

Related issues: [#31](https://github.com/alecjacobson/warpack/issues/31),
[#28](https://github.com/alecjacobson/warpack/issues/28),
[#10](https://github.com/alecjacobson/warpack/issues/10).

**2026-10-19 follow-up:** the [normal-product experiment](../experimental/NORMAL_POD.md)
uses a full basis in the small snapshot dimension and reaches 15.41 ms for the
100,000×48 clustered k20 case, against CuPy 74.78 ms. It remains experimental;
the augmented formulation and its independent-chain guard remain the production
default. The follow-up also fixes stale transpose values in changed-matrix SVD
replay and preserves a rejected early-exploration attempt. The tables below are
the earlier augmented-product campaign.

Production changes are confined to `warpack/svd.py`. Long rectangular CSR
products distribute work across GPU threads; very long rows use multiple blocks
and a second device reduction. The same products accelerate final SVD residual
evaluation. Scratch is allocated at construction. Arithmetic and reductions
are Warp kernels, without cuDSS, CuPy, CPU numerical work or convergence
readbacks during replay. The new row-product kernels use no atomics.
Short sparse rows retain the scalar path.

The augmented symmetric formulation, restart budget, tolerance, independent-chain
exploration guard and multiplicity fixes are unchanged. This is a product-kernel
optimization, not a relaxed stopping test. The discarded short-vector
orthogonalization experiments were not merged.

## Matrices and independent acceptance

Four fixtures use 48 snapshots and either 10,000 or 100,000 spatial samples:

- Tall correlated snapshots with independently generated orthonormal left/right
  factors, geometric singular spectrum and two zero singular values.
- Their wide transpose, exercising the opposite long-row orientation.
- A correlated fixture with a complete threefold leading singular cluster.
- Equally spaced snapshots of an analytic fixed-end unit string, with 22 excited
  spatial sine modes and their cosine time responses over a fundamental period.
  Spatial/time normalization gives an independent singular-value reference.

The last fixture is a physics-derived unweighted sampled snapshot problem.
General consistent-mass weighted POD preprocessing and complex SVD are separate
API work; these results do not establish support for those contracts.

Every trial checks requested singular values, both SVD equations, left/right
orthogonality, both complete requested subspaces, captured energy and optimal
rank-k reconstruction energy. Subspace checks allow rotations inside repeated
clusters. Reconstruction truncation error is reported separately from solver
error. All numerical acceptance bounds remain 1e-8. Warp uses internal tolerance
1e-11, and both left/right strict convergence counts are checked independently.
The reported augmented product count is derived from the device restart counter
and the captured schedule; it is not a separate runtime instrumentation counter.

## Four-mode comparisons

L40 48 GB; pinned Warp 1.18.0/FSAI environment; CuPy 14.2.0; SciPy 1.15.0;
Spectra `db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95`; four BLAS/OMP threads.
Three synchronized warm solve trials, milliseconds:

| Fixture / spatial samples | SciPy ARPACK | Spectra | CuPy | Warp |
|---|---:|---:|---:|---:|
| Tall / 10,000 | 29.258 | 29.115 | 14.971 | **12.374** |
| Tall / 100,000 | 356.223 | 446.646 | 60.151 | **35.899** |
| Wide / 10,000 | 34.987 | 23.119 | 14.394 | **12.579** |
| Wide / 100,000 | 421.410 | 367.209 | **16.178** | 35.720 |
| Repeated cluster / 10,000 | 28.901 | 29.332 | **14.472** | 14.781 |
| Repeated cluster / 100,000 | 354.622 | 459.557 | 59.926 | **42.597** |
| String / 10,000 | 29.669 | 36.910 | 14.194 | **9.899** |
| String / 100,000 | 350.373 | 556.754 | 61.113 | **28.895** |

All 32 configurations pass every numerical trial. All eight Warp profiles report
full left/right counts and have no HOST callback or host-transfer nodes in their
nested graph DOT. No competing GPU process was observed. Warp's worst normalized
SVD equation residual is 1.54e-14; worst requested-value error 1.50e-14;
left/right subspace errors below 5.9e-15. DOT inspection is not an API trace.

All methods explicitly request k=4, ncv=32. Warp's augmented operator has
nrow+ncol unknowns; the reference methods use normal-equation products on the
smaller dimension. These are the native algorithmic formulations, not equivalent
internal work budgets. Warp restores seed 42; SciPy/CuPy restore reference seed
2026. Spectra's `PartialSVDSolver` uses its built-in deterministic initialization
and a fresh native solver for each trial. CuPy's adapter now passes ncv explicitly;
earlier reports using its default ncv are not identical-setting comparisons.

Setup, capture, warmup, fixture construction and validation are excluded from
solve time. For tall 100,000×48, setup is 1.49 ms ARPACK, 105.50 ms Spectra,
364.72 ms CuPy and 154.75 ms Warp; Warp capture is another 229.79 ms. These are
setup-inclusive library paths, not claims about total time to create snapshots.
Peak memory was not measured in this campaign.

The [composite report](../../results/pod_comparisons.json) retains the 24 native
reference rows from the complete [initial comparison](../../results/pod_cooperative_comparisons.json)
and replaces only its eight Warp rows with the [final split-row run](../../results/pod_split_comparisons.json).
Reference adapter code did not change between those runs. Each row records its
measured source; this is not presented as a new simultaneous 32-profile rerun.

## Twenty-mode extraction

A separate [16-profile run](../../results/pod_twenty_modes.json) requests k=20,
ncv=40, maximum 60 Warp cycles. Fourteen pass the numerical gates; the two
native CuPy string cases fail and remain in the raw report. All four Warp
profiles pass both strict counts and the graph audit. Milliseconds:

| Fixture / spatial samples | ARPACK | Spectra | CuPy, ncv40 | Warp |
|---|---:|---:|---:|---:|
| Repeated cluster / 10,000 | 52.641 | 41.676 | 17.168 | 80.469 |
| Repeated cluster / 100,000 | 488.675 | 546.624 | 74.694 | 198.916 |
| String / 10,000 | 192.823 | 56.731 | rejected | 76.761 |
| String / 100,000 | 464.580 | 741.506 | rejected | 197.908 |

Warp takes 25–26 cycles and 19 breakdown recoveries. It remains slower than
CuPy on accepted 20-mode profiles, and Spectra wins the 10,000-row k=20 cases.
The CPU advantage here is on a tall matrix whose smaller dimension is only 48;
row count alone is not a sufficient predictor of useful GPU work.

A [native CuPy-only probe](../../results/pod_cupy_probe.json), with no Warp
import or stream interop, reproduces failure at ncv40 on both string sizes.
The normal-equation `eigsh` stage has non-finite eigenvalues. Exact `svds`
outputs differ between runs/construction paths: zeros or very large invalid
values. This localizes the failure to native CuPy, without establishing the
internal cause. Other ncv settings can also fail; it is not merely an adapter
convergence-count disagreement.

The [ncv24 follow-up](../../results/pod_cupy_wave_followup.json) passes both
string cases at 11.751 / 46.954 ms. These use a different basis size from the
ncv40 table and are explicitly tuned profiles, not identical-settings
regression comparisons. They still leave a meaningful Warp performance gap.
The original rejected rows are not replaced by the follow-up.

## Identical-settings regression

The [new regression run](../../results/pod_performance_regression.json) versus
the [unchanged main control](../../results/pod_before.json) retains k=4,
ncv=32, maximum 30 cycles, original starts, matrices and tolerances:

| Snapshot shape | Main control (ms) | Final Warp (ms) | Speedup |
|---|---:|---:|---:|
| 80×40 | 8.446 | 7.293 | 1.16× |
| 1,024×48 | 21.566 | 10.770 | 2.00× |
| 10,000×48 | 119.217 | 12.389 | 9.62× |
| 100,000×48 | 1814.334 | 35.883 | 50.56× |

All four pass the matched accuracy, graph and 1.25× timing-regression gates.
[Raw checks](../../results/pod_before_after.json) include comparability guards.
Against the older pre-multiplicity-fix baseline, the 1,024×48 case now passes
at 1.161× its original 9.276 ms. The 80×40 case remains at 2.012× its original
3.625 ms and fails that unchanged historical allowance. No old performance
failure was recategorized as a pass. Neither this optimization nor the small
case justifies weakening the multiplicity checks.

## What remains

The wide 100,000 case still favors CuPy by about 2.2×. Warp still performs five
restart cycles / three breakdown recoveries on the tall and wide k=4 cases,
six cycles on the repeated cluster and four on the string. The guard was not
bypassed to reduce these counts. Adaptive sparse eigensolver throughput is not
changed by this SVD-only patch. #31 remains open.

The four original Spectra SVD fixtures retain their previous outcomes: three
pass, and `SVD_1` reports 4/5 at the strict 1e-12 requested tolerance despite
meeting its original 1e-9 numerical assertions. This is a remaining strict
convergence gap under #10, not four remaining wrong-singular-value failures.
The focused [upstream record](../../results/pod_upstream_spectra.json) preserves
that failure. This campaign does not rerun the full upstream inventory or Anvil.

## Validation

The final production source passes **322 local tests in 177.25 s**, plus lint,
formatting and wheel build. New regressions cover repeated tall/wide singular
subspaces, both sides of the 4096-entry dispatch threshold, uneven and empty
long rows, changed device inputs on replay, analytic snapshot references and
rejection of accurate-but-wrong requested triplets. The existing multiplicity
suite remains intact. [Source and validation provenance](../../results/pod_validation.json)
records the complete and focused runs separately.

## Reproduction

Use the [pinned environment](README.md) and compile the extended native adapter:

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
g++ -O3 -DNDEBUG -shared -fPIC -I/usr/include/eigen3 -Ibuild/upstream-spectra/include \
  benchmarks/applications/spectra.cpp -o build/application_spectra.so
python -m pytest -q
python -m benchmarks.applications.pod_compare --out results/pod_comparisons.json
python -m benchmarks.applications.pod_compare --cases clustered,wave --modes 20 \
  --ncv 40 --steps 60 --out results/pod_twenty_modes.json
python -m benchmarks.applications.pod_compare --cases wave --methods cupy --modes 20 \
  --ncv 24 --steps 60 --out results/pod_cupy_wave_followup.json
python -m benchmarks.applications.pod_cupy_probe --out results/pod_cupy_probe.json
python -m benchmarks.applications.compare --classes I --sizes 80,1024,10000,100000 \
  --methods warp --baseline results/application_coverage.json \
  --out results/pod_performance_regression.json
python -m benchmarks.upstream.bounded spectra --families SVD --fresh --timeout 90 \
  --out results/pod_upstream_spectra.json
```

The historical regression command intentionally returns nonzero while the
80×40 case still exceeds the unchanged original 1.25× performance allowance.
Its numerical results remain valid; a failed performance gate is not omitted.

The ncv40 20-mode comparison also returns nonzero because of the two retained native CuPy failures.
