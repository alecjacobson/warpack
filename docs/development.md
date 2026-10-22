# Testing and benchmarking

[Back to README](../README.md) · [Installation](installation.md)

## Local tests

The complete numerical suite needs a CUDA GPU, the optional cuDSS and reference
packages, and the pinned Warp FSAI/batched-CG environment. From the repository root:

```bash
python -m pip install -e '.[test,cudss,benchmark]'
# Install the pinned overlay as described in the installation guide first.
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m pytest -q
python -m pip install ruff build
ruff check warpack tests benchmarks examples scripts
ruff format --check warpack tests benchmarks examples scripts docs
python -m build
```

Stock Warp supports a smaller compatibility subset; unsupported upstream
features are skipped or explicitly rejected. The [release validation](../results/release_v0_2_0/README.md)
records exactly which environments and scopes were checked, including final
focused checks after setup-only changes. Hosted CI covers packaging and lint;
GPU results are recorded separately. Test counts are dated evidence, not a
promise that every upstream API is supported.

Tests check original equations, eigenvalues and complete requested invariant
subspaces, repeated/null modes, mass orthogonality, constitutive derivatives,
changed values/constraints/starts on replay, failure/status behavior and CUDA
graph nodes. Independent NumPy/SciPy calculations are reference code outside
the library. Solver arithmetic remains Warp, plus optional cuDSS sparse solves.

## Comparison harnesses

| Coverage | Protocol / reproduction | Results |
|---|---|---|
| Application matrix families | [FEM, Hodge, geometry, dynamics and POD](../benchmarks/applications/README.md) | [Application reports](../benchmarks/applications/RESULTS.md) |
| Upstream CuPy and Spectra | [Pinned sources, fixtures and adapters](../benchmarks/upstream/README.md) | [CuPy refresh](../benchmarks/upstream/CUPY_REFRESH.md), [Spectra refresh](../benchmarks/upstream/SPECTRA_REFRESH.md), [generalized follow-up](../benchmarks/upstream/SPECTRA_REGULAR.md) |
| Optional FEM completeness | [API and limitations](completeness.md) | [Newton cases and overhead](../results/completeness/README.md) |
| Shape/mass updates and FSAI refits | [Replay contract and commands](../results/fem_shapes/README.md) | [Eight monitored configurations](../results/fem_shapes/summary.json) |
| Coarse-to-fine FEM starts | [Nested P1 beam protocol and results](../results/fem_hierarchy/README.md) | [20 modes](../results/fem_hierarchy/consistent-20.summary.json), [50 modes](../results/fem_hierarchy/consistent-50.summary.json) |
| Coarse accuracy and full-width starts | [Seven monitored profiles](../results/fem_hierarchy_tuning/README.md) | [20-mode full loose](../results/fem_hierarchy_tuning/full-loose.summary.json), [50-mode control](../results/fem_hierarchy_tuning/k50-full-loose.summary.json) |
| Low-rank force-covariance ROM | [Procedural forces, equations and limits](../results/force_dual/README.md) | [cuDSS](../results/force_dual/consistent-cudss.summary.json), [pure Warp](../results/force_dual/consistent-cg.summary.json) |
| Experimental FSAI workspace | [Packed-storage patch and memory evidence](../results/fsai_packed/README.md) | Four uncontended FEM comparisons, original and packed widths 32/64 |
| Candidate FSAI updates | [Dependency qualification and reproduction](../results/fsai_current/README.md) | Captured refits, deferred factors and changing FEM matrices |
| Current FSAI dependency | [Setup fix, installation and regression protocol](../results/fsai_setup_regression/README.md) | [Three fresh Anvil comparisons](../results/fsai_setup_regression/anvil/summary.json), full library and upstream tests |
| Fast public FEM path | [Families, configurations and gates](../benchmarks/applications/FAST_BLOCK.md) | [FEM qualification](../results/fem_fast_block_v020/README.md) |
| Coarse FEM correction | [Experimental driver](../benchmarks/applications/fem_coarse_correction.py) | [Matched comparison and negative result](../results/fem_coarse_correction/README.md) |
| Fast public Anvil path | [Release comparison command](../results/release_v0_2_0/README.md#anvil-promotion-regression) | [Every accepted/rejected release trial](../results/release_v0_2_0/README.md) |
| CPU Anvil factors | [PARDISO / SuperLU / Eigen audit](../benchmarks/ANVIL_FACTORIZATION.md) | [ARPACK](../results/anvil_pardiso_noir_confirmed.json), [Spectra](../results/anvil_pardiso_spectra_confirmed.json) |

Run numerical GPU jobs sequentially. For timing, use the occupancy wrapper:

```bash
python -m benchmarks.monitor_gpu --out build/run.occupancy.json \
  --log build/run.log -- python -m YOUR_BENCHMARK_MODULE YOUR_ARGUMENTS
```

The command above is a template; the linked reports give executable benchmark
commands. Do not infer a speedup from a failed reference, a contended sample,
or an eigenpair set that has small residuals but misses requested modes.

Record setup/factorization, JIT/capture, warm replay, memory and validation
separately. Compare original physical residuals, orthogonality, requested
spectrum/subspaces and full counts. Native stopping conventions differ between
libraries, so equal tolerance arguments do not necessarily mean equal accuracy.
Restore starts for each timed trial. Retain failed and timed-out samples.

The Anvil and ROM follow-up reporters record hashes of the loaded Warp linear
and FSAI modules, in addition to the wheel version. Patched overlays can share
the same wheel version while running different numerical code. The installer
writes a checksum manifest, which the reporters check against the loaded files.
Regression comparisons reject changed dependencies, stale manifests and older
baselines without this provenance. The [Anvil input protocol](../results/anvil_inputs_142/README.md)
also requires a verified common assembled fixture, unchanged reference data and
method-specific start/shifted-matrix identities. Use the historical checkout to reproduce an
older report, or establish and review a new baseline; do not copy dependency
metadata onto an old measurement.

Graph DOT audits check callback nodes and transfer directions; they are not
complete API traces. The cuDSS vendor metadata exception remains explicit.
Application setup, host file I/O and independent references stay outside the
timed/captured solver.

## Package checks

`python -m build` builds the wheel from the source distribution. Verify installed
APIs outside the source checkout so benchmark imports cannot accidentally mask
missing package files. [`scripts/verify_release_wheel.py`](../scripts/verify_release_wheel.py)
imports all core modules and exercises changed-matrix replays while rejecting
application/reference/optional-cuDSS imports. Its header documents the required
installed-wheel path; `--require-fast` additionally checks the pinned fast path.

## Historical evidence

The [v0.2.0 README](https://github.com/alecjacobson/warpack/blob/v0.2.0/README.md)
preserves the previous detailed experiment index. [Historical reports](../historical/README.md)
and the per-experiment pages retain earlier settings, unsuccessful approaches
and source provenance. Current user-facing documentation should link to those
reports rather than accumulate their chronological updates.
