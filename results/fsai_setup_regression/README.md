# FSAI setup regression and promotion

Addresses [#139](https://github.com/alecjacobson/warpack/issues/139) and
[#141](https://github.com/alecjacobson/warpack/issues/141), while retaining the
memory improvement from [#131](https://github.com/alecjacobson/warpack/issues/131).
Measured on NVIDIA L40 (48 GB), driver 570.158.01, Xeon Platinum 8362,
2026-10-22. CPU workers use cores 8–11 and four OpenBLAS/OMP threads.

## Cause and fix

The newer upstream FSAI implementation selects supports directly from BSR
storage. Its scalar-entry lookup loaded an entire 3×3 block for every requested
entry. This dominated the support-selection kernel on the 283,857-DOF Anvil,
although the smaller procedural FEM qualification did not expose the regression.
Packing the Cholesky workspace alone did not fix it.

The corrected dependency reads entries through a scalar view of the original
values array. A three-dimensional view preserves block strides; flattening would
unnecessarily restrict inputs to contiguous values. This changes neither the
support search nor its arithmetic order, and creates no scalar-CSR copy or new
device allocation. Construction, ordinary refits and deferred selection use the
same access method. Solver arithmetic remains entirely Warp.

The [patch](block-view.patch) applies to the packed-only source. The combined
source is published at upstream revision
`f339a9fc7f231afa3405f15da9d7a45ce3fc4bcf`, FSAI SHA256
`8ef59b8b5ee4e008c9b40f622f004c850dd560c335dea0cb03c24a18364fa0c7`.
The two linear modules are unchanged from the v0.2.0 overlay. The installer
verifies all three files and records their hashes.

Live recipes no longer pass the removed `apply_lanes` constructor option.
The fast Anvil path uses Warpack's coalesced factor application, so this option
was unused there. Reproduce older scalar-application timings from their recorded
source checkout as well as their dependency profile.

## Isolated construction profile

One assembled Anvil matrix, one warmup and three constructions per variant,
CUDA activity instrumentation enabled. [Raw profile](profile.json) and
[occupancy record](profile.occupancy.json) show uncontended execution.
Every factor has **bitwise-identical supports and values** across all variants.

| Source / entry access | Median FSAI construction (s) |
|---|---:|
| v0.2.0 / scalar expansion | 1.0226 |
| New upstream / whole BSR block | 2.4798 |
| Packed workspace / whole BSR block | 2.5324 |
| Packed workspace / scalar expansion | 0.9454 |
| Packed workspace / flat scalar view | 0.6241 |
| Packed workspace / strided scalar view (selected) | 0.6305 |

Scalar expansion additionally costs 0.0159 s in this profile. The selected
strided view needs no expansion. These are instrumented setup measurements on a
shared assembled matrix, not end-to-end comparisons or independent memory
measurements. `screening.*` is an earlier contended run and is excluded.

## Full Anvil regression

Three alternating fresh processes per profile; each captures the public fast
block solve and measures three restored-start warm replays. FSAI32, FP32 CG25,
block width 24, requested 20 elastic modes with six known rigid constraints.
All six workers are uncontended and accepted. All 18 replay measurements pass
native status/count, original physical residuals, orthogonality, complete
requested spectrum and subspace checks. Graph audits contain kernels, conditional
nodes, memsets and DtoD copies; no HOST/HtoD/DtoH nodes. DOT inspection is not a
complete CUDA API trace.

| Profile | FSAI setup (s) | Setup + capture (s) | Warm solve (s) | Setup/capture device-memory decrease (GiB) |
|---|---:|---:|---:|---:|
| v0.2.0 | 1.0695 | 2.1572 | 1.8849 | 4.1484 |
| Corrected packed + scalar view | 0.6447 | 1.7368 | 1.8872 | 3.3438 |

Values are medians of three workers; warm solve is the median of each worker's
three trials, then the median across workers. Total setup includes factor,
solver/operator/CG workspace preparation and capture; it excludes mesh assembly,
reference work and host random-start generation. Setup includes any JIT work
encountered. Memory is the observed free-device-memory decrease through setup
and capture, including runtime/pool allocations, **not peak or live factor bytes**.

FSAI setup improves **39.7%**, total setup + capture **19.5%**, and the measured
memory decrease **19.4%**. Warm solve changes **+0.12%**, within variation; no warm
speedup is claimed. These records compare dependency profiles, not refreshed
measurements of all six README methods. The README's dated comparison table
continues to identify the v0.2.0 overlay that produced it.

The [summary](anvil/summary.json) records every worker and a reproducible promotion
gate: all accuracy/occupancy checks, matched code/configuration and unchanged
non-FSAI dependency sources, no more than 10% regression in median setup+capture
or warm solve, and reduced setup/capture memory. The same local mesh/reference
were used throughout; the fast-block worker's missing input fingerprints are
tracked in [#142](https://github.com/alecjacobson/warpack/issues/142) and must not
be retroactively invented for these records.

## Correctness validation

The exact installed candidate passes all **89 upstream tests** on CPU/GPU
([log](installed-upstream89.log)): 85 existing registrations and four new
FP32/FP64 × CPU/GPU strided-input registrations. Independent dense local solves
and scalar-CSR factors check the added cases. Tests cover changed-value refits
on scalar/2×2/3×3 storage and captured CUDA replay.

The full library suite passes **1,643 tests, five explicit skips** in 598.33 s
([log](full-tests.log)). Four skips concern the historical old-source-only
workspace patch, and one is a stock-runtime negative control. The suite includes
the installed profile's captured changing-K/M, deferred-factor, failed-update
and recovery tests. This runtime is validation duration, not a performance claim.

Additional layout checks compare strided captured refits bitwise against the
preceding packed implementation, inspect the graph for host/transfer nodes,
and cover explicitly typed 1×1 matrix blocks as well as scalar/2×2/3×3 values
([log](strided-tests.log)). A separate current-profile test run removes the
historical downloaded candidate file temporarily, verifying that installed-profile
coverage does not silently skip on a fresh checkout ([log](installed-refit-tests.log)).
These final additions do not retroactively change the full-suite count above.

The final installer/provenance/reporting checks pass **36 tests**
([log](reporting-tests.log)), including deliberate setup/replay regressions,
contention, rejected accuracy, changed dependencies and changed implementation
hashes. Wheel/source builds and the installed-wheel import/replay smoke are also
checked; the latter forbids reference/application imports and requires the fast
public path ([report](wheel.json), [log](wheel.log)).

## Reproduce

Install the official Warp 1.18.0+cu12 wheel into **two separate targets**, as in
the [installation guide](../../docs/installation.md), then select their profiles:

```bash
python scripts/install_fsai_overlay.py build/warp-legacy --profile v0.2.0
python scripts/install_fsai_overlay.py build/warp-current --profile current
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.experimental.fsai_anvil_regression \
  --legacy-overlay build/warp-legacy --candidate-overlay build/warp-current \
  --cpu-affinity 8-11 --out build/fsai-anvil-regression
PYTHONPATH="$PWD/build/warp-current:$PWD" python -m \
  benchmarks.experimental.fsai_current_suite --installed --download
PYTHONPATH="$PWD/build/warp-current:$PWD" python -m pytest -q
```

The regression command requires a prepared Anvil reference; follow the
[release comparison prerequisites](../release_v0_2_0/README.md#anvil-promotion-regression).
It refuses to overwrite existing measurements. Occupancy failures require a new
run directory; do not silently discard slow trials. Use `--summarize-only` with
an existing directory to recheck its gates.

For setup profiling, explicitly download the older candidate once using
`benchmarks.experimental.fsai_current.load(download=True)`, then run
`python -m benchmarks.experimental.fsai_anvil_setup --out build/fsai-profile.json`
under the legacy overlay and the occupancy wrapper. Its checksummed experimental
loaders retain all compared variants. The setup profile intentionally uses
instrumentation; the full regression uses ordinary wall timing.

Smaller-memory GPU qualification remains open in #131. Optional regularized
refits are not enabled by this upgrade.
