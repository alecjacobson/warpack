# Profiling first-whitening conditionals (#177)

The diagnostic active-step profiler is repaired. Ordinary solver numerics and
captured solve graphs are unchanged.

After #171, its `metric.diagonalize` wrapper could record an external timing
event inside the first-whitening canonical-fallback conditional body. CUDA
rejects that operation. The [retained reproducer](validation/reproducer.log)
and [original source identities](validation/reproducer-source.json) establish
the failure on the small wide-block FEM fixture; uninstrumented solves were
passing before profiling failed.

The profiler now times each complete first/second whitening helper at its
boundary and suspends nested event instrumentation while constructing its
conditional bodies. It restores both wrappers afterward. Each helper has a
separate event span; spans overlap enclosing `whiten` and step events and must
not be added. Canonical metric sweep counts may describe an earlier step when
both fast paths succeed; the report now says so explicitly.

Six focused checks pass ([transcript](validation/tests.log)), including both
history policies, legacy small profiles, both conditional helpers, nonnegative
finite event spans, wrapper restoration, and independent complete original
spectrum/subspace/count checks before and after profiling.

The [fixed diagnostic](fixed.json), [configuration](config.json) and [log](fixed.log)
replay two selected steps twice. Every first/second-helper span is positive;
uninstrumented solves pass before and afterward. The uninstrumented graph has
no host nodes and only device-to-device copies. This small validation run is
not a performance comparison; host selection of diagnostic branches is explicit.

```bash
python -m benchmarks.experimental.profile_active_block \
  --config results/first_whitening_profile/config.json \
  --checkpoints 0,1 --repeats 2 --out build/profile-fixed.json
python -m pytest tests/test_active_block_profile.py -q
```

Use the pinned Warp/FSAI environment. The diagnostic requires upstream FSAI;
production solvers do not acquire a profiling dependency.
