# Installed register-blocked triangular application (#173)

All **56 complete first/later outputs pass**. Wide FEM warm solves improve
1.3–3.4% in fresh paired controls; Anvil retains its smaller-width path.
The independent prototype and rejected alternatives are retained in the
[preceding qualification](README.md).

## Implementation and review

For Ada (SM 89) and Ritz widths 96–256, the block solver interleaves eight serial
triangular row recurrences per thread. Each previous solved value is loaded once
and reused across those eight recurrences. Each row retains its original
subtraction order. Partial panels are guarded; the unused upper factor triangle
is never read. There is no inverse, shared scratch/barrier or added global
workspace. Both complete QR passes and all rank/stopping thresholds remain.
Other widths and architectures retain the original scalar application.

The installed numerical kernel's AST exactly matches the independently qualified
prototype. Review covered arithmetic order, panel tails, buffer ownership,
launch guards, eager/capture preparation, cached-image updates, active branch
composition and explicit canonical benchmark controls. No host numerical
decision, replay synchronization or vendor call is introduced. All new numerical
work is Warp. Kernel tests require exact agreement with the original recurrence;
full solves may differ slightly because of atomic Gram reductions and stopping.

## Regression and package checks

**2,958 tests pass, with 11 skips** in the clean full run
([transcript](validation/installed-full-tests.log)). This collection preceded
the twelve separate fresh-image prototype tests for #179; those are not part
of this installed implementation.

[Focused checks](validation/installed-focused-tests.log),
[failure/recovery](validation/installed-failure-tests.log),
[stock Warp](validation/installed-stock-tests.log),
[retained first full attempt](validation/installed-first-full-attempt.log),
[corrected dispatch tests](validation/cholesky-dispatch-tests.log),
[wheel identity](validation/installed-wheel.json),
[public wheel replays](validation/installed-wheel-public.json),
[projected outputs](validation/installed-wheel-projected.json),
[wide wheel replays](validation/installed-wheel-wide.json).

- 105 initial focused checks pass, followed by two explicit invalid-QR/recovery
  tests. Coverage includes guard boundaries/other architectures, poisoned unused
  triangles, strided/tail storage, dense even large shapes, changed/scaled inputs,
  coupled/ill-scaled mass, cached images, constraints, repeated modes, both history
  policies and changed-input replay. Negative mass, nonfinite and zero input
  remain rejected; healthy input recovers on the same graph.
- Stock Warp passes **97 tests with 10 expected skips** for upstream FSAI features.
  These include the two supplemental failure/recovery tests.
- The first full attempt had 2,934 passes, 11 skips and 14 stale metadata-test
  failures: a Cholesky test assumed triangular application could never specialize
  and supplied a fake device without architecture metadata. It now checks a truly
  unrelated operation. All 45 Cholesky tests pass after that update. The failed
  full transcript remains archived; no numerical gate was weakened.
- The isolated wheel contains **41 core modules**, each byte-identical to the
  tested source. SHA256: `2af40b5f96ab5c4b9dc13f28d4c19be3b591b18fd861740aee68d024b5115af2`.
- With benchmark/application/SciPy/CuPy/cuDSS imports forbidden, the wheel passes
  three public replays, sixteen projected outputs and nine wide generalized
  replays (640 unknowns, width 104, 40 modes). Wide tests change stiffness/mass
  on the same graph, retain constraints and repeated modes, and exercise both
  history policies and compact graphs. DOT inspection confirms the installed
  register kernel executes and no HOST/HtoD/DtoH nodes occur. Maximum absolute
  original residual is 2.06e-9; mass-orthogonality error is below 1.3e-15.

## Complete workflow controls

L40, pinned Warp/FSAI, four pinned Xeon 8362 cores, 2026-10-23. Each worker
constructs its own preconditioner and solver. Only independently verified CPU
reference arrays are cached outside backend timing. All current source, input,
dependency and reference identities match within each pair; continuous GPU
occupancy is clean. Graphs contain no host nodes or non-device-to-device copies.

`--triangular canonical` disables only this new dispatch; `default` uses it.
Both use the installed whitening and projected solvers. First-use compute is
setup + capture/eager preparation + the first captured solve; mesh assembly,
CPU references, imports, fingerprinting and host validation are excluded. Warm
medians use three later replays. Setup/capture are single cached-compilation
observations, not established setup improvements. Every first/later result must
pass original equations, requested spectrum/subspace, M-orthogonality, native
status/count/stopping and retained rigid modes.

| Problem | Triangular | Setup s | Capture s | First-use compute s | Warm median s | Iterations |
|---|---|---:|---:|---:|---:|---|
|[anvil](production/anvil-canonical-a1.json)|canonical|0.7852|0.6159|3.0035|1.6016|16 / 16 / 16|
|[anvil](production/anvil-default-a1.json)|default|0.7989|0.6016|3.0108|1.6021|16 / 16 / 16|
|[100](production/100-canonical-a1.json)|canonical|0.4393|1.4276|4.5067|2.5255|55 / 54 / 55|
|[100](production/100-default-a1.json)|default|0.4292|1.4330|4.4232|2.4391|55 / 54 / 54|
|[200](production/200-canonical-a1.json)|canonical|0.6354|3.9397|13.4404|8.4205|58 / 58 / 58|
|[200](production/200-default-a1.json)|default|0.6400|3.9575|13.2300|8.1651|57 / 58 / 58|
|[200-compact](production/200-compact-canonical-a1.json)|canonical|0.6567|1.7139|10.8729|8.4933|58 / 58 / 58|
|[200-compact](production/200-compact-default-a1.json)|default|0.6468|1.6917|10.6074|8.2119|58 / 58 / 58|
|[free](production-families/free-canonical-a1.json)|canonical|0.4713|1.5614|4.5759|2.3963|36 / 36 / 36|
|[free](production-families/free-default-a1.json)|default|0.4613|1.5636|4.5060|2.3429|36 / 36 / 36|
|[disconnected](production-families/disconnected-canonical-a1.json)|canonical|0.5178|1.8895|7.0275|4.4665|32 / 32 / 32|
|[disconnected](production-families/disconnected-default-a1.json)|default|0.5141|1.8941|6.9752|4.4101|32 / 32 / 32|
|[heterogeneous](production-families/heterogeneous-canonical-a1.json)|canonical|0.4296|1.4152|4.7275|2.7516|51 / 51 / 51|
|[heterogeneous](production-families/heterogeneous-default-a1.json)|default|0.4327|1.4163|4.6633|2.6894|51 / 51 / 51|

Warm gains: 3.4% for 100 modes, 3.0% for 200, 3.3% for compact 200,
2.2% for free, 1.3% for disconnected, and 2.3% for heterogeneous FEM. First-use
compute also improves in these wide controls, with all phase costs retained
above. The original normalized residuals remain below 1e-11; requested-space
error is below 5.7e-11. All six/twelve rigid modes remain in the free/disconnected
cases. Anvil's unchanged-path control is 1.6016/1.6021 s, with identical graph
node summaries and 16 outer iterations throughout; no Anvil speedup is claimed.
The README's historical six-backend Anvil table is not replaced by these controls.

This is a modest complete-solve improvement. The larger matched-preconditioner
CuPy gap (#147) remains a separate target. To reproduce, run
`benchmarks.experimental.compare_register_whole` with `--public-api`, an explicit
`--reference-cache`, and `--triangular canonical` or `default`; each adjacent
occupancy JSON retains the complete command. `--compact-graphs` selects the
separately qualified compact composition.
