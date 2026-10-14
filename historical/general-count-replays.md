# General-solver count replays, 2026-10-14

Follow-up to [#35](https://github.com/alecjacobson/warpack/issues/35), after PR #33.
These diagnostics preserve the original full-run failures; no favorable replay
replaces those rows. Production numerical code is `e0a4f58` (now on main).
All numerical kernels remain Warp except optional cuDSS factor solves.

| Exact Spectra fixture | Captured replay counts | Max absolute residual range | Upstream bound |
|---|---|---|---|
| ComplexEigs_16 | 19, 19, 19 | 2.71e-12–3.17e-12 | 1e-9 |
| GenEigsRealShift_12 | 20, 20, 19 | 1.64e-12–2.43e-12 | 1e-8 |

All replays start from the same device-generated initialization. The direct
solver uses 300 adaptive cycles and the shifted solver 30 fixed cycles.
The shifted case shows count variability on the same captured graph at the
requested internal tolerance `1e-12`. Its final replay's largest normalized
residual is `1.006e-12`, just above that threshold. The direct case stayed at
19/20 throughout this diagnostic, with worst normalized residual about
`1.5e-12`; this run does not establish its count variability. Neither fixture
checks requested-spectrum selection in the generic upstream assertion.

[Direct data](../results/refinement_direct_replays.json) and
[shifted data](../results/refinement_shift_replays.json) include every replay,
individual normalized residuals and actual cycles. Times include the first
replay; this is an accuracy diagnostic, not a matched-accuracy performance
comparison. The direct solver does not call cuDSS, so its result cannot simply
be attributed to inverse correction. The old baseline and current run are not
a controlled source-version A/B experiment.

Reproduce after preparing the pinned fixtures/environment:

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.upstream.replay_probe --case ComplexEigs_16 --repeats 3 \
  --out /tmp/direct-replays.json
python -m benchmarks.upstream.replay_probe --case GenEigsRealShift_12 --repeats 3 \
  --out /tmp/shift-replays.json
```

The recorded run used the same numerical sequence in an initial local version
of this runner before it was wrapped in `main()` and given explicit CLI case
choices and environment metadata. No production fix is claimed; #35 remains
open. Do not relax the tolerance or count the best replay only.
