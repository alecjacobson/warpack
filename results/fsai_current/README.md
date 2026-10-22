# FSAI dependency update qualification

Tracked in [#139](https://github.com/alecjacobson/warpack/issues/139). The shipped
fast-path pin remains `04427b997fe7ff07bc6ed1dde5ca9813496c6b55`. This report
qualifies an isolated candidate, not a dependency upgrade or new performance claim.

Candidate: [alecjacobson/warp at 429dccd](https://github.com/alecjacobson/warp/blob/429dccd0a133973c0809bdd974d81fcfc744919f/warp/_src/optim/fsai.py),
FSAI module SHA256 `584e0fb277ae78d0f6e382b551898831680275396aacbcfb10e948961e141c8c`.
The loader verifies this exact file and imports it under a separate module name;
it does not replace installed Warp modules. Qualification uses the existing
Warp 1.18 overlay on L40, 2026-10-22. Other dependency changes are not implied.

## Reproduced bug and compatibility fix

The pinned factor rejects refitting an unchanged SPD matrix when growth order
and sorted factor storage encounter different pivot safeguards. A 3-by-3
regression uses diagonal 1, A01=sqrt(0.0179), A12=0.99, A02=0 and pivot_floor=0.01.
Construction succeeds, but pinned `update()` raises. The candidate preserves
the growth order during refits and passes this case. This nondefault safeguard
case does not establish an error in the released Anvil results.

The candidate can defer support selection until the first captured update.
Its G factor then uses padded rows with device row counts. Warpack's row-major
application previously ignored those counts, and its column-major application
rejected padded factors. Both now use count-bounded kernels. Existing compressed
factor kernels and the public sparse operator's compressed-input contract are
unchanged. No new host synchronization is introduced.

## Validation

Eighteen candidate qualification tests pass: the nine cases below run against
both the unmodified candidate and its packed-workspace variant:

- The pinned failure above and the candidate's unchanged-factor refit.
- Row/column factor layouts with ordinary/deferred factors, captured update and
  CG, changed values, stable factor addresses and independent sparse equations.
- Invalid diagonal updates report failure in `update_status[1]`, preserve the
  preceding factor and recover on a later valid replay. Failed-update solves
  are not accepted.
- Lumped/consistent FEM mass with ordinary/deferred factors; changing K and M
  across two shape sequences, refitting inside the eigensolve graph, and checking
  complete requested spectra/subspaces and original physical accuracy.

The deferred CG test verifies that support selection first happens on replay.
BlockLOBPCG's initializer warmup selects its pattern before graph capture;
subsequent refits occur inside the captured solve. Graph audits find no HOST,
HtoD or DtoH nodes. DOT checks are not a full API trace.

The broader overlay run passes 138 tests, with two opt-in width-64 memory
comparisons skipped; a separate explicit opt-in run passes all five storage
patch tests. Eight padded-application tests cover changing row counts and values,
poisoned padding, aliasing, FP32/FP64 work and partial RHS tiles. Together with
32 existing compressed/coalesced tests, all 40 pass on the pinned overlay.
Stock Warp 1.15 passes the four column-layout padded tests; four row-layout
tests require the newer batched linear API and explicitly skip there.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
python -c 'from benchmarks.experimental.fsai_current import load; load(download=True)'
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 taskset -c 8-11 python -m pytest -q \
  tests/test_fsai_current.py tests/test_padded_factor_application.py \
  tests/test_coalesced_fsai.py
```

Without the explicit downloaded candidate, its tests skip. The source loader
rejects a different checksum. The candidate removes `apply_lanes`; the benchmark
adapter explicitly removes that option. Updating all production recipes and
installer provenance, broader upstream regression tests, matched setup/refit/
solve timing, and smaller-device qualification remain before changing the pin.
The [packed-workspace patch](fsai-packed-current.patch) preserves the candidate's
refit and deferred-update behavior in those same tests; its memory/timing on this
revision still needs separate measurement. Regularized refits are not enabled. Numerical testing during GPU contention
makes no timing claim.

Packed candidate SHA256: `aa4f526521c35f87ddfc1610f18bbda395784b636a4a8c844bce8f40ef4ab8b2`.

## Procedural FEM comparison attempt

Four fresh workers also solve the same 10,890-DOF consistent-mass beam as the
[original-pin memory experiment](../fsai_packed/README.md), requesting 20 modes
with row widths 32/64, current/packed setup, one warmup and three repeats.
All four pass native counts/status and physical residual, spectrum,
orthogonality and requested-subspace checks. Only `fem-current32` is
uncontended; a foreign GPU process appeared during the other three workers.
The raw JSON/log/occupancy files are retained, but do not form a matched timing
comparison. This does not justify changing the dependency pin.

```bash
python -m benchmarks.monitor_gpu --out build/current32.occupancy.json \
  --log build/current32.log -- taskset -c 8-11 \
  python -m benchmarks.experimental.fsai_current_fem \
  --row-size 32 --out build/current32.json
# Add --packed for the patched candidate, or use --row-size 64.
```

For isolated memory/compiler-property qualification of this revision:

```bash
python -m benchmarks.monitor_gpu --out build/current-memory.occupancy.json \
  --log build/current-memory.log -- python -m benchmarks.experimental.fsai_packed_storage \
  --current --width 32 --gpu packed --out build/current-memory.json
```
