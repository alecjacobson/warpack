# Register-blocked triangular recurrence (#173)

The guarded installed implementation is now covered by the
[full-suite, wheel and fresh complete-solve qualification](VALIDATION.md).

## Independent prototype qualification (28ee6ee)

Eight interleaved serial right-hand-side rows reuse each earlier solved entry in
registers. Every row retains the original subtraction order; there is no triangular
inverse, matrix-product regrouping, shared scratch/barrier or new global workspace.
Only triangular application is replaced, at widths 96–256 in the whole-solve
control. Both complete QR passes, initial scaling, cached AX updates and final MX
remain. All other kernels, settings and acceptance gates are identical.

This is a different approach from the [rejected tiled/shared experiments](../triangular_rotate/README.md).
The independent prototype remains outside the installed package for reproducible
controls. This section records its pre-promotion results at commit `28ee6ee`;
installed regression, package and performance evidence is linked above.

**14 pinned kernel/stage tests and 12 pinned FEM tests pass.** The kernel checks
require bitwise equality with the original recurrence on widths 7–256, changed
inputs/scales (including zero), NaN-poisoned unused upper factors and strided
sentinel storage. Large even identity solves exercise 70,656 rows. Full-stage
checks include coupled/scaled mass and cached images; FEM checks include five
families, both history policies and changed constrained clustered pencils. The
14 kernel/stage tests also pass stock Warp (the combined stock transcript includes
18 additional shared-recurrence tests). [Kernel](validation/kernel-tests.log),
[FEM](validation/fem-tests.log), [stock](validation/stock-tests.log).

All **96 component outputs and 56 complete first/later outputs pass** with current
source/input/dependency/reference identities and clean occupancy. Kernel-level
bitwise comparisons do not imply identical full-solver trajectories: atomic Gram
reductions and stopping can produce small iteration differences. No threshold
is relaxed. Each whole-solve worker constructs its own preconditioner and solver;
only independently verified CPU reference arrays are cached outside backend
timers. Graph inspection finds no host nodes or non-device-to-device transfers.

### Component controls

| Rows | Width/input | Original stage ms | Register stage ms |
|---:|---|---:|---:|
|34560|104/random|6.544|5.519|
|34560|104/near_orthogonal|6.541|5.510|
|34560|120/random|8.686|7.285|
|34560|120/near_orthogonal|8.703|7.288|
|34560|204/random|26.212|21.519|
|34560|204/near_orthogonal|26.211|21.599|
|70656|120/random|14.753|14.232|
|70656|120/near_orthogonal|14.745|14.229|

[Small raw control](../triangular_rotate/register-a1.json),
[large raw control](../triangular_rotate/register-large-a1.json).

### Complete solves

L40, pinned Warp/FSAI, four pinned Xeon 8362 cores. Seconds; warm median of three
replays, with a separately checked first solve. First-use compute includes setup,
capture/eager preparation and the first captured solve. Setup/capture are single
observations with cached compilation, not established setup improvements.

| Case | Triangular | Setup s | Capture s | First-use compute s | Warm median s | Iterations |
|---|---|---:|---:|---:|---:|---|
|[100](whole/100-canonical-a1.json)|canonical|0.4407|1.4423|4.4948|2.5045|55 / 54 / 54|
|[100](whole/100-register-a1.json)|register|0.4489|1.4195|4.4280|2.4381|54 / 54 / 54|
|[200](whole/200-canonical-a1.json)|canonical|0.6314|3.9606|13.4529|8.3878|58 / 58 / 58|
|[200](whole/200-register-a1.json)|register|0.6346|3.9901|13.2439|8.1355|58 / 58 / 57|
|[200-compact](whole/200-compact-canonical-a1.json)|canonical|0.6504|1.6789|10.9056|8.4667|57 / 58 / 58|
|[200-compact](whole/200-compact-register-a1.json)|register|0.6526|1.6741|10.6299|8.1609|58 / 57 / 57|
|[free](families/free-canonical-a1.json)|canonical|0.4696|1.5694|4.5693|2.4031|36 / 36 / 36|
|[free](families/free-register-a1.json)|register|0.4506|1.5546|4.4777|2.3381|36 / 36 / 36|
|[disconnected](families/disconnected-canonical-a1.json)|canonical|0.5081|1.8753|6.9915|4.4550|32 / 32 / 32|
|[disconnected](families/disconnected-register-a1.json)|register|0.5118|1.8898|6.9622|4.4131|32 / 32 / 32|
|[heterogeneous](families/heterogeneous-canonical-a1.json)|canonical|0.4305|1.4142|4.7104|2.7580|51 / 51 / 51|
|[heterogeneous](families/heterogeneous-register-a1.json)|register|0.4472|1.4315|4.6914|2.6986|51 / 51 / 51|

Warm gains are 2.7% for 100 modes, 3.0% for 200, 3.6% for compact 200,
2.7% for free, 0.9% for disconnected and 2.2% for heterogeneous FEM.
The disconnected gain survives a reverse-order pair: **4.4613 → 4.3885 s (1.6%)**,
with all eight first/later outputs accepted, 32 iterations throughout and all
twelve rigid modes retained. [Canonical](recheck/disconnected-canonical-a1.json),
[register](recheck/disconnected-register-a1.json). Raw JSON/log/occupancy files retain
individual samples, full commands, original physical checks and provenance.

These are modest whole-solve gains, not closure of the larger matched-preconditioner
CuPy gap (#147). First-use results and iteration differences are retained above.
The historical README Anvil six-backend table is not replaced by these controls.
