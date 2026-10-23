# Installed paired-update controls (#172)

All **32 complete first/later outputs pass** in matched legacy/installed
controls. L40, pinned Warp/FSAI, four Xeon 8362 cores, 2026-10-23. Each pair has
identical operators, initial vectors, references, source and dependency/version
identities. Native count/status/stopping and independent original-pencil,
requested-spectrum/subspace and orthogonality gates all pass. Internal and
continuous external occupancy are clean.

| Problem | Setup s, legacy / installed | Capture s, legacy / installed | First-compute s, legacy / installed | Warm median s, legacy / installed |
|---|---:|---:|---:|---:|
|100 modes, compact R/P history|0.4101 / 0.4154|1.4073 / 1.4146|4.6186 / 4.5843|2.6826 / 2.6643|
|200 modes, compact R/P history|0.6209 / 0.6147|4.5164 / 3.9337|14.6343 / 13.7336|8.9116 / 8.7447|
|200 modes, compact R/P history and compact projected graphs|0.6288 / 0.6343|1.6624 / 1.6663|11.3206 / 11.1980|9.0020 / 8.8279|
|Anvil, retained R/P history|0.7899 / 0.7789|0.6127 / 0.6160|3.0233 / 3.0769|1.6180 / 1.6685|

The installed 200-mode median reductions are **1.9% ordinary / 1.9% compact**,
consistent with the modest benefit in [prototype controls](../runtime/README.md).
This does not close the broader CuPy gap. Three measured replay samples follow
a separately validated first captured solve in each fresh worker.

All Anvil projected dimensions remain below the new dispatch guard. Its
16/17-step variability explains the observed difference: legacy16/16/16,
installed17/17/16. No Anvil speedup or regression is attributed to this change.
The ordinary 200-mode counts are58/58/58 versus57/58/58; compact58/58/57 versus
58/58/58. All samples and achieved errors are retained in adjacent reports.

Setup independently constructs the preconditioner and solver. First-compute
includes setup, capture and the first validated solve, excluding assembly,
independent CPU reference, imports and host validation with cached kernels.
Setup/capture are single observations; the ordinary capture difference is not
a claim of a new graph-construction optimization. These narrow controls do not
replace the six-backend Anvil showcase table.

Graphs have no host nodes or non-device-to-device copies. [Broader FEM checks](../families/README.md)
add12 accepted outputs, bringing installed qualification to **44**.
[Full suite and isolated-wheel validation](../VALIDATION.md) cover the core change.
