# Installed normalization controls (#174)

The installed fix passes all **32 complete outputs** in these paired controls:
first captured solve plus three measured replays in each fresh process. L40,
pinned Warp/FSAI, four Xeon 8362 cores, 2026-10-23. Every pair has identical
matrices, initial vectors, reference spectra/subspaces, source and dependency
identities. Both internal and continuously monitored occupancy are clean.

| Problem | Setup s, legacy / installed | Capture s, legacy / installed | First-compute s, legacy / installed | Warm median s, legacy / installed |
|---|---:|---:|---:|---:|
|100 modes, compact R/P history|0.4068 / 0.4384|1.3943 / 1.4402|4.5798 / 4.6730|2.6622 / 2.6826|
|200 modes, compact R/P history|0.6092 / 0.6043|4.0007 / 3.8784|13.9505 / 13.8017|8.9297 / 8.9133|
|200 modes, compact R/P history **and compact projected graphs**|0.6120 / 0.5982|1.6557 / 1.6622|11.2836 / 11.3500|8.9949 / 8.9657|
|Anvil, retained R/P history|0.7929 / 0.7940|0.6066 / 0.6140|3.0938 / 3.1005|1.6850 / 1.6445|

These qualify an accuracy fix, not a speedup. The legacy control disables only
final column normalization; rotations, symmetry restoration, sweep budget and
stopping criterion remain identical. Small projected paths are unchanged.
Anvil's ordinary 16/17-iteration variation remains: legacy replay counts
16/17/17, installed 17/16/16. Its median difference is not attributed to the fix.

Setup includes independent preconditioner and solver construction. First-compute
includes that setup, graph capture and the validated first captured solve.
Assembly, independent CPU references, imports and host validation are excluded;
kernels are cached. Setup and capture are single observations per worker, not
stability claims. These narrow Warp controls do not replace the six-backend
Anvil showcase table. Commands, exact inputs, all samples, gates and source
identities are in the adjacent JSON and occupancy reports.

Every output passes native count/status/stopping and independent original-pencil,
orthogonality, requested-spectrum and subspace checks. The synthetic cases have
worst normalized residual below 1e-11. Anvil retains its own original-residual
and MAC/rigid-subspace gates; its worst original residual is 1.89e-8. These
residual formulas differ and should not be compared as one metric. Graph audits
show no host nodes or non-device-to-device copies in these pure-Warp solves.

[Broader FEM families](../families/README.md) add 12 accepted outputs, bringing
the installed-fix qualification to **44**. [Full validation](../VALIDATION.md)
records 2,566 passing tests, both Warp runtimes and installed-wheel verification.
