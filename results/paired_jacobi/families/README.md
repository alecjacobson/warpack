# Broader FEM checks for installed paired updates (#172)

All **12 complete first/later outputs pass** with the installed path: one first
captured solve plus three measured replays for each family. These are current
accuracy checks and observations, not paired speed comparisons. L40, pinned
Warp/FSAI, four Xeon8362 cores,2026-10-23; compact R/P history and ordinary
projected graphs. Internal and continuous external occupancy are clean.

| Family | Unknowns | Modes | Setup s | Capture s | First-compute s | Warm median s | Worst normalized residual | Worst requested-subspace error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Free body|35,328|100 positive +6 rigid|0.4330|1.5402|4.6529|2.5530|9.32e-12|2.80e-11|
|Disconnected bodies|70,656|100 positive +12 rigid|0.5052|1.8931|7.1304|4.5842|1.00e-11|2.60e-11|
|Heterogeneous material|34,560|100 positive|0.4206|1.4219|4.8357|2.8868|9.96e-12|3.33e-11|

Each first/later output passes native counts/status/stopping, original equations,
orthogonality, complete requested spectra/subspaces and expected rigid nullity.
Graphs have no host nodes or non-device-to-device copies. Adjacent reports retain
all samples, commands, exact input/source/dependency identities and occupancy.

Setup includes independently constructed preconditioner and solver. First-compute
also includes capture and the first accepted solve, excluding assembly,
independent reference, imports and host checks with cached kernels. Setup/capture
are single worker observations. Together with [paired controls](../production/README.md),
these complete **44** accepted outputs for the installed update.
