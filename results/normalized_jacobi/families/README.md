# Broader installed normalization checks (#174)

All **12 outputs** pass: first captured solve and three measured replays for
each family. These are accuracy checks and current observations, not paired
speed comparisons. L40, pinned Warp/FSAI, four Xeon 8362 cores, 2026-10-23;
clean internal and continuous external occupancy. Compact R/P history,
ordinary projected graphs, unchanged native and independent acceptance gates.

| Family | Unknowns | Returned modes | Setup s | Capture s | First-compute s | Warm median s | Worst normalized residual | Worst requested-subspace error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Free body|35,328|100 positive + 6 rigid|0.4359|1.5746|4.7175|2.5778|8.23e-12|2.38e-11|
|Disconnected bodies|70,656|100 positive + 12 rigid|0.5009|1.8845|7.1649|4.6408|8.83e-12|2.42e-11|
|Heterogeneous material|34,560|100 positive|0.4167|1.4149|4.8562|2.9082|9.92e-12|3.32e-11|

All first/later outputs recover the expected rigid nullity and pass complete
requested-spectrum/subspace, original-pencil, orthogonality, native count and
status checks. No host graph nodes or non-device-to-device transfers occur.
Exact input, source/dependency identities, commands, raw samples and occupancy
are retained in adjacent reports. Setup includes independently built
preconditioner and solver; first-compute also includes capture and the first
validated solve, excluding assembly/reference/imports/host validation with
cached kernels. Setup/capture are single worker observations.

Together with [paired production controls](../production/README.md), these
complete 44 accepted outputs for the installed accuracy fix.
