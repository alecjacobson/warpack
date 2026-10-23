# Complete exact-padding controls (#171)

All **24 complete first/later outputs pass**: first captured solve and three
measured replays per fresh worker. L40, pinned Warp/FSAI, four Xeon 8362 cores,
2026-10-23. Each canonical/padded pair has identical inputs, initial vectors,
reference data, source and dependency/version identities. Both internal and
continuous external occupancy are clean.

| Problem | Setup s, canonical / padded | Capture s, canonical / padded | First-compute s, canonical / padded | Warm median s, canonical / padded |
|---|---:|---:|---:|---:|
|100 modes, compact R/P history|0.4182 / 0.4443|1.4904 / 1.4500|4.6695 / 4.5313|2.6643 / 2.5071|
|200 modes, compact R/P history|0.6168 / 0.6166|3.9965 / 3.9736|13.8220 / 13.4010|8.7636 / 8.3614|
|Anvil, retained R/P history|0.7890 / 0.8049|0.6147 / 0.6139|3.0916 / 3.0594|1.6331 / 1.6366|

Warm replay improves about **5.9% / 4.6%** for100/200 modes. Outer counts remain
visible:100 canonical55/55/55 versus padded55/54/54;200 canonical58/58/58 versus
57/57/57. Both native and independent original-pencil, complete requested-spectrum/
subspace and orthogonality gates remain unchanged. The Anvil path is below the
candidate guard, with16steps for every measured replay and no speed claim.

On the last replay,100/200 modes use the wide certificate17/23times respectively,
with one early canonical fallback each. Smaller branches retain canonical
whitening. Anvil invokes the new helper zero times. Original physical rank and
padded zeros are preserved; auxiliary identity entries do not become modes.

Setup independently constructs the solver and preconditioner. First-compute
includes setup, capture and the first accepted solve; assembly, independent
reference work, imports and host checks are excluded, with cached kernels.
Setup/capture are single observations. [Verified CPU reference reuse](../../reference_cache/README.md)
reduces worker turnaround only: pairs use exactly the same independent reference
manifest, and no backend factor/preconditioner is shared. Anvil retains its own
fixture protocol.

Graphs have no host nodes or non-device-to-device transfers. All commands,
phases, raw samples, native/physical gates, reference-cache records and exact
identities are retained beside this report. [Broader paired families](../families/README.md)
add24 accepted outputs, for **48** total. These are prototype controls on
healthy SPD problems; #176 tracks a separately discovered failed-normalization
routing guard before promotion. The outer failure flag is not cleared by that
edge case, and none of these healthy controls exercises it.

The subsequently fixed [#176 guard](../README.md) rejects an already-failed
normalization before attempting the certificate. These healthy SPD controls
retain their original source identities; no performance claim is made for the
invalid-input routing fix.
