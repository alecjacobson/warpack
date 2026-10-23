# Broader exact-padding controls (#171)

All **24 complete first/later outputs pass**, with matching canonical/padded
input, start, reference, source and dependency identities. L40, pinned Warp/FSAI,
four Xeon8362 cores,2026-10-23; compact R/P history, ordinary projected graphs,
three measured replays and one separately checked first captured solve per
fresh worker. Internal and continuous external occupancy are clean.

| Family | Unknowns | Modes | Setup s, canonical / padded | Capture s, canonical / padded | First-compute s, canonical / padded | Warm median s, canonical / padded |
|---|---:|---|---:|---:|---:|---:|
|Free body|35,328|100 positive +6 rigid|0.4340 / 0.4503|1.5529 / 1.5766|4.6630 / 4.5726|2.5489 / 2.3977|
|Disconnected bodies|70,656|100 positive +12 rigid|0.5120 / 0.5388|1.8837 / 1.9542|7.1341 / 7.0833|4.6028 / 4.4446|
|Heterogeneous material|34,560|100 positive|0.4139 / 0.4290|1.3898 / 1.4268|4.8024 / 4.7232|2.8892 / 2.7490|

The observed median replay reductions are **5.9%,3.4%,4.9%** respectively.
Outer counts: free36throughout; disconnected32/32/32 versus32/33/32;
heterogeneous51throughout. All first/later outputs recover the expected rigid
nullity and pass native status/count/stopping plus independent original equations,
complete requested spectra/subspaces and mass orthogonality. Worst normalized
residual stays below1e-11. No tolerance or acceptance gate is loosened.

Each last candidate replay records17wide certificates and one early canonical
fallback. Original inactive masks/ranks remain physical; the separate auxiliary
identity is only a certificate device. Both projection/whitening passes remain.
Graphs have no host nodes or non-device-to-device copies.

Setup, capture and first-compute use the same scope as the [primary controls](../whole/README.md).
Verified independent CPU references may be reused outside timing; each worker
still constructs its own backend/preconditioner. Setup/capture are single
observations, not a significance claim. Adjacent reports retain all samples,
commands, quality gates and source/reference provenance. Together these qualify
**48** complete outputs; failed-normalization routing (#176) is a separate
review finding to resolve before core promotion.

The subsequently fixed [#176 guard](../README.md) rejects an already-failed
normalization before attempting the certificate. These healthy SPD controls
retain their original source identities; no performance claim is made for the
invalid-input routing fix.
