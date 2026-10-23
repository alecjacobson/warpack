# Exact procedural block-comparison inputs (#156)

New FEM/Hodge block-comparison records hash the actual device operator, mass,
and shifted operator after verifying them against the host validation pencil.
They also identify reference values, requested and null subspaces, excluded
guard values, and the Warp/CuPy starting columns. Identity mass is explicit.
The fingerprint downloads and hashing are outside setup/capture/replay timers.
Anvil retains its shared-fixture protocol.

`--initial-pool-width` lets different block widths take contiguous prefixes of
the same seeded pool; the producer verifies the exact prefix. Omitting it
preserves the previous starting values. The wide FEM regression summarizer now
rejects missing identities and comparisons with different input identities.
Older raw reports remain unchanged; use their historical checkout to reproduce
their configured-fixture summaries. Matching two absent input records never
certifies equal input bytes.

Targeted regression coverage: 71 passes on the pinned Warp 1.18 overlay;
69 passes and two upstream-preconditioner-dependent skips on stock Warp 1.15.
Tests cover scalar/block sparse coefficients, consistent/identity mass, changed
operator/mass/shift/reference/start rejection, common prefixes, empty nullspace
hashes, complete CLI runs and device-only graph checks. The existing Anvil
fixture tests exposed #157 on stock Warp: notify the sparse runtime after
replacing buffers so cached zero counts cannot overwrite the uploaded count.
The other direct fixture constructor, skinning stiffness, already does this.

Fresh matched baseline and candidate measurements are pending. No exact-input
equivalence is retroactively claimed for older procedural records.
