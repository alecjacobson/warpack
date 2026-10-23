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

## Fresh matched qualification

Three uncontended workers on the L40 / four pinned Xeon threads solve the
34,560-DOF consistent-mass cantilever for 100 modes with starting width 104.
All three workers have identical actual operator, mass, shifted operator,
reference/guard/subspace and starting-pool identities, identical source hashes
and dependencies, and pass every requested-mode, physical and native stopping
gate in all three measured repeats per solver (18 measured solves total).

| Projected factor | Warp setup + capture (s) | Warp warm median (s) | CuPy warm median (s) |
| --- | ---: | ---: | ---: |
| Forced old serial | 4.5982 | 3.7083 | 3.5725 |
| Production cooperative | 4.2403 | 3.1820 | 3.6316 |
| Production repeat | 4.2843 | 3.2353 | 3.8180 |

Only the factor selector changes in the first comparison. Production reduces
Warp warm time by 14.2%; its independent repeat differs by 1.7%. CuPy, which
does not use this factor, varies by 1.7% then 5.1%. These are warm solve times;
setup/capture is a single construction per worker and warmup is recorded
separately. CuPy uses the same supplied Warp FSAI/CG preconditioner and has an
additional adapter/upload cost of 0.084–0.098 s, not standalone setup. Warp
requires normalized residual 1e-11 on the requested 100 columns; CuPy requires
absolute residual 1e-10 on all 104 starting columns, in addition to the common
physical checks. Padding and stricter-tolerance controls are separate work.

The [comparison record](controls/factor-comparison.json) verifies exact inputs
and the intentionally changed factor selector. The strict repeat regression
gate passes without any configuration exception. Each worker has raw,
occupancy and validated summary records:
[serial](controls/wide-serial-a1.json),
[production](controls/wide-default-a1.json),
[repeat](controls/wide-repeat-a1.json), and
[environment](controls/environment.json).
No exact-input equivalence is retroactively claimed for older procedural records.
