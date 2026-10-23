# Library validation after paired projected updates

On2026-10-23, L40 and four Xeon8362 cores:

- **Full pinned suite:2,638 passed,11 skipped**,945.31seconds.
- **Affected tests:154 pass on both runtimes**:115.74seconds pinned,
  117.94seconds stock Warp.
- A clean wheel contains exactly39core modules matching the tested source.
  Isolated public changed-input replay passes while forbidding benchmark,
  application, SciPy, CuPy and optional cuDSS imports.
- A separate isolated installed-wheel projected check passes16changed/scaled
  replays of256/408-dimensional ordinary and compact solves against analytic
  tridiagonal spectra, full equations and orthogonality. This supplements the
  small public smoke; it explicitly exercises private projected components.

[Full suite](validation/installed-full-tests.log),
[pinned focused tests](validation/installed-focused-tests.log),
[stock focused tests](validation/installed-stock-tests.log),
[exact-source wheel](validation/wheel.json),
[public replay](validation/wheel-public-smoke.json),
[wide projected replay](validation/wheel-projected-smoke.json).

The installed update uses one thread per disjoint2×2 output block for existing
even global projections of dimension256–1024 on Ada89, with64threads/block.
Ordinary and compact graph policies share the same arithmetic. Round indices
come from immutable partner data, never the counter advanced by another lane.
All per-output FP64 expressions, pair order, symmetry restoration, final column
normalization, the30-sweep bound and2e-15 stopping criterion remain unchanged.
Other size/device paths retain their existing update. Kernel variants are
prepared during setup; initial preparation and pre-capture synchronization
order remain intact. No solver buffers, host decisions/readbacks, vendor calls
or synchronization were added to replay.

Installed-versus-legacy controls preserve bit-identical H,V,sweeps for ordinary
and compact schedules. They cover254/256dispatch boundaries,408/612/1024 sizes,
repeated/null spectra, scaling1e±80, zero/recovery, strides/sentinels, original
spectra/equations/orthogonality and graph transfer audits. Nonfinite bounded
replay and recovery are also covered. Full-suite coverage retains original
sparse equations, constraints, coupled mass, repeated/null modes, callbacks,
changed inputs and device-controlled stopping.

[Experimental whole/component qualification](runtime/README.md) preceded core
promotion. Installed complete-solve timing and broader-family controls are
separate and still being collected; #172 remains open until they qualify.
This validation supersedes the2,566-pass normalization checkpoint as the latest
full-suite result. It does not claim closure of the wider CuPy gap or all
upstream conformance issues.
