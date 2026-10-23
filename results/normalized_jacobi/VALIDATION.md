# Library validation after projected column normalization

On 2026-10-23, pinned Warp/FSAI runtime, L40 and four Xeon 8362 cores:

- **Full suite: 2,566 passed, 11 skipped**, 883.18 seconds.
- Affected suite: **243 passed**, 117.92 seconds.
- Stock Warp affected suite: **230 passed, 13 expected dependency skips**,
  107.37 seconds. Skips require upstream FSAI/batched-CG features absent from stock.
- A clean wheel contains exactly **39 core modules matching source**. Isolated
  installed-wheel public API replay passes after changed inputs while forbidding
  imports of benchmarks, examples, SciPy, CuPy and the optional cuDSS package.

[Full transcript](validation/full-tests.log),
[pinned focused transcript](validation/pinned-focused-tests.log),
[stock focused transcript](validation/stock-focused-tests.log),
[wheel source verification](validation/wheel.json),
[isolated wheel replay](validation/wheel-smoke.json),
[source/dependency identities](validation/source.json).

The installed global Jacobi path normalizes finite, nonzero eigenvector columns
once after the entire bounded loop for projected dimensions ≥256. Compact
projected graphs inherit the same finalization. Matrix/eigenvalues, rotations,
symmetry restoration, the 30-sweep cap and 2e-15 stopping criterion are unchanged.
Smaller and odd projected paths retain their implementation. The kernel adds
no solver buffers or host decision, readback, vendor call or synchronization.
Initial preparation and pre-capture synchronization/callback order are unchanged.

The formerly failing 1,024-dimensional repeated/null regression now passes the
same 1e-11 full-basis orthogonality gate; its strict-xfail marker is removed.
Ordinary/compact controls preserve identical H and sweep counts against the
explicit legacy path. Tests cover the 254/256 boundary, 612/1024 dimensions,
strided buffers/sentinels, scaling by 1e±80, zero/nonfinite/changed replay, original
equations, complete spectra and graph host-transfer audits. Invalid columns are
preserved for existing failure detection rather than replaced with valid-looking
vectors. Independent column-norm and off-diagonal diagnostics distinguish the
repaired drift from an arbitrary loss of orthogonality.

Complete-suite coverage also includes original sparse equations, requested
spectra/subspaces, coupled mass, constraints, repeated/null modes, changed inputs,
callbacks and graph replay. The isolated wheel smoke exercises a small public
solve; byte-for-byte source verification connects the packaged wide kernel to
the dedicated large-dimension regressions.

This supersedes the 2,445-pass symmetry-recovery baseline as current full-suite
evidence. Performance controls are separate; this does not claim complete
CuPy/Spectra conformance or closure of the other open issues.

[Installed production controls](production/README.md) and
[broader FEM families](families/README.md) now qualify **44 complete first/later
outputs**, including ordinary and compact projected graphs. All source and
input identities and native/independent gates were checked; all timing workers
were uncontended. This completes #174 qualification without changing thresholds.

The subsequent [paired-update validation](../paired_jacobi/VALIDATION.md) records
2,638 full-suite passes and supersedes this as the latest complete test run.
