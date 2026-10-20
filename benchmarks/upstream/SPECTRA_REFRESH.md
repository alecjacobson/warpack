# Complete original Spectra numerical-fixture refresh

All **360 exported cases** have been rerun at the adapter's original widths,
budgets and tolerances. The frozen run records **170 passes, 14 failures and
176 unsupported cases**, with no timeouts. All 118 original passes remain;
52 former failures/timeouts now pass. No previously passing checked numerical
bounds regress, no case is missing, and no success label contradicts the
recorded assertions/counts/status.

This is an accuracy inventory, not a performance comparison or a claim of full
upstream conformance. Generic upstream assertions do not always check the
requested spectrum; `requested_spectrum_checked` identifies where the runner
adds that check. Independent large-general spectrum coverage remains separate
in [GENERAL_SEARCH](../experimental/GENERAL_SEARCH.md) and issue #104.

## Remaining outcomes at these original profiles

| Outcome | Cases |
|---|---|
| Compact shifted SM nonconvergence | `SymEigsShift_2`, `_7` |
| Compact general SM numerical failures | `GenEigs_3`, `ComplexEigs_21` |
| Narrow Davidson LA absolute-bound failures | `DavidsonSymEigs_0`, `_2` |
| External numerical bounds pass, internal count short | `SymEigsShift_17`, `GenEigsRealShift_18`, `_30`, `ComplexEigs_3`, `SVD_1`, `Example1Shift_1`, `_3`, `_5` |

The eight count-only failures stay rejected. For example, the three shifted
examples have original residuals below 2.4e-16 but only 2/3, 4/5 and 5/6
inverse-operator counts at their internal tolerance. An excellent physical
residual does not silently replace a declared strict-count condition.

Explicit wider/longer profiles in [SHIFTED_SELECTION](SHIFTED_SELECTION.md),
[GENERAL_SM](GENERAL_SM.md) and [DAVIDSON_SVD](DAVIDSON_SVD.md) retain their own
accepted evidence. They do not turn these unchanged-profile failures into
passes. All 30 ordinary symmetric and all 30 Hermitian original fixtures pass.
The unchanged native Spectra suite separately passes all 31 executables; some
native tests expressly allow nonconvergence, so executable success is not a
claim that every native matrix case converged.

## Protocol and provenance

The frozen source is `90c19200e9443ffdd084e98ef8462d92104e48d6` on
`fix/upstream-width-contracts`, with pinned Spectra
`db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95`, Warp1.18 and the L40/570.158.01.
This predates the optional SVD-start addition; its default mathematical path
is the one actually fingerprinted. No later candidate changes entered this
checkout. Each whole worker has a 120-second cap including setup/capture and
validation. Bulk exported matrices are hashed and deleted after each completed
family to keep disk use bounded; they can be regenerated from the pinned source.

Each case launches two graph replays, but **this frozen harness validates only
the final replay**. It must not be read as evidence that both returned outputs
were independently checked. [Issue #122](https://github.com/alecjacobson/warpack/issues/122)
tracks per-replay validation. All 184 evaluated final outputs record zero device
status; status checks were added in #118 and happen outside capture.

A scratch scheduling overlap affected portions of the first general-real-shift
and ordinary-symmetric runs. Both complete families (66 cases) were subsequently
rerun alone, after checking that regenerated matrix hashes matched. The published
inventory contains only those serial replacement rows. No timing measurements
were collected. The serial replacement report is retained separately.

[Issue #121](https://github.com/alecjacobson/warpack/issues/121) tracks an adapter
coverage gap found during this audit: 27 generalized LM/LA/SA fixtures can use
an already supported public API but still appear unsupported in this baseline.
The [completed follow-up](SPECTRA_REGULAR.md) enables all 27 successfully,
checks every replay, and remains separate from this frozen baseline.

## Evidence and reproduction

- [Complete 360-case report](../../results/performance_followup/upstream-refresh-spectra-warp.json)
- [Original-to-current assertion comparison](../../results/performance_followup/upstream-refresh-spectra-comparison.json)
- [Source, environment and protocol manifest](../../results/performance_followup/upstream-refresh-spectra-manifest.json)
- [Fixture sizes and SHA-256 hashes](../../results/performance_followup/upstream-refresh-fixture-hashes.json)
- [Complete serial replacement families](../../results/performance_followup/upstream-refresh-spectra-serial-controls.json)

```bash
python -m benchmarks.upstream.prepare_spectra --out build/spectra-refresh-fixtures
python -m benchmarks.upstream.bounded spectra \
  --fixtures build/spectra-refresh-fixtures --accuracy-only --timeout 120 \
  --out build/spectra-refresh.json
```

Use the pinned environment in the main README and the recorded frozen source
to reproduce this exact protocol. Export/run one family at a time when disk
space is limited. Run numerical GPU jobs serially; retain failures and unknown
contracts rather than relabeling them as successes.
