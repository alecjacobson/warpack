# General residual-directed restart

The public solver fixes [#35](https://github.com/alecjacobson/warpack/issues/35)
by expanding each restarted basis from an unfinished requested Ritz residual.
The exact Spectra fixtures retain their
matrices, starting policy, k20/ncv50, requested tolerance `1e-12`, and original
300 adaptive / 30 fixed cycle budgets. Historical failures remain failures.

## Two stalled finite-dimensional searches

The new direct `ComplexEigs_16` baseline returns 19/20 on all three replays,
exhausting 300 cycles at normalized original residuals `1.13e-12`–`1.48e-12`.
An instrumented 400-cycle control still returns 19/20. Its residual has already
flattened near `1.30e-12` by cycle 100. Recomputing the eigenvalue as a Rayleigh
quotient does not improve it. The last projected solve has maximum normalized
residual `2.10e-14`, and sparse-product differences from a CPU product have
norm about `3e-14` before eigenvalue normalization. Neither explains the larger
outer plateau.

The last retained guard vector is accurate to about `1.15e-14` relative, while
the last requested mode is only accurate to `1.30e-12`. Expanding from that
guard can break down; replacing the missing direction with the same random
completion every cycle can leave the unfinished requested direction unresolved.

For shifted `GenEigsRealShift_12`, the original 30-cycle profile remains close
to its physical tolerance. Three current baseline replays happen to pass, but
the first recovery prototype still gives 20/20, 20/20, 19/20. Increasing the
unchanged algorithm to 40 cycles leaves the same plateau, with one physical
residual `9.9997e-13`. Projected residuals are at most `3.56e-17`; the cached
projection differs from an independent dense inverse projection by only
`2.76e-17` in Frobenius norm. Recovery restricted to numerical breakdown does
not help this case: its guard directions are still changing. The preserved
near-threshold controls are not a robust count fix.

## Public residual-directed restart

`GeneralEigensolver` retains the selected Ritz/guard vectors, but uses the
largest normalized requested Ritz residual as the first new search direction
after restart. Cached operator images provide that residual; this adds **no
operator or inverse applications**. The candidate is normalized and projected
against the growing basis. If it lies in that basis or is invalid after
projection, random completion is retained so a zero row cannot enter the basis.
The same candidate is available for later breakdowns. Initialization resets
the candidate and counters on device for each captured replay.

Residual selection is a search heuristic, not a new convergence test. In
particular, a tiny inverse-problem residual can still correspond to an
unfinished physical shifted problem. Existing fresh final residuals, counts,
status, requested-spectrum assignment and vector-norm gates remain authoritative.

Three warmed, synchronized replays of the public implementation pass both
exact cases, including all physical and transformed convergence counts,
status, original residual, requested-spectrum and vector-norm gates:

| Case | Executed cycles | Count | Max normalized original residual | Median replay (s) |
|---|---:|---:|---:|---:|
| Direct SR | 79–80 | 20/20 | `9.55e-13` | 17.4965 |
| Shifted nearest, sigma 100 | 30 | 20/20 | `2.17e-14` | 6.6888 |

The failed original direct solver is **not an accepted timing baseline**.
Separate matched controls at `tol=1e-10` pass before and after the change:
direct median 15.8926 → 15.5151 s (ratio 0.9762), shifted 6.6464 → 6.6874 s
(ratio 1.0062). Matrix, start, budgets, runtime and acceptance gates match.
These are regression checks, not a claim of a major general-solver speedup.

The full pinned suite passes **1,143 tests**; the general subset passes **95**
on stock Warp 1.15. New tests cover manufactured normal/nonnormal stalled
searches at three scales, changed captured starts, unusable candidates,
all six selectors after complete-space convergence, and trace-hook restoration.
The direct graph contains no host nodes or host transfers. The shifted graph
has no host nodes or device-to-host copies, but retains cuDSS's 6,172 vendor
host-to-device metadata copies ([#29](https://github.com/alecjacobson/warpack/issues/29)).

## General performance remains separate

A three-sample instrumented eager restart takes 221–223 ms. The serial complex
Schur kernel accounts for 198.6–201.1 ms, about 90% of that kernel time; general
CSR products account for about 18.4 ms in the first sample. This establishes
the projected solve as a bottleneck for this fixture under #14. Event overhead
and the different instrumentation scope prevent interpreting these as full
accepted solve timings. This change does not optimize the Schur kernel.

## Reproduction and evidence

Prepare the pinned Spectra fixtures with the normal exporter:

```bash
python -m benchmarks.upstream.prepare_spectra \
  --families ComplexEigs GenEigsRealShift --max-size 1000
python -m benchmarks.experimental.general_count_probe \
  --case ComplexEigs_16 --out build/direct.json --repeats 3
python -m benchmarks.experimental.general_count_probe \
  --case GenEigsRealShift_12 --out build/shift.json --repeats 3
```

The original prototype and plateau diagnosis are preserved at commit `afaba1b`;
use that revision to reproduce the old failures. `--profile`
instruments one eager restart after the final replay. Trace hooks run during
setup/capture only; the graph contains device copies and kernels, not Python
callbacks. Use separate processes and run GPU work sequentially. Dense
reference/Rayleigh diagnostics are outside the production solve.

- [Public direct](../../results/general-counts-direct-public.json) and
  [public shifted](../../results/general-counts-shift-public.json) repeated results.
- [Matched performance controls](../../results/general-counts-performance-regression.json).
- [Validation](../../results/performance_followup/general-restart-validation.json)
  and [clean wheel](../../results/performance_followup/general-restart-wheel.json).
- [Direct baseline](../../results/counts-direct-baseline.json) and
  [400-cycle trace](../../results/counts-direct-trace.json).
- [Shifted baseline](../../results/counts-shift-baseline.json),
  [40-cycle control](../../results/counts-shift40-baseline.json),
  [shifted projection trace](../../results/counts-shift-trace.json).
- [Conditional direct recovery](../../results/counts-direct-recovery.json),
  [thresholded shifted recovery](../../results/counts-shift-recovery.json),
  [unthresholded](../../results/counts-shift-recovery-unthresholded.json),
  [all-breakdown control](../../results/counts-shift-recovery-all.json).
- [Directed direct trial](../../results/counts-direct-directed.json) and
  [directed shifted trial](../../results/counts-shift-directed-corrected.json).
- [Kernel profile](../../results/counts-kernel-profile.json).

The diagnostic CLI records the actual public core class and source hashes.
An intermediate local prototype CLI wiring mistake
selected the public shifted core despite its prototype flag; that run is not
used as evidence for residual-directed iteration. The standalone earlier
recovery/trace scripts used the intended classes; this mistake did not affect
their results or the installed library. The prototype module was removed after
promotion to the public solver; its historical source remains at `afaba1b`.
