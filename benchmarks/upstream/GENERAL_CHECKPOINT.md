# Preserving completed general eigenpairs during fixed budgets

`GeneralEigensolver` and the general path of `ShiftInvertEigensolver` now keep
a complete converged result while a fixed restart budget continues searching.
This repairs the lost-output behavior in #108. At explicit widths 40/60 and
80 fixed cycles, all 20 original Spectra general benchmark matrices pass all
60 reset replays. This is separate from the compact adaptive profile's known
requested-spectrum failures in #104.

## Behavior and limits

Fixed `solve()` and fixed `capture()` enable preservation by default. Set
`preserve_converged=False` in the constructor to reproduce the prior policy.
Adaptive graphs continue to use their existing stopping/selection rule and do
not preserve snapshots. The Hermitian shift-invert path is unchanged.

A candidate must contain all k accepted pairs. The device compares the sum of
the requested LM/SM/LR/SR/LI/SI selection scores; later candidates can improve
the saved set. Resolved score improvements are compared using observed residuals
and floating-point slack, not the potentially much looser requested tolerance.
Within that slack it prefers the smaller worst inverse/operator residual.
This is a comparison heuristic, not a rigorous nonnormal eigenvalue-error bound
or a certificate that all requested modes have been discovered.

The direct solver uses operator images already computed in the current cycle
to assess a candidate. The shifted solver additionally converts candidate
inverse eigenvalues to physical eigenvalues and evaluates fresh **original
operator products** before saving. An inverse Ritz residual alone cannot
qualify a shifted candidate. Original-equation acceptance can differ from the
inverse core's convergence count, as in the existing physical result contract.

At finalization, the saved vectors and values receive fresh operator products
and the ordinary residual/count checks. The working projected matrix, spectrum,
ordering and search basis remain intact. Failed final physical equations,
invalid starts and projected-QR failure are not converted into successful
results. Check `status`, `converged`, and original residuals as before.

One k-by-n complex vector snapshot plus small scalar/value/order arrays is
allocated before the first fixed solve/capture. Every replay resets saved state;
changed matrices or starts cannot inherit an old successful result. The decision,
copies, residual calculations and resets are Warp operations on device. There
are no new host convergence reads or numerical library calls. Shift-invert still
uses the selected cuDSS/GMRES inverse. cuDSS's vendor HtoD metadata nodes remain
tracked separately in #29.

The application-specific damped-coordinate adapter explicitly retains its old
opt-out policy: its balancing, polishing and velocity recovery require a
matching candidate guard before using snapshots. This does not change its
published numerical configuration.

## Spectrum and correctness evidence

[Initial complete wide audit](../../results/performance_followup/public-checkpoint-wide-initial.json):
original matrices, original v0, k=10, tol=1e-10, n=100/ncv=40 or n=1,000/ncv=60,
80 fixed cycles and three replays each. All 20 cases pass original equations,
complete target assignment including required boundary modes, vector norms,
numerical independence, invariant subspaces, status and full counts. No timing
claim is made for this audit. The later ownership/counter repair changes no
numerical checkpoint kernel or selection criterion.

New public regressions cover every selector, changed problems on one graph,
partial-space preservation under extra restarts, fresh final rejection after a
matrix change, lazy allocation and opt-out, an existing adaptive graph after a
fixed graph is created, resolved clustered improvements, and rejection of a
wrong physical pair even when its supplied inverse Ritz residual is zero.
Existing start/budget/shift/replay tests remain applicable.

Two additional bookkeeping problems were repaired during review:

- #113: ordinary `solve()` left the device iteration counter at zero or at an
  earlier captured count. An 8x8 reproduction observed 0/3/3 after ordinary
  two-cycle solve / fixed-three replay / ordinary one-cycle solve. The repaired
  counter reports 2/3/1, including through general shift-invert.
- #114: the unmerged checkpoint candidate retained owners through reference
  cycles. State now receives its solver only for each call; the shifted guard
  binds operator/array data rather than a bound owner method. Direct and shifted
  solver deletion is tested without a cyclic-GC pass. No unbounded leak or OOM
  claim is made.

[Reproduction and focused bookkeeping validation](../../results/performance_followup/public-checkpoint-bookkeeping.json)
retain the failure evidence and 53 passing focused checks after repair.

The [fresh compact adaptive inventory](../../results/performance_followup/public-checkpoint-compact-inventory.json)
retains exactly the preceding 11 accepted and nine rejected cases. Neither
its acceptance gates nor its original ncv=20/30, k=10, tol=1e-10, budget-300
configuration is relaxed. Issue #104 remains open.

## Performance controls

Warm synchronized solve medians on the L40, with clean occupancy records:

| Configuration | Preservation off, s | Preservation on, s | Change |
|---|---:|---:|---:|
| Original general case 20, n=100, ncv=40, fixed 80 cycles | 1.117040 | 1.117231 | +0.02% |
| Original shifted case 12, n=1,000, k=20, ncv=50, fixed 30 cycles, cuDSS | 1.161401 | 1.173901 | +1.1% |

[Direct control and preservation](../../results/performance_followup/public-checkpoint-direct-overhead.json),
[shifted control](../../results/performance_followup/public-checkpoint-shift-control.json),
and [shifted preservation](../../results/performance_followup/public-checkpoint-shift-preserved.json)
retain every sample and corresponding occupancy records.

The direct comparison alternates order over five samples per variant on identical
matrix/start buffers. The shifted controls run sequentially, three samples each.
All timed results pass independent original equations, requested spectrum and
strict count/status checks. Setup/capture/warmup and CPU validation are excluded.
These controls measure overhead, not a claim that more fixed cycles outperform
adaptive stopping or that all general matrices now converge.

The direct graph adds 161 kernels and six memsets over the 80-cycle control,
without adding copies or host nodes. Shifted graphs have no HOST callback nodes;
the existing vendor HtoD metadata copies remain. The original checkpoint
prototype's rejected timing control is retained in the
[historical diagnostic report](../experimental/GENERAL_SELECTION.md).

## Full validation

The [full pinned-environment suite](../../results/performance_followup/public-checkpoint-validation.json)
passes **1,332 tests in 494.94s**. The focused general/projected/shifted subset
passes **172 tests in 58.30s on stock Warp 1.15**. Python lint, formatting and
whitespace checks pass. This does not relabel the nine known compact-spectrum
failures as conformance passes.

## Reproduction

```bash
python -m benchmarks.experimental.general_selection_profiles \
  --ncv-multiplier 2 --fixed --budget 80 --repeats 3 \
  --out build/public-checkpoint-wide.json
python -m benchmarks.experimental.compare_general_checkpoint \
  --public --case benchmark_gen_20 --ncv 40 --budget 80 --repeats 5 \
  --out build/checkpoint-direct-overhead.json
python -m benchmarks.experimental.general_count_probe \
  --case GenEigsRealShift_12 --iterations 30 --repeats 3 --warmup \
  --out build/checkpoint-shift-preserved.json
```

Use the pinned Warp environment and original Spectra fixture export. Supply
`--fixtures` to the first two commands if needed. Wrap timing runs with
`benchmarks.monitor_gpu`; add `--no-preserve` to the shifted probe for its
control. The old `--checkpoint` profile option selects the retained experimental
prototype; omit it to test the public policy.
