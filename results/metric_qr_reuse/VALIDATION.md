# Installed metric-checked Ritz QR reuse (#183)

All **64 complete first/later outputs pass** in fresh original/default FEM controls. The installed policy uses a fresh device mass-Gram check to avoid unnecessary Ritz-basis QR, retaining complete original QR whenever the check cannot establish the qualified computed-metric bound. All original native stopping, rank and physical acceptance gates remain.

## Implementation and review

The default guard requires CUDA Ada SM89, block width 96–256, `gram_chunk=4096`, `tol >= 1e-11`, `reuse_operator_images=True`, `symmetric_products=True`, and an owned Warp scalar/batched sparse or identity mass operator with its original apply method. Eager execution and unqualified configurations retain full QR. Anvil uses width 24 and its original graph remains unchanged. No performance claim is made for unmeasured architectures.

On capture/replay, form fresh MX and G=X^T MX. The maximum absolute row/column sum of G-I, inflated against summation roundoff, must be finite and <= 1e-13 with zero status. Then preserve X/cached AX and retain fresh MX; otherwise run both original QR passes. This bounds the computed Gram defect, not exact-arithmetic physical matrix products. Independent original-product validation remains essential.

Review covered source/branch ownership, shared scratch, status propagation, cached AX/fresh MX, invalid-to-healthy transitions, custom operators and conditional-graph placement. The two kernels match the qualified prototype AST. The original QR body is unchanged in its separate fallback method. Mutually exclusive branches share 24 logical bytes of control/counter device values, without new dense workspace or replay host readback. All numerical operations remain Warp. No external numerical backend is introduced. The diagnostic profiler times this conditional helper at its boundary and suppresses nested event nodes in child graphs.

## Regression and clean-wheel validation

- Full suite: **3237 passed, 11 skipped in 1062.60s (0:17:42)** ([log](validation/full-tests.log)).
- Focused regressions: **71 pass** ([log](validation/focused-tests.log)), covering scope guards, branch sharing, exact reuse/fallback, nonfinite/zero/negative/scaled/coupled mass, cached images, changed stiffness/mass/constraints/clusters, both history settings, custom operators and conditional profiling.
- Stock Warp 1.15: **64 passed, 7 skipped in 18.77s** ([log](validation/stock-tests.log)); skips require the pinned upstream FSAI feature set.
- The isolated wheel contains **43 byte-identical core modules** ([inventory](validation/wheel-identity.json)), SHA256 `32d6dd3954bb7d7f702d28a212ac291c35d10921dd348e6371767a254dfed7d6`. It passes three public replays, sixteen projected outputs and nine changed-pencil wide replays, rejecting benchmark/example/SciPy/CuPy/cuDSS imports. This is a test wheel, not a new published release.
- The wide installed smoke uses n=640, width=104 and k=40 with repeated modes, constraints, changed K/M, both history settings and compact graphs. All nine replays exercise QR reuse; maximum absolute residual is 2.028e-09, mass-orthogonality error 3.572e-14. DOT inspection requires the installed metric-check, Gram and triangular kernels, and rejects HOST/HtoD/DtoH nodes ([outputs](validation/wheel-wide.json)).

## Complete workflow controls

L40 / Xeon 8362, four pinned CPU cores, checksum-pinned Warp 1.18 FSAI/batched-CG overlay, 2026-10-23. Every worker constructs its own factors/preconditioner and solver; only independently checked CPU references are cached outside compute timers. All first and later outputs pass requested spectrum/subspace, original residual, mass orthogonality, expected nullity and native status/count/stopping. Within pairs, input/source/dependency/reference identities match and all occupancy records are clean. Pure-Warp graphs contain no HOST or host/device-transfer nodes.

Each paired cell is **original / default**, seconds. First-use compute is setup + capture/eager preparation + first captured solve. Warm medians use three later replays. The qualified wide cases improve 3.4–8.0%; this is not a claim that the remaining CuPy gap is closed. Setup/capture are single cached-compilation observations, not setup stability distributions. Assembly, imports, references and host validation are outside these timers.

| Problem | Setup | Capture | First-use compute | Warm median | Warm change |
|---|---:|---:|---:|---:|---:|
| 200 | 0.6299 / 0.6407 | 3.8922 / 3.8516 | 11.9462 / 11.3389 | 6.9630 / 6.4222 | -7.77% |
| 100 | 0.4392 / 0.4652 | 1.4108 / 1.4372 | 4.1744 / 4.0732 | 2.2183 / 2.0488 | -7.64% |
| 200-compact | 0.6840 / 0.6436 | 1.6110 / 1.5937 | 9.3669 / 8.7663 | 7.0225 / 6.4612 | -7.99% |
| 100-history | 0.4401 / 0.4508 | 1.6778 / 1.6480 | 4.5601 / 4.5127 | 2.2821 / 2.2053 | -3.36% |
| anvil | 0.7999 / 0.7898 | 0.6188 / 0.6171 | 3.0446 / 2.9660 | 1.5673 / 1.6133 | +2.93% |
| free | 0.4553 / 0.4579 | 1.5651 / 1.5747 | 4.2390 / 4.1274 | 2.0779 / 1.9588 | -5.73% |
| disconnected | 0.5102 / 0.5248 | 1.8425 / 1.8636 | 6.3170 / 6.1096 | 3.7915 / 3.5834 | -5.49% |
| heterogeneous | 0.4376 / 0.4375 | 1.4187 / 1.4140 | 4.4203 / 4.2381 | 2.4433 / 2.2762 | -6.84% |

Repository-wide Ruff lint passes, and formatting checks pass for all ten modified Python files. A broader format-only check also reported five unchanged historical/older implementation files; they are not rewritten as part of this numerical qualification.

All raw records are in [production](production/); [qualified summaries](validation/installed-qualified.json) retain every sample, iteration count, QR reuse/fallback count and maximum checked error. Anvil has no candidate QR controls and its complete graph summary is identical between policies, so its measured change is run variability. Slower samples and iteration changes are retained.

The [pilot](README.md) preserves all predeclared threshold choices, and the [wide prototype follow-up](WIDE.md) preserves reverse-order/compact/history/nullspace/heterogeneity controls and initial test-fixture failures. No failed threshold or source record is replaced.

Reproduce original QR with `python -m benchmarks.experimental.compare_metric_qr --metric-defect 0 --reference-cache CACHE --public-api ...`; add `--installed-default` for the qualified current default. Exact remaining arguments and commands are in each raw JSON/occupancy report. #147 remains the broader wide-FEM performance gap.
