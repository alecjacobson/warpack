# Installed FP64 Gram tiles (#180)

All **72 complete first/later outputs pass** in fresh installed original/default controls. Warm FEM solves improve **3.5–14.2%**, including **3.8% on Anvil**. The larger 200-mode comparison still trails matched-preconditioner CuPy; this change narrows, rather than closes, #147. The independent prototype, reversed controls, failed initial implementation and slower component variants remain in [README.md](README.md) and [WHOLE.md](WHOLE.md).

## Implementation and review

For Ada CUDA devices (SM 89), block widths 24–256 and `gram_chunk=4096`, `BlockLOBPCG` uses 16×16 FP64 output tiles with 128 threads. Other architectures, widths and chunk settings retain the original 8×8 implementation. Qualification used an L40; no throughput claim is made for unmeasured hardware. The private `_wide_gram_enabled=False` flag provides an explicit original-kernel benchmark control. It is inherited by the shallow-copied active-search branches.

The installed numerical kernel body exactly matches the qualified prototype AST; its factory fixes the qualified tile and reduction sizes. A rolled loop reuses tile storage. Review covered partial/strided views, output zeroing and atomic accumulation, mirrored symmetric off-diagonal tiles, independently rounded entries within diagonal tiles, buffer ownership, branch dispatch, eager/captured execution and invalid-input recovery. No global workspace, host numerical decision, synchronization or vendor call is added. Both complete mass-QR passes, rank thresholds and original stopping criteria remain. All new numerical work is Warp.

## Regression and package checks

- **3,161 tests pass, 11 skip** in the [full run](validation/full-tests.log), including the installed numerical path and dispatch guards. The 11 separate wheel-audit tests were added after that collection. The subsequent diagnostic-only profiler fix (#182) is covered by six focused profile tests; it does not change core arithmetic.
- **185 focused tests pass** ([log](validation/focused-tests.log)): guard boundaries and other architectures, independent original products, strided/tail storage, changed/scaled/zero replays, complete two-pass QR with cached images, changed coupled pencils/constraints, repeated modes, both history settings, and negative-mass/nonfinite/zero rejection followed by recovery on the same graph.
- Stock Warp 1.15 passes **185 tests**, with 17 expected skips requiring the upstream FSAI/batched-CG overlay ([log](validation/stock-tests.log)).
- The clean wheel contains exactly **42 core modules**, each byte-identical to source and the isolated installation ([identity](validation/wheel-identity.json)). SHA256: `97068955f1f38b2ab0582a3a6806d482f7a7e998b3e9b926f535c99c7b979dd9`. A contaminated direct local build was rejected; [#181 evidence](../wheel_audit/README.md) preserves that failure and the new automatic inventory audit.
- With benchmark/application/SciPy/CuPy/cuDSS imports forbidden, the wheel passes [three public replays](validation/wheel-public.json), [sixteen projected outputs](validation/wheel-projected.json), and [nine wide generalized replays](validation/wheel-wide.json). The wide control changes stiffness/mass on one graph (640 unknowns, width 104, 40 modes), retains constraints and repeated modes, and exercises both history settings and compact graphs. DOT inspection confirms the installed Gram and register-triangular kernels run, with no HOST/HtoD/DtoH nodes. Maximum absolute residual is 2.06e-9 and mass-orthogonality error below 1.2e-15. These are test wheels, not a new release.

## Complete workflow controls

L40 / Xeon 8362, four pinned CPU cores, pinned Warp 1.18 FSAI/CG overlay, 2026-10-23. Every worker constructs its own factors/preconditioner and solver. Only independently verified CPU references are cached outside timers. Input, source, dependency and reference identities match within each pair; every external occupancy record is clean. Every first and later output passes original equations, requested spectrum/subspace, mass orthogonality, expected nullities, native count/status/stopping and device graph audits.

Each cell below is **original / default**, in seconds. Setup excludes capture. **First-use compute = setup + capture/eager preparation + first captured solve**. Warm compute is the median of three later replays. Setup/capture are single cached-compilation observations, not established setup speedups. Assembly, imports, references and host validation are excluded.

| Problem | Setup | Capture | First-use compute | Warm median | Warm reduction |
|---|---:|---:|---:|---:|---:|
| [anvil](production/anvil-default-a1.json) | 0.7979 / 0.7821 | 0.6107 / 0.6162 | 3.0214 / 3.0036 | 1.6163 / 1.5547 | 3.81% |
| [100](production/100-default-a1.json) | 0.4456 / 0.4304 | 1.4227 / 1.3978 | 4.4233 / 4.1524 | 2.4390 / 2.2283 | 8.64% |
| [200](production/200-default-a1.json) | 0.6340 / 0.6428 | 3.9471 / 3.9423 | 13.1665 / 12.0776 | 8.1459 / 6.9904 | 14.19% |
| [200-compact](production/200-compact-default-a1.json) | 0.6419 / 0.6548 | 1.7000 / 1.5682 | 10.6709 / 9.3408 | 8.1820 / 7.0328 | 14.05% |
| [48](production/48-default-a1.json) | 0.2874 / 0.2845 | 0.6978 / 0.7227 | 1.8850 / 1.8823 | 0.8699 / 0.8397 | 3.47% |
| [60](production/60-default-a1.json) | 0.3096 / 0.3061 | 0.8571 / 0.8695 | 2.2334 / 2.1745 | 1.0208 / 0.9539 | 6.56% |
| [free](production/free-default-a1.json) | 0.4470 / 0.4553 | 1.5688 / 1.5472 | 4.4940 / 4.2184 | 2.3379 / 2.0754 | 11.23% |
| [disconnected](production/disconnected-default-a1.json) | 0.5252 / 0.5408 | 1.9124 / 1.8691 | 7.0075 / 6.3999 | 4.3984 / 3.8112 | 13.35% |
| [heterogeneous](production/heterogeneous-default-a1.json) | 0.4329 / 0.4288 | 1.4176 / 1.4143 | 4.6486 / 4.3971 | 2.6801 / 2.4332 | 9.22% |

All raw original/default JSON, logs and occupancy samples are in [production](production/); [qualified summaries](validation/installed-qualified.json) retain every time, iteration count and worst checked quality. Small one-iteration differences remain in some adaptive runs. The 48/60-mode, free, disconnected and heterogeneous controls have identical iteration counts. Original normalized residuals remain below 1e-11 for the wide FEM controls; Anvil retains its existing physical gates.

Reproduce with `python -m benchmarks.experimental.compare_gram_whole --gram-tile 0 --reference-cache CACHE --public-api ...`; add `--installed-default` to enable current installed dispatch. Exact problem/solver arguments are in every raw JSON. The existing README six-backend Anvil table remains a separately dated comparison until all its rows are refreshed.
