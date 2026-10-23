# Fresh matched-preconditioner CuPy comparison after QR reuse

All **21 checked outputs pass**: the first Warp solve plus three Warp and three CuPy measured solves in each of three predeclared profiles. CuPy warmups are not included in that count. Each output passes original physical/native gates; occupancy is clean and sources/dependencies/independent references are verified. The numerical core is `d5f5b1e`.

This is a **warm solve comparison**. CuPy uses the supplied Warp FSAI/base CG and separately allocated active-width adapters. Its recorded adapter setup excludes the supplied factors and is not an independent total setup cost. Use the [Anvil comparison](../anvil_showcase_180/README.md) for independently constructed complete setup rows.

| Modes | CuPy native absolute tolerance | Warp warm median s | CuPy warm median s | Warp / CuPy |
|---|---:|---:|---:|---:|
| 100 | 1e-10 | 2.051914 | 2.849719 | 0.720 |
| 100 | 5e-11 | 2.050992 | 2.751095 | 0.746 |
| 200 | 1e-10 | 6.393108 | 3.764676 | 1.698 |

The 200-mode gap is now about **1.70×**, compared with 1.87× in the [previous fresh comparison](../gram_tiles/CUPY.md). It remains open under #147. Both 100-mode profiles pass here, but the earlier native 1e-10 boundary failure remains valid evidence under #69; one successful rerun does not establish a fix in CuPy. Both tolerance profiles were predeclared and every sample is retained. A tighter tolerance does not imply a monotonic timing comparison between these separate adaptive runs.

Warp uses widths 104/204 and normalized native tolerance 1e-11; CuPy uses widths 100/200 and the declared absolute tolerance. Both must pass shared independent original-pencil and requested-spectrum/subspace gates, plus their own stopping criteria. These are explicit algorithm/block/stopping differences. QR usage counters are collected outside timing only for Warp, with no counter read or attribution during CuPy validation. Seven driver tests cover restoration, rejected reports, policy selection and that counter separation ([log](cupy-current/driver-tests.log)).

[Every timing and quality maximum](cupy-current/qualified.json), with all adjacent raw worker, occupancy and command logs in [cupy-current](cupy-current/). Exact as-run sources are captured in each report and this publishing revision; no initial/reference/factor numerical settings are silently substituted.
