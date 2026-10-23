# Fresh matched-preconditioner CuPy comparison

The accepted 200-mode control is **6.9766 s Warp / 3.7315 s CuPy**, a **1.87×** remaining warm-solve gap. The preceding installed comparison was 2.20× ([report](../register_triangular/CUPY.md)); the Gram change narrows #147. It does not close it.

| Modes | Warp normalized tolerance | CuPy absolute tolerance | Warp median s | CuPy median s | Native/physical status |
|---:|---:|---:|---:|---:|---|
| 100 | 1e-11 | 5e-11 | 2.2097 | 2.7014 | all checked outputs pass |
| 200 | 1e-11 | 1e-10 | 6.9766 | 3.7315 | all checked outputs pass |

Both use the same supplied Warp FSAI32/FP32 CG preconditioner. Warp widths are 104/204; CuPy widths are 100/200 without padding. Starts use identical prefixes and the physical matrices and independent reference are shared. CuPy is an external reference, not part of Warpack computation. All fourteen outputs in these two accepted workers pass (two first Warp solves plus twelve timed outputs); every full returned CuPy block meets its separately declared native absolute tolerance. Source/input/dependency/reference identities and clean occupancy are verified.

These are **warm comparisons**, not independently constructed setup comparisons. CuPy adapter setup borrows the existing factor/base CG from the worker and excludes their cost. It must not be presented as a standalone setup time or copied into the README Anvil setup column. Every timed sample, phase scope and accuracy field is retained in [raw workers](cupy-current/) and [qualified summaries](validation/cupy-qualified.json).

## Retained native-stopping failure (#69)

The initial 100-mode worker at CuPy native 1e-10 is **rejected**: one of its three CuPy outputs has fresh absolute residual 1.0006013782027795e-10 despite final history maximum 9.994517389942498e-11. The physical requested-space checks pass, but native stopping does not. The entire paired worker remains rejected and supplies no speed ratio ([raw report](cupy-current/100-default-a1.json)). Its external samples show no foreign GPU activity; the monitor marks timing ineligible because the accuracy failure produced exit 1.

A predeclared separate 100-mode profile requested native 5e-11; all three CuPy outputs pass, with maximum fresh native residual 4.894e-11. This is a different explicit tolerance profile, not a replacement or relabeling of the failed run. The stopping issue remains open. Small tolerance changes can change adaptive trajectories, so solve time need not vary monotonically.

For the accepted 200-mode pair, maximum original absolute residuals across timed outputs are 2.404e-12 Warp and 2.319e-12 CuPy. Their normalized residuals are 7.227e-12 and 3.886e-11. The difference in native tolerance definitions is explicit; the remaining timing gap is not explained by a grossly inaccurate reference result. No installed CuPy code was modified.
