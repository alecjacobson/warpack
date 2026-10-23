# Current matched-preconditioner CuPy comparison (#147)

All21 checked outputs pass: each profile checks the first Warp output, three measured Warp outputs and three measured CuPy outputs. Inputs/source/dependencies/independent references match; occupancy is clean. Warp graphs have no HOST or host/device transfer nodes. As-run numerical source: `0c5ecfc`.

These are warm comparisons. CuPy uses the supplied Warp FSAI/base-CG preconditioner, so its adapter setup is **not an independent total setup cost**. No setup/table cost is borrowed from another row. The caller-native stopping definitions differ (Warp normalized residual, CuPy absolute residual); both must pass the common original-equation, spectrum/subspace and mass-orthogonality gates, and each recorded native gate remains mandatory.

| Requested modes | CuPy absolute tolerance | Warp median s | CuPy median s | Warp / CuPy |
|---|---:|---:|---:|---:|
| 100 | 1e-10 | 1.9830 | 2.8183 | 0.704 |
| 100 | 5e-11 | 1.9746 | 2.5828 | 0.764 |
| 200 | 1e-10 | 6.1123 | 3.7290 | 1.639 |

Both100-mode profiles were predeclared and are retained; the stricter CuPy threshold happens to be faster in this run, which does not establish a general tolerance/speed relationship or erase the older #69 failures. The current200-mode gap is **1.64×**, versus1.70× in the previous QR-reuse comparison. The same-run installed original/default controls establish the rotation improvement separately; this cross-checkpoint comparison does not attribute every change to one kernel.

Raw first/later accuracy, iteration/accounting, setup scope and timings are in [cupy-current](cupy-current/); [qualified summary](cupy-qualified.json) retains all measured samples and error maxima. #147 remains open.
