# Current matched-preconditioner CuPy comparison (#147)

All **14 checked outputs pass**: two first Warp solves, six later Warp solves
and six timed CuPy solves. CuPy's preliminary warmup is not part of this count.
Source/dependency identities and exact prepared inputs are recorded; GPU occupancy
is clean. This measures warm solver iteration with the same supplied upstream
Warp FSAI/CG preconditioner. **It is not an independent CuPy setup comparison.**
The raw adapter setup scope explicitly excludes the borrowed factor/base-CG
construction; no end-to-end setup ratio is claimed.

Both use the same physical matrices, CPU reference, and corresponding starting
columns. Warp widths are 104/204 with compact history; CuPy widths are exactly
100/200. Warp's normalized native tolerance is 1e-11; CuPy's absolute tolerance
is 1e-10 on its entire block. Both native rules and the shared original-equation,
spectrum/subspace and orthogonality gates must pass. Achieved errors are shown
because the native stopping definitions differ.

Seconds; medians of three warm trials, L40 and pinned Warp/FSAI, 2026-10-23.

| Modes | Solver | Warm median s | Max absolute residual | Max normalized residual | Requested-space error |
|---:|---|---:|---:|---:|---:|
|100|Warp|2.4390|1.383e-12|7.732e-12|3.728e-11|
|100|CuPy|2.8047|2.486e-12|2.454e-11|6.077e-10|
|200|Warp|8.1507|3.161e-12|9.948e-12|5.642e-11|
|200|CuPy|3.7009|2.502e-12|3.884e-11|1.207e-10|

Warp is about 13% faster at 100 modes in this control, but remains **2.20× slower
at 200**. The 200-mode CuPy absolute residual is slightly smaller than Warp's;
the observed large gap cannot simply be explained as CuPy returning a grossly
looser solution. This does not establish a universal backend ordering.

[100-mode raw worker](cupy-current/100-default-a1.json),
[200-mode raw worker](cupy-current/200-default-a1.json),
[qualified summary](validation/cupy-qualified.json). Adjacent logs/occupancy files
retain commands, individual samples, fresh full-block CuPy residuals and exact
adapter/memory scope. The [installed optimization controls](VALIDATION.md) compare
canonical and specialized Warp paths independently of this backend comparison.
