# Dominant generalized spectral reconstruction coverage

[#136](https://github.com/alecjacobson/warpack/issues/136) now has procedural
point/tet fixtures, independent energy tests, complete repeated-cluster checks,
and ARPACK / Warp-cuDSS / pure-Warp comparisons. No external meshes or paper
code are copied. See the [equations, API and reproduction protocol](../../benchmarks/applications/SPECTRAL_SURFACE.md).

The pencil maximizes sampled normal-alignment energy A under an SPD
point/gradient/biharmonic regularization metric B. **B is not physical mass.**
A is PSD and singular; this requests its largest generalized modes, rather
than the lowest stiffness/mass modes typical of vibration analysis.

## Accuracy and geometry

All **18 final configurations / 72 solves including warmups** pass fresh
original-equation residual, metric orthogonality, eigenvalue and complete
requested-subspace checks. Maximum normalized residual is `3.15e-13`;
maximum B-weighted subspace error is `6.03e-12`. GPU results also pass native
zero status/full count. [Validation and matched matrix/source identities](validation.json).

Cases cover:

- 4,913 variables / one dominant mode, default regularization.
- 35,937 variables / one and four dominant modes.
- Two identical disconnected copies, 9,826 variables / the complete two-fold
  dominant cluster; individual vectors in that eigenspace are not unique.
- Gradient-only regularization with alpha=10, gamma=0.
- Stronger anchoring/biharmonic regularization with alpha=1000, gamma=0.01,
  anisotropy=30.

Small tests additionally vary sphere/ellipsoid sampling, sample count at fixed
resolution, regularization and captured anisotropy. They independently verify
P1 interpolation/gradients, volume/stiffness energy, PSD A, SPD B, and rejection
of unanchored/singular metrics. Small eigen references use dense generalized
LAPACK. Large references use an independently computed one-copy ARPACK basis
with four guard pairs and a larger subspace; no reference vectors seed timed
solvers. The shared ARPACK reference is a cross-solver check, not a proof of
all possible spectrum/completeness behavior.

The simple dominant mode defines a zero set. Its geometry metrics agree across
all backends within `1e-8`, separately from the eigen-accuracy checks:

| Background | Sample distance RMS (world units) | Zero-set relative radial RMS | Mean absolute normal alignment |
|---|---:|---:|---:|
| 16 cells/axis, 4,913 variables | 0.0029474 | 0.0076140 | 0.992071 |
| 32 cells/axis, 35,937 variables | 0.0009120 | 0.0020809 | 0.997853 |

Zero-set radial error is measured at exact P1 edge crossings against the analytic
ellipsoid. It is a reconstruction/discretization error, not an eigensolver
residual; an exactly solved discrete eigenproblem need not reconstruct the
analytic surface exactly.

## Uncontended timing

Measured 2026-10-22, NVIDIA L40 / Xeon Platinum 8362, four BLAS/OMP threads
pinned to cores 8–11, current verified Warp 1.18.0/FSAI overlay. Each row has
one warmup and three measured solves. Factors/workspaces are independently
constructed in each worker; nothing is reused from another method's row.

| Variables / modes | Backend | Setup / factorization (s) | Graph capture (s) | Warm solve median (s) |
|---|---|---:|---:|---:|
| 4,913 / 1 | SciPy ARPACK + SuperLU | 0.5900 | — | 0.1481 |
| 4,913 / 1 | Warp + cuDSS | 0.2282 | 0.3198 | 0.0639 |
| 4,913 / 1 | Pure Warp + FSAI/CG | 0.2924 | 0.5071 | 0.1593 |
| 35,937 / 1 | SciPy ARPACK + SuperLU | 39.3642 | — | 3.6292 |
| 35,937 / 1 | Warp + cuDSS | 0.9476 | 0.3491 | 0.2053 |
| 35,937 / 1 | Pure Warp + FSAI/CG | 0.3087 | 1.0369 | 0.7160 |

Setup includes conversion and factor/solver construction. Capture includes its
warmup/JIT work. Fixture assembly, independent reference computation, output
copies and validation are excluded and recorded separately. Setup is a single
observed construction per worker, not a stable cold-process median. This CPU
row measures SciPy with SuperLU; it is not a comparison against the best CPU
sparse factorization. The large metric has 2,077,527 nonzeros, versus 19,955
for alignment A, so this differs substantially from a diagonal-mass problem.

Both GPU paths use ncv=32. cuDSS runs four fixed outer cycles; pure Warp stops
on device after one for the simple dominant mode. The differing capture policy
is explicit, not a hidden CPU convergence readback.

The large CG result spends **13,453 inner iterations / 32 inverse applications**
(up to 425 per RHS). All recursive residuals pass the requested inner tolerance
`1e-13`, but all 32 fresh residual audits miss it: maximum fresh relative
residual `5.46e-13`. The original eigenproblem nevertheless passes independently
at about `1.98e-13`. This is not a claim that every strict inner inverse passed.
The cost and diagnostic distinction are tracked in
[#144](https://github.com/alecjacobson/warpack/issues/144).

Device/context free-memory decreases after setup/capture are 608 MiB for the
large cuDSS worker and 1,010 MiB for CG. These are snapshots, **not peak memory
or live factor sizes**; allocator/context behavior contributes.

The altered regularization controls are also uncontended:

| 4,913-variable control | ARPACK setup / solve (s) | cuDSS setup / capture / solve (s) | Warp CG setup / capture / solve (s) |
|---|---:|---:|---:|
| Gradient-only | 0.1304 / 0.0940 | 0.1870 / 0.3064 / 0.0392 | 0.1546 / 0.4099 / 0.1197 |
| Stronger regularization | 0.5929 / 0.1457 | 0.2296 / 0.3251 / 0.0639 | 0.2925 / 0.5593 / 0.2105 |

All three large-four-mode workers and all three repeated-cluster workers
encountered other GPU processes. Their accuracy is accepted, but their timings
are excluded from conclusions. The twelve uncontended workers have gated
`.summary.json` records in [qualified/](qualified/); all raw occupancy/log files
remain available. Earlier one-repeat [screening](screening/) is diagnostic and
does not meet the three-repeat timing gate.

## Graphs, tests and remaining work

Every pure-Warp graph has no HOST/HtoD/DtoH nodes. cuDSS graphs have no HOST or
DtoH nodes, but retain vendor HtoD scalar metadata copies under
[#29](https://github.com/alecjacobson/warpack/issues/29). Reports expose this as
`graph_device_only=false`; they cannot be presented as transfer-free graphs.
DOT inspection is not a CUDA API trace.

**38 focused tests passed:** [36-test run](tests.log) plus
[two additional fixed-resolution sampling tests](sampling-tests.log).
These include accuracy/energy checks, changed replay, repeated clusters and
reporter rejection of native failures, wrong spectra, metric errors, missing
trials, contention, invalid provenance, hidden host transfers and a 30% injected
slowdown. Four tests also exercise the experimental public `BlockLOBPCG(-A, B)`
composition for #144; that prototype is outside the installed core and is not
included in the timing tables above. The full library suite was not rerun:
no `warpack/` implementation changed.

Future work includes closing the pure-Warp inverse cost (#144), unknown/noisy
normal estimation, more complex point clouds and matrix-free consistent-mass
biharmonic regularization. Those extend this fixture's initial contract; they
are not silently claimed by the current tests.
