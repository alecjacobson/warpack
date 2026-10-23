# Additional 100-mode FEM families (#128, #147)

These procedural P1 tetrahedral cases use the public solver after the guarded
cooperative-factor change in `cda0f53`. All matrices are assembled independently
on the host, with expanded ARPACK references. No external meshes are copied.
Every accepted solve passes original equations, requested spectrum/subspace,
mass orthogonality, rigid-mode count/subspace and native stopping checks.

## Accuracy coverage

| Problem | DOFs | Requested elastic + rigid modes | Warp width | Worst Warp residual | Worst requested-subspace error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Free beam | 35,328 | 100 + 6 | 112 | 8.02e-12 | 1.53e-11 |
| Two disconnected free beams | 70,656 | 100 + 12 | 120 | 6.93e-12 | 2.24e-11 |
| Heterogeneous clamped beam | 34,560 | 100 + 0 | 104 | 8.47e-12 | 1.34e-11 |

Each case has three accepted Warp replays. Disconnected and heterogeneous cases
also have separate three-replay Warp-only accuracy runs. The disconnected
reference explicitly duplicates the single-body spectrum and includes complete
boundary clusters; all 12 known rigid modes are retained. Heterogeneous material
has a 20:1 stiffness contrast and a 3:1 density contrast. The final checks use
the original coupled mass matrices. No tolerance was relaxed.

## Uncontended configured-fixture comparisons

L40, pinned current FSAI/CG overlay, CPU cores 8–11, three measured solves per
method. All rows below pass the strict reporter and full-worker occupancy gate.

| Problem | Method | Warm median (s) | Worst normalized original residual |
| --- | --- | ---: | ---: |
| Free beam | Warp | 3.1777 | 8.02e-12 |
| Free beam | CuPy + supplied Warp CG/FSAI | 2.2394 | 2.62e-11 |
| Heterogeneous beam | Warp | 3.5974 | 8.47e-12 |
| Heterogeneous beam | CuPy + supplied Warp CG/FSAI | 2.1077 | 8.04e-11 |

Warp uses normalized tolerance 1e-11 on the requested modes; CuPy uses absolute
1e-10 on its entire starting block. The preconditioner is FSAI32 with FP32 CG,
25 iterations and relative inner tolerance 0.01, with independent RHSs and a
power-of-two chunk adapter for CuPy. These are different native stopping
criteria, so this is a workflow comparison, not an identical-accuracy speed
ratio. CuPy remains faster in these two observed profiles; #147 stays open.

Single-worker Warp construction plus capture is 4.9391 s for free and 4.2672 s
for heterogeneous. The CuPy preparation adds 0.0876 / 0.0974 s while reusing the
supplied preconditioner; this is not independently constructed CuPy setup.
Assembly, independent reference generation and warmup are excluded. Warmup time
is recorded but its physical checks are not saved.

These workers record source/dependency and deterministic fixture configuration,
but procedural input array fingerprints are missing (unlike Anvil).
[#156](https://github.com/alecjacobson/warpack/issues/156) tracks exact matrix,
reference and start identities; no exact-byte equality claim is made here.

Evidence: [free summary](free-a2.summary.json),
[heterogeneous summary](heterogeneous-a1.summary.json), with corresponding raw
JSON/log/occupancy files. `free-a1` is contended and excluded from timings.

## Rejected disconnected CuPy control

`disconnected-a1` contains three accepted Warp solves and one fully accepted
CuPy solve. The other two CuPy returns pass the original physical checks and
all first 112 requested absolute residuals, but fail the native full-width
criterion: padding column 118 has absolute residual 3.052e-10 / 3.619e-10,
above 1e-10. The last recorded history also exceeds the bound. These rows are
rejected; no timing ratio is reported. This extends the native stopping
coverage tracked in [#69](https://github.com/alecjacobson/warpack/issues/69).
The worker was also contended.

`disconnected-accuracy-a2` and `heterogeneous-accuracy-a2` are accepted Warp-only
correctness runs under contention, with no performance claim. Their `a1`
predecessors failed CLI validation before solving because the orchestration
retained a CuPy-only adapter flag after disabling CuPy; the corrected runs
explicitly disable that flag. These harness errors are retained and are not
solver failures.
