# Anvil block preconditioner shift follow-up

A subsequent [inner-budget follow-up](ANVIL_BUDGET.md) improves the accepted
Warp profile to 2.385s with CG25; the shift-only results below are retained.

The existing approximate block preconditioner benefits from a smaller positive
shift. On the 283,857-DOF Anvil, **2.545s** replaces the preceding **3.111s**
experimental profile. A fresh control with the old shift takes **3.140s**, so
this is about a **19%** reduction against that control. No solver arithmetic,
upstream algorithm, stopping tolerance, mesh, mode count or start policy changes.
This is an Anvil-specific setting, not a new universal default.

The block method solves the original mass-normalized elasticity problem. Its
preconditioner applies 15 intentionally truncated upstream FP32 CG steps to
`A + shift*I`, using upstream FSAI32 with FP32 factors. Unlike the production
shift-invert Krylov route, this inverse is only a preconditioner: moving its
shift changes convergence, not the requested eigenvalue transformation.
Six analytic rigid motions are projected out; all 20 elastic modes are solved
and all 26 are checked against the existing cached reference. No CPU Anvil
factorization was rerun, and reference modes never enter the start vectors.

## Uncontended, accuracy-gated comparisons

NVIDIA L40; three warm synchronized trials per profile; Warp width 24 / CuPy
width 20, same supplied upstream FSAI32/FP32-CG preconditioner, 200 outer-step
limit. Warp retains its normalized tolerance 1e-8; CuPy retains its native
absolute tolerance 1e-5. Independent original residual, returned eigenvalues,
mode shapes and orthogonality pass on every trial. Warp reports all 20 with
status zero; CuPy's fresh full-block absolute bound passes on every trial.

| Positive preconditioner shift | Inner CG limit | Warp median, s | Warp outer steps | CuPy median, s |
|---:|---:|---:|---:|---:|
| 10000, fresh control | 15 | 3.14012 | 28–29 | 3.23662 |
| 1000 | 15 | 2.60986 | 23–24 | 2.65446 |
| 100 | 15 | 2.54485 | 22–23 | 2.71816 |
| 10 | 15 | 2.54944 | 23 | 2.69881 |
| 100 | 10 | 3.09208 | 33 | 3.16409 |

The smaller shift improves both outer solvers. Reducing the inner budget to ten
steps saves scheduled CG work but needs more outer steps and is slower overall.
The shift-10 and shift-100 Warp results are effectively similar; there is no
claim that this sweep establishes a universal optimum. CuPy trajectories vary,
so the small Warp/CuPy difference is not a universal ranking either.

At shift 100 / CG15, Warp samples are 2.47592/2.54485/2.55025s, with
6,840–6,960 scheduled inner vector iterations versus 8,580–8,760 in the fresh
old-shift control. The largest original elastic residual is 1.8404e-8; the
returned-pair residual is 9.2021e-9, below the declared normalized 1e-8 bound.
These stopping formulas and final residuals differ from the tighter production
shift-invert results. Do not interpret a lower time as identical final error.

Total factor/workspace setup is 1.2135s, capture .6526s, and setup/capture device
free-memory decrease 4.308GB, not peak memory. Fixture assembly/reference load
and CuPy adapter setup are separately excluded, as recorded in the raw schema.
The graph audit finds 19 conditional nodes, 50 DtoD copies, and no HOST/HtoD/DtoH
nodes. Post-profile physical checks, counts and status pass as well.

The preceding accepted cuDSS control is .4656s. Thus the remaining iterative
gap is still about 5.5 times; [#15](https://github.com/alecjacobson/warpack/issues/15)
remains open. The public block method is unchanged. Device-selected batched CG
and cached operator images remain experimental benchmark adapters.

## Evidence and reproduction

All five profiles are retained, including the slower CG10 attempt:
[old-shift control](../../results/performance_followup/anvil-shift10000-control.json),
[shift1000](../../results/performance_followup/anvil-shift1000.json),
[shift100](../../results/performance_followup/anvil-shift100.json),
[shift10](../../results/performance_followup/anvil-shift10.json),
[shift100/CG10](../../results/performance_followup/anvil-shift100-cg10.json).
Their sibling `.occupancy.json` files record sequential execution and no sampled
competing processes. All exit zero, with numerical acceptance and timing
eligibility true. [Validation and input hashes](../../results/performance_followup/anvil-shift-validation.json).
This is a settings/benchmark follow-up; no new numerical source or test code is
introduced, and no redundant full-suite run is claimed.

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.experimental.compare_block --case anvil --width 24 \
  --steps 200 --shift 100 --fsai-width 32 --repeats 3 --tol 1e-8 \
  --cupy --cupy-width 20 --gram-chunk 4096 --symmetric-products \
  --fused-updates --paired-rotation --reuse-x-image --batched-products \
  --inner-cg 15 --batched-cg --active-cg-bucket-size 4 --inner-fp32 \
  --inner-tol .01 --profile --out build/anvil-shift100.json
```
