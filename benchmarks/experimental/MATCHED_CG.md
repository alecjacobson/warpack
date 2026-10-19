# Matched CG preconditioning and related FEM starts

2026-10-19 follow-up to [#59](https://github.com/alecjacobson/warpack/issues/59),
[#15](https://github.com/alecjacobson/warpack/issues/15), and
[#66](https://github.com/alecjacobson/warpack/issues/66). These remain experimental
solvers/adapters outside the installed package. The production API is unchanged.

Both outer methods use the **same upstream Warp FSAI32 factors and batched
FP32 CG**: ten inner steps, relative tolerance .01, `atol=0`, FP64 outer
arithmetic. The comparison is Warp's captured block iteration versus CuPy's
LOBPCG with a supplied Warp preconditioner. **Neither iterative route uses
cuDSS.** This is not a native CuPy sparse factorization comparison.

## Fixing active-block comparison overhead

CuPy removes converged columns from its active residual block. The first
adapter padded that block with zeros to Warp's fixed width. Although correct,
its sparse products still processed the zero RHSs: the Anvil width 24 run did
about 24,000 scheduled vector iterations versus about 8,640 after removing the
padding. Those initial runs are retained as explicitly biased diagnostics;
they do not justify a Warp speedup claim.

The corrected adapter preallocates one captured upstream CG state per active
width, sharing the shifted working matrix and FSAI factors. It dispatches using
array shape, with no numerical host decision inside CG. All workspace creation,
JIT and inner capture happen before timed solves. Each state processes only its
active RHSs. Scheduled counts still include RHSs masked after early convergence
within a batch; they are not sums of individual first-convergence counts.

This costs extra setup memory: the Anvil width 20 comparison records about
**3.02 GB** device free-memory decrease and **.147 s** extra adapter setup with
compiled kernels cached. Width 24 records **4.10 GB** and **3.694 s**, including
compilation encountered in that earlier process. These are single process-local
observations, not independent end-to-end CuPy setup or peak-memory measurements.
The states share numerical factor/matrix storage, but have separate CG/graph
workspaces; their total vector capacity grows as `sum(1..width)`.

Four regressions compare shrinking/growing blocks in FP32/FP64, including zero
and scaled RHSs and in-place output. They check fresh residuals, shared matrix
storage and exact active RHS accounting. All **29 batched-CG tests pass**.

## Fresh Anvil and Hodge comparisons

NVIDIA L40, three synchronized warm trials with starts restored. Every listed
trial passes independent physical residual, requested spectrum/subspace and
orthogonality gates. Warp additionally returns full device counts and zero
status. CuPy's absolute stopping criterion differs from Warp's normalized
criterion; acceptance here refers to the shared original-problem gates.
Setup/capture and reference construction are outside solve timing.

| Problem / configuration | Warp block (s) | CuPy LOBPCG + supplied Warp CG (s) |
|---|---:|---:|
| Anvil, 283,857 DOFs, 20 modes; width 24 on both sides | 4.5970 | 3.4533 |
| Anvil, Warp width 24 / CuPy width 20 | 4.3531 | **3.2289** |
| Disconnected Hodge, 39,366 DOFs, six modes, width 6 | **.07900** | .11265 |

Anvil uses shift 10000 and Gram chunk 4096. The two sides begin with the same
columns; Warp's width 24 includes four extra directions when CuPy requests 20.
Warp stops on its first 20 modes, while CuPy works on its entire requested
block. The width 24 CuPy run reaches the 100-step budget with unconverged guard
columns; its first 20 modes nevertheless pass the independent gates. This is
not a claim that all 24 native residuals converged. CuPy uses absolute tolerance
1e-5; Warp uses normalized tolerance 1e-8. Individual samples and residuals are
retained rather than selecting the most favorable replay. In the final width 20
run, two CuPy trials meet its absolute tolerance; one reaches the budget with a maximum absolute residual of 1.017e-5 and
fails that native criterion, while still passing every shared physical gate.
The 3.2289 s median includes all three trials and must not be interpreted as
three native-tolerance successes.

Removing padding reverses the initial Anvil conclusion: CuPy is faster even
with the same iterative preconditioner. The final run takes 35–37 Warp steps. The earlier width 20 comparison
records 8,880–9,120 scheduled Warp inner vector iterations, versus 8,000–8,050
for CuPy. The remaining outer-work gap is tracked in
[#67](https://github.com/alecjacobson/warpack/issues/67). Reducing Warp to width 20
is an accurate but slower single-trial screen: 81 steps and 7.714 s; it is not
promoted as an improvement. The production cuDSS path remains much faster
than either iterative method (the earlier fresh Anvil control is .4671 s).

Hodge uses shift .001 and Gram chunk 256. Both methods recover all four harmonic
modes and two positive guards. Warp takes 24 outer steps / 1,440 scheduled inner
vector iterations; CuPy's residual history has 22 entries and its preconditioner
processes 1,150 vector iterations. Warp's graph-only outer iteration is faster
here despite somewhat more inner work. The native cuDSS-preconditioned Hodge
comparison in [ADAPTIVE_CG.md](ADAPTIVE_CG.md) is a different preconditioner and
remains clearly labeled.

CuPy's last residual-history entry need not describe its returned vectors:
the implementation can perform another Ritz update after appending that entry.
The reports label those values `last_history_absolute_residuals`; they are not
used for physical acceptance. The current runner also recomputes fresh returned
residual norms outside timing and records whether they meet CuPy's requested
absolute tolerance, separately from the shared numerical gate.

## Computed starts after a regional stiffness change

The source problem is a clamped tetrahedral beam. A Warp block solve computes
its first 20 modes and an oversampled 24-row block, then stiffness increases by 2%
in the right half of the beam. All changed-problem strategies use the same
newly constructed FSAI32/CG10 preconditioner. No reference eigenvector enters
any start. The comparison includes a fresh random block, previous 20 modes with
four random padding vectors, and the previous full 24-row block.

Both methods use the nominal tolerance 1e-9, with their respective normalized
(Warp) and absolute (CuPy) stopping formulas. The table reports medians of three
warm solves in seconds, with every trial passing the shared accuracy gates.
The 10,890-DOF CuPy trials also all meet their native returned-vector tolerance.
The larger run predates that additional native-tolerance field; its fresh
original-equation validation is recorded.

| Changed problem / start | Warp block | CuPy LOBPCG + supplied Warp CG |
|---|---:|---:|
| Consistent mass, 10,890 DOFs, fresh random | .3286 | .4200 |
| Consistent mass, previous 20 + four random | .2597 | .3271 |
| Consistent mass, previous full 24 | **.1786** | .3779 |
| Lumped mass, 79,380 DOFs, fresh random | .8961 | 1.1430 |
| Lumped mass, previous 20 + four random | .6442 | .7454 |
| Lumped mass, previous full 24 | **.5787** | .6356 |

The Warp source solves cost .3403 s and .9939 s, respectively, before the
material change. Full-block reuse reduces changed-problem Warp solve time by
46% and 35%. It does not uniformly improve CuPy: on the consistent-mass case,
its previous-full-block run is slower than previous-20-plus-padding. Retain
that result rather than assuming warm starts always help.

Fresh production controls on these same changed matrices take .06680 s and
.21672 s with cuDSS, and .63165 s and 2.86178 s with FSAI-CG. All return 20
accepted pairs. These use shift -.01, ncv 64 and two cycles; the block solver
uses width 24 and adaptive stopping.

Every accepted trial checks original equations, M-orthogonality, the full
requested spectrum/subspace and all 20 Warp convergence counts. The consistent
mass is used as a coupled sparse matrix, without diagonal approximation. Source
solve, mass/operator setup, preconditioner/CG setup, per-solver workspace setup,
capture and warmup are recorded separately. Bootstrap cost matters if an
application does not already have the previous solution; warm-start solve time
must not be presented as a complete first solve.

Production Warp+cuDSS and Warp+FSAI-CG controls solve the same changed pencils
with ncv 64 and two Krylov cycles. These are explicit accepted control profiles,
not exhaustive tuning of those solvers. Their starts are fresh; the public
Krylov API does not consume the computed full block. cuDSS controls retain the
known vendor HtoD metadata transfers (#29), which affect the graph contract
but do not invalidate their independently checked numerical results.

## Validation

The full local suite passes **522 tests in 242.96 s**. Lint and formatting pass;
a wheel built from a clean source copy contains only the 18 production modules,
excluding these experiments. [Validation provenance](../../results/performance_followup/matched-cg-validation.json)
records commands and source hashes. This is not a rerun of the full upstream
CuPy/Spectra parameter battery.

## Records and reproduction

Raw reports and per-run occupancy records are in
[`results/performance_followup/`](../../results/performance_followup/).
`anvil-active-cg-final.json`, `anvil-active-cg-width24.json` and
`hodge-active-cg.json` are the active-width controls. Files containing
`matched-cg` and `related-consistent-matched.json` retain the initial padded
adapter diagnostics with an explicit caveat. `anvil-warp-width20-screen.json`
retains the slower block-width screen. Related-matrix tables use
`related-consistent-active-tol1e-9.json` and `related-lumped-large-active.json`.
The earlier consistent-mass report uses a tighter CuPy tolerance (1e-10) and
is kept separately. Occupancy sampling detected no competing
compute process; it is not a GPU reservation and may miss brief jobs.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu --out build/anvil.occupancy.json --log build/anvil.log -- \
  python -m benchmarks.experimental.compare_block --case anvil --width 24 \
  --cupy --cupy-width 20 --steps 100 --shift 10000 --tol 1e-8 --fsai-width 32 \
  --inner-cg 10 --batched-cg --inner-fp32 --inner-tol .01 \
  --gram-chunk 4096 --batched-products --repeats 3 --out build/anvil-active.json
python -m benchmarks.monitor_gpu --out build/related.occupancy.json --log build/related.log -- \
  python -m benchmarks.experimental.fem_starts --case clamped_lumped --side 20 \
  --modes 20 --width 24 --steps 200 --inner-cg 10 --inner-fp32 --batched-products \
  --gram-chunk 1024 --cupy --cupy-tol 1e-9 --baselines --baseline-ncv 64 \
  --baseline-steps 2 --repeats 3 --out build/related-active.json
```

Use `--cupy-padding` only to reproduce the diagnostic comparison overhead.
