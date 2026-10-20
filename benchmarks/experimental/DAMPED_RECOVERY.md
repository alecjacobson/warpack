# Physical recovery for the large damped fixture (#111)

The accepted 65,536-unknown profile combines an exact coordinate similarity,
one extra shifted inverse application per returned vector, and recovery of
velocity from displacement. All arithmetic in the Warp adapter is on device;
only the existing cuDSS sparse solves leave Warp. The helper lives in the
benchmark tree because it uses the structure of this quadratic FEM problem.
It does not change the public nonsymmetric solver's stopping or restart policy.

For state x=[u;v], use y=Sx with S=diag(8 I,I). The solver sees S A S^-1 and
the consistently transformed initial vector. The binary scale introduces no
coefficient rounding from division by a non-power-of-two scale. After the
solve, apply the same shifted inverse once to each returned mode. Recover
original coordinates, set v=lambda*u, normalize, and freshly evaluate both
A x=lambda*x and (K+lambda*(C+G)+lambda^2*M)u=0. The second check prevents
velocity reconstruction from hiding an incorrect displacement mode. Count,
requested spectrum/subspace, rank and norm checks remain mandatory.

The reference adapters use the same coordinate and extraction steps. ARPACK
uses SuperLU and Spectra uses native complex Eigen SparseLU/GenEigsSolver,
with one explicit sparse-solve residual correction. Spectra's iteration and
operation counters exclude extraction operations, but every solve time
includes them. The Warp inverse already uses one correction by default.
CuPy remains unsupported for this nonsymmetric problem.

The accepted profile uses k=20,ncv=64,two fixed Warp cycles, one inverse polish,
Warp's physical tol=2e-8 and reference iterative tol=1e-12. **The common
independent physical/quadratic/spectrum/subspace/norm bound stays 1e-8.** The
original tol=1e-10 profile is retained as a rejection; these results do not
claim to meet that stricter requested count. The reference's own FP64 physical
residual is around 3e-9, including when stored data is evaluated in extended
precision. That diagnoses a precision concern, not a formal lower bound.

All nine timed supported-backend samples pass. Median solves on the L40 host
are Warp+cuDSS 0.218344 s, ARPACK 1.119790 s and Spectra 2.106111 s. Maximum physical
residuals are 2.91e-9, 3.01e-9 and3.07e-9 respectively. Setup, capture and warmup
are separate; these are warmed solve comparisons, not cold-start timings.
[Full accepted control](../../results/damped-large-balanced8.json),
[occupancy](../../results/damped-large-balanced8.occupancy.json).

## Controls that did not pass

All use the same original matrix, target and requested 20 modes. None is a
performance win; unchanged failed controls remain reviewable.

| Control | Finding | Record |
|---|---|---|
| Original unbalanced,20 cycles,tol 1e-10 | All backends fail physical accuracy | [Original](../../results/damped-nonproportional-large.json) |
| Scale16385,20 cycles,tol 1e-10 | Mesh-resolution scaling does not repair accuracy | [Scaling](../../results/damped-large-balanced-tight.json) |
| One inverse polish,20 cycles,tol 1e-10 | Warp residual improves to about5e-9, but strict count fails | [Polish](../../results/damped-large-polished-tight.json) |
| One polish,2 cycles,tol 2e-8,reference correction1 | Physical residual/count is inconsistent across replays | [Short](../../results/damped-large-polished-physical.json) |
| Two polishes,reference correction2 | Individual vectors drift toward modes nearer the shift | [Repeated polish](../../results/damped-large-polished2.json) |
| One polish,20 cycles,tighter reference tol | Longer budget/tighter internal reference stopping does not solve extraction error | [Longer](../../results/damped-large-polished-accurate.json) |
| One polish and velocity recovery,scale1 | All Warp samples pass; CPU quadratic residuals still fail | [Unscaled recovery](../../results/damped-large-kinematic.json) |

Repeated inverse-only polishing is a power iteration and can amplify unwanted
components associated with modes nearer the shift. Do not interpret it as
monotonic refinement of every eigenpair. The full original-equation and
requested-set gates detect the failure. Scaling by 8 plus one polish and
structured recovery is an explicit validated fixture profile, not a universal
parameter recommendation.

The small fixture/recovery suite passes 21 tests on pinned Warp (22.76s) and
stock 1.15 (23.46s). Tests include replay, balanced/unbalanced recovery, all four
damping/mass variants, pure-Warp GMRES, and a deliberately wrong displacement
that still fails after velocity recovery.
[Validation](../../results/performance_followup/damped-recovery-validation.json).

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  PYTHONPATH="$PWD/build/warp-fsai:$PWD" python -m benchmarks.monitor_gpu \
  --out build/damped-balanced8.occupancy.json --log build/damped-balanced8.log -- \
  python -m benchmarks.applications.damped_compare --nodes 16384 --k 20 \
  --ncv 64 --cycles 2 --balance 8 --polish 1 --kinematic \
  --reference-refinement 1 --reference-tol 1e-12 --tol 2e-8 \
  --out build/damped-balanced8.json
```


## Replication and further large variants

The final native ABI guard is included in a fresh nonproportional replication
and two additional 65,536-unknown variants. Every supported-backend sample
passes again (27 samples total); all three occupancy monitors are eligible.
The repeated nonproportional Warp median is 0.218532 s, consistent with the first
0.218344s. The gyroscopic and coupled-mass variants use the exact same profile,
without retuning tolerances, basis width or iteration budget.

| Case | ARPACK solve(s) | Spectra solve(s) | Warp+cuDSS solve(s) |
|---|---:|---:|---:|
| nonproportional | 1.155124 | 2.221034 | 0.218532 |
| gyroscopic | 1.190982 | 2.076426 | 0.218731 |
| coupled-mass | 1.878366 | 3.356669 | 0.199320 |

[nonproportional raw trials](../../results/damped-large-balanced8-repeat.json), [occupancy](../../results/damped-large-balanced8-repeat.occupancy.json).
[gyroscopic raw trials](../../results/damped-large-gyroscopic.json), [occupancy](../../results/damped-large-gyroscopic.occupancy.json).
[coupled-mass raw trials](../../results/damped-large-coupled-mass.json), [occupancy](../../results/damped-large-coupled-mass.occupancy.json).

The coupled-mass fixture still has lumped spatial mass with non-diagonal 2x2
polarization blocks. These tests do not add a general nonsymmetric pencil or
left-mode API. The historical tol 1e-10 failure remains correctly reported as
nonconvergence; the validated practical profile is explicit above. Earlier
records' `versions.warp-lang` describes the installed distribution; the
`warp_runtime` field in newer records identifies the imported pinned overlay.
The runtime source hashes identify the actual code in both cases.
