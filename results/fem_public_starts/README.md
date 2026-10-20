# Computed starts with the public fast FEM solver

This follow-up to [#128](https://github.com/alecjacobson/warpack/issues/128)
qualifies the installed v0.2.0 fast APIs for related FEM problems. Each source
pencil is solved by Warp. Young's modulus then increases by 2% in the beam's
right half; mass stays fixed. No reference eigenvectors are supplied as starts.
The changed-pencil comparison uses the same rebuilt FSAI factor for every start.

Both mass types have **34,560 DOFs**, 20 requested modes and width 24. Three
start policies are compared: random; the previous 20 modes plus four random
rows; and the previous full 24-row block (padding rows need not have converged
to the requested tolerance in the source solve).

All 36 timed trials—two mass types, three starts, two solvers, three repeats—pass
original residual/spectrum/subspace/mass-orthogonality gates and their respective
stopping-count/tolerance checks. Runs were serial and occupancy monitoring saw
no competing GPU processes or errors. The source solves also pass. The Warp
graph audits contain only device-to-device copies and no host nodes.

| Mass | Start | Warp median ms | Warp iterations | CuPy median ms |
|---|---|---:|---|---:|
| Consistent | Random | 341.54 | 24–25 | 655.31 |
| Consistent | Previous 20 modes | 258.46 | 19–20 | 430.65 |
| Consistent | Previous 24-row block | 201.25 | 15 | 445.60 |
| Lumped | Random | 358.91 | 26–27 | 833.04 |
| Lumped | Previous 20 modes | 258.39 | 20 | 525.73 |
| Lumped | Previous 24-row block | 212.94 | 16 | 393.24 |

Using the previous full block reduces Warp warm replay time by about **41%**
in both cases. This measures reuse of a solution that already exists. The
source solve, changed-factor setup, graph capture, host assembly and independent
references are excluded from those medians and recorded separately in each JSON.
It is not a first-solve speedup or a claim that warm starts always help.

CuPy is native LOBPCG with the same supplied Warp FSAI/CG preconditioner and
identical starts. It converges its entire width-24 block to absolute `1e-9`;
Warp stops on the first 20 at normalized `1e-10`. Both must pass the shared
`1e-8` original physical gates. These stopping conventions are deliberately
explicit rather than equating their tolerance arguments.

Raw records: [consistent](clamped_consistent_s15_k20.json),
[lumped](clamped_lumped_s15_k20.json). Occupancy JSONs contain exact commands.
Source hashes and package versions are recorded. Hardware is NVIDIA L40 48 GB,
driver 570.158.01, pinned Warp 1.18.0+cu12/FSAI overlay, CuPy 14.2.0.

## Reproduction

Use the [pinned installation](../../docs/installation.md#fast-block-solves).
Change `--case` to `clamped_lumped` for the other mass type.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu --out build/starts.occupancy.json \
  --log build/starts.log -- python -m benchmarks.experimental.fem_starts \
  --case clamped_consistent --side 15 --modes 20 --width 24 --steps 300 \
  --fsai-width 64 --change .02 --tol 1e-10 --repeats 3 \
  --inner-cg 25 --active-cg-bucket-size 4 --inner-fp32 --inner-tol .01 \
  --batched-products --gram-chunk 4096 --symmetric-products --fused-updates \
  --paired-rotation --reuse-x-image --public-api --active-search-bucket-size 4 \
  --active-search-keep-history --fsai-layout columns --matrix-layout columns \
  --cupy --cupy-tol 1e-9 --out build/starts.json
```

FSAI64 has a substantial setup/context memory cost on this GPU; see
[#131](https://github.com/alecjacobson/warpack/issues/131). This is separate from
factor storage and from the released Anvil recipe, which already uses FSAI32.

## Same-graph update regression

`tests/test_fem_fast_replay.py` separately checks both mass types on small
independent FEM fixtures. One graph is captured and replayed through four
regional stiffness states, copying the computed modes into separate start
storage on device. It retains the initial SPD FSAI as an approximate
preconditioner while updating both the outer stiffness and the inner-CG matrix.
All four physical eigensystems are independently verified. The sparse pattern
reserves structural zeros that a later material edit may un-cancel.

This checks fixed-address replay and FP32 working-matrix refresh. It is separate
from the timings above, which rebuild changed-pencil factors and capture each
start policy. Factor refit, larger changes, large meshes and material contrasts
remain separate qualification work. Numerical solver code is unchanged.

Focused validation: **56 passed, 1 skipped** in the pinned environment, including
the new replay regressions, the earlier FEM families, release API tests and
reference-error/graph gates. Stock checks: **6 passed, 2 skipped**. The legacy
non-public harness profile also passes a small independent three-start smoke
check. This is not a new full-library or upstream conformance run.
