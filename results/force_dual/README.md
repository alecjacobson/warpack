# Force-aware low-rank covariance ROM

This is procedural application coverage for
[Force-Dual Modes](https://arxiv.org/html/2505.23969v1), equations 7–14 and Appendix C,
under [#135](https://github.com/alecjacobson/warpack/issues/135). No paper mesh,
force data or implementation is copied. Geometry and force priors live under
benchmarks, outside the installed solver core.

For SPD physical stiffness H, SPD mass M and force covariance Sigma_F=F Fᵀ,
compute displacement responses G=H⁻¹F. The desired modes are the dominant
M-orthonormal eigenvectors of C M, where C=G Gᵀ. Solve the small symmetric
problem Gᵀ M G v=lambda v, then recover u=Gv/sqrt(lambda). Equivalently, the
physical symmetric generalized pencil is (M C M, M). The possibly singular
force covariance is never inverted, and C is never materialized at large n.

The first captured stage solves the sparse RHSs, applies M and forms the small
Gram matrix in Warp. The second uses public KrylovSchur with `which="LA"` and
lifts the physical modes in Warp. cuDSS is used only for the optional sparse
solve. The pure-Warp route uses upstream FSAI32 and FP64 batched CG at 1e-13,
with a 1,000-step cap and explicit native completion checks. Outer Gram work is
FP64. Supplied force values may change between replays; H/M/factor topology and
storage are fixed. Both graphs can be launched consecutively without host
numerical decisions.

## Matched results

L40, Xeon Platinum 8362, four CPU threads on cores 8–11; 2026-10-22. A generated
clamped P1 beam has 10,890 free DOFs and 32 independent localized force columns
on tip/contact regions. Request 20 dominant modes. The matrix categories are
real symmetric SPD stiffness and mass, with a low-rank PSD force covariance.

One warmup and three repeats per fresh worker, all uncontended. Runtime columns
are medians; setup includes host assembly, device conversion, sparse factor and
solver construction, warmup and graph capture. Setup is a single observation,
not a cold-process median. References and validation are outside solver timing.

| Mass / sparse backend | Assembly + setup + capture (s) | Responses + Gram (s) | Small eigenproblem + physical lift (s) | Warm total (s) |
|---|---:|---:|---:|---:|
| consistent-cudss | 0.9318 | 0.0019 | 0.0018 | 0.0037 |
| consistent-cg | 1.0435 | 0.0345 | 0.0018 | 0.0363 |
| lumped-cudss | 0.9637 | 0.0018 | 0.0017 | 0.0036 |
| lumped-cg | 1.0398 | 0.0346 | 0.0017 | 0.0363 |

All 16 warmup/timed workflows pass sparse response residuals, dominant covariance
spectrum, original covariance equations, M-orthogonality, complete requested
subspaces and expected reconstruction energy. Native eigensolver status/count,
positive rank and CG completion are also required. Independent SciPy sparse
responses and dense small Gram references are validation only; their runtime
is recorded separately and is not a matched ARPACK/CuPy performance comparison.

Small tests additionally compare a full dense generalized covariance pencil,
exercise dependent force columns and changed priors on replay, and verify that
Sigma_F=M recovers ordinary modal subspaces with covariance eigenvalues 1/κ²,
where Hφ=κMφ. Rank exhaustion and a zero force prior reject the requested positive
mode count, and a valid later replay recovers on the same graphs.

The pure-Warp response/eigen graphs have no HOST, HtoD or DtoH nodes. cuDSS's
response graph retains the known vendor HtoD metadata copies in
[#29](https://github.com/alecjacobson/warpack/issues/29); the eigen/lift graph has
none. DOT inspection is not an API trace. Setup and host verification synchronize
explicitly. All physical numerical response/reduction/lift work stays on device.

## Limits and reproduction

An optional `--reference-method svd` now independently checks the modes with
the SVD of mass-weighted responses, avoiding a Gram eigenproblem in the
reference. Diagonal mass needs only row scaling; small coupled mass uses a
CPU dense Cholesky factor, capped at 2,048 variables. That cap prevents silent
large dense mass construction. This is validation code, not a new GPU weighted
SVD API. Singular-value rank uses relative cutoff 1e-12, explicitly distinct
from the existing covariance-eigenvalue cutoff 1e-12. Tests recover a known
snapshot singular direction of size 1e-8 that the latter policy excludes.
The runtime Gram path and its documented cutoff are unchanged.

Eight local-force/replay cases now use this independent weighted-SVD reference
and still compare against a full dense covariance pencil. Together with six
new CPU reference tests and the existing modal/rank tests, all 17 pass. Full-rank
implicit priors and a GPU weighted-SVD alternative remain open under #135.

The [10,890-variable / 32-force / 20-mode CG validation](svd-reference/)
also passes all four workflows against the weighted-SVD reference. Worst
source residual is `7.37e-13`, covariance residual `1.98e-12`, M-orthogonality
error `3.59e-10`, and complete-subspace error `1.74e-10`. All positive/native
counts pass. Another GPU process was present, so its raw timings are excluded
from performance conclusions. This uses the current FSAI overlay and does not
replace the earlier dependency-specific timing table.

The [combined 89-test regression run](svd-reference/tests.log) and
[31-test source/report follow-up](svd-reference/source-tests.log) include
the independent reference and #145 fingerprint changes. No installed solver
arithmetic changed.

The archived measurements above used a single-driver source fingerprint.
Under [#145](https://github.com/alecjacobson/warpack/issues/145), new workers
record transitive local imports and the full Warpack source package. Do not
treat the historical fingerprint as proof of identical helper/core sources.

`SnapshotROM` in
[`force_dual.py`](../../benchmarks/applications/force_dual.py) is a checkout-only
application driver. It is not a new public covariance API. The Gram formulation
squares snapshot conditioning and uses an explicit relative covariance cutoff
1e-12 for positive rank. It does not promise to recover arbitrarily weak modes.
The nullity of Sigma_F is distinct from a singular physical mass matrix.

The source-response approach preserves a symmetric PSD covariance even when
responses are approximate; original-H residuals and independent physical checks
bound the error in these cases. This does not prove that a tolerance-stopped CG
inverse inside a repeatedly applied implicit covariance operator is linear or
symmetric. Large full-rank priors, repeated implicit covariance applications,
condition-sensitive weighted SVD and their comparisons remain open in #135.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_force_dual.py tests/test_rom_followup_report.py
python -m benchmarks.monitor_gpu --out build/force.occupancy.json \
  --log build/force.log -- taskset -c 8-11 \
  python -m benchmarks.applications.force_dual --mass consistent \
  --backend cg --side 10 --columns 32 --modes 20 --out build/force.json
# Record truthful hardware metadata next to the report, then:
python -m benchmarks.applications.rom_followup_report build/force.json --kind forces \
  --baseline results/force_dual/consistent-cg.summary.json
```

Use `--backend cudss` for the other sparse backend. The reporter rejects
contended/incomplete runs, failed native or physical checks, insufficient rank,
unconverged source CG, invalid timings and unexpected host graph operations.
The 25% warm-workflow regression gate requires compatible source/matrices,
configuration and environment; setup is not gated.
