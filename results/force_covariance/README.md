# Implicit force-covariance qualification

[Application, equations, references and reproduction](../../benchmarks/applications/FORCE_COVARIANCE.md).
All fixtures are procedural. No paper data or meshes are included.

The 15 device tests and 32 initial reporter tests pass. Device tests include coupled
mass, singular force priors, changed-prior first replay, independent dense
subspaces, numerical symmetry probes and mixed inverse backends.

## Qualified implicit comparison

Fresh measurements on 2026-10-23: NVIDIA L40, Xeon Platinum 8362, four pinned
CPU cores, prepopulated kernel cache, verified Warp 1.18 CUDA 12 FSAI overlay.
The procedural clamped cantilever has 10,890 unknowns, lumped mass, 20 requested
modes and Krylov width 64. Each row has an accepted warmup and three accepted,
uncontended repeats. All nine rows pass the current strict reporter.

| Prior | Solver | Complete setup (s) | Median warm solve (s) | Worst covariance residual |
|---|---|---:|---:|---:|
| Full-rank nonuniform diagonal | CPU ARPACK + SuperLU | 1.0271 | 1.3900 | 4.53e-14 |
| Full-rank nonuniform diagonal | Warp + cuDSS | 1.2121 | 0.08820 | 3.74e-13 |
| Full-rank nonuniform diagonal | Pure Warp FSAI/CG | 4.6656 | 1.29935 | 5.20e-13 |
| `S=M` sanity check | CPU ARPACK + SuperLU | 1.0477 | 1.36060 | 4.53e-14 |
| `S=M` sanity check | Warp + cuDSS | 1.2051 | 0.08821 | 3.66e-13 |
| `S=M` sanity check | Pure Warp FSAI/CG | 4.5837 | 1.29688 | 5.23e-13 |
| Rank-32 localized prior | CPU ARPACK + SuperLU | 1.0587 | 1.42569 | 1.43e-13 |
| Rank-32 localized prior | Warp + cuDSS | 1.2393 | 0.09158 | 6.48e-13 |
| Rank-32 localized prior | Pure Warp FSAI/CG | 4.5963 | 1.29291 | 8.79e-13 |

Setup includes procedural assembly, device uploads/factors/preconditioners, and
graph capture/initialization. No factors are shared between rows or borrowed
from the reference. For example, diagonal-prior CG setup is 0.1826 s assembly,
0.5883 s solver construction and 3.8948 s capture/initialization. ARPACK has no
graph capture. Reference construction and output validation are outside both
setup and solve timings. Raw stage timings, samples, provenance and occupancy
are in [`qualified/`](qualified/); `.summary.json` files are reporter outputs.
`diagonal-arpack-a1` is contended and excluded; the independently repeated `a2`
row is used above.

GPU rows explicitly use **one fixed cycle**. They pass independent complete
requested-subspace checks, with worst subspace error below 9e-13 across all
warmups and repeats. The full-rank rows have zero recoveries and do not satisfy
the generic adaptive exploration guard. The rank-32 localized rows have 29–30
recoveries and do satisfy that guard, but still use explicit fixed capture.
ARPACK uses its native stopping. Core tolerances differ from the shared physical `1e-8` acceptance
gate; all counts and requested positive ranks must also pass.

These are comparisons of the **implicit covariance formulation**. For `S=M`,
ordinary elastic modes avoid this work entirely. For rank-32 localized forces,
the [snapshot reduction](../../benchmarks/applications/WEIGHTED_SNAPSHOT_SVD.md)
is the better established route; its earlier separate comparison is not a
matched timing baseline for the table above. Full-rank diagonal priors do not
have either shortcut. The reciprocal alternative is being qualified in
[#152](https://github.com/alecjacobson/warpack/issues/152).

Preserved evidence:

| Directory | Role and limits |
|---|---|
| `screening/` | First large mass-prior runs; contended, one repeat. Accuracy diagnostics only. |
| `final/` | Earlier three-repeat mass/localized comparisons. They lack explicit exploration-policy metadata and are not accepted by the current strict reporter. The directory name predates that gate. |
| `budget-screen/` | One-cycle accuracy screens, one repeat; lack explicit exploration-policy metadata. Not performance baselines. |
| `exploration-control/` | Deliberate adaptive one-cycle rejection: all 20 physical modes match the independent reference, but zero recoveries leave the adaptive exploration guard incomplete. Worker exit 1 is expected and retained. |
| `qualified/` | The nine accepted three-repeat comparisons above, plus the preserved contended first ARPACK diagonal attempt. |

The exploration control makes a useful distinction: a full native residual
count is not the same as adaptive completion. An explicit fixed-budget solve
can be validated against the independently requested full subspace for this
fixture, without weakening the generic solver's exploration safeguard.

The reference for `S=M` uses ordinary elastic modes and inverse squared
eigenvalues; the localized reference uses an independent snapshot reduction.
The nonuniform diagonal reference uses a larger independently initialized
ARPACK implicit solve. Small dense tests provide a separate mathematical oracle.
Reported CPU timing constructs new H/M factors after reference construction;
its setup includes its own factorization. Neither competitor times validation.
