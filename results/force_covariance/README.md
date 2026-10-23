# Implicit force-covariance qualification

[Application, equations, references and reproduction](../../benchmarks/applications/FORCE_COVARIANCE.md).
All fixtures are procedural. No paper data or meshes are included.

The 15 device tests and 32 reporter tests pass. Device tests include coupled
mass, singular force priors, changed-prior first replay, independent dense
subspaces, numerical symmetry probes and mixed inverse backends. The large
comparison's final performance qualification is pending idle GPU windows;
no timing table is promoted from preliminary or contended runs.

Preserved evidence:

| Directory | Role and limits |
|---|---|
| `screening/` | First large mass-prior runs; contended, one repeat. Accuracy diagnostics only. |
| `final/` | Earlier three-repeat mass/localized comparisons. They lack explicit exploration-policy metadata and are not accepted by the current strict reporter. The directory name predates that gate. |
| `budget-screen/` | One-cycle accuracy screens, one repeat; lack explicit exploration-policy metadata. Not performance baselines. |
| `exploration-control/` | Deliberate adaptive one-cycle rejection: all 20 physical modes match the independent reference, but zero recoveries leave the adaptive exploration guard incomplete. Worker exit 1 is expected and retained. |
| `qualified/` | Fresh three-repeat comparisons with explicit outer policy, full physical/native gates and occupancy monitoring. Only reporter-accepted rows may become timing baselines. |

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
