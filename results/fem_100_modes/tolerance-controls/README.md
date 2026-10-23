# Explicit tolerance tradeoff on the 100-mode FEM case (#147)

This study keeps the exact matrices, reference, 104-column starting pool,
preconditioner settings, source and dependency identities fixed. Warp uses 104
columns; CuPy uses the same pool's first 100 and retains absolute tolerance 1e-10.
Only Warp's declared normalized stopping tolerance changes. No production
default or physical validation threshold changes.

All original spectrum, requested subspace, mass orthogonality and residual
gates remain 1e-8; native stopping is checked separately. The accepted workers
have three measured replays per solver, with no observed GPU competition.

| Warp tolerance | Warp setup + capture (s) | Warp warm median (s) | CuPy warm median (s) | Worst Warp residual | Worst Warp subspace error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-11 | 4.3167 | 3.2455 | 2.8331 | 7.58e-12 | 2.00e-11 |
| 1e-9 | 4.2483 | 2.6512 | 2.7896 | 8.34e-10 | 1.49e-9 |

The looser profile needs 27 outer iterations and reduces Warp's warm median by
18.3%. Both methods meet the common physical target, and this particular
profile reaches similar runtime, but **their achieved accuracies differ**.
CuPy's worst normalized residual stays near 2.45e-11; its subspace errors are
below 6.50e-10. The strict Warp profile still has a performance gap. This is a
measured accuracy/work tradeoff for this fixture, not a solver optimization or
a universal recommended tolerance. Setup/capture is a single construction per
worker; CuPy adapter/upload additionally costs 0.090/0.084 s and uses the worker's
supplied preconditioner, so the table does not claim standalone CuPy setup.

The intermediate Warp 1e-10 worker is retained but rejected as a complete
comparison: all Warp solves pass, but one CuPy solve returns maximum absolute
residual 1.00227e-10 against its unchanged native 1e-10 request. Its physical
checks pass. This is another #69 native-stopping miss; no timing claim or ratio
uses that rejected worker. All three workers had no observed competing GPU job.

[Strict control](tol1e-11-a1.json), [looser profile](tol1e-09-a1.json),
[rejected intermediate worker](tol1e-10-a1.json),
[controlled-difference checks](comparison.json), [environment](environment.json).
Adjacent summaries exist only for accepted workers; the regression summarizer
rejects the intermediate one. Logs and external occupancy records retain every
attempt.
