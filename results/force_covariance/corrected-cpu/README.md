# Independently charged reciprocal CPU recovery (#153)

The diagnostic inverse-step/covariance-Ritz correction now has a standalone
benchmark adapter. It constructs and charges its own explicit
`Q = H S^-1 H` and SuperLU factor, plus its own H factor. It receives no factor
from the independent covariance reference. Each measured solve includes native
explicit-Q ARPACK followed by two H block solves for the inverse step and two
more for covariance projection, plus sparse products and dense Ritz recovery.
This is an explicitly corrected CPU comparison, not native ARPACK and not
installed Warpack solver code.

The procedural clamped tet beam has 10,890 DOFs and 20 requested modes, with a
positive diagonal force prior. Four sequential workers use four pinned Xeon
threads (8–11), one independently validated warmup and three measured repeats.
Both masses pass every original covariance and reciprocal residual, spectrum,
subspace, mass-orthogonality and count check at the unchanged 1e-8 gate after
recovery. External occupancy records show no GPU competition during any worker;
this is CPU execution, not a GPU speed result.

| Mass | Independent Q/H setup (s) | Complete warm median (s) | Recovery-stage median (s) | Worst covariance residual | Worst Q residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lumped | 3.2399 | 1.5680 | 0.4175 | 3.23e-14 | 4.96e-10 |
| Consistent | 3.2137 | 1.6572 | 0.4155 | 3.23e-14 | 1.29e-9 |

Setup includes Q assembly/factorization (2.3632/2.3383 s) and recovery H
factorization (0.8767/0.8754 s). The H factors contain 9,025,644 combined L/U
nonzeros and Q factors 14,668,536. Assembly of FEM inputs, independent reference
generation, and validation are excluded from setup/solve and separately recorded.
Setup is one construction per worker, not a repeated median.

Native explicit-Q outputs are retained before correction and fail the common
gates in every trial. Independent native implicit-covariance ARPACK controls
have tiny covariance residuals (below 4.6e-14) and remain valid for the original
covariance task, but fail this stronger **two-equation** comparison: explicit-Q
residuals reach 1.05e-8 (lumped) and 1.13e-7 (consistent). Those complete workers
are rejected and no speed ratio is claimed against them. A small covariance
residual alone does not control the residual after applying the ill-conditioned
reciprocal operator. This does not invalidate their separately qualified
covariance-only results.

The corrected and implicit workers have identical H/M/prior/Q coefficients,
reference values/bases, initial vectors, source hashes and versions within each
mass. Native stopping tolerance is 1e-12, with ncv 64 and the same fixed start.
The adapter is a qualified workaround; it does not repair native explicit
assembly/factorization conditioning. #153 remains open for that distinction.

Raw records and their adjacent logs/occupancy records:
[lumped corrected](lumped-corrected-a1.json),
[consistent corrected](consistent-corrected-a1.json),
[rejected lumped implicit](lumped-implicit-a1.json),
[rejected consistent implicit](consistent-implicit-a1.json).

Reproduce with `python -m benchmarks.applications.force_reciprocal_corrected
--backend corrected --side 10 --modes 20 --repeats 3 --out NEW.json`, optionally
`--consistent`; `--backend implicit` retains the failing native control.
Use four BLAS/OpenMP threads and CPU affinity 8–11 as above. Twelve focused CPU
tests cover diagnostic and standalone recovery, dense independent spectra,
coupled masses, independent factor ownership, four charged recovery solves and
both physical equation gates.
