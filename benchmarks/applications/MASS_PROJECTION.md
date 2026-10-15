# Mass-weighted projection of known rigid modes

`MassOrthogonalComplementInverse` addresses [#42](https://github.com/alecjacobson/warpack/issues/42):
request the elastic block directly in a free-body generalized FEM problem,
including a non-diagonal consistent mass. This does not discover nullity or
remove modes automatically. Hodge/Betti applications must retain their harmonic
space; their counting diagnostics remain a separate contract in #39.

## Operator and lifecycle

For symmetric K, SPD M and a known nullspace basis with `Q.T M Q = I`, define
`P = I - Q Q.T M`. The wrapper applies

```text
P (K - sigma M)^-1 P.T
```

Generalized shift-invert composes this with M. Primal P and dual P.T differ
for a non-diagonal mass. Applying the same Euclidean projector on both sides
would solve the wrong problem. Tests compare to an independent dense product,
check M-orthogonality, and use a mass for which P and P.T visibly differ.
The original K/M residual and complete positive eigenspace remain the acceptance
criteria. A supplied arbitrary constraint space is not necessarily invariant:
this API does not implement arbitrary constrained variational eigenproblems.
Request at most n-rank(Q) retained modes; the inverse wrapper does not choose
the eigensolver capacity or certify completeness of the remaining spectrum.

The default requires mass-orthonormal FP64 basis rows on the inverse device.
`normalize=True` instead copies raw rows and runs two mass CholeskyQR passes
entirely in Warp. The input remains unchanged. A squared Cholesky pivot at most
1e-12 of that row's metric diagonal is treated as numerical rank deficiency;
ill-conditioned input bases may therefore be rejected. An orthonormality check
uses `basis_tolerance` (default 1e-10). The device `basis_status` is zero for a
valid metric basis and one otherwise. Invalid bases produce NaN output rather
than silently projecting with incorrect rows. This status does not prove that
K annihilates the supplied space; the caller owns that mathematical contract.

Construction preallocates scratch. `apply` supports multiple RHS rows through
a one-RHS inverse and exact input/output aliasing. Other overlapping views and
concurrent use on independent streams are unsupported. `refresh()` recomputes
MQ, normalization and validation after changing M or basis values; it can be
captured along with applications. Normalization starts from the original input
each time. Captures bind storage, so replacing arrays requires recapture.
Updating K, the shifted matrix and its numerical factors remains the caller's
responsibility. The wrapper does not own or close the inverse and forwards
inverse-history reset when supported.

All projection, basis normalization, metric products and status decisions are
Warp kernels with no numerical host readback. Only the optional cuDSS inverse
leaves Warp. Host reference assembly and validation in the benchmark are
outside setup/capture/replay timing boundaries.

## Example: one connected free body with consistent mass

The rigid-motion constructor below is an application helper from the source
checkout; the mass projection and generalized eigensolver are library APIs.

```python
import warp as wp
from warpack import (
    CuDSSInverse,
    GeneralizedKrylovSchur,
    MassOrthogonalComplementInverse,
    SparseOperator,
)
from examples.problems.elasticity import rigid_body_basis

# positions is a device vec3d array; K, M and shifted = K - sigma*M
# are compatible FP64 Warp sparse matrices. Choose sigma < 0.
a, mass = SparseOperator(K), SparseOperator(M)
# Unit nodal weights give the physical rigid span. Normalize in the actual
# consistent mass below, not in fictitious lumped coordinates.
rigid = rigid_body_basis(
    positions, wp.ones(positions.size, dtype=wp.float64, device=positions.device)
)
inverse = CuDSSInverse(shifted, 1, mtype="spd")
projected = MassOrthogonalComplementInverse(inverse, mass, rigid, normalize=True)
solver = GeneralizedKrylovSchur(a, mass, projected, k=20, ncv=64, target=sigma)
graph = solver.capture(5)  # Explicit budget; check physical accuracy for the input.
wp.capture_launch(graph)
# eigenvectors contains elastic modes directly; no six-row rigid prefix.
```

The same wrapper accepts `CGInverse` with upstream block-Jacobi or FSAI.
Disconnected bodies need separate six-dimensional rigid blocks per component.
The benchmark constructs these from known component geometry on device, never
from reference elastic eigenvectors or reference solver output.

## Validation and comparisons

Every measured
trial checks original-pencil residual, requested spectrum, M-orthogonality and
the complete reference positive subspace at 1e-8. Reported counts at the solver's
stricter 1e-11 tolerance are kept separately. No failed timing establishes a
performance win. Setup includes analytic basis construction and normalization;
capture and replay are measured separately. Fixed shifts and cycle budgets are
explicit experiments, not adaptive defaults.

On the same NVIDIA L40 / pinned upstream Warp FSAI environment as the FEM
report, the [composite](../../results/mass_projection_coverage.json) retains
**30 configurations: 27 numerical passes and three failed aggressive-shift
runs**. This combines separate experiments with original source/provenance;
it is not one full rerun. All accepted timed trials were uncontended.

Median replay times for six elastic modes, in milliseconds (three trials):

| Free consistent-mass beam / settings | Warp + cuDSS | Warp + block-Jacobi CG | Warp + FSAI CG |
|---|---:|---:|---:|
| 11,253 DOFs, projected, sigma=-1, ncv=32, one cycle | 17.570 | 505.274 | 199.523 |
| 42,483 DOFs, projected, sigma=-1, ncv=32, one cycle | 41.808 | 924.300 | 512.538 |
| 11,253 DOFs, projected, sigma=-100, ncv=64, one cycle* | 39.521 | 262.145 | 108.594 |
| 11,253 DOFs, projected, sigma=-100, ncv=32, two cycles* | 30.351 | 222.043 | 89.939 |

The first two rows reach **6/6 at solver tolerance 1e-11 on every trial for
all three backends**; maximum physical residuals are below 4.4e-12.
*The shifted CG rows pass external 1e-8 accuracy but report 0/6 at the tighter
1e-11 tolerance. Direct ncv=64 reports 6/6; direct ncv=32/two cycles reports
3/6. The latter budget has physical residual around 6.2e-9 for all backends.
These faster application-tolerance settings must not be presented as equal
strict convergence. Inner CG tolerance remains 1e-13 throughout.

Projection does **not** by itself substantially lower the cost of each CG
solve. At the same sigma=-0.01, ncv=32, one-cycle budget:

| Policy (same six elastic modes) | Warp + cuDSS | Block-Jacobi CG | FSAI CG |
|---|---:|---:|---:|
| Retain six rigid + six elastic | 17.074 | 760.429 | 310.099 |
| Project six rigid, return six elastic | 17.498 | 745.941 | 295.857 |

The extra projections add about 0.4 ms to direct replay here. FSAI executes
8,453 versus 8,825 inner iterations across the same 32 RHS calls, a modest
change. At ncv=64 and this shift, projected and retained FSAI times are likewise
similar (594.736 versus 605.111 ms). Both comparisons pass physical accuracy;
retained strict counts are incomplete, including direct counts 11/12, 11/12,
12/12 in the ncv=32 trials. Projected direct returns 6/6 on all trials.

The practical gain is in feasible shifted/outer budgets: sigma=-100,
ncv=64/one cycle **fails** the retained-rigid problem, recovering only two
rigid modes in the prior FEM report. The projected problem passes and returns
the complete elastic block. It does not make all short budgets safe:
sigma=-100, ncv=32/**one** cycle fails for all three backends (residual about
0.87), and those rejected timings remain in the composite. No parameter is
promoted to a universal default. Adaptive budgeting remains #31 and CG cost
remains #15; this does not rerun or improve the Anvil result.

The 1,950-DOF disconnected comparison removes twelve known rigid modes and
returns six modes spanning complete doubled elastic clusters. All three
backends pass the original-pencil and full-subspace gates; direct and FSAI
report 6/6, while block-Jacobi reports 4/6 at the stricter tolerance. This
exercises independent per-component rigid spaces rather than a single global
six-mode space.

The complete local suite passes **256 tests in 158.94 seconds**. Lint, Python
formatting and wheel build pass. The saved 10,890-DOF clamped direct benchmark
also passes accuracy and graph gates, with a replay ratio of 1.00013; see the
[regression report](../../results/mass_projection_existing_regression.json).
This is not a full upstream or Anvil rerun.

The new projection/normalization graph tests contain no HOST, HtoD or DtoH
nodes. Full iterative solve graphs likewise have no host callbacks/transfers;
cuDSS solves retain their vendor HtoD nodes (#29). These are DOT graph audits,
not a host synchronization API trace. Tests cover invalid/zero/nonfinite bases,
recovery after replacement, non-diagonal mass, aliasing and replay lifecycle.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_mass_projection.py
python -m benchmarks.applications.fem_compare --cases free --side 10 \
  --project-rigid --methods warp,warp-cg-bj,warp-cg-fsai --ncv 32 --steps 1 \
  --shift -1 --out results/mass_projection_free.json
```

See [FEM fixture and reference details](FEM.md) and the [pinned environment](README.md).
