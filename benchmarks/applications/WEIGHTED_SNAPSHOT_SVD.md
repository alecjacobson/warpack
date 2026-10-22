# Lumped-mass weighted snapshot SVD

The checkout-only [composition](weighted_snapshot_svd.py) extends the
[force-response ROM example](force_dual.py) for #135. Given response columns
G=H⁻¹F and positive diagonal mass M, it computes the leading singular triplets
of W=sqrt(M)G using public `PartialSVD`. Physical modes are u=M⁻¹/²U, with
covariance eigenvalues λ=σ². They are M-orthonormal and solve G Gᵀ M u=λu.
The force covariance F Fᵀ may be singular; neither it nor an n-by-n covariance
matrix is formed or inverted.

This avoids forming the normal matrix GᵀMG. Tests include σ=1e-6 alongside
σmax=2: the singular triplet passes physical checks even though its squared
value falls below the Gram example's 1e-12 relative eigenvalue cutoff. This is
a useful conditioning/rank distinction, not a guarantee of relative accuracy
for arbitrarily small singular values. Both paths require independent physical
residual, subspace, energy and source-response checks.

All numerical weighting, augmented SVD and physical recovery run in Warp.
Optional cuDSS only solves the sparse H responses. Storage for the rectangular
snapshot matrix is O(nr), represented as compact scalar CSR; the augmented
operator is implicit. This representation has substantial indexing/storage
cost compared with dense snapshots. The current recipe requires diagonal mass;
coupled mass whitening and implicit full-rank force priors remain open.

For existing device matrices and row-major FP64 force arrays:

```python
import warp as wp
from benchmarks.applications.force_dual import SnapshotROM

# H and M: Warp sparse SPD matrices; forces: Warp FP64 array (r,n).
# diagonal_mass: Warp FP64 array (n,), containing the positive diagonal of M.
rom = SnapshotROM(H, M, forces, k=20, backend="cg", diagonal_mass=diagonal_mass)
rom.capture()                       # setup, including graph initialization
wp.capture_launch(rom.response_graph)
rom.solve_modes()                   # no inter-stage host synchronization
# rom.vectors is (k,n); rom.svd.values are covariance eigenvalues.
# Check source-solve quality and original physical covariance equations.
```

Use `backend="cudss"` for sparse direct responses. The CG route uses the pinned
upstream Warp FSAI implementation with strict FP64 response solves. Force
values may change at the same address before replay. The standalone
`LumpedSnapshotSVD` also accepts changed response and positive mass values.
Invalid mass/nonfinite responses and insufficient positive rank produce a
nonzero device status; later valid replays can recover on the same graphs.
A supplied `diagonal_mass` must agree with M; this caller contract is not a
numerical comparison performed by the wrapper.

## Graph initialization and validation

The first integration test exposed #148: with lazy graph instantiation, the
first cuDSS response followed by a newly uploaded weighting graph returned an
incorrect weighted matrix on L40 / driver 570.158.01. The SVD correctly solved
that wrong matrix, so native singular-triplet counts alone did not detect the
physical error. A [minimal diagnostic](../experimental/response_graph_ordering.py)
reproduces the weighting error without an eigensolver. Its lazy first replay
has max error 1.5152; later replays and separately initialized controls have
zero weighting error. See [raw controls](../../results/force_dual/weighted-svd/ordering-minimal.json)
and [full-workflow controls](../../results/force_dual/weighted-svd/ordering-full-workflow.json).

The example now completes each graph's initial launch during **setup**, before
caller replay. That cost belongs in capture/setup timing. Changed-input tests
exercise the first caller replay; no synchronization was inserted between
stages during normal replay. This is containment of an observed runtime
interaction, not a claimed fix of the driver issue tracked in #61.
All three SVD graphs audit as device-only. The optional cuDSS response graph
retains the known vendor HtoD metadata copies tracked in #29.

The 10,890-DOF, 32-force, 20-mode
[screening case](../../results/force_dual/weighted-svd/screen-cudss.json) passes
against an independent weighted-SVD reference: worst original covariance
residual 8.36e-13 and physical subspace error 3.08e-13. This one-repeat screening
run is not a timing comparison. The strict paired benchmark command is:

```bash
python -m benchmarks.applications.force_dual --side 10 --columns 32 --modes 20 \
  --mass lumped --backend cudss --reference-method svd --reduction svd \
  --repeats 3 --out results/weighted-svd.json
```

Compare `--reduction gram` with identical inputs and references, and repeat with
`--backend cg`. The reporter checks all warmup/timed results, native counts,
rank, graph stages, occupancy and source/dependency fingerprints before allowing
a timing or regression claim.
