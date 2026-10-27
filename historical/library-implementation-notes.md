# Library implementation and qualification notes

Archived from the usage guide on 2026-10-27. These paragraphs describe specific
implementations and recorded experiments, not the public API contract. See the
[current documentation](../docs/index.md) for supported usage and the linked
reports for exact revisions, settings and limitations.

For the tested 100/200-mode FEM configurations, `keep_history=False` reduces
the projected work enough to outweigh additional outer iterations. It keeps all
current Ritz vectors and compacts the R/P directions with the active residuals.
[History-policy controls](../results/fem_100_modes/history-current/README.md)
include setup, first solve, replay, CuPy comparisons and rigid/null-mode cases.
The Anvil recipe retains `keep_history=True`; its workload favors that tradeoff.

Device-selected CG buckets share maximum-width packing and adapter application
storage. Upstream CG's internal workspaces and public `for_width` variants
remain independent. This reduces memory without changing stopping tests;
[allocation controls](../results/cg_adapter_storage/README.md) quantify the savings.

Active-search capture directly records all branch kernels for built-in
`SparseOperator`, `BatchedSparseOperator` and `Identity` applications. This avoids
an extra numerical solve of every branch during preparation. Custom stiffness
or mass operators (including subclasses and instance overrides) retain eager
per-width preparation. The initial eager solve and synchronization before
capture remain. [Preparation controls](../results/capture_preparation/production/README.md)
report the first-solve benefit; replay arithmetic is unchanged.

For one-time ROM generation, `BlockLOBPCG(..., compact_projected_graphs=True)`
reduces the graph construction work for wide projected solves. It records
16 Jacobi rounds per device-loop iteration, preserving FP64 rotation order and
convergence tests. The default remains `False`: compact graphs can cost more
per warmed replay, so repeated solves may favor the default. Only existing
even-dimensional global projected solves of size 96–1024 change; the Anvil
recipe and small/odd projected problems keep their established paths.
Both global graph policies restore projected symmetry before testing convergence;
[regressions](../results/projected_symmetry/VALIDATION.md) include repeated and null spectra.
For projected dimensions ≥256, a final device kernel normalizes eigenvector
columns to contain accumulated length drift. Eigenvalues and convergence tests
are unchanged; [validation](../results/normalized_jacobi/VALIDATION.md) includes
the large repeated/null regression and both graph policies.
On Ada GPUs, even global projections of size 256–1024 reuse each 2×2 output
block within a Jacobi round. Both graph policies retain the original FP64
expressions and rotation order; [validation](../results/paired_jacobi/VALIDATION.md)
includes bit-identical legacy controls. Smaller projections keep their existing
update because reduced parallelism offsets the reuse there.
[Complete setup/capture/replay controls](../results/compact_jacobi/README.md)
report the tradeoff separately from the main six-backend comparison.

The current pinned FSAI supports `reuse_pattern=True` and captured `update()`.
After changing matrix values at stable storage addresses, enqueue the update
before the solve on the same stream. Check `update_status` after replay:
a failed refit preserves the preceding factor, which is not a valid replacement
for a successful update of the changed problem. Fixed topology and factor
storage are required for replay. `defer_pattern=True` selects supports on its
first update; Warpack accepts the resulting padded factors, but initializer
warmup can trigger that first selection before capture. Regularized refits are
an explicit upstream option, not enabled by Warpack's examples.

Upstream FSAI setup can consume much more device memory than its stored factors.
The [current dependency qualification](../results/fsai_setup_regression/README.md)
reduces this cost with packed workspace and scalar BSR entry access. The older
[row-size comparison](../results/fsai_memory/README.md) remains useful context;
neither report qualifies smaller-memory GPUs or makes a peak-memory guarantee.



The [FEM reuse example and measured comparison](../results/fem_public_starts/README.md)
show computed starts for related material configurations, including the setup
costs and the fixed-address replay contract. The
[shape-change qualification](../results/fem_shapes/README.md) additionally updates
K and M together, exercises upstream FSAI refits at fixed storage addresses, and
checks mode crossings and complete repeated clusters. For an opt-in response to
a rejected warm start, see the [bounded recovery example](../docs/completeness.md#opt-in-recovery-after-a-rejected-warm-start).
The [coarse-to-fine example](../results/fem_hierarchy/README.md) transfers computed
tetrahedral modes with the device initializer hook and includes the cost of
building and solving the coarse problem. Such starts need not improve total time.
The [100-mode FEM comparison](../results/fem_100_modes/README.md) records complete
requested-space checks, native stopping differences and the remaining wide-block
replay gap against CuPy.
The [projected-factor follow-up](../results/parallel_cholesky/README.md) measures
a faster pure-Warp Cholesky path for wide blocks, while retaining the existing
small-block path used by Anvil. CuPy's iteration variability remains explicit.
The [200-mode qualification](../results/global_cholesky/README.md) extends that
factor to CUDA widths 129–512 with a setup-time device workspace of
`8 * width**2` bytes, shared by active-search branches. Widths 48–128 retain the
qualified packed shared-memory path when capacity permits; other widths and
non-CUDA devices retain the serial fallback. No allocation or host convergence
decision is added to replay.
On Ada GPUs, Ritz widths 96–256 also use a register-blocked triangular
application, retaining each row's subtraction order when QR runs.
It adds no global workspace, inverse or replay synchronization. Other widths and
architectures retain the scalar application. The [installed qualification](../results/register_triangular/VALIDATION.md)
includes changed-input/failure recovery, clean-wheel replay and complete FEM
controls; Anvil uses the original triangular kernel.

With `gram_chunk=4096`, Ada devices use 16×16 FP64 Gram output tiles for block
widths 24–256. This path adds no global workspace or replay synchronization;
the QR policy and all rank/stopping thresholds are independent of this tile choice. Other settings
retain the original Gram implementation. The [Gram qualification](../results/gram_tiles/VALIDATION.md)
records complete FEM comparisons and original-equation/replay checks.

With `paired_rotation=True`, Ada SM89 devices use eight output rows per FP64
Ritz-rotation tile for `n >= 32768` and block widths 24–256. The inner tile stays
eight columns wide; separate bounded basis views preserve both the updated
vectors and search directions. Other configurations retain the four-row
kernel. This adds no workspace, host readbacks or accuracy-policy changes.
The [rotation qualification](../results/paired_rotation_tiles/VALIDATION.md)
records whole-solver controls and installed-package checks.

For Ada block widths 96–256, `gram_chunk=4096`, `tol >= 1e-11`, and both
`reuse_operator_images=True` and `symmetric_products=True`, captured execution
checks a fresh mass Gram before reorthogonalizing Ritz vectors. If the finite
computed metric defect is at most `1e-13`, it retains the basis and cached
operator images with a fresh mass image. Otherwise it runs both original QR
passes. This uses owned Warp sparse/identity mass operators; custom operators,
stricter tolerances, eager execution and other configurations retain full QR.
The bound concerns the computed Gram, not exact-arithmetic mass products;
original-equation stopping and rank thresholds remain unchanged. Mutually
exclusive search branches share 24 bytes of control/counter values and add no
dense workspace or replay host readback. The [QR reuse qualification](../results/metric_qr_reuse/VALIDATION.md)
records complete physical/replay and packaging checks. Anvil's width 24 retains
full Ritz-basis QR.

For Ada CUDA devices, search metrics of size 96–512 use certified Cholesky
whitening during capture. Both whitening passes remain. Each fast path must
establish a fresh conservative spectral bound above the unchanged relative
rank threshold; uncertain or failed attempts use canonical rank revelation on
device. The first pass also handles exact zero padding: an auxiliary identity
helps certify the active block, but its added directions never become physical
modes. Failed normalization retains the original failure status and canonical
fallback.

Both passes share three FP64 square scratch arrays across mutually exclusive
search branches (`24 * metric_size**2` bytes for the largest eligible metric,
plus small flags). The first pass adds no dense workspace, host numerical
decision, or transfer. Smaller metrics, other architectures and eager solves
retain canonical whitening. See the [first-pass qualification](../results/first_whitening/VALIDATION.md)
and [second-pass qualification](../results/certified_whitening/README.md).
The [coarse accuracy/width follow-up](../results/fem_hierarchy_tuning/README.md)
measures that tradeoff while holding final fine-grid accuracy fixed.

For force-aware ROMs, the [low-rank covariance example](../results/force_dual/README.md)
solves sparse stiffness responses with cuDSS or pure-Warp CG, forms a small
symmetric Gram problem and lifts its dominant modes into physical coordinates.
It accepts singular force covariances without inverting them; physical stiffness
and mass remain SPD. This is an application recipe outside the solver core.
For diagonal mass, the [weighted snapshot SVD recipe](../benchmarks/applications/WEIGHTED_SNAPSHOT_SVD.md)
avoids forming the Gram matrix and tests weak positive singular directions.

