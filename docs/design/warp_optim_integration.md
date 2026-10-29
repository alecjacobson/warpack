# Proposal to integrate sparse eigensolvers into warp.optim

**Status:** Draft for discussion. **Updated:** 2026-10-29.

[Documentation](../index.md) · [Current API](../api/README.md) · [Current execution contracts](../library_contracts.md)

## Recommendation

Add `warp.optim.eigen`, initially for real symmetric standard and generalized
partial eigenproblems. Expose `KrylovSchur` and `LOBPCG` as public reusable
solver states, with matching `krylov_schur` and `lobpcg` convenience functions
and Warp-array results on the selected CPU or CUDA device. Support both CPU and
CUDA from the first public integration. Reuse Warp's sparse matrices, linear
operators, CG, GMRES and preconditioners. Keep all eigensolver arithmetic in Warp; allow an
optional cuDSS provider for sparse factorization and solves.

`warp.optim.eigen` will not construct problem matrices. Callers provide assembled
matrices or matrix-free operators; examples and tests will use `warp.fem` for
matrix construction as much as possible. Include a reusable stateful linear
system solver abstraction in `warp.optim.linear`, so eigen routines can borrow
pure-Warp or optional cuDSS solve state through the same contract. Pure Warp is
the default numerical backend.

The primary workflow is reduced-order model construction:

$$K u_i = \lambda_i M u_i, \qquad u_i^T M u_j = \delta_{ij},$$

where stiffness $K$ is symmetric and mass $M$ is symmetric positive definite
(SPD). Support both an iterative block method for low modes and a Krylov method
with an explicitly supplied shifted solve. This also serves many geometry
processing problems with stiffness-like operators and mass metrics.

Integrate in stages rather than exposing all 31 current warpack exports in Warp.
Preserve the existing implementation and comparison suite as migration controls.
Complex problems, constrained pencils and SVD need separate frontend reviews;
their existing warpack implementations need not disappear while that happens.

This proposal changes no executable API. Names and code below describe the
proposed interface, not calls available in current Warp or warpack. The audience
is Warp maintainers and warpack contributors deciding scope and contracts before
implementation.

## Motivation and scope

Applications already assembling and simulating in Warp should be able to compute
modes without exporting matrices to a host eigensolver or introducing a second
GPU array runtime. A reusable state should compose with simulation CUDA graphs,
consume changed device values and starting subspaces, and expose convergence
information without a host readback.

| Problem class | Matrix properties and requested output | Proposed treatment |
|---|---|---|
| Elastic and Laplacian low modes | Symmetric $K$, SPD $M$; often semidefinite $K$ with a known nullspace | First stage: generalized block LOBPCG, smallest algebraic modes |
| Interior or near-zero symmetric modes | Symmetric $K$, SPD $M$; invertible $K-\sigma M$ | First stage: generalized Krylov–Schur with supplied inverse action |
| Standard symmetric spectrum ends | Symmetric $A$, identity mass | First stage: Krylov–Schur; algebraic and magnitude selections |
| Largest generalized modes | Symmetric $K$, SPD $M$ | First stage: metric Krylov–Schur with supplied mass solve |
| Equality-constrained physical modes, including skinning formulations | Symmetric physical Hessian, SPD physical mass, independent constraint rows; augmented mass singular | Follow-up: dedicated constrained frontend with KKT solve |
| Connection or conformal operators with complex coefficients | Hermitian or general complex operator, possibly a generalized pencil | Follow-up after complex storage and result contracts are agreed |
| Damped dynamics and other general pencils | Nonsymmetric $A,B$; regular pencil, possibly singular $B$ | Follow-up general eigensolver state; finite right eigenpairs, explicit selection and failure policy |
| Snapshot/POD compression | Rectangular matrix; singular triplets | Follow-up partial SVD state, requiring forward and adjoint operator actions |

Smallest algebraic (`SA`) and smallest magnitude (`SM`) are different requests.
They coincide for a positive semidefinite operator, but not for an indefinite
stiffness or Hessian. Free bodies require inclusion or deliberate deflation of
rigid modes; an inverse at exactly zero is invalid if those modes make $K$
singular. A negative shift can make the shifted matrix SPD for semidefinite $K$.

The initial contribution is not a full SciPy/ARPACK compatibility layer, a dense
eigensolver API, a mesh assembly library, a distributed solver, or a promise of
global spectral completeness from residuals alone. CPU and CUDA execution are
required for the initial public solver algorithms, with the same mathematical
contracts and frontend. CPU support is a release requirement, not an optional
follow-up. Both paths use Warp numerical kernels; neither invokes a hidden
NumPy, SciPy or other host numerical fallback.

Choose execution from the operands' Warp device. GPU-specific tiled kernels,
cooperative reductions and graph control flow need CPU-compatible Warp paths
where their implementation cannot run on CPU. These may use different kernels
while preserving selection, normalization, residual and failure semantics.
Results and workspaces remain on the selected device. Here, device-resident
arrays include Warp CPU arrays; the API does not implicitly convert either
backend's results to NumPy or Python scalars.

CUDA graph capture is a CUDA execution capability. The CPU path executes the
same prepared solver eagerly and does not require graph emulation. Optional
cuDSS remains CUDA-only; requesting that provider on CPU raises a clear backend
error, while the pure-Warp solvers and inverse paths remain available on CPU.

### Matrix construction belongs to the caller

The eigen module consumes matrices and operators; it does not assemble stiffness,
mass, shifted, constraint/KKT or application-specific matrices. In particular,
the caller supplies the system $S=K-\sigma M$ to a linear solver before passing
that solver into an eigen routine. Omitting a mass matrix uses an identity action,
not an assembled identity matrix. A matrix-free composition may apply a shift
without materializing it. A direct solver may require an explicit matrix, which
the caller must supply.

This boundary concerns problem matrices. Internal Gram and projected Ritz
matrices, basis storage and numerical workspace remain part of the eigensolver.
Backend storage conversion and factorization belong to the linear solver; they
do not authorize application matrix assembly inside `warp.optim.eigen`.

Examples and tests should use `warp.fem` as much as possible to assemble their
stiffness and mass matrices, including procedural elasticity and Laplacian cases,
consistent and lumped mass, and boundary conditions. Keep assembly in example
drivers or test fixtures, with reusable assembly functionality in `warp.fem`
and sparse operations in `warp.sparse`. The eigen implementation must not depend
on `warp.fem`, mesh loaders or application assembly helpers.

Retain directly constructed diagonal/block fixtures and independent small dense
references where they isolate selection, multiplicity or failure behavior better
than a FEM problem. Using `warp.fem` for realistic fixtures must not make it the
sole accuracy oracle. Exercise representative FEM assembly-to-solve workflows on
both CPU and CUDA, and report assembly cost separately from solver setup.

## Current implementation and integration work

warpack currently has FP64 outer solvers, row-oriented vector blocks, complex128
storage represented by `wp.vec2d`, and optional FP32 work in block CG/FSAI
preconditioning. It has no supported eigensolver adjoint. Several wrappers have
different result fields and residual conventions. Its fast paths also depend on
specific upstream Warp features; the installed version string alone does not
identify that environment. See [installation](../installation.md),
[the solver guide](../guide.md) and [validation](../development.md).

The frontend should remove those inconsistencies without treating every
implementation option as a permanent public contract. In particular, graph
buckets, projected factorization choices, cached products and fused updates
should be implementation policy. The fast Anvil configuration must remain
reproducible through documented solver and preconditioner settings, with the
necessary optimized dispatch covered by regression tests.

## Alignment with Warp conventions

The API audit uses upstream revision
[`0c1ed78f`](https://github.com/NVIDIA/warp/tree/0c1ed78f7b4ee397534e5bd1cea632af618d900f),
checked on 2026-10-27. Its public
[`warp.optim.linear` exports](https://github.com/NVIDIA/warp/blob/0c1ed78f7b4ee397534e5bd1cea632af618d900f/warp/optim/linear.py#L10-L21)
include algorithm-specific state classes **and** matching convenience functions.
The nonconstructible `LinearSolverState` supplies their shared interface. Each
function constructs its corresponding class, returning the state for
`run=False` or invoking it for `run=True`. Instances execute through `__call__`.
See the [state and CG implementation](https://github.com/NVIDIA/warp/blob/0c1ed78f7b4ee397534e5bd1cea632af618d900f/warp/_src/optim/linear.py#L1280-L1680).

| Existing Warp API | Proposed eigen counterpart |
|---|---|
| `CG`, `CR`, `BiCGSTAB`, `GMRES` | Public `KrylovSchur`, `LOBPCG` states |
| `cg`, `cr`, `bicgstab`, `gmres` | `krylov_schur`, `lobpcg` convenience functions |
| Nonconstructible `LinearSolverState` | Nonconstructible `EigenSolverState` sharing lifecycle and result contracts |
| `cg(..., run=False)` returns a `CG` | `lobpcg(..., run=False)` returns a `LOBPCG` |
| Calling a preallocated state runs it | `result = solver()` executes a prepared eigen state |

Warp also publicly exposes algorithm-specific
[`Adam` and `SGD`](https://github.com/NVIDIA/warp/blob/0c1ed78f7b4ee397534e5bd1cea632af618d900f/warp/optim/__init__.py#L23-L26).
These are evidence of current practice, not a written prohibition on generic
frontends. The recommendation follows the closest existing subsystem: iterative
linear solvers.

**Revised decision:** expose the algorithms themselves. The earlier draft's
`eigsh(method=...)` and constructible generic `EigenSolver` would have hidden
public algorithm choices behind a dispatch layer. That is a weaker match for
Warp and combines incompatible options in one signature: `ncv` versus block
width, shifted inverse versus correction preconditioner, and a single starting
vector versus a starting subspace. A small public API need not hide the algorithm.

Do not add `eigsh` to the initial contribution. If users later need a SciPy-style
adapter, layer it over the public states with documented dispatch and no separate
implementation. It must not become the sole way to select or reuse an algorithm.
The shared `EigenResult`, explicit `prepare()` step and device-only result default
remain new eigen-specific proposals; they are not claimed to be existing Warp
standards. Linear solvers instead accept the solution array from the caller and
can return host or device diagnostics according to `check_every`. Review output
ownership and capture preparation explicitly rather than copying that API only
in name.

## Proposed frontend

### Entry points

Initially export `KrylovSchur`, `LOBPCG`, `krylov_schur`, `lobpcg`,
`EigenSolverState` and `EigenResult` from `warp.optim.eigen`. Concrete states
are directly constructible; `EigenSolverState` is a common base, not a dispatcher.
The convenience functions have these proposed signatures:

```python
# Proposed signatures, not implementations.
def krylov_schur(
    A, k, *, mass=None, which="LM", sigma=None,
    inverse=None, mass_inverse=None, initial_vector=None,
    ncv=None, tol=None, atol=0.0, maxiter=100, run=True,
): ...


def lobpcg(
    A, k, *, mass=None, preconditioner=None, initial_vectors=None,
    deflation=None, block_size=None, tol=None, atol=0.0,
    maxiter=100, run=True,
): ...
```

`KrylovSchur(...)` and `LOBPCG(...)` have the arguments of their corresponding
functions except `run`, and do not run automatically. For either function,
`run=False` returns its concrete state; `run=True` constructs, prepares and calls
that state, returning `EigenResult` with a reference that keeps storage alive.
`solver()` returns the same result view exposed as `solver.result`. It does not
allocate a new result on each invocation or read convergence to the host.

The public `KrylovSchur` covers both standard symmetric and SPD generalized
problems, retaining the mass inner product when supplied. Its kernels and
specialized workspaces may remain separate internally. `LOBPCG` is the block
algorithm; no separate `BlockLOBPCG` alias is needed upstream. Neither class
silently switches to the other algorithm.

| Shared argument | Contract |
|---|---|
| `A`, `mass` | Warp sparse matrix, supported Warp array, or `linear.LinearOperator`; `mass=None` means identity. Both square with matching scalar dimensions, device and supported precision. Symmetry and SPD mass are mathematical preconditions. Use `mass` to avoid confusing it with the preconditioner named `M` in Warp's linear solvers. |
| `k` | Number of requested eigenpairs, `0 < k < n`. No implicit dense fallback for `k == n`. |
| `tol`, `atol` | Relative and absolute original-equation residual tolerances, defined below. FP64 default `tol=1e-8`, `atol=0`. These are proposed defaults, not a change to existing warpack defaults. |
| `maxiter` | Fixed upper bound on outer iterations per call; a restart cycle for Krylov–Schur, a block update for LOBPCG. Inner solve budgets belong to the inverse/preconditioner state. |

| KrylovSchur argument | Contract |
|---|---|
| `which` | Without a shift, standard problems support `LM`, `SM`, `LA`, `SA`, `BE`; generalized regular mode initially supports `LM`, `LA`, `SA`. Unsupported combinations raise during setup. |
| `sigma`, `inverse` | A real shift and a borrowed `linear.LinearSystemSolver` bound to the caller-supplied $A-\sigma M$ system, or a compatible inverse action. Both supplied together. Initially require `which="LM"`, meaning largest transformed magnitude, hence nearest physical eigenvalues to `sigma`. The inverse applies only the solve; the eigensolver applies the mass matrix. |
| `mass_inverse` | Required for generalized regular mode when `mass` is supplied; absent for shift-invert. A borrowed linear system solver bound to the mass matrix, or compatible action applying $M^{-1}$. The algorithm retains the mass inner product. |
| `initial_vector` | Optional starting vector, a scalar array of shape `(n,)`, copied into private workspace on each call. The initial implementation accepts one start, not a block. If absent, use a deterministic device generator. |
| `ncv` | Search dimension, `k < ncv <= n`; default `min(n, max(2*k, k+8))`. |

| LOBPCG argument | Contract |
|---|---|
| `preconditioner` | Correction action, distinct from a shifted inverse. May be an SPD approximate inverse or a documented inexact block solve. It need not equal $(A-\sigma M)^{-1}$. |
| `initial_vectors` | Optional independent starting rows, up to `block_size`, copied into private workspace on each call. Missing rows use a deterministic device generator. |
| `deflation` | Known invariant subspace, such as rigid modes. Copied and mass-orthonormalized into private storage; rank failure is reported. Arbitrary equality constraints are a different problem. |
| `block_size` | Working block width, `k <= block_size`, `3*block_size <= n`; default `k`. |

Initially `LOBPCG` computes smallest algebraic modes only. Its signature does not
accept `which`, `sigma`, `inverse`, `mass_inverse` or `ncv`; those are not ignored
options. Largest-mode support can later add an explicit selection parameter with
its own tests. In both algorithm contracts, mathematical $M$ denotes `mass`.

Validate metadata and incompatible arguments before launching or altering state.
Report numerical failures on the device. Do not synchronously scan a matrix to
prove symmetry, definiteness or rank; offer explicit diagnostics when useful.
Reject unused parameters rather than silently ignoring them.

For equal selection keys, order by eigenvalue, then stable internal index.
Return selected real eigenvalues in ascending algebraic order, independently of
the selection criterion. For `BE`, an odd `k` selects the extra value from the
high end. Individual bases within a repeated eigenspace are not fixed by this
ordering. These rules must be tested, not inherited accidentally from a projected
eigensolver's output order.

### Generalized low modes

```python
# Proposed API. The example driver assembles K and M using warp.fem.
import warp as wp
from warp.optim import eigen

solver = eigen.LOBPCG(
    K, k=20, mass=M,
    block_size=24, preconditioner=P, initial_vectors=initial_modes,
    deflation=rigid_modes, tol=1e-8, maxiter=200,
)
solver.prepare()  # Allocation, compilation and warmup happen outside capture.

if K.device.is_cuda:
    with wp.ScopedCapture(device=K.device) as capture:
        solver()
    wp.capture_launch(capture.graph)
else:
    solver()

values = solver.result.eigenvalues      # Warp array (20,) on K.device
vectors = solver.result.eigenvectors    # Warp array (20, n), vectors are rows
```

`P` is caller-configured: for example an upstream block Jacobi or FSAI action,
or preallocated batched CG using that action. FP32 inner work is a preconditioner
option; the outer problem remains FP64. Reuse upstream implementations rather
than adding a second CG or FSAI implementation in the eigen module.

### Shift-invert with an explicit backend

```python
# Proposed API, including the new matrix-bound linear solver construction.
# The driver uses warp.fem for K and M and warp.sparse to form S=K-sigma*M.
# This example assumes S is SPD; use GMRES for a general/indefinite system.
from warp.optim import eigen, linear

shifted_solve = linear.CG(S, tol=1e-11, atol=0.0, maxiter=1000, check_every=0)
shifted_solve.prepare(nrhs=1)

solver = eigen.KrylovSchur(
    K, k=20, mass=M, sigma=sigma, inverse=shifted_solve,
    which="LM", ncv=48, tol=1e-9, maxiter=100,
)
solver.prepare()
result = solver()
```

The default path above is pure Warp on CPU or CUDA. For an explicit optional
direct solve on CUDA, replace only the linear solver construction:

```python
# Proposed optional provider; S is already assembled on CUDA.
shifted_solve = linear.CuDSS(S, matrix_type="spd")
shifted_solve.prepare(nrhs=1)
# Pass this state as inverse=shifted_solve to the same KrylovSchur constructor.
```

Matrix-bound construction without `b` and `x`, `prepare(nrhs=...)`, and
`linear.CuDSS` are proposed additions, not existing Warp API calls. The initial
implementation must preserve the existing `CG(A, b, x, ...)` and callable-state
interfaces while adding the reusable system-solve contract below.

The same eigen call accepts either state.
The frontend does not choose a sparse factorization, assemble the shift, or
invoke a vendor package implicitly. CG is only valid for an SPD shifted matrix;
an interior shift can require GMRES or an appropriate direct factorization.
Inexact inverse tolerances affect the outer method: check inner solve failures
and fresh physical residuals. Reaching a truncated CG budget while computing a
LOBPCG preconditioner is not automatically an eigenproblem failure.

### Result and state lifecycle

`EigenResult` contains fixed-shape device arrays:

| Field | Meaning |
|---|---|
| `eigenvalues`, `eigenvectors` | `(k,)` real values and `(k,n)` rows; rows are orthonormal in the mass metric, or Euclidean metric for identity mass |
| `residual_norms`, `relative_residuals` | `(k,)` fresh original-equation residual diagnostics |
| `valid` | `(k,)` mask identifying usable output pairs; no inference from uninitialized or stale slots |
| `converged`, `iterations` | Device integer scalars: accepted pair count and executed outer iteration count |
| `status` | Device enum: `NOT_RUN`, `CONVERGED`, `ITERATION_LIMIT`, `BREAKDOWN`, `INVALID_METRIC`, `INNER_SOLVE_FAILED` or `NONFINITE` |

Use `valid` with `status`: on iteration exhaustion, accurate partial results may
be returned, but a breakdown must not leave a previous solve's success visible.
`CONVERGED` requires all `k` pairs to pass the residual and independence gates;
it does not certify no lower or nearer mode was missed. `result.check()` is an
explicit synchronizing convenience which raises for an unsuccessful solve;
device consumers read the arrays directly.

Represent the counters and status as `wp.int32` arrays of shape `(1,)` and the
validity mask as `wp.bool` of shape `(k,)`. Expose host enum constants for device
kernels to compare against; reading those constants does not read solver state.

`solver()` resets state from the supplied `initial_vector` or `initial_vectors`
and runs the configured budget, with an optional device-side early exit where
supported. Warm starts are explicit copies into separate input storage; they do
not alias output workspace. The result is a view of state-owned buffers and is
overwritten by the next solve. Retain state, graphs
and borrowed factors until all queued work completes. A state is not reentrant;
concurrent solves require independent workspaces.

Use one residual acceptance rule for standard and SPD generalized problems:

$$r_i=A u_i-\lambda_i M u_i,\quad
  s_i=\|A u_i\|_2+|\lambda_i|\,\|M u_i\|_2,$$
$$\|r_i\|_2\leq\text{atol}+\text{tol}\,s_i.$$

The relative diagnostic is $\|r_i\|_2/s_i$ for positive $s_i$; for $s_i=0$
report zero only if the residual is zero, otherwise infinity. `atol` supplies a
physical absolute floor when needed near null modes. Normalize first, check
finiteness and independence, and evaluate fresh products before accepting
results. During iteration, cheaper cached estimates may guide stopping, but
they cannot substitute for final validation. This changes warpack's existing
denominator conventions: migration must compare raw residuals and explicitly
translate tolerances, never claim equal accuracy just from equal `tol` values.

## Operators and sparse storage

Use the existing `warp.optim.linear.LinearOperator` as the common single-vector
interface. Add an optional block-product capability, with a default adapter
that records single-vector applications for a fixed number of rows. A native
block implementation shares sparse reads and reduces launch overhead. An
optional transpose/conjugate-transpose action is required before supporting
general SVD or nonsymmetric adjoints.

Block products should preserve the linear operator's scaling semantics:
`Z = alpha * apply_rows(A, X) + beta * Y`, with rows representing vectors.
They are multiple vectors for the **same** operator, not the independent
subproblems described by `LinearOperator.batch_offsets`. Initially reject
nontrivial batch partitions at the eigen frontend; batched eigenproblems need
their own result and workspace design. Extending `LinearOperator` is a small
separate upstream proposal, not a prerequisite for the scalar-product fallback.

Accept scalar BSR (CSR) and qualified block BSR layouts through `warp.sparse`.
The initial optimized path covers the existing scalar and 3×3 cases. Other block
sizes need a tested generic adapter or explicit rejection. Normalize metadata
to scalar dimensions without copying numerical data to the host. Support full
symmetric storage first; no ambiguous inference that a missing triangle should
be mirrored. The eigen module should not introduce another public CSR container.

Keep public eigenvector rows contiguous in the scalar degree-of-freedom index.
This preserves warpack's efficient block layout, but differs from SciPy's column
outputs; state it prominently and provide an explicit conversion example.
Reject unsafe overlap and unsupported strides before dispatch. Matrix-free
operators must overwrite outputs, preallocate scratch and be capturable if the
solver is captured. They are responsible for their mathematical properties.

## Setup, capture and changing inputs

Construction allocates shape-dependent buffers. `prepare()` compiles and warms
kernels, prepares internal child graphs on CUDA and validates backend capabilities.
Both may allocate and synchronize, and are forbidden inside active capture.
CPU preparation must not create CUDA resources or require a CUDA device or driver.
Calling a prepared CPU state executes its Warp kernels directly.
On CUDA, the prepared `solver()` must record into the caller's capture without
nested standalone capture or allocation. There should be one internal preparation and
capture policy shared by all eigen algorithms.

Replay must not read device convergence on the host, allocate according to
numerical rank, perform Python callbacks based on device data, or run a host
projected eigensolve. Fixed-budget masked execution is the fallback when native
device-controlled graph loops are unavailable. Record the chosen execution
policy in state metadata so iteration timing is interpretable.

| Change between solves | Required action |
|---|---|
| Start vectors or matrix values in the same buffers | Ordered device update; refresh every dependent cast, preconditioner and factor before solving |
| Factor values with unchanged sparsity | Provider's qualified refactor path, if available; otherwise rebuild factor and dependent graphs |
| Topology, buffer addresses, dimensions, dtype or working width | Rebuild affected state and graphs |
| Iteration budget or algorithm policy | Reprepare and recapture |
| Constraint values | Recompute the relevant normalization/rank data; changing rank can require new allocation |

Arbitrary writes to a caller-owned matrix cannot be detected cheaply. The caller
must manage updates explicitly; debug generation checks can detect registered
updates but must not imply that stale factors are automatically prevented.
Graph dependencies include operand addresses and native handles, not just shapes.

The current [graph-upload issue #61](https://github.com/alecjacobson/warpack/issues/61)
and [captured mass-QR investigation #198](https://github.com/alecjacobson/warpack/issues/198)
are integration qualification gates. Existing workarounds complete first launch
during setup before changed-input work is queued. The new frontend must either
qualify an upstream fix or include that setup requirement in its supported-stack
contract and measured cost. Do not hide a device synchronization in every replay.

Native factors retain their stream and lifetime rules. Validate construction,
application and capture streams; permit cross-stream replay only with established
ordering and tested provider support. Closing an owned resource while captured
work can still reference it is an error.

## Stateful linear system solvers

Add a common `LinearSystemSolver` contract in `warp.optim.linear` for a prepared
system $A x=b$ with changing right-hand sides. It owns or retains reusable
workspace, iterative/preconditioner state or direct factors, and binds to a
caller-supplied matrix or operator. It is useful to eigen routines and other
clients; it does not belong in `warp.optim.eigen`.

Extend or adapt the existing `LinearSolverState`, `CG` and `GMRES` implementations
to satisfy this contract, preserving their current interfaces. Do not implement
a second CG or GMRES. Expose an optional `CuDSS` state satisfying the same
contract. Whether `LinearSystemSolver` is a base class or a protocol is an
implementation review decision; consumers must not require one concrete solver
class or test for the presence of cuDSS.

| Member | Proposed contract |
|---|---|
| System metadata | Bound operator, scalar shape, dtype, device, prepared RHS count and capability metadata; no numerical host readback to discover them |
| `prepare(nrhs=1)` | Allocate and warm state outside capture; prepare native analysis/factors when needed. Pure-Warp implementations work on CPU and CUDA. Provider-specific factorization settings remain on the concrete state. |
| `solve(rhs, out)` | Solve the bound system into caller-owned storage and return reusable device diagnostics. For one RHS, arrays have shape `(n,)`; block solves use `(nrhs,n)` rows. Use fixed prepared layouts, reject unsupported aliasing, and overwrite `out` without consuming an uninitialized initial guess. |
| Device diagnostics | Per-RHS status, iteration counts and residual information with documented freshness; reset on each call. No implicit host exception based on convergence. Numerical failures can propagate to the eigen result on the device. |
| `update_values()` | Explicitly refresh state after changes to the bound matrix values. Refactor or refresh dependent preconditioners as supported; otherwise require reconstruction. Report any need to reprepare/recapture through setup metadata, never silently reuse stale factors. |
| Lifetime | Caller owns the passed state. Eigen routines borrow it and do not close it. Retain it through graph completion; do not share mutable scratch between concurrent solves. Resource release is explicit for native state. |

The matrix-bound construction mode is the proposed extension that allows
`CG(A, ...)` or `GMRES(A, ...)` to be prepared before the first RHS is known.
`solve(rhs, out)` complements their existing bound-RHS callable interface.
Preserve existing constructor and return behavior for existing callers rather
than globally changing the meaning of `LinearSolverState`.

The default backend is pure Warp. Use CG when the caller knows the system is
SPD; use GMRES for a general system. Any convenience factory must also default
to pure Warp and must not select cuDSS merely because it is installed. Request
the direct solver explicitly. Matrix-free operators work with iterative states;
the cuDSS state requires compatible assembled sparse storage.

Eigen routines use `solve` through an internal inverse-action adapter that
supplies their prepared RHS/output buffers. `inverse` supplies the shifted solve;
`mass_inverse` supplies the mass solve. The eigensolver itself applies any mass
product required by the spectral transformation. A block solve can also provide
a LOBPCG preconditioning action with explicitly chosen inner tolerances and
budgets. This does not imply that a truncated iterative solve is an exact linear
inverse or that it has the same failure policy as a preconditioning application.

An iterative state resets its initial guess by default. A warm-start policy must
be explicit because history-dependent inverse errors can affect the outer
algorithm. Validate final eigenpairs against the original equations regardless
of backend. Multiple RHSs refer to one shared system, not independent matrices;
shared factors and optimized block products should be reusable across them.

All solver states expose the same CPU eager/CUDA capture composition rules where
their backend is supported. Preparation and value updates are setup operations;
after preparation, `solve` allocates no workspace and performs no host convergence
checks during CUDA replay. Graphs retain stable buffer and factor addresses.
Any unsupported update invalidates the dependent preparation/graph explicitly.

### Optional cuDSS provider

Keep cuDSS behind the inverse-action interface, with lazy optional imports and
an explicit provider installation. The public stateful solve abstraction and
thin `linear.CuDSS` adapter are included in this proposal. Native bindings may
remain in the optional `warp-cuDSS` provider; bundling the native dependency in
Warp is a separate packaging decision. Neither importing the linear/eigen modules
nor constructing a pure-Warp state requires cuDSS.

The adapter owns symbolic analysis, numerical factorization, native workspace,
solve status and any supported refactor lifecycle. Warp owns RHS assembly,
residual checks and iterative refinement orchestration. Requesting inertia is
an explicit setup diagnostic, not a convergence query inside replay. No cuBLAS,
cuSOLVER, CuPy, SciPy or NumPy numerical eigensolve enters the implementation.

Qualify versions and CUDA stacks explicitly. Current cuDSS graph metadata
transfers ([#29](https://github.com/alecjacobson/warpack/issues/29)) and cuDSS 1.0
qualification ([#189](https://github.com/alecjacobson/warpack/issues/189)) remain
separate from pure-Warp graph guarantees. A captured native solve containing
small metadata HtoD nodes must not be described as having no host-device
transfers, even when it has no host convergence decisions.

## Precision and differentiation

### Precision

Start with the implemented real FP64 outer problem, then add FP32 after a
dedicated accuracy and performance qualification. Dtype follows the inputs;
unsupported combinations fail rather than silently promote, downcast or
reinterpret them. A Warp namespace alone must not imply every Warp dtype works.

| Precision | Initial status | Follow-up requirement |
|---|---|---|
| Real FP64 outer solve | Initial integration on CPU and CUDA | Preserve original-residual and orthogonality gates on both devices |
| FP32 inner preconditioning with FP64 outer solve | Explicit supported configuration for qualified preconditioners | Refresh casts on value changes, report work precision and validate final FP64 residuals |
| Real FP32 outer solve | Deferred | Dtype-aware tolerances, reductions, projected solves, stagnation and conditioning coverage |
| Complex128 outer solve | Exists in warpack, deferred from initial upstream API | Agree storage/conjugation conventions and consistent complex-dimensional workspace sizing |
| Complex64, FP16 or BF16 outer solve | Deferred | Separate design and evidence; no inferred support from low-precision preconditioner storage |

Track broader precision through [#17](https://github.com/alecjacobson/warpack/issues/17).
Do not promote an experimental low-precision projected eigensolver solely from
a microbenchmark; require an end-to-end improvement at matched accuracy.

### Implicit adjoints

Reserve a custom implicit differentiation path, tracked in
[#212](https://github.com/alecjacobson/warpack/issues/212). Initial forward-only
entry points must explicitly reject unsupported differentiation requests rather
than let a Tape silently produce incomplete gradients. Warp's
[custom differentiation facilities](https://nvidia.github.io/warp/latest/user_guide/differentiability.html)
provide the integration mechanism; differentiating all iterative kernels is
not the proposed derivative contract.

For an isolated simple eigenvalue with $u^T M u=1$:

$$d\lambda=u^T(dK-\lambda\,dM)u.$$

First support losses on these eigenvalues, with gradients accumulated into
stored sparse values of both matrices and then propagated through Warp assembly.
Next support isolated eigenvector losses using a projected or bordered adjoint
solve with the normalization contribution included. The shifted operator is
singular and generally indefinite away from the lowest mode; ordinary CG is
not a universal backward solver.

Repeated modes require a basis-invariant loss on an isolated complete cluster,
or an explicit unsupported/ill-conditioned diagnostic. A sorted individual
eigenvector, and generally an individual eigenvalue branch at a collision, has
no unique smooth derivative. A cluster must have a gap to the excluded spectrum.

The backward implementation must save the accepted forward state, forbid its
overwrite before backward, accumulate gradients correctly across calls and
honor Tape reset semantics. Matrix-free operators need an explicit parameter
VJP or differentiable application hook; an opaque `matvec` is insufficient.
Sparse gradients must respect duplicate entries and symmetry conventions
without constructing dense $n\times n$ arrays. Differentiate the converged
physical problem, not the arbitrary preconditioner or iteration path, and gate
gradients on forward and adjoint solve accuracy. Parameter-dependent constraints
need their own derivatives and are outside the first adjoint stage.

Qualify forward and backward graph replay separately, including retained buffers
and changing parameters. Tape callbacks may record device work during backward
construction; they must not become host numerical callbacks during graph replay.
Finite-difference directional checks and Taylor tests cover stiffness, mass,
normalization, sign/basis invariance and near-degenerate failures.

## Source organization and migration

Use Warp's public forwarding-module convention and private implementation tree:

```text
warp/optim/eigen.py                  public exports
warp/optim/linear.py                 shared stateful system-solve contract and exports
warp/_src/optim/eigen/               frontend, state, operators, algorithms, kernels
warp/_src/optim/                     reusable iterative states and optional direct adapter
warp/tests/                         focused eigen correctness and graph tests
warp/examples/optim/                procedural modal example assembled with warp.fem
asv/                                performance benchmarks in Warp's harness
docs/                               API reference, guide, support matrix
```

Confirm exact test and benchmark paths against upstream conventions when opening
implementation PRs. Share projected algebra internally where needed, without
promoting it into a public dense linear algebra API. Linear solve adapters and
block operator improvements should be reusable outside eigenproblems.

| Current warpack exports | Proposed destination |
|---|---|
| `KrylovSchur`, `GeneralizedKrylovSchur`, `BlockLOBPCG` | Public `KrylovSchur` for standard/generalized problems and `LOBPCG` for the block method; common `EigenSolverState` and matching convenience functions |
| `SymmetricEigensolver`, `GeneralizedEigensolver` | Retained warpack compatibility implementations; no new public upstream aliases |
| `SparseOperator`, `BatchedSparseOperator` | Existing Warp operator interface plus private block adapters |
| `CGInverse`, `GMRESInverse`, `BatchedCGPreconditioner` | Shared `warp.optim.linear` system-solve contract implemented using upstream iterative states; private eigen inverse/preconditioning adapters |
| `CuDSSInverse`, `InertiaResult` | Optional `linear.CuDSS` state and provider-specific diagnostics behind the common system-solve contract |
| `EigenpairEvaluation` | Common result finalization and explicit validation utilities |
| `OrthogonalComplementOperator`, `MassOrthogonalComplementInverse` | Internal deflation machinery; expose reusable projection utilities only with an independent use case |
| `ConstrainedKrylovSchur`, `independent_constraints`, `saddle_point_matrix` | Follow-up constrained solve/diagnostics API consumes caller-supplied systems; KKT construction stays in caller code using warp.fem/warp.sparse or separate assembly helpers, outside eigen |
| `HermitianCSR`, `ComplexCSR`, `HermitianEigensolver`, `GeneralEigensolver`, `ComplexShiftInverse`, `ShiftInvertEigensolver`, `GeneralizedShiftInvertEigensolver` | Follow-up complex storage review and explicit Hermitian/general solver states |
| `BucklingEigensolver`, `CayleyEigensolver` | Follow-up explicit transformation policy with its own selection semantics |
| `PartialSVD`, `NormalSVD` | Follow-up public partial SVD solver states; normal-product conditioning tradeoff stays explicit |
| `NumericalNullspaceDiagnostics`, `LowestModeCertificate` | Optional diagnostic layer; distinguish supplied-span classification from completeness certification |

Keep FEM and skinning energy assembly, mesh loading, rendering and application
data outside `warp.optim.eigen`; use `warp.fem` from examples and tests wherever
possible. CuPy/SciPy/Spectra adapters, external suite translations,
timing workers and archived experiments remain in the warpack comparison project.
Move small independent regression fixtures into upstream tests, with provenance
and licensing reviewed. No test or comparison imports from the installed module.

Retain warpack's released API during the transition. After upstream qualification,
warpack can delegate supported calls and continue supplying deferred features.
Use an explicit compatibility matrix and deprecation period; do not silently
change eigenvector orientation, result semantics, tolerance meaning or factor
ownership. Preserve MIT notices for imported warpack code and audit third-party
source and fixture licenses under Warp's contribution process. Do not copy the
Anvil asset into upstream source as an incidental solver dependency.

## Validation and performance acceptance

Every migration stage needs a matched before/after run at a pinned source and
dependency revision. Existing reports are baselines, not proof that the new
frontend is equivalent. Test the first solve as well as warmed replay; the
remaining benchmark migration is tracked in
[#208](https://github.com/alecjacobson/warpack/issues/208).

| Coverage | Acceptance evidence |
|---|---|
| CPU and CUDA support | Run the initial public algorithms, selections and pure-Warp inverse/preconditioner paths on both devices; check original residuals and invariant subspaces rather than bitwise equality; exercise CPU-only CI without a CUDA driver |
| Assembly boundary | Assemble representative elasticity/Laplacian fixtures with warp.fem on CPU and CUDA; keep analytic fixtures and independent references; check that eigen source does not import FEM or application matrix-construction helpers |
| Algebra and selection | Analytic diagonal/block problems and independent small dense references; requested spectrum, not just residuals; all implemented selection modes |
| Generalized metrics | Diagonal and consistent SPD mass, indefinite stiffness, scaling, near-zero modes, noncommuting $K,M$ and mass orthogonality |
| Multiplicity and deflation | Repeated/clustered modes, disconnected nullspaces, independent requested subspace, known rigid-mode exclusion and rank-deficient starts |
| Backends | Same matrices and accuracy gates for iterative and cuDSS inverses; deliberate inner failure; SPD versus indefinite shifts |
| Reusable linear state | Pass different concrete solver states to the same eigen call; repeat RHS/block solves, update matrix values, check status reset and factor lifetime, exercise pure-Warp operation without cuDSS installed, and regression-test existing linear APIs |
| Sparse and operator interfaces | Scalar and qualified block storage, matrix-free products, overlap/stride rejection, dtype/device errors, no runtime test imports |
| Lifecycle and capture | First solve, warm solve, changed values and starts, stale factor prevention through documented update workflow, reset after failure, stream ownership, eager/captured agreement |
| Device execution | Graph-node inspection plus CUDA API tracing for unexpected allocations, synchronizations and transfers; no host numerical decisions in replay |
| Accuracy failures | Preserve regressions for wrong requested spectra, ill-scaled pencils, orthogonality loss and full-count-but-invalid outputs |
| Application scale | Procedural elasticity with lumped/consistent mass, constrained skinning follow-up, geometry operators and existing Anvil fixture in external benchmarks |

Keep upstream CuPy/Spectra suites as external compatibility evidence; unsupported
tests remain classified rather than rewritten to pass. Known general selection
failure [#104](https://github.com/alecjacobson/warpack/issues/104) gates a stable
general eigensolver contribution. Ill-scaled pencil recovery
[#123](https://github.com/alecjacobson/warpack/issues/123) constrains the advertised
generalized support envelope. Do not make an upstream release depend on fixing
unrelated bugs in competitor libraries; preserve those comparison failures.

Report application assembly, compilation, matrix conversion, preconditioner/factor setup, graph
preparation including first upload, first accepted solve, repeated solve and
peak memory separately. For a caller deciding total cost, also give the sum for
one solve and for repeated solves. Charge each backend its complete setup cost
even if a comparison process could borrow another row's factor or preconditioner.

For performance qualification, use uncontended workers, identical requested
modes and physical acceptance gates, and retain every timing sample. A proposed
review trigger is a repeatable greater-than-5% regression in median solve or
setup time against the same current-warpack configuration; investigate it rather
than accepting it as wrapper overhead. Small-case launch overhead and peak
memory need separate checks. The large-FEM and Anvil paths must retain their
optimized block operations and mixed preconditioning. No GPU is universally
faster across every size, method and matrix family.

Establish separate CPU correctness and performance baselines where current
warpack lacks a qualified CPU path. Track CPU setup, solve time and memory on
representative small and medium problems; do not apply a CUDA timing threshold
to CPU execution. GPU performance specialization must retain a tested CPU path.

## Delivery milestones

| Priority | Deliverable | Exit gate |
|---|---|---|
| P0 | Maintainer discussion and frontend contract | Agree namespace, state/results, row orientation, required CPU and CUDA FP64 scope, residual policy and optional backend boundary |
| P0 | Core operator/state integration | Scalar operator fallback plus qualified block fast path; prepare/call composition; device status; import and ownership tests |
| P0 | Shared linear system solver abstraction | Matrix-bound iterative states in warp.optim.linear, caller-owned RHS/output solves, update/lifetime contracts, backwards compatibility and default pure-Warp operation |
| P0 | Symmetric standard and SPD generalized algorithms | Krylov–Schur and LOBPCG supported on CPU and CUDA with fresh validation, accurate selection and device-specific performance baselines |
| P0 | Optional inverse adapters and fast preconditioning | Upstream CG/GMRES/FSAI reuse through the shared linear contract; optional cuDSS state qualification; first-use and update lifecycle tests; reproduce the fast Anvil path |
| P0 | Initial upstream documentation and release qualification | Executable CPU and CUDA examples using warp.fem assembly, CPU-only CI and CUDA qualification, assembly boundary checks, optional-provider support matrix, packaging/license audit, graph issue disposition and before/after benchmark report |
| P1 | Real FP32 and implicit eigenvalue adjoints | Separate precision and Tape tests; no silent unsupported gradients in the initial release |
| P1 | Constrained physical modes and isolated-eigenvector adjoints | KKT matrix-category tests, constraint residuals and backward bordered solves |
| P2 | Complex Hermitian, general pencils, buckling/Cayley and SVD | Separate API reviews; resolve correctness gates for each advertised family |
| P2 | Cluster adjoints, multilevel reuse and batched eigenproblems | Explicit mathematical contracts and application-scale evidence |

Each milestone should be split into reviewable PRs. A first symmetric-only merge
does not close warpack's broader coverage issues. Published docs must say exactly
which stage and precision combinations shipped.

Warp's [contribution guide](https://github.com/NVIDIA/warp/blob/0c1ed78f7b4ee397534e5bd1cea632af618d900f/CONTRIBUTING.md)
asks for discussion of new features before implementation, focused contributions,
tests and documentation. The first deliverable is a discussion proposal with
these contracts and a bounded initial implementation plan.

## Decisions for maintainer review

1. **Namespace:** prefer `warp.optim.eigen` alongside `linear`. A later SVD API
   could share internals while using a separately agreed public name. Avoid a
   wholesale `warp.optim.warpack` namespace transplant.
2. **State and results:** expose `KrylovSchur` and `LOBPCG` directly, with
   matching functions and a shared nonconstructible `EigenSolverState`, following
   Warp linear solver practice. Review the proposed `EigenResult` storage
   ownership and device-only return policy separately; defer `eigsh` dispatch.
3. **Block products:** prefer an optional common operator capability over a
   competing eigen-only operator hierarchy. Agree aliasing and transpose rules
   before extending `LinearOperator`.
4. **Capture ownership:** prefer caller-owned composition and a common explicit
   preparation step. Resolve first-upload behavior at the runtime layer wherever
   possible rather than adding solver-specific replay barriers.
5. **Precision and Tape:** permit an explicitly forward-only FP64 first stage,
   with unsupported differentiation rejected. Agree whether FP32 or a first
   eigenvalue adjoint is a prerequisite for the first public upstream release.
6. **Linear solve state and cuDSS:** include the common stateful system-solve
   abstraction in `warp.optim.linear`, adapting existing iterative states and
   adding an explicit optional cuDSS state. Pure Warp is the default; native
   cuDSS packaging remains optional. Review the common protocol, matrix-bound
   construction and backwards compatibility with existing linear solver calls.
7. **Matrix construction:** `warp.optim.eigen` consumes caller-provided matrices
   and operators. Use `warp.fem` as much as possible in examples and test fixtures;
   keep shifted/KKT construction and all application assembly outside eigen.

