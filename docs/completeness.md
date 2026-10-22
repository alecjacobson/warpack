# Optional lowest-mode completeness checks

Small residuals do not establish that an eigensolver found every requested low
mode. It can return exact higher modes while omitting a repeated or newly lower
one. `LowestModeCertificate` combines numerical inertia with independent checks
of the returned subspace. It is optional and does not change ordinary solves.

The supported pencil is real symmetric `K x = lambda M x`, with SPD `M` (or
identity). `K` may be indefinite or have rigid zero modes. Both matrices must be
compressed scalar FP64 Warp CSR on the same CUDA device. Symmetry and positive
definiteness of `M` are caller preconditions. Singular-mass skinning pencils and
general nonsymmetric eigenproblems are outside this contract.

```python
import warp as wp
from warpack import LowestModeCertificate

# Physical eigenpairs, not eigenvalues of a shift-invert operator.
# values: FP64 device array (k,); vectors: FP64 device array (k,n).
# upper_bound lies strictly between the last included eigenvalue and the next.
certificate = LowestModeCertificate(
    K,
    vectors,
    values,
    mass=M,
    upper_bound=upper_bound,
    nullspace=rigid_modes,  # optional (q,n), zero modes omitted by the solver
)
certificate.run()  # warm the Warp verification kernels
with wp.ScopedCapture() as capture:
    certificate.run()
wp.capture_launch(capture.graph)
# certificate.certified and certificate.flags are device int32 arrays of size 1.
```

Choose a bound in a resolved spectral gap. Requesting extra guard modes can help
find that gap, but the guard eigenpair alone is not a completeness certificate.
Include an entire repeated boundary cluster. The supplied `nullspace` must be
an independent, M-orthonormal basis of zero modes; its residuals and mutual
orthogonality with the returned vectors are checked too. No supplied integer
nullity is trusted in place of that basis.

Construction factors `K - upper_bound*M` with cuDSS and explicitly synchronizes
to query its inertia. This is an additional factorization, outside capture.
Its negative count is compared with the number of independent accepted rows
`k+q`. `run()` recomputes original-pencil residuals, M-orthogonality and a
conservative test that the whole supplied subspace lies below the bound. It
uses Warp kernels and device copies only; it has no host query or synchronization.
The [Newton benchmark](../results/completeness/README.md) measures these two
costs separately, for modes computed with either cuDSS or pure Warp.

A value of one in `certified[0]` means these **numerical** checks pass under the
matrix preconditions. It is not an interval-arithmetic proof. The subspace gap
test uses a Gershgorin bound and can conservatively reject an approximate basis,
even when its individual Rayleigh quotients lie below the bound. There is no
automatic gap search, missing-mode recovery, or retry.

| `flags` bit | Meaning |
|---|---|
| `COUNT_UNAVAILABLE` (1) | No qualified inertia count; includes `backend=None` |
| `COUNT_INCONCLUSIVE` (2) | cuDSS error metadata, nonzero pivot counter, or invalid count |
| `COUNT_MISMATCH` (4) | Count below the bound differs from `k+q` |
| `INVALID_BASIS` (8) | M-Gram matrix differs too much from identity |
| `UNCONVERGED` (16) | At least one original residual exceeds tolerance |
| `GAP_UNRESOLVED` (32) | Whole-subspace separation from the bound is unresolved |
| `INVALID_DATA` (64) | Nonfinite values or numerical products |

Flags may combine. Any nonzero flag prevents certification. `residuals` includes
nullspace rows first, followed by eigenvector rows. `orthogonality_error` is the
largest absolute M-Gram error; `gap_margin` is the conservative separation from
the bound. The host `inertia` snapshot holds status, raw positive/negative counts,
INFO, pivot counter and explanation. Failed factorization calls may raise rather
than return a snapshot; failure always invalidates an older certificate first.

Residuals use `||Kv-lambda Mv|| / max(1, ||Kv||+|lambda| ||Mv||)`, including the
zero modes. Choose appropriate matrix units and tolerances: the absolute floor
matters especially for rigid modes. Scaling `K` and `values`/`upper_bound` by the
same positive factor preserves the generalized spectrum's ordering and vectors.
The default tolerances are `1e-8` for residuals and orthogonality and `1e-10` for
the gap relative to `max(1, abs(upper_bound), max(abs(values)))`.

After changing K or M **values**, call `certificate.refresh_count()` outside
capture before replaying checks. This refactors the current pencil, invalidates
the previous result, and updates count metadata in stable device buffers. The
class cannot detect unannounced matrix mutations. Changes to eigenvector,
eigenvalue or nullspace buffer contents only require another `run()` or replay.
Matrix topology, input buffer identities and the bound must stay fixed; create a
new certificate when these change. Do not mutate inputs during execution or
share an instance across concurrent streams.

## Backend limitations

`CuDSSInverse.inertia()` also exposes the explicit metadata query for an existing
symmetric/Hermitian indefinite factor. General LU, SPD declarations and global
COLAMD reorderings return `unavailable`. Closed factors and capture-time queries
raise. The supported cuDSS wrapper uses int32 CSR indices; the query is tested
with cuDSS 0.8.0.10. See NVIDIA's
[`CUDSS_DATA_INERTIA`, `CUDSS_DATA_INFO`, and `CUDSS_DATA_NPIVOTS` documentation](https://docs.nvidia.com/cuda/cudss/0.8.0/types.html).

cuDSS can perturb zero pivots and return INFO=0, so successful factorization alone
is insufficient. We conservatively treat any nonzero pivot counter as
inconclusive, even when the numerical count happens to agree with a reference.
No zero count is inferred. Missing metadata is unavailable.

Pure Warp does not currently have an inertia backend. `backend=None` runs the
device diagnostics but always reports `COUNT_UNAVAILABLE` and never certifies.
Modes computed entirely with Warp can optionally use cuDSS just for the separate
certificate. `NumericalNullspaceDiagnostics` retains its existing contract: it
does not certify an unknown nullity or topological Betti number.

## Opt-in recovery after a rejected warm start

A known failure case is two independent elastic members exchanging lengths:
the old lowest modes can remain exact eigenvectors but move above the requested
range. An unperturbed full starting block can converge to that wrong invariant
space. The certificate rejects it; seeded exploration recovers the correct
modes in the [regression fixture](../tests/test_fem_shapes.py). A perturbation is
not a general completeness guarantee.

The checkout example [`examples/certified_recovery.py`](../examples/certified_recovery.py)
composes existing APIs into a bounded retry policy. Prepare a finite list of
candidate solves, for example an ordinary warm start, a perturbed warm start,
and a cold start. They may use different starting block widths or iteration
budgets but must target the same k physical eigenpairs of the same K/M pencil.
Keep all solver resources alive, preferably through `Attempt.owner`:

```python
from examples.certified_recovery import Attempt, recover

# warm/retry are prebuilt BlockLOBPCG instances for the same K, M and k.
# Their graphs and initial-vector buffers are already prepared.
# retry can use initial_perturbation=1e-3 or fresh random starts.
attempts = [
    Attempt(
        "warm",
        warm_graph,
        warm.eigenvalues,
        warm.eigenvectors,
        warm.status,
        warm.converged,
        owner=warm,
    ),
    Attempt(
        "perturbed",
        retry_graph,
        retry.eigenvalues,
        retry.eigenvectors,
        retry.status,
        retry.converged,
        owner=retry,
    ),
]
certificate = LowestModeCertificate(
    K,
    wp.zeros_like(warm.eigenvectors),
    wp.zeros_like(warm.eigenvalues),
    mass=M,
    upper_bound=upper_bound,
    nullspace=rigid_modes,
)
report = recover(attempts, certificate)
if not report["accepted"]:
    raise RuntimeError(report)  # Do not consume rejected modes as a valid ROM.
# Accepted physical outputs now reside in certificate.values/vectors.
```

This **policy driver is not graph capturable**: it reads flags/native status on
the host between attempts. Each eigensolve and each fresh verification is a
captured device computation. There is no CPU numerical solve and no core solver
API change. The example is provided in the source checkout, not the wheel.

K, M, nullspace and the gap must remain fixed throughout a call. Output storage
must be independent of all candidate inputs/workspaces. The single inertia
factorization is paid when constructing the certificate; retries reuse that
count, with no further factorization or query. Report that setup and every
candidate's setup/capture cost separately. `check_capture_seconds` measures
verification warmup/capture, and each recorded attempt's `seconds` includes its
solve, output copies, verification and explicit status readback.

Unavailable/inconclusive inertia or a count incompatible with k+nullity stops
before any attempt. Changing a random seed cannot fix a wrong output capacity
or a boundary through a repeated cluster. Invalid data/bases/native status and a supplied nullspace that fails its zero-mode
residual check stop the sequence; changing the candidate start cannot fix that
fixed constraint basis. Only unresolved separation, residual failure or native iteration
exhaustion permits another supplied attempt. The list bounds the work; exhausting
it returns `accepted=False`. Acceptance requires **both** a zero certificate flag
and successful native status/count; the certificate's mathematical diagnostics
alone do not override a native solver failure. No graph is run after success.
