# Application contracts and comparisons

[Recorded results and remaining failures](RESULTS.md) accompany this protocol.

This suite turns the API/matrix taxonomy in [research issue #19](https://github.com/alecjacobson/warpack/issues/19)
into reproducible numerical fixtures. These are initial representatives, not
complete coverage of the publications or a claim of upstream conformance.
The [priority tracker](https://github.com/alecjacobson/warpack/issues/18) links
implementation bugs and the remaining coverage requirements.

| Class / priority | Fixture and numerical contract | Coverage issue |
|---|---|---|
| A / P0 | Clamped/free tetrahedral FEM, disconnected bodies, material contrast, lumped/consistent mass and repeated Hermite bending; complete low-mode clusters | [#20](https://github.com/alecjacobson/warpack/issues/20) |
| B / P1 | [Coupled FEM pencils](INTERIOR.md), interior/near-eigenvalue shifts, consistent mass and repeated bands; complete nearest physical set | [#21](https://github.com/alecjacobson/warpack/issues/21) |
| C / P2 | Indefinite tangent stiffness, SPD M; most-negative eigenvalues via a shift below the spectrum | [#22](https://github.com/alecjacobson/warpack/issues/22) |
| D / P0 | Sphere/disk/annulus/tori Hodge 1-Laplacians; disconnected and weak components, independent topology/subspace references, positive guard; [Krylov/CG profiles](HODGE_CG.md) | [#23](https://github.com/alecjacobson/warpack/issues/23) |
| E / P2 | Complex Hermitian connection ring with nontrivial holonomy and known spectrum; standard mass | [#24](https://github.com/alecjacobson/warpack/issues/24) |
| F / P2 | P1 Cauchy–Riemann energy and centered boundary norm; reciprocal spectral-conformal pencil with SPD regularized metric | [#25](https://github.com/alecjacobson/warpack/issues/25) |
| G / P2 | Coupled SPD stiffness / indefinite geometric-stiffness pencil; analytic buckling factors | [#26](https://github.com/alecjacobson/warpack/issues/26) |
| H / P3 | Damped coupled chain, standard first-order nonsymmetric linearization; analytic complex frequencies | [#27](https://github.com/alecjacobson/warpack/issues/27) |
| I / P2 | [Large tall/wide POD and analytic string snapshots](POD.md); repeated leading clusters, rank deficiency, both singular subspaces and reconstruction energy | [#28](https://github.com/alecjacobson/warpack/issues/28) |

The FEM bar, ring and load-factor pencil have analytic spectra, so comparison
does not depend on trusting another iterative solver. The small conformal case
uses dense LAPACK generalized `eigh` as an independent reference; larger cases
use SciPy ARPACK. Fixture assembly and validation run on the host outside all
solver timings. Production Warpack imports none of these reference libraries.

## What “Warp” means here

A/B/C use `GeneralizedKrylovSchur`: restarted Krylov iteration with two-pass
orthogonalization in the M inner product. In shift-invert mode its one-vector
inverse solves `(K - sigma M) y = x`; Warp forms Mx and all other numerical
operations. Regular mode applies `M^-1 K` while preserving the M metric. It
supports physical LM/LA/SA selection. F uses this regular mode on the reciprocal
pencil; this does **not** add support for an arbitrary singular mass matrix.

D uses block `GeneralizedEigensolver` for its repeated nullspace, with
[device-controlled metric QR recovery](WEAK_BLOCK.md) for ill-conditioned inverse images.
G uses `BucklingEigensolver`. E/H use the existing complex
`ShiftInvertEigensolver`, and I uses `PartialSVD`. All inverse paths in `warp`
use cuDSS. SVD requires no sparse inverse. `warp-block` selects the older block
method; `warp-cg` uses upstream Warp CG with diagonal preconditioning for the
known-SPD shifted systems A/C/D. This is a controlled comparison, not the tuned
Anvil FSAI configuration. The expanded [elastic ROM suite](FEM.md) separately
compares upstream block-Jacobi and FSAI on the natural displacement blocks,
with per-trial inner iteration statistics and independent full-subspace gates.

B/G now use explicit matched general cuDSS factors without correction; the
other application profiles retain their existing settings. See the
[factorization regression report](../upstream/MATCHING.md) for the recovered
interior-mode overhead, per-trial accuracy and remaining vendor graph copies.

The [generalized-transform report](../upstream/TRANSFORMS.md) resolves the
remaining #7 cases with restarted Krylov iteration, including buckling and
Cayley variants. All 21 supported generalized Spectra cases pass their original
bounds and full counts. Additional exact fixtures check requested spectra,
metric orthogonality and invariant subspaces. The comparison includes native
ARPACK/Spectra and explicit fixed-budget sweeps, including failed short budgets;
this does not establish completeness for arbitrary repeated spectra.

## Reference paths and acceptance

* SciPy uses ARPACK with SuperLU factored outside solve timing, correct mass
  operators, shift/buckling modes, or `svds`. The harness now passes `ncv`
  explicitly; older reference timing rows that used SciPy defaults are not
  identical-budget baselines.
* The native C++ Spectra adapter uses Eigen SparseLU and the real generalized
  shift, regular-inverse, buckling and Cayley APIs. E/H/I are explicitly unsupported
  by this adapter; the separate upstream suite covers those Spectra APIs.
* CuPy uses native `eigsh` for E and `svds` for I. For A/C/D/F it uses generalized
  **LOBPCG with the mass matrix and a cuDSS preconditioner**. Those rows are not
  CuPy `eigsh`. The expanded Hodge suite separately compares native `eigsh`
  on identity-mass fixtures. FEM adds diagonal-mass native `eigsh` through symmetric
  mass scaling and a `cupy-batched` LOBPCG adapter with one padded block cuDSS
  preconditioner call, alongside the original columnwise adapter. B/G/H are explicitly unsupported by this adapter. Applying
  Euclidean `eigsh` directly to a non-symmetric `M^-1 K` would be an invalid
  reference.

LOBPCG uses absolute tolerance `1e-8`; the other reference adapters request
`1e-11` under their own stopping conventions. The stricter LOBPCG `1e-10`
setting over-iterated the 10,000-DOF bar cases into inaccurate results; those
attempts are retained. All paths still face the same independent acceptance gate.

Every measured trial must pass original-equation residual, spectrum and
orthogonality checks at `1e-8`; a small residual alone is insufficient. The original D fixture must recover both known zeros; the expanded
[Hodge suite](HODGE.md) also checks independently referenced nullspaces and
positive guards for its counting contract. H checks unit vector norms rather than orthogonality because
its matrix is nonnormal. SVD checks both equations and both orthogonality
conditions. Failed timings are retained as diagnostics and excluded from wins.
The damped-chain Warp residual now passes with Warp correction of cuDSS solves;
see the [correction report](../upstream/REFINEMENT.md) for both tested sizes and
the previous interior-mode overhead; the matching report above records its recovery.
The conformal CuPy LOBPCG comparison has shown failed residuals and a projected
Cholesky failure; that result is recorded, not treated as a Warpack speedup.

Setup/factorization, capture and warmup are reported separately. Timed GPU calls
are synchronized; output copies and CPU validation are outside those timings.
References restore their starting spaces and Warp reinitializes on each replay.
Different algorithms use different starting spaces. Fixed Warp cycle budgets
are explicit and are not tuned automatically to each reference's stopping rule.
Graph DOT inspection reports host callback nodes and transfer kinds separately
from numerical acceptance. `contract_accepted` also requires the device-only
graph check. A captured cuDSS solve can contain vendor HtoD nodes even though
Warp's iteration does not read numerical data on the host; DOT inspection is
not an API trace or proof of absence of host synchronization.

GPU process inventories bracket each sample. Contended samples cannot establish
a performance regression or improvement. Process polling may miss transient
activity; use an exclusive GPU for publishable timing comparisons. The runner
records exact samples, sizes, sparsity, matrix hashes, versions and source state.

## Reproduction

Use the pinned environment in the root README and install benchmark extras.
The Spectra checkout must be at `db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95`.

```bash
g++ -O3 -DNDEBUG -shared -fPIC -I/usr/include/eigen3 -Ibuild/upstream-spectra/include \
  benchmarks/applications/spectra.cpp -o build/application_spectra.so
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q tests/test_applications.py
python -m benchmarks.applications.generalized_regression
python -m benchmarks.applications.fem_compare
python -m benchmarks.applications.hodge_compare
python -m benchmarks.applications.audit_inverse
python -m benchmarks.applications.compare --sizes 80,1024 --repeats 3 \
  --out results/application_coverage.json
python -m benchmarks.applications.compare --classes ABC --sizes 10000 \
  --steps 5 --repeats 3 --out results/application_coverage_large.json
python -m benchmarks.applications.compare --classes ACD --sizes 1024 \
  --methods warp,warp-block,warp-cg --out results/application_inverse_comparison.json
```

Compare a subsequent run with `--baseline path/to/baseline.json`. Accepted,
uncontended rows with identical matrices and solver parameters are checked for
slowdowns above `--regression-limit` (default 1.25); any accuracy regression
fails the command. Missing/unsupported methods stay visible. Baseline comparison
requires matching GPU/runtime and does not convert inconclusive timings into a
pass. Fixture sizes are requested resolutions: torus/conformal/dynamics dimensions
are rounded, and their actual dimensions are recorded.
