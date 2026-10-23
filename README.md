# warpack

**Sparse eigenproblems on the GPU, written in NVIDIA Warp.**

warpack computes vibration modes, reduced bases and other partial eigensystems
for large sparse problems in graphics and physics. Numerical iteration,
projected eigensolves and convergence checks run in Warp, with an optional
cuDSS backend for sparse direct solves. Solves can be captured and replayed as
CUDA graphs.

[Release v0.2.0](https://github.com/alecjacobson/warpack/releases/tag/v0.2.0)
· [User guide](docs/guide.md)
· [Examples](#examples)
· [Issues](https://github.com/alecjacobson/warpack/issues)

![The first 20 elastic vibration modes of an anvil](results/anvil_teaser.gif)

*283,857 degrees of freedom; the first 20 elastic modes, each shown at 5% of the
bounding-box diagonal. [Model credit and visualization details](#anvil-showcase).*

## Installation

Requires **Python 3.10+**, an NVIDIA CUDA GPU, and **Warp 1.15+**. Device-controlled
conditional graphs require a driver supporting CUDA 12.4 or later. Choose a
Warp build compatible with your driver; the tested CUDA 12 environment is
specified in the [installation guide](docs/installation.md).

Install the released library:

```bash
python -m pip install 'https://github.com/alecjacobson/warpack/releases/download/v0.2.0/warpack-0.2.0-py3-none-any.whl'
```

For examples, development, or optional cuDSS support:

```bash
git clone https://github.com/alecjacobson/warpack.git
cd warpack
python -m pip install -e '.[test]'
# Optional sparse direct solver:
python -m pip install -e '.[cudss]'
```

**The fast Anvil configuration additionally requires a pinned upstream Warp
FSAI/batched-CG overlay.** It is not installed by the commands above. Follow
[the fast-path setup](docs/installation.md#fast-block-solves) to install it in a
separate directory. The example below works with stock Warp 1.15.

## Quick start: generalized low modes

For FEM mode extraction, solve `K x = λ M x`, where `K` is symmetric stiffness
and `M` is a symmetric positive-definite mass matrix. `BlockLOBPCG` directly
computes the smallest algebraic modes and accepts an upstream Warp
preconditioner.

This complete example builds a small diagonal pencil; replace `K` and `M` with
your assembled sparse matrices for a real problem.

```python
import warp as wp
import warp.sparse as sparse
from warp.optim import linear

from warpack import BlockLOBPCG, SparseOperator

wp.init()
with wp.ScopedDevice("cuda:0"):
    n, k = 64, 4
    indices = wp.array(list(range(n)), dtype=wp.int32)
    K = sparse.bsr_from_triplets(
        n,
        n,
        indices,
        indices,
        wp.array([float(i + 1) for i in range(n)], dtype=wp.float64),
    )
    M = sparse.bsr_from_triplets(
        n,
        n,
        indices,
        indices,
        wp.array([1.0 + 0.01 * i for i in range(n)], dtype=wp.float64),
    )
    solver = BlockLOBPCG(
        SparseOperator(K),
        k=k,
        width=8,
        mass=SparseOperator(M),
        preconditioner=linear.preconditioner(K, ptype="diag"),
        tol=1e-8,
    )
    graph = solver.capture(200)
    wp.capture_launch(graph)

    # Results stay on device. Eigenvectors are ROWS: shape (k, n).
    values, vectors = solver.eigenvalues, solver.eigenvectors

    # Explicit host readback for this example, outside the captured solve.
    wp.synchronize()
    if solver.status.numpy()[0] != 0 or solver.converged.numpy()[0] != k:
        raise RuntimeError("The requested modes did not all converge")
    print(values.numpy())
```

Real sparse operators accept compact **FP64 scalar CSR or 3×3 BSR** matrices
from `warp.sparse`. Block iteration requires `k <= width` and `3*width <= n`.
For free bodies, supply known rigid modes through `constraints`; general
linear equality constraints use `ConstrainedKrylovSchur`. See
[starts, constraints and residuals](docs/guide.md#generalized-low-modes).

## Choose a solver

| Problem | API | Main requirement |
|---|---|---|
| Lowest symmetric FEM modes | `BlockLOBPCG` | Symmetric K, SPD M; optional preconditioner and known invariant constraints |
| Standard symmetric eigenpairs | `KrylovSchur` | LM / LA / SA / SM / BE selection |
| Generalized or interior symmetric modes | `GeneralizedKrylovSchur` | SPD metric (e.g. mass); supply a metric inverse or shifted inverse |
| Symmetric block inverse iteration | `SymmetricEigensolver`, `GeneralizedEigensolver` | Supply an iteration operator that amplifies the desired modes |
| Equality-constrained / skinning modes | `ConstrainedKrylovSchur` | SPD physical mass, independent constraints, KKT inverse |
| Complex Hermitian eigenpairs | `HermitianEigensolver` | Complex FP64 data |
| General real/complex eigenpairs | `GeneralEigensolver`, `ShiftInvertEigensolver` | See the requested-spectrum limitation below |
| Generalized nonsymmetric pencils | `GeneralizedShiftInvertEigensolver` | Regular pencil; finite nearest right modes; singular B is permitted |
| Buckling and Cayley transforms | `BucklingEigensolver`, `CayleyEigensolver` | Transform-specific definiteness and selection rules |
| Real partial SVD / POD | `PartialSVD`, `NormalSVD` | `NormalSVD` targets resolved largest nonzero triplets and squares conditioning |

Inverse backends are `CuDSSInverse`, `CGInverse` for SPD systems and
`GMRESInverse` for indefinite/nonsymmetric systems. For fast block iteration,
`BatchedCGPreconditioner` and `BatchedSparseOperator` share work across RHSs.
Truncated CG is a preconditioner, not an exact shift-invert operator.
[Solver usage and contracts →](docs/guide.md)

## Graph replay and results

`capture()` performs setup, compilation and warmup, then returns a reusable
CUDA graph. Call `wp.capture_launch(graph)` for each solve. Setup may synchronize;
the pure-Warp replay performs no host numerical decisions or residual readbacks.

- Keep the solver, inputs, graph and inverse resources alive during execution.
- Matrix values and supplied starts can change at fixed storage addresses.
  Topology changes require new setup. Maintain inverse factors and preconditioners
  consistently with the updated matrices.
- Each solver owns mutable scratch; do not use the same instance concurrently.
- Check **both** `status == 0` and `converged == k`, plus the residuals appropriate
  to the original problem. A fixed budget can expire without convergence.
- A small residual does not establish that every requested mode was found.
  Warm starts need particular care after mode crossings.

cuDSS graphs use fixed outer budgets and currently contain small vendor metadata
HtoD transfers. The Warp iteration adds no host convergence readback.
[Capture policies, lifetimes and diagnostics →](docs/guide.md#graph-replay)

## Examples

Run these from a source checkout; application assembly helpers are separate
from the installed library.

| Example | What it shows |
|---|---|
| [`python -m examples.block_reuse`](examples/block_reuse.py) | Consistent-mass elastic bar; reuse computed modes after changing stiffness; stock Warp |
| [`python -m examples.anvil_block`](examples/anvil_block.py) | First 20 elastic Anvil modes with the fast pure-Warp configuration; requires the pinned overlay |
| [Skinning modes](benchmarks/applications/SKINNING.md) | Rig constraints and the physical-space KKT formulation |
| [Spectral surface reconstruction](benchmarks/applications/SPECTRAL_SURFACE.md) | Largest generalized modes with an SPD regularization metric; direct and pure-Warp block routes |
| [Force-aware reduced bases](benchmarks/applications/FORCE_COVARIANCE.md) | Sparse implicit covariance and snapshot reductions; full-rank or singular force priors with SPD mass |
| [Application guide](benchmarks/applications/README.md) | FEM, Hodge/nullspaces, conformal maps, connection modes, buckling and damped dynamics |

## Anvil showcase

The GIF uses the supplied [`anvil for export-ftetwild.mesh`](anvil%20for%20export-ftetwild.mesh):
94,619 vertices and 508,903 tetrahedra. The example assembles a stable
Neo-Hookean rest Hessian and lumped mass, excludes six rigid modes, and solves
for the first 20 elastic modes. The longest rest extent is normalized to one
metre, with illustrative material parameters `E = 100,000 Pa`, `ν = 0.3`,
`ρ = 1,000 kg/m³`. Color follows instantaneous displacement.

Measurements from 2026-10-23 on an NVIDIA L40 and Xeon Platinum 8362
(four pinned CPU cores), using the [v0.2.0 overlay](docs/releases/v0.2.0.md)
for the FSAI rows:

| Solver | Setup (s) | Warm solve (s) | Max original residual, all 26 modes |
|---|---:|---:|---:|
| SciPy ARPACK + MKL PARDISO | 7.0288 | 12.0792 | 4.040e-09 |
| Spectra + MKL PARDISO | 7.0222 | 6.9091 | 9.068e-09 |
| CuPy eigsh + cuDSS | 2.1396 | 0.4589 | 2.856e-09 |
| Warp + cuDSS | 2.3825 | 0.4655 | 2.693e-09 |
| CuPy LOBPCG + Warp FSAI/CG | 0.9617 | 2.3365 | 1.728e-09 |
| **Warp block + FSAI/CG** | **1.3984** | **1.5432** | **1.943e-08** |

Setup is the median of three independent constructions, including factorization
or preconditioning, workspaces and graph capture where applicable. Solve is the
median of 15 warm runs. Mesh assembly, process startup and validation are
excluded; compiled kernels are cached. All 90 timed solves passed the accuracy
checks, with run-to-run solve medians varying by less than 2%.

CuPy's cuDSS and Warp preconditioners are supplied by the benchmark. The pure-Warp
block setting uses width 24, FSAI32 and FP32 CG capped at 25 iterations, with FP64
outer work.

All rows use the same verified assembled matrices and reference. Each row
includes independently constructed factors or preconditioners in its setup time.

[Timing samples and accuracy checks](results/anvil_showcase_180/README.md)
· [Current comparison setup](results/anvil_inputs_142/README.md)
· [Rendering recipe and downloadable assets](docs/anvil.md)

Anvil adapted from [“Blacksmith's Anvil”](https://sketchfab.com/3d-models/blacksmiths-anvil-4b11b76fe069496ebd8ccd98f7d37ade)
by [m4rcu5j4ck50n](https://sketchfab.com/m4rcu5j4ck50n),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [Acknowledgments](ACKNOWLEDGMENTS.md) for mesh and visualization attribution.

## Validation and benchmarking

Validation checks original equations, requested eigenvalues and invariant
subspaces, mass orthogonality, repeated/null modes, changed-input replay and
graph residency. Reference libraries are used outside library computation.

The [latest full-suite checkpoint](results/gram_tiles/VALIDATION.md)
records **3,161 passes and 11 skips**, with separate stock-Warp and isolated-wheel
checks. This includes the memory, graph-preparation and projected-solve regressions.
The [v0.2.0 release report](results/release_v0_2_0/README.md) includes clean-wheel
replay checks and the release review. This is not full upstream conformance.

[Run tests and comparisons](docs/development.md)
· [CuPy coverage](benchmarks/upstream/CUPY_REFRESH.md)
· [Spectra coverage](benchmarks/upstream/SPECTRA_REFRESH.md)
· [Historical experiments](historical/README.md)

## Current limitations

The primary supported precision is FP64; FP32 inner CG does not imply a general
FP32 eigensolver API. Complex SVD and public left/infinite eigenmodes are not
supported. General nonsymmetric solves have a known
[requested-spectrum gap](https://github.com/alecjacobson/warpack/issues/104),
and strict generalized-pencil accuracy has
[remaining limitations](https://github.com/alecjacobson/warpack/issues/123).
See the [user guide](docs/guide.md#limitations) before choosing those paths.

## License

The library is [MIT licensed](LICENSE). Example meshes and artwork retain their
separate licenses in [Acknowledgments](ACKNOWLEDGMENTS.md).
