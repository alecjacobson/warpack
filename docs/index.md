# warpack documentation

warpack solves partial sparse eigenproblems and real partial SVDs on NVIDIA
GPUs using Warp, with optional cuDSS sparse direct solves. The primary use case
is extracting modes of a symmetric stiffness matrix with an SPD mass matrix.

This documentation follows **current main**. The v0.2.0 wheel predates some
options described here; use the documentation in your source checkout or
[release tag](https://github.com/alecjacobson/warpack/tree/v0.2.0).

## Start here

1. [Install warpack](installation.md), with stock Warp or the optional backends.
2. [Run the quick start](../README.md#quick-start-generalized-low-modes).
3. [Choose a solver and interpret its outputs](guide.md).
4. Look up signatures, parameters, device outputs and methods in the
   [API reference](api/README.md).

The [small runnable API examples](../examples/api_quickstart.py) cover standard,
generalized, complex Hermitian and SVD problems without application data or
reference-library dependencies. Run `python -m examples.api_quickstart` from
a source checkout; add `--cudss` to exercise the direct inverse backend.

## API reference

| Area | Entry points |
|---|---|
| [Symmetric and generalized](api/symmetric.md) | BlockLOBPCG, KrylovSchur, generalized and transformed solvers |
| [Complex and nonsymmetric](api/complex.md) | Hermitian, general and shift-invert solvers |
| [SVD](api/svd.md) | PartialSVD, NormalSVD |
| [Operators and inverses](api/operators.md) | Sparse storage, cuDSS, CG, GMRES, batched preconditioning |
| [Constraints](api/constraints.md) | KKT systems, independent rows, invariant-space projection |
| [Diagnostics](api/diagnostics.md) | Fresh residuals, numerical nullspaces, lowest-mode completeness |

## Execution and examples

- [Arrays, ownership, streams and CUDA graph capture](library_contracts.md)
- [Completeness and recovery after mode crossings](completeness.md)
- [FEM and geometry applications](../benchmarks/applications/README.md)
- [Anvil setup and visualization](anvil.md)
- [Release notes and the fast Anvil configuration](releases/v0.2.0.md)

## Development and evidence

[Testing, benchmarking and maintaining the reference](development.md) describes
contributor workflows. [Recorded validation](../results/library_review/README.md)
and [historical experiments](../historical/README.md) retain measurements and
implementation history separately from the user-facing API.

## Design proposals

[Integrating sparse eigensolvers into warp.optim](design/warp_optim_integration.md)
proposes an upstream API, execution contracts and staged migration. It describes
future interfaces, not the current Warp or warpack API.
