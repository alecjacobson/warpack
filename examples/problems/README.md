# Application inputs

These helpers construct application-specific inputs for warpack. They are shared
by examples, benchmarks, and regression tests, and require a source checkout.
Run drivers from the repository root with `python -m examples.dragon_modes` or
`python -m benchmarks.anvil`; the `examples` package is not included in the wheel.

| Module | Responsibility |
|---|---|
| [elasticity.py](elasticity.py) | Medit mesh loading, tetrahedral rest stiffness and lumped mass, graph-captured assembly updates, mass normalization, and rigid-motion bases |
| [skinning.py](skinning.py) | Scalar weight stiffness from elastic Hessian blocks and coordinate-weighted rig constraint rows |

Numerical assembly uses Warp. Mesh parsing uses NumPy on the host. Independent
CPU reference assembly and accuracy checks remain in `benchmarks/applications/`.

The library accepts the resulting matrices without knowing their physical origin.
`ConstrainedKrylovSchur`, `saddle_point_matrix`, `independent_constraints`, and
mass-orthogonal projections remain in `warpack`: they operate on general matrix
and constraint inputs, not a particular mesh, material, or rig.

Imports have moved from `warpack.fem` to `examples.problems.elasticity`, and from
`warpack.skinning` to `examples.problems.skinning`. `RestElasticity` and
`rigid_body_basis` are no longer top-level library exports. The generic row
reduction helper formerly in `warpack.skinning` is now
`from warpack import independent_constraints` (implemented in
`warpack.constraints`). Existing numerical benchmark results predate this
relocation; their source hashes and measurements are preserved.
