# Supported fast block solver: FEM qualification

This follow-up to v0.2.0 exercises the installed `BlockLOBPCG`,
`BatchedSparseOperator`, and `BatchedCGPreconditioner` APIs on the independent
[FEM fixtures](FEM.md). It is the first increment of
[coverage issue #128](https://github.com/alecjacobson/warpack/issues/128), not a
universal configuration or a replacement for the separate Anvil comparison.

## Scope and contracts

The six families cover clamped tetrahedra with lumped or consistent mass,
free bodies, two disconnected bodies, material/density contrast, and two
identical bending polarizations. The heterogeneous fixture uses 20:1 Young's
modulus and 3:1 density contrasts. The tetrahedral mass is consistent except
in `clamped_lumped`; bending uses consistent Hermite mass.

`--modes` requests **elastic modes**. Free/disconnected cases retain 6/12 rigid
modes in addition. A request at a repeated eigenvalue includes its entire
cluster; the report records the actual count and the first excluded eigenvalue.
These cases do not project out the rigid space. Choose `width >= actual k`
and `3*width <= n`. Tetrahedral preconditioning uses natural 3x3 displacement
blocks; bending uses scalar CSR rather than regrouping displacement/rotation
pairs into xyz blocks.

Every trial must pass original-equation residual, spectrum, mass orthogonality,
full requested-subspace, and known-nullspace checks at `1e-8`. The independent
references use dense generalized LAPACK on small fixtures and expanded-basis
ARPACK shift-invert on the larger ones. Reference assembly and factorization
are outside the timings. Warp must also return status zero and all requested
pairs converged. CuPy must meet its own declared absolute tolerance on its
**entire starting block**, checked with fresh matrix products.

The tests in `tests/test_fem_fast_block.py` use two requested mode counts (3 and
6) on each family, two graph replays, independent complete-space checks, and
DOT checks for host nodes and transfer directions. The odd bending request
explicitly exercises completion of a double eigenvalue. Stock Warp skips this
suite when upstream FSAI/batched CG is unavailable.

## Configuration and reproduction

Use the [pinned upstream environment](../../docs/installation.md#fast-block-solves)
and benchmark extras. All solver computation is Warp; there is no cuDSS in this
profile. The CuPy reference uses native LOBPCG with the same supplied Warp
preconditioner. It is not native CuPy `eigsh`, nor an independent CuPy setup.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.monitor_gpu --out build/fem.occupancy.json \
  --log build/fem.log -- python -m benchmarks.experimental.compare_block \
  --case clamped_consistent --side 10 --modes 6 --width 12 \
  --steps 300 --tol 1e-11 --public-api --gram-chunk 4096 \
  --symmetric-products --fused-updates --paired-rotation --reuse-x-image \
  --batched-products --inner-cg 25 --batched-cg --active-cg-bucket-size 4 \
  --inner-fp32 --inner-tol .01 --active-search-bucket-size 4 \
  --active-search-keep-history --fsai-layout columns --matrix-layout columns \
  --cupy --cupy-tol 1e-10 --repeats 3 --out build/fem.json
```

FSAI width is 64. The preconditioned matrix is `K + .01 M`. FP32 CG is a
truncated preconditioner; outer products and projected solves remain FP64.
For 6 elastic modes, widths are 12 (clamped/heterogeneous), 16 (free), and
24 (disconnected). For 20, they are 24, 32, and 36 respectively. The bending
follow-up uses side 6, width 12, Warp tolerance `1e-8` and CuPy absolute tolerance
`1e-7`; its original stricter run is retained separately.

Exact commands, source hashes, package versions, setup/capture scope, every
trial, and approximately one-second GPU occupancy records are in
[`results/fem_fast_block_v020`](../../results/fem_fast_block_v020/).
Re-run serially; identical nominal tolerance arguments are not identical
stopping rules across libraries.

## Results and limits

See the [recorded results](../../results/fem_fast_block_v020/README.md).
These intermediate-size tetrahedral cases have 10,890–22,506 DOFs. The bending
case is a small conditioning/multiplicity test, not a large performance case.
This sweep does not qualify different GPU architectures, substantially larger
material contrasts, mesh aspect ratios, or all large mode counts.

The strict bending profile stalls at 4/6 counted modes at `1e-11` despite
passing the `1e-8` physical checks; [#130](https://github.com/alecjacobson/warpack/issues/130)
documents the [precision study](../../results/bending_floor/README.md). A 60-digit
reference rounded to FP64 also misses this particular `1e-11` target; this is
evidence of precision sensitivity, not a proof of the best attainable residual. Do not silently
relax a caller's requested tolerance. The separate practical profile passes
all six Warp modes at `1e-8`.

CuPy can miss its full-block native stopping bound on disconnected bodies
([#69](https://github.com/alecjacobson/warpack/issues/69)) and crashes in residual
orthogonalization on bending ([#90](https://github.com/alecjacobson/warpack/issues/90)).
The harness now retains completed rows and an explicit failure row when the
optional reference raises ([#129](https://github.com/alecjacobson/warpack/issues/129)).
Rejected comparisons cannot establish a speedup.

## Related material updates

The [public warm-start comparison](../../results/fem_public_starts/README.md)
adds 34,560-DOF, 20-mode consistent/lumped FEM pencils after a 2% regional
stiffness change. It separates computed-start reuse from first-solve cost and
includes same-graph material-update regression tests.

## Setup memory

The [FSAI memory study](../../results/fsai_memory/README.md) isolates substantial
upstream CUDA context usage at row size 64. A fresh 20-mode comparison on two
consistent-mass FEM fixtures qualifies row size 32 with similar replay speed,
about 3.5x lower setup time and 7.5x lower device-memory increase. Specify
`--fsai-width 32` to try that profile; verify accuracy on the actual matrix.
The historical commands above use 64 and are unchanged. The released Anvil
configuration already uses 32.

## Shape updates and fixed-support refits

The [shape-family qualification](../../results/fem_shapes/README.md) changes both
K and M at fixed topology, compares cold/computed/perturbed starts and retained
versus refitted FSAI, and includes explicit crossing/repeated-cluster regressions.
Its 34,560-DOF cases use factor row cap 32; all setup, source-solve and changed-state
costs are separate. This extends the material-only reuse coverage.

## Coarse-to-fine starts

The [nested P1 beam comparison](../../results/fem_hierarchy/README.md) adds
computed 1,620-DOF coarse bases, Warp barycentric transfer, and fine-mass
refinement on 34,560 DOFs for 20/50 modes. It checks Galerkin identities against
independent assembly and the full requested fine subspace. Transfer can run in
the same graph as refinement through `BlockLOBPCG.capture(initializer=...)`.
Fine iterations decrease modestly, but recomputing the coarse solution costs
more than the saved refinement time in these cases. This is a starting-basis
baseline; multilevel preconditioning remains work under
[#134](https://github.com/alecjacobson/warpack/issues/134).
