# Exact Anvil input provenance

[#142](https://github.com/alecjacobson/warpack/issues/142) adds a verified shared
assembled fixture to the six-method Anvil comparison. Factorizations,
preconditioners, solver workspaces and captures remain independently constructed
and charged to each row. Only the mathematical inputs are shared.

Repeated Warp assembly of this mesh produced identical stiffness values but
13,310 differing per-vertex mass entries, with relative maximum difference
3.30e-16. The resulting normalized stiffness differed at 1,646,368 entries
(relative maximum 4.45e-17). This is rounding from parallel mass accumulation,
not evidence of inaccurate modes ([diagnostic](assembly-identity.log)). Exact
matrix hashes therefore require a fixed assembled snapshot rather than a mesh
filename alone.

`benchmarks.anvil_inputs` assembles with Warp once and stores normalized positions,
compact 3x3 stiffness structure/values, lumped masses, normalized stiffness and
rigid constraints. Every load verifies array dtype/shape/content hashes and the
mesh, assembly source, material and normalization identity. Existing fixtures
are never silently overwritten. The cached reference's bytes and arrays are
fingerprinted; starting vectors and actual shifted matrices are recorded per
method. CPU CSR/CSC and GPU BSR fingerprints use canonical scalar CSR semantics.
CPU factors reject a stale matrix cache that differs from the shared fixture.

The showcase gate requires identical physical inputs and reference across all
methods, and identical starts/shifted matrices across repetitions of each
method. Different methods may use different starts or shifts. The dependency
promotion gate additionally requires identical complete inputs between its two
profiles. Older reports missing this evidence must be processed with their
historical checkout, not backfilled with invented hashes.

## Validation

All six production comparison paths passed native and independent original
residual/spectrum/subspace checks on the verified 283,857-DOF fixture:
SciPy ARPACK/PARDISO, Spectra/PARDISO, CuPy eigsh/cuDSS, Warp/cuDSS,
CuPy LOBPCG/Warp FSAI-CG and Warp block/FSAI-CG. Each ran one warmup and one
measured solve. All sampled GPU occupancy was contended; **no timing claim** is
made and the README timing table is unchanged.

[Validation summary](validation.json), [fixture manifest](fixture.manifest.json),
and `screening/` retain every worker's exact provenance, results and occupancy.
The 205 MiB derived fixture is an ignored local benchmark cache, not a new mesh
or a committed binary. No external meshes or data were copied.

[Tests](tests.log) check changed mass/stiffness/reference/starts, stale and relocated
caches, missing historical provenance, sparse format equivalence, exact CPU/GPU
shift agreement, changed dependencies and rejected performance regressions.
Fingerprinting and file I/O occur outside solver setup/replay timing and CUDA
capture. No installed solver code changes.

## Prepare and compare

Use the current pinned Warp overlay and the existing Anvil reference/initial
vector and PARDISO/Spectra library described in the
[comparison prerequisites](../anvil_showcase_132/README.md#environment-and-reproduction).
From the repository root:

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export WARPACK_ANVIL_INPUTS="$PWD/build/anvil_inputs.npz"
python -m benchmarks.anvil_inputs
python -m benchmarks.anvil_factor_audit --backend assemble \
  --out build/anvil-factor-audit/verified-matrix.json
# Set the MKL runtime library path/preload as in the prerequisites if needed.
python -m benchmarks.anvil_showcase --out build/anvil-verified --rounds 3 --repeats 5
```

The CPU shifted cache defaults to the fixture path with suffix `.shift1.npz`;
`--matrix` can override it but its coefficients must still match the fixture.
`WARPACK_ANVIL_INPUTS` permits keeping large derived caches on another filesystem.
Keep one verified fixture for a baseline/candidate comparison. Regenerating it
may change rounding-level coefficients; that creates a new input identity and
requires a new baseline. The user-facing Anvil example still assembles directly
from the mesh and does not require this comparison cache.
