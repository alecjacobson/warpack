"""Verified, shared Anvil inputs for comparisons; all hashing stays outside timing.

Build once with ``python -m benchmarks.anvil_inputs``. Comparison workers read
WARPACK_ANVIL_INPUTS (default build/anvil_inputs.npz), never silently reassemble.
Parallel mass assembly is not bitwise deterministic, so sharing only a mesh
filename is insufficient for exact input comparison.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

MESH = Path("anvil for export-ftetwild.mesh")
REFERENCE = Path("build/anvil_reference.npz")
ASSEMBLY_SOURCE = Path("examples/problems/elasticity.py")
MATERIAL = dict(young=1e5, poisson=0.3, density=1000.0)
ARRAYS = ("positions", "offsets", "columns", "stiffness", "mass", "normalized", "rigid")


def array_identity(array):
    a = np.ascontiguousarray(array)
    return dict(
        shape=list(a.shape),
        dtype=a.dtype.str,
        # A contiguous NumPy buffer also handles empty (n, 0) nullspace bases;
        # memoryview.cast rejects multidimensional views with a zero extent.
        sha256=hashlib.sha256(a).hexdigest(),
    )


def matrix_identity(matrix):
    """Canonical scalar CSR: storage format/index width and explicit zeros do not matter."""
    a = sp.csr_matrix(matrix, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    return dict(
        shape=list(a.shape),
        offsets=array_identity(a.indptr.astype("<i8")),
        columns=array_identity(a.indices.astype("<i8")),
        values=array_identity(a.data),
    )


def parameters(mesh=MESH):
    return dict(
        mesh_sha256=hashlib.sha256(Path(mesh).read_bytes()).hexdigest(),
        assembly_source_sha256=hashlib.sha256(ASSEMBLY_SOURCE.read_bytes()).hexdigest(),
        material=MATERIAL,
        normalization="(x - arithmetic vertex mean) / longest rest extent",
        energy="stable Neo-Hookean rest Hessian",
        mass="lumped per-vertex mass, repeated for xyz",
    )


def reference():
    data = REFERENCE.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        values, vectors = z["values"].copy(), z["vectors"].copy()
    return (vectors, values), dict(
        file_sha256=hashlib.sha256(data).hexdigest(),
        values=array_identity(values),
        vectors=array_identity(vectors),
    )


def fixture_path():
    return Path(os.environ.get("WARPACK_ANVIL_INPUTS", "build/anvil_inputs.npz"))


def save(path, arrays, configuration):
    manifest = dict(
        schema=1,
        configuration=configuration,
        arrays={key: array_identity(arrays[key]) for key in ARRAYS},
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently changing inputs between workers.
    with path.open("xb") as stream:
        try:
            np.savez(stream, **arrays, manifest=np.array(json.dumps(manifest, sort_keys=True)))
        except BaseException:
            path.unlink()
            raise
    return manifest


def load(path=None, configuration=None):
    path = fixture_path() if path is None else Path(path)
    if not path.exists():
        raise ValueError(
            f"Missing verified Anvil fixture: {path}; run python -m benchmarks.anvil_inputs"
        )
    with np.load(path, allow_pickle=False) as z:
        manifest = json.loads(str(z["manifest"]))
        arrays = {key: z[key].copy() for key in ARRAYS}
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported Anvil fixture schema")
    if manifest["configuration"] != (parameters() if configuration is None else configuration):
        raise ValueError(
            "Anvil fixture mesh/assembly/material identity changed; create a new fixture"
        )
    if manifest["arrays"] != {key: array_identity(value) for key, value in arrays.items()}:
        raise ValueError("Anvil fixture array fingerprint mismatch")
    return arrays, manifest


def host_matrix(arrays, name="normalized"):
    n = 3 * len(arrays["positions"])
    return sp.bsr_matrix((arrays[name], arrays["columns"], arrays["offsets"]), shape=(n, n)).tocsr()


def inputs(manifest, reference_identity, initial, shifted, *, constraints=None):
    return dict(
        schema=1,
        common=dict(fixture=manifest, reference=reference_identity),
        method=dict(
            initial=array_identity(initial),
            shifted_matrix=matrix_identity(shifted),
            constraints=None if constraints is None else array_identity(constraints),
        ),
    )


def signature(record, *, common_only=False):
    if (
        not isinstance(record, dict)
        or record.get("schema") != 1
        or not record.get("common")
        or not record.get("method")
    ):
        raise ValueError(
            "Missing verified Anvil input provenance; use historical checkout or collect a new baseline"
        )
    return record["common"] if common_only else record


def device_matrix(arrays, name="normalized", device="cuda:0"):
    import warp as wp
    import warp.sparse as ws

    n = len(arrays["positions"])
    a = ws.bsr_zeros(n, n, wp.mat33d, device=device)
    a.offsets = wp.array(arrays["offsets"], dtype=wp.int32, device=device)
    a.columns = wp.array(arrays["columns"], dtype=wp.int32, device=device)
    a.values = wp.array(arrays[name], dtype=wp.mat33d, device=device)
    a.nnz = len(arrays["columns"])
    # Replacing offsets invalidates the empty matrix's cached nnz transfer.
    # Stock Warp may otherwise reuse that stale zero on the next nnz_sync().
    a.notify_nnz_changed(nnz=a.nnz)
    return a


def shifted_device(arrays, shift, device="cuda:0"):
    # Scaling by unit mass preserves cached normalized coefficients exactly;
    # the existing Warp kernel adds the requested diagonal shift on device.
    import warp as wp

    from examples.problems.elasticity import mass_normalized

    a = device_matrix(arrays, device=device)
    ones = wp.ones(len(arrays["positions"]), dtype=wp.float64, device=device)
    return a, mass_normalized(a, ones, shift)


def host_device_matrix(a):
    a.nnz_sync()
    return sp.bsr_matrix(
        (a.values.numpy()[: a.nnz], a.columns.numpy()[: a.nnz], a.offsets.numpy()),
        shape=a.shape,
    ).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=fixture_path())
    args = parser.parse_args()
    if args.out.exists():
        parser.error("fixture already exists; choose a new path instead of overwriting evidence")
    import warp as wp

    from examples.problems.elasticity import (
        assemble_rest,
        mass_normalized,
        read_mesh,
        rigid_body_basis,
    )

    configuration = parameters()
    x, t = read_mesh(MESH)
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, _, mass = assemble_rest(x, t, **MATERIAL)
    a = mass_normalized(h, mass)
    h.nnz_sync()
    a.nnz_sync()
    rigid = rigid_body_basis(wp.array(x, dtype=wp.vec3d), mass)
    arrays = dict(
        positions=x,
        offsets=h.offsets.numpy(),
        columns=h.columns.numpy()[: h.nnz],
        stiffness=h.values.numpy()[: h.nnz],
        mass=mass.numpy(),
        normalized=a.values.numpy()[: a.nnz],
        rigid=rigid.numpy(),
    )
    if configuration != parameters():
        raise ValueError("Mesh or assembly source changed during fixture construction")
    manifest = save(args.out, arrays, configuration)
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
