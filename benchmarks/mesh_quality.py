"""Device tetrahedron shape metrics and host connectivity/summary diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
from scipy.sparse.csgraph import connected_components

from warpack.fem import read_mesh


@wp.kernel
def tet_metrics(x: wp.array[wp.vec3d], t: wp.array[wp.vec4i], out: wp.array2d[wp.float64]):
    e = wp.tid()
    ids = t[e]
    p = wp.matrix(shape=(4, 3), dtype=wp.float64)
    for i in range(4):
        for j in range(3):
            p[i, j] = x[ids[i]][j]
    dm = wp.matrix_from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0])
    det = wp.determinant(dm)
    volume = wp.abs(det) / wp.float64(6.0)
    edge_sum = wp.float64(0.0)
    for i in range(4):
        for j in range(i + 1, 4):
            edge_sum += wp.length_sq(p[j] - p[i])
    # Mean ratio: 1 for a regular tetrahedron and 0 for a collapsed one.
    out[e, 0] = (
        wp.float64(12.0)
        * wp.pow(wp.float64(3.0) * volume, wp.float64(0.6666666666666666))
        / edge_sum
    )
    out[e, 1] = volume
    normals = wp.matrix(shape=(4, 3), dtype=wp.float64)
    for i in range(4):
        a, b, c = (i + 1) % 4, (i + 2) % 4, (i + 3) % 4
        normal = wp.cross(p[b] - p[a], p[c] - p[a])
        if wp.dot(normal, p[i] - p[a]) > wp.float64(0.0):
            normal = -normal
        normals[i] = wp.normalize(normal)
    angle = wp.float64(180.0)
    for i in range(4):
        for j in range(i + 1, 4):
            cosine = wp.clamp(-wp.dot(normals[i], normals[j]), wp.float64(-1.0), wp.float64(1.0))
            angle = wp.min(angle, wp.acos(cosine) * wp.float64(57.29577951308232))
    out[e, 2] = angle
    regular = wp.mat33d(
        wp.float64(1.0),
        wp.float64(0.5),
        wp.float64(0.5),
        wp.float64(0.0),
        wp.sqrt(wp.float64(3.0)) / wp.float64(2.0),
        wp.sqrt(wp.float64(3.0)) / wp.float64(6.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.sqrt(wp.float64(0.6666666666666666)),
    )
    jac = dm * wp.inverse(regular)
    out[e, 3] = wp.sqrt(wp.ddot(jac, jac) * wp.ddot(wp.inverse(jac), wp.inverse(jac))) / wp.float64(
        3.0
    )
    out[e, 4] = det


def summarize(values):
    percentiles = [0, 0.1, 1, 5, 50, 95, 99, 99.9, 100]
    return {str(p): float(v) for p, v in zip(percentiles, np.percentile(values, percentiles))}


def inspect(path):
    x, t = read_mesh(path)
    extent = np.ptp(x, axis=0)
    x = (x - x.mean(0)) / extent.max()
    data = wp.empty((len(t), 5), dtype=wp.float64)
    wp.launch(tet_metrics, len(t), [wp.array(x, dtype=wp.vec3d), wp.array(t, dtype=wp.vec4i), data])
    values = data.numpy()
    edges = np.column_stack([np.repeat(t[:, 0], 3), t[:, 1:].ravel()])
    adjacency = sp.coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(len(x), len(x))
    ).tocsr()
    count, labels = connected_components(adjacency, directed=False)
    masses = np.bincount(t.ravel(), weights=np.repeat(values[:, 1] / 4, 4), minlength=len(x))
    return {
        "file": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "vertices": len(x),
        "tetrahedra": len(t),
        "dofs": 3 * len(x),
        "original_extent": extent.tolist(),
        "normalized_volume": float(values[:, 1].sum()),
        "connected_components": int(count),
        "component_vertices": np.bincount(labels).tolist(),
        "unreferenced_vertices": int(np.count_nonzero(masses == 0)),
        "mean_ratio_percentiles": summarize(values[:, 0]),
        "minimum_dihedral_degrees_percentiles": summarize(values[:, 2]),
        "regular_reference_condition_percentiles": summarize(values[:, 3]),
        "tet_volume_percentiles": summarize(values[:, 1]),
        "nodal_lumped_volume_percentiles": summarize(masses),
        "mean_ratio_below": {
            str(v): int(np.count_nonzero(values[:, 0] < v)) for v in [0.001, 0.01, 0.05, 0.1, 0.2]
        },
        "minimum_dihedral_below": {
            str(v): int(np.count_nonzero(values[:, 2] < v)) for v in [0.1, 1, 5, 10]
        },
        "zero_volume_tets": int(np.count_nonzero(values[:, 1] == 0)),
        "negative_reference_orientations": int(np.count_nonzero(values[:, 4] < 0)),
        "orientation_note": "Negative signed reference volume indicates vertex ordering, not by itself an invalid tetrahedron.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("meshes", nargs="+", type=Path)
    parser.add_argument("--out", default="results/mesh_quality.json")
    args = parser.parse_args()
    report = [inspect(path) for path in args.meshes]
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
