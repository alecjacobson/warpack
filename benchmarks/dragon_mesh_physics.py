"""Check whether low-quality elements dominate the displayed elastic modes."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.mesh_quality import tet_metrics
from warpack.fem import read_mesh


@wp.kernel
def modal_energy(
    x: wp.array[wp.vec3d],
    t: wp.array[wp.vec4i],
    u: wp.array2d[wp.vec3d],
    scales: wp.array[wp.float64],
    energy: wp.array2d[wp.float64],
    peak_det: wp.array2d[wp.float64],
):
    mode, e = wp.tid()
    ids = t[e]
    dm = wp.matrix_from_cols(x[ids[1]] - x[ids[0]], x[ids[2]] - x[ids[0]], x[ids[3]] - x[ids[0]])
    du = wp.matrix_from_cols(
        u[mode, ids[1]] - u[mode, ids[0]],
        u[mode, ids[2]] - u[mode, ids[0]],
        u[mode, ids[3]] - u[mode, ids[0]],
    )
    grad = du * wp.inverse(dm)
    strain = (grad + wp.transpose(grad)) * wp.float64(0.5)
    mu = wp.float64(38461.53846153846)
    lam = wp.float64(57692.30769230769)
    energy[mode, e] = (
        wp.abs(wp.determinant(dm))
        / wp.float64(6.0)
        * (
            wp.float64(2.0) * mu * wp.ddot(strain, strain)
            + lam * wp.trace(strain) * wp.trace(strain)
        )
    )
    peak_det[mode, e] = wp.determinant(wp.identity(n=3, dtype=wp.float64) + scales[mode] * grad)


def inspect(mesh, modes_file, teaser):
    data = np.load(modes_file)
    _, t = read_mesh(mesh)
    scales = np.array(json.loads(Path(teaser).read_text())["peak_visual_scales"])
    x = wp.array(data["vertices"], dtype=wp.vec3d)
    dt = wp.array(t, dtype=wp.vec4i)
    quality = wp.empty((len(t), 5), dtype=wp.float64)
    wp.launch(tet_metrics, len(t), [x, dt, quality])
    quality = quality.numpy()
    energy = wp.empty((20, len(t)), dtype=wp.float64)
    determinants = wp.empty_like(energy)
    wp.launch(
        modal_energy,
        (20, len(t)),
        [
            x,
            dt,
            wp.array(data["modes"][6:26], dtype=wp.vec3d),
            wp.array(scales, dtype=wp.float64),
            energy,
            determinants,
        ],
    )
    energy, determinants = energy.numpy(), determinants.numpy()
    totals = energy.sum(axis=1)
    fractions = {}
    for name, mask in [
        ("mean_ratio_below_0.1", quality[:, 0] < 0.1),
        ("dihedral_below_5_degrees", quality[:, 2] < 5),
        ("dihedral_below_10_degrees", quality[:, 2] < 10),
    ]:
        fractions[name] = {
            "elements": int(mask.sum()),
            "modal_energy_fractions": (energy[:, mask].sum(axis=1) / totals).tolist(),
        }
    return {
        "mesh": mesh,
        "elastic_eigenvalues_from_element_energy": totals.tolist(),
        "energy_vs_rayleigh_max_relative_error": float(
            np.max(abs(totals / data["eigenvalues"][6:26] - 1))
        ),
        "low_quality_element_energy": fractions,
        "exaggerated_teaser_peak_minimum_jacobian": determinants.min(axis=1).tolist(),
        "exaggerated_teaser_peak_inverted_tets": (determinants <= 0).sum(axis=1).tolist(),
        "note": "Peak Jacobians describe deliberately amplified linear modal poses, not the rest mesh or a nonlinear simulation.",
    }


def main():
    rows = [
        inspect(*inputs)
        for inputs in [
            (
                "../bbw-comparison/dragon-H/dragon.mesh",
                "results/dragon_optimized.npz",
                "results/dragon_teaser.json",
            ),
            (
                "xyzrgb_dragon-720K-ftetwild.mesh",
                "results/dragon_ftetwild_modes.npz",
                "results/dragon_ftetwild_teaser.json",
            ),
        ]
    ]
    Path("results/dragon_mesh_physics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
