"""Check the entire sinusoidal cycle against tetrahedral inversions on device."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from warpack.fem import read_mesh


@wp.func
def jacobian(a: wp.float64, b: wp.float64, c: wp.float64, t: wp.float64):
    return wp.float64(1.0) + t * (a + t * (b + t * c))


@wp.kernel
def minimum_jacobian(
    x: wp.array[wp.vec3d],
    tets: wp.array[wp.vec4i],
    u: wp.array2d[wp.vec3d],
    scale: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    mode, e = wp.tid()
    t = tets[e]
    dm = wp.matrix_from_cols(x[t[1]] - x[t[0]], x[t[2]] - x[t[0]], x[t[3]] - x[t[0]])
    du = (
        wp.matrix_from_cols(
            u[mode, t[1]] - u[mode, t[0]],
            u[mode, t[2]] - u[mode, t[0]],
            u[mode, t[3]] - u[mode, t[0]],
        )
        * scale[mode]
    )
    g = du * wp.inverse(dm)
    a = wp.trace(g)
    b = (a * a - wp.trace(g * g)) * wp.float64(0.5)
    c = wp.determinant(g)
    low = wp.min(jacobian(a, b, c, wp.float64(-1.0)), jacobian(a, b, c, wp.float64(1.0)))
    aa = wp.float64(3.0) * c
    bb = wp.float64(2.0) * b
    if wp.abs(aa) > wp.float64(1.0e-30):
        disc = bb * bb - wp.float64(4.0) * aa * a
        if disc >= wp.float64(0.0):
            root = wp.sqrt(disc)
            if bb < wp.float64(0.0):
                root = -root
            q = -wp.float64(0.5) * (bb + root)
            t1 = q / aa
            if wp.abs(t1) <= wp.float64(1.0):
                low = wp.min(low, jacobian(a, b, c, t1))
            if wp.abs(q) > wp.float64(1.0e-30):
                t2 = a / q
                if wp.abs(t2) <= wp.float64(1.0):
                    low = wp.min(low, jacobian(a, b, c, t2))
    elif wp.abs(bb) > wp.float64(1.0e-30):
        tt = -a / bb
        if wp.abs(tt) <= wp.float64(1.0):
            low = wp.min(low, jacobian(a, b, c, tt))
    wp.atomic_min(out, mode, low)


def main():
    d = np.load("results/dragon_modes.npz")
    _, t = read_mesh("../bbw-comparison/dragon-H/dragon.mesh")
    u = d["modes"][6:26]
    ids = np.unique(d["faces"])
    scales = 0.025 / np.linalg.norm(u[:, ids], axis=2).max(axis=1)
    out = wp.full(20, 1.0, dtype=wp.float64)
    wp.launch(
        minimum_jacobian,
        (20, len(t)),
        [
            wp.array(d["vertices"], dtype=wp.vec3d),
            wp.array(t, dtype=wp.vec4i),
            wp.array(u, dtype=wp.vec3d),
            wp.array(scales, dtype=wp.float64),
            out,
        ],
    )
    mins = out.numpy()
    report = dict(
        minimum_jacobian_over_whole_cycle=mins.tolist(),
        visual_scales=scales.tolist(),
        maximum_surface_displacement_m=0.025,
        inverted=bool(np.any(mins <= 0)),
    )
    Path("results/animation_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert mins.min() > 0, "Display amplitude inverts at least one tetrahedron"


if __name__ == "__main__":
    main()
