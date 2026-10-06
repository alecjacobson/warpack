"""Independent constitutive/assembly check and mode-09 localization diagnosis.

Run after generating the original and fTetWild numerical mode archives.
All assembly and alternative eigenproblem solves use Warp + optional cuDSS;
NumPy/SciPy below perform explicit offline diagnostics only.
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
import warp.sparse as ws
from scipy.sparse.csgraph import connected_components, dijkstra

from benchmarks.elasticity_reference import assemble_consistent_mass, assemble_stvk_rest
from warpack import (
    CuDSSInverse,
    EigenpairEvaluation,
    GeneralizedEigensolver,
    KrylovSchur,
    SparseOperator,
)
from warpack.fem import RestElasticity, mass_normalized, read_mesh


def vector_agreement(a, b, mass):
    aa, bb = a * np.sqrt(mass)[None, :, None], b * np.sqrt(mass)[None, :, None]
    return (
        np.sum(aa * bb, axis=(1, 2)) ** 2
        / (np.sum(aa**2, axis=(1, 2)) * np.sum(bb**2, axis=(1, 2)))
    ).tolist()


def localization(x, t, modes, masses, display_scales):
    u = modes[8]
    norm = np.linalg.norm(u, axis=1)
    tip = int(norm.argmax())
    edges = np.unique(
        np.sort(np.concatenate([t[:, [i, j]] for i in range(4) for j in range(i + 1, 4)]), axis=1),
        axis=0,
    )
    lengths = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
    graph = sp.coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(len(x), len(x))).tocsr()
    distance = dijkstra(graph, directed=False, indices=tip, limit=0.08)
    region = distance <= 0.08
    # Face-connected components detect attachments through only an edge/vertex.
    faces = np.concatenate([t[:, [1, 2, 3]], t[:, [0, 2, 3]], t[:, [0, 1, 3]], t[:, [0, 1, 2]]])
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    pair = np.column_stack([order[starts[counts == 2]], order[starts[counts == 2] + 1]]) % len(t)
    tg = sp.coo_matrix(
        (np.ones(len(pair)), (pair[:, 0], pair[:, 1])), shape=(len(t), len(t))
    ).tocsr()
    components, _ = connected_components(tg, directed=False)
    rms = np.sqrt(np.sum(masses * norm**2) / masses.sum())
    return {
        "mode": 9,
        "tip_vertex_zero_based": tip,
        "tip_position_in_normalized_input_coordinates": x[tip].tolist(),
        "tip_peak_over_mass_weighted_rms": float(norm[tip] / rms),
        "region_definition": "vertices within 0.08 m weighted tet-edge graph distance of mode-09 displacement maximum; longest body extent is 1 m",
        "region_vertices": int(region.sum()),
        "region_mass_fraction": float(masses[region].sum() / masses.sum()),
        "region_mode09_kinetic_fraction": float(
            np.sum(masses[region] * norm[region] ** 2) / np.sum(masses * norm**2)
        ),
        "region_bbox_extent_m": np.ptp(x[region], axis=0).tolist(),
        "tip_incident_tets": int(np.count_nonzero(np.any(t == tip, axis=1))),
        "face_connected_tet_components": int(components),
        "faces_with_more_than_two_incident_tets": int(np.count_nonzero(counts > 2)),
        "displayed_peak_displacement_m": float(norm[tip] * display_scales[8]),
        "displacement_at_2point5percent_body_length_scale_m": 0.025,
        "display_amplification_relative_to_2point5percent": float(
            norm[tip] * display_scales[8] / 0.025
        ),
    }, region


def inspect(mesh, modes_file, teaser, consistent=False):
    print("Auditing", mesh, flush=True)
    data = np.load(modes_file)
    x, t = read_mesh(mesh)
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    np.testing.assert_allclose(x, data["vertices"], rtol=0, atol=1e-15)
    model = RestElasticity(x, t)
    model.update()
    with wp.ScopedCapture() as captured:
        model.update()
    wp.capture_launch(captured.graph)
    hs, ms = assemble_stvk_rest(x, t)
    hs.nnz_sync()
    model.hessian.nnz_sync()
    np.testing.assert_array_equal(hs.offsets.numpy(), model.hessian.offsets.numpy())
    np.testing.assert_array_equal(
        hs.columns.numpy()[: hs.nnz], model.hessian.columns.numpy()[: model.hessian.nnz]
    )
    v = hs.values.numpy()[: hs.nnz]
    original = model.hessian.values.numpy()[: hs.nnz]
    error = float(np.linalg.norm(v - original) / np.linalg.norm(original))
    mass_error = float(np.max(abs(ms.numpy() / model.mass.numpy() - 1)))
    assert error < 1e-12 and mass_error < 1e-12
    del v, original
    a = mass_normalized(hs, ms)
    inv = CuDSSInverse(mass_normalized(hs, ms, shift=1), 1)
    solver = KrylovSchur(inv, 26, ncv=64, which="LA", tol=1e-12)
    ev = EigenpairEvaluation(SparseOperator(a), solver.eigenvectors, tol=1e-7)
    graph = solver.capture(1, adaptive=False, finalizer=ev.run)
    wp.capture_launch(graph)
    vals = ev.values.numpy()
    order = np.argsort(vals)
    vals = vals[order]
    vectors = (
        solver.eigenvectors.numpy()[order].reshape(26, len(x), 3)
        / np.sqrt(ms.numpy())[None, :, None]
    )
    residual = float(ev.residuals.numpy().max())
    agreement = vector_agreement(data["modes"][6:26], vectors[6:], data["mass"])
    freq_error = float(np.max(abs(np.sqrt(vals[6:] / data["eigenvalues"][6:26]) - 1)))
    assert residual < 1e-7 and min(agreement) > 0.99999 and freq_error < 1e-8
    del graph, solver, ev
    inv.close()
    report = {
        "mesh": mesh,
        "stvk_vs_production_hessian_relative_frobenius": error,
        "stvk_vs_production_mass_max_relative_error": mass_error,
        "production_graph_reassembly_checked": True,
        "stvk_solve_max_original_residual": residual,
        "stvk_max_relative_frequency_change": freq_error,
        "stvk_elastic_mode_MAC": agreement,
    }
    report["mode09"], region = localization(
        x,
        t,
        data["modes"][6:26],
        data["mass"],
        json.loads(Path(teaser).read_text())["peak_visual_scales"],
    )
    if consistent:
        cm = assemble_consistent_mass(x, t)
        shifted = ws.bsr_copy(hs)
        ws.bsr_axpy(cm, shifted)
        inv = CuDSSInverse(shifted, 48)
        solver = GeneralizedEigensolver(
            SparseOperator(hs), SparseOperator(cm), inv, 26, ncv=48, tol=1e-8
        )
        graph = solver.capture(40, adaptive=False)
        wp.capture_launch(graph)
        vals = solver.eigenvalues.numpy()
        vec = solver.eigenvectors.numpy().reshape(26, len(x), 3)
        residual = float(solver.residuals.numpy().max())
        assert residual < 1e-7, residual
        ref = data["modes"][6:26]
        weights = data["mass"]
        correlations = np.einsum("mvi,nvi,v->mn", ref, vec[6:], weights) ** 2 / (
            np.einsum("mvi,mvi,v->m", ref, ref, weights)[:, None]
            * np.einsum("nvi,nvi,v->n", vec[6:], vec[6:], weights)[None, :]
        )
        mode = int(np.argmax(correlations[8]))
        u = vec[6 + mode]
        mag = np.linalg.norm(u, axis=1)
        report["consistent_mass"] = {
            "max_generalized_residual": residual,
            "frequencies_hz": (np.sqrt(vals[6:]) / (2 * np.pi)).tolist(),
            "frequency_changes_percent": (
                100 * (np.sqrt(vals[6:] / data["eigenvalues"][6:26]) - 1)
            ).tolist(),
            "mode_matching_original_mode09": mode + 1,
            "mode09_lumped_mass_MAC": float(correlations[8, mode]),
            "mode09_peak_over_lumped_mass_rms": float(
                mag.max() / np.sqrt(np.sum(weights * mag**2) / weights.sum())
            ),
            "mode09_peak_vertex": int(mag.argmax()),
        }
        del graph, solver
        inv.close()
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    rows = []
    for mesh, modes, teaser, consistent in [
        (
            "xyzrgb_dragon-720K-ftetwild.mesh",
            "results/dragon_ftetwild_modes.npz",
            "results/dragon_ftetwild_teaser.json",
            True,
        ),
        (
            "../bbw-comparison/dragon-H/dragon.mesh",
            "results/dragon_optimized.npz",
            "results/dragon_teaser.json",
            False,
        ),
    ]:
        rows.append(inspect(mesh, modes, teaser, consistent))
        Path("results/elasticity_audit.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
