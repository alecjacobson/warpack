"""Assemble and solve the full dragon; host I/O and validation follow GPU solve."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from warpack import CuDSSInverse, KrylovSchur, SparseOperator, SymmetricEigensolver
from warpack.fem import assemble_rest, mass_normalized, read_mesh
from warpack.solver import EigenpairEvaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", default="../bbw-comparison/dragon-H/dragon.mesh")
    p.add_argument("--out", default="results/dragon_modes.npz")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--method", choices=["krylov", "block"], default="krylov")
    args = p.parse_args()
    start = time.perf_counter()
    x, t = read_mesh(args.mesh)
    # Normalize longest bounding-box extent to one metre, preserve aspect ratio.
    scale = np.ptp(x, axis=0).max()
    x = (x - x.mean(axis=0)) / scale
    print(f"Mesh: {len(x)} vertices, {len(t)} tetrahedra", flush=True)
    h, m, mass = assemble_rest(x, t)
    h.nnz_sync()
    a = mass_normalized(h, mass)
    shifted = mass_normalized(h, mass, shift=1.0)
    wp.synchronize()
    assembly = time.perf_counter() - start
    print(f"Assembly + IO: {assembly:.3f}s; {h.nnz} blocks", flush=True)
    start = time.perf_counter()
    inverse = CuDSSInverse(shifted, args.width if args.method == "block" else 1)
    wp.synchronize()
    setup = time.perf_counter() - start
    print(f"cuDSS setup: {setup:.3f}s", flush=True)
    if args.method == "block":
        solver = SymmetricEigensolver(
            SparseOperator(a),
            26,
            ncv=args.width,
            iteration_operator=inverse,
            tol=1.0e-7,
        )
    else:
        solver = KrylovSchur(inverse, 26, ncv=args.width, which="LA", tol=1.0e-12)
    graph = solver.capture(args.iterations, adaptive=False)
    evaluation = EigenpairEvaluation(SparseOperator(a), solver.eigenvectors, tol=1.0e-7)
    evaluation.run()
    wp.synchronize()
    with wp.ScopedCapture(device="cuda:0") as final_cap:
        evaluation.run()
    start = time.perf_counter()
    wp.capture_launch(graph)
    wp.capture_launch(final_cap.graph)
    wp.synchronize()
    first_replay = time.perf_counter() - start
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        wp.capture_launch(graph)
        wp.capture_launch(final_cap.graph)
        wp.synchronize()
        timings.append(time.perf_counter() - start)
    elapsed = float(np.median(timings))
    vals = evaluation.values.numpy()
    err = evaluation.residuals.numpy()
    u = solver.eigenvectors.numpy()
    iterations = int(solver.iterations.numpy()[0])
    masses = mass.numpy()
    modes = u.reshape(26, len(x), 3) / np.sqrt(masses)[None, :, None]
    orth = np.max(np.abs(u @ u.T - np.eye(26)))
    # Boundary faces; remove interior pairs and orient outward using opposite vertex.
    faces = np.concatenate([t[:, [1, 2, 3]], t[:, [0, 3, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 1]]])
    opposite = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 3]])
    _, first, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    ix = first[counts == 1]
    f = faces[ix]
    o = opposite[ix]
    normal = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    flip = np.sum(normal * (x[o] - x[f[:, 0]]), axis=1) > 0
    f[flip] = f[flip][:, [0, 2, 1]]
    report = dict(
        vertices=len(x),
        tetrahedra=len(t),
        dofs=len(x) * 3,
        blocks=h.nnz,
        young=1.0e5,
        poisson=0.3,
        density=1000.0,
        length_m=1.0,
        mass="lumped",
        boundary="free",
        assembly_and_io_seconds=assembly,
        factor_seconds=setup,
        solve_seconds=elapsed,
        first_replay_seconds=first_replay,
        repeats=3,
        iterations=iterations,
        method=args.method,
        ncv=args.width,
        eigenvalues=vals.tolist(),
        relative_residuals=err.tolist(),
        mass_orthogonality_error=float(orth),
        cholesky_status=int(solver.status.numpy()[0]),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, vertices=x, faces=f, eigenvalues=vals, modes=modes, mass=masses)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    inverse.close()
    if err.max() > 1.0e-7 or orth > 1.0e-8:
        raise RuntimeError("Modes saved for diagnosis but validation has NOT passed")


if __name__ == "__main__":
    main()
