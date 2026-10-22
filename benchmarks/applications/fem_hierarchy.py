"""Nested P1 tet-beam starts for BlockLOBPCG (not multilevel Lanczos).

Host assembly/topology and independent references are benchmark code. Computed
coarse vectors are prolonged on device; fine initialization uses the fine mass.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.audit_profile import graph_summary

from .fem_cases import beam_mesh, elastic_matrices, reference
from .fem_shapes import ShapeReplay, elapsed
from .problems import Problem


def beam_problem(side, modes, consistent=True):
    vertices, tets = beam_mesh(side)
    dofs = np.flatnonzero(np.repeat(vertices[:, 0] > 0, 3))
    a, b = elastic_matrices(vertices, tets, consistent=consistent)
    return Problem(
        f"nested_beam_{side}", "A", a[dofs, :][:, dofs], b[dofs, :][:, dofs], modes, -0.01
    )


def attach_reference(p):
    values, vectors, label = reference(p.a, p.b, p.k + 8)
    if len(values) <= p.k or values[p.k] - values[p.k - 1] < 1e-7 * max(1, values[p.k]):
        raise ValueError("requested boundary splits a cluster; increase modes")
    p.expected, p.requested_basis, p.notes = values[: p.k], vectors[:, : p.k], label


def transfer_stencil(coarse, fine):
    """Four barycentric weights in the coarse Freudenthal tetrahedron.

    Indices address xyz DOFs after eliminating x=0. A -1 index contributes zero
    Dirichlet motion. Sorting local coordinates selects one of the six P1 tets;
    trilinear interpolation would describe a different finite element space.
    """
    if coarse < 1 or fine < coarse or fine % coarse:
        raise ValueError("fine side must be a positive integer multiple of coarse side")
    shape = (3 * fine + 1, fine + 1, fine + 1)
    grid = np.array(list(np.ndindex(shape)), dtype=np.int64)[(fine + 1) ** 2 :]
    ratio = fine // coarse
    cell = np.minimum(grid // ratio, np.array([3 * coarse, coarse, coarse]) - 1)
    local = (grid - cell * ratio) / ratio
    order = np.argsort(-local, axis=1, kind="stable")
    descending = np.take_along_axis(local, order, axis=1)
    weights = np.column_stack(
        [
            1 - descending[:, 0],
            descending[:, 0] - descending[:, 1],
            descending[:, 1] - descending[:, 2],
            descending[:, 2],
        ]
    )
    offsets = np.eye(3, dtype=np.int64)[order].cumsum(axis=1)
    corners = cell[:, None, :] + np.concatenate([np.zeros((len(grid), 1, 3), int), offsets], axis=1)
    nodes = np.ravel_multi_index(
        corners.transpose(2, 0, 1), (3 * coarse + 1, coarse + 1, coarse + 1)
    )
    indices = 3 * (nodes[:, None, :] - (coarse + 1) ** 2) + np.arange(3)[None, :, None]
    indices = np.where(corners[:, None, :, 0] == 0, -1, indices)
    return indices.reshape(-1, 4).astype(np.int32), np.repeat(weights, 3, axis=0)


@wp.kernel
def _prolong(
    indices: wp.array2d[wp.int32],
    weights: wp.array2d[wp.float64],
    coarse: wp.array2d[wp.float64],
    fine: wp.array2d[wp.float64],
):
    row, i = wp.tid()
    value = wp.float64(0)
    for j in range(4):
        source = indices[i, j]
        if source >= 0:
            value += weights[i, j] * coarse[row, source]
    fine[row, i] = value


class BeamTransfer:
    """Fixed geometry transfer; updates of source values are visible on replay."""

    def __init__(self, coarse_side, fine_side, source, target):
        ids, weights = transfer_stencil(coarse_side, fine_side)
        coarse_n = 9 * coarse_side * (coarse_side + 1) ** 2
        if (
            source.ndim != 2
            or target.shape != (source.shape[0], len(ids))
            or source.shape[1] != coarse_n
        ):
            raise ValueError("incompatible row-major beam mode arrays")
        if (
            source.dtype != wp.float64
            or target.dtype != wp.float64
            or source.device != target.device
        ):
            raise ValueError("source and target must be FP64 on the same device")
        self.indices = wp.array(ids, dtype=wp.int32, device=source.device)
        self.weights = wp.array(weights, dtype=wp.float64, device=source.device)
        self.source, self.target = source, target

    def apply(self):
        wp.launch(
            _prolong,
            self.target.shape,
            [self.indices, self.weights, self.source, self.target],
            device=self.source.device,
        )


def audit(graph, path):
    wp.capture_debug_dot_print(graph, str(path))
    result = graph_summary(path)
    path.unlink()
    assert "HOST" not in result["node_types_including_conditional_body"]
    assert not any(kind.startswith(("HtoD", "DtoH")) for kind in result["memcpy_kinds"])
    return result


def matrix_hash(p):
    digest = hashlib.sha256()
    for a in (p.a, p.b):
        a.sort_indices()
        for data in (a.indptr, a.indices, a.data):
            digest.update(data.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=int, default=5)
    parser.add_argument("--fine", type=int, default=15)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--mass", choices=("consistent", "lumped"), default="consistent")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.coarse, args.fine, args.modes, args.repeats) < 1
        or args.fine % args.coarse
        or args.fine <= args.coarse
        or args.width < args.modes
        or 3 * args.width > 9 * args.coarse * (args.coarse + 1) ** 2
    ):
        parser.error(
            "require fine>coarse, integer refinement, positive sizes and k<=width<=coarse_n/3"
        )
    setup, problems, drivers = {}, {}, {}
    for level, side in (("fine", args.fine), ("coarse", args.coarse)):
        start = time.perf_counter()
        p = beam_problem(side, args.modes, args.mass == "consistent")
        setup[level + "_assembly_seconds"] = time.perf_counter() - start
        problems[level] = p
    start = time.perf_counter()
    for p in problems.values():
        attach_reference(p)
    reference_seconds = time.perf_counter() - start
    for level, p in problems.items():
        start = time.perf_counter()
        drivers[level] = ShapeReplay([(0.0, p, 0.0)], args.width, refit=False)
        wp.synchronize()
        setup[level + "_setup_capture_seconds"] = time.perf_counter() - start
    fine, coarse = drivers["fine"], drivers["coarse"]
    start = time.perf_counter()
    transfer = BeamTransfer(args.coarse, args.fine, coarse.solver.eigenvectors, fine.starts)
    transfer_graph = fine.solver.capture(200, initializer=transfer.apply)
    # A retained *computed* fine basis provides the same-resolution upper bound.
    retained = wp.empty_like(fine.starts)
    wp.synchronize()
    setup["transfer_setup_capture_seconds"] = time.perf_counter() - start
    rows, warmup = [], []
    for trial in range(args.repeats + 1):
        sequence = []
        for strategy in ("random", "coarse", "retained"):
            coarse_seconds = 0.0
            coarse_quality = None
            if strategy == "coarse":
                coarse_seconds = elapsed(coarse.solve)
                coarse_quality = coarse.quality(problems["coarse"])
                transfer_seconds = 0.0  # included in the combined transfer/refine graph
            else:
                source = fine.random if strategy == "random" else retained
                transfer_seconds = elapsed(lambda: wp.copy(fine.starts, source))
            graph = transfer_graph if strategy == "coarse" else fine.graph
            solve_seconds = elapsed(lambda: wp.capture_launch(graph))
            quality = fine.quality(problems["fine"])
            if strategy == "random":
                wp.copy(retained, fine.solver.eigenvectors)
            row = dict(
                strategy=strategy,
                coarse_seconds=coarse_seconds,
                transfer_seconds=transfer_seconds,
                solve_seconds=solve_seconds,
                total_seconds=coarse_seconds + transfer_seconds + solve_seconds,
                quality=quality,
                coarse_quality=coarse_quality,
                cg_stats=fine.cg_statistics(),
            )
            sequence.append(row)
            print(trial, strategy, row, flush=True)
        if trial:
            rows.append(sequence)
        else:
            warmup = sequence
    args.out.parent.mkdir(parents=True, exist_ok=True)
    graphs = {
        level: audit(driver.graph, args.out.with_suffix(f".{level}.dot"))
        for level, driver in drivers.items()
    }
    graphs["transfer_refine"] = audit(transfer_graph, args.out.with_suffix(".transfer.dot"))
    sources = [
        Path(__file__),
        Path(__file__).with_name("fem_shapes.py"),
        Path(__file__).with_name("fem_cases.py"),
        Path("warpack/block.py"),
    ]
    report = dict(
        arguments=vars(args) | {"out": str(args.out)},
        setup=setup,
        reference_seconds=reference_seconds,
        warmup=warmup,
        rows=rows,
        graphs=graphs,
        dimensions={level: p.n for level, p in problems.items()},
        matrix_sha256={level: matrix_hash(p) for level, p in problems.items()},
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        warp=wp.__version__,
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
        accepted=all(
            r["quality"]["accepted"]
            and (r["coarse_quality"] is None or r["coarse_quality"]["accepted"])
            for seq in [warmup] + rows
            for r in seq
        ),
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
