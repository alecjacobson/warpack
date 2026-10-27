"""Procedural Newton FFRF eigenproblems; no upstream geometry or matrix data.

The spring model follows the specification of Newton's sparse FEM test, but
is assembled here through a bar-extension matrix. The unchanged, hash-checked
upstream modal_sparse.py supplies the CPU reference solver.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
import subprocess
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.applications.fem_cases import rigid_modes
from benchmarks.graph_audit import graph_summary
from benchmarks.upstream.adapters import matrix
from examples.problems.elasticity import rigid_body_basis
from warpack import (
    BatchedCGPreconditioner,
    BatchedSparseOperator,
    BlockLOBPCG,
    CuDSSInverse,
    KrylovSchur,
    OrthogonalComplementOperator,
)

NEWTON_COMMIT = "360a9b29a469af04eae6909f6d1709460a8dc35e"
SOURCE_SHA256 = "85722355f33692b800e659070e9839c813246e3196c127b6b8af0f1d458d12d3"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/skulkar22/newton/{NEWTON_COMMIT}"
    "/newton/_src/sim/modal_sparse.py"
)
SPACING, SPRING, NODE_MASS = 0.01, 2e9, 0.008
EIGENVALUE_SCALE = SPRING / (SPACING * NODE_MASS)


def versions():
    result = {"warp": wp.__version__}
    for name in ("numpy", "scipy", "nvidia-cudss-cu12"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def lattice(shape=(40, 20, 12), boundary="free"):
    """Braced axial springs, K = D.T diag(EA/length) D, M=m I.

    Fixing both end faces gives a fixed-interface interior eigenproblem;
    this does not implement Newton's MPC/RBE constraint reduction.
    """
    if len(shape) != 3 or min(shape) < 2 or boundary not in ("free", "fixed"):
        raise ValueError("need three node counts >=2 and boundary free or fixed")
    if boundary == "fixed" and shape[0] < 3:
        raise ValueError("fixed end faces require at least one interior node layer")
    grid = np.indices(shape).reshape(3, -1).T
    points = SPACING * (grid - grid.mean(axis=0))
    edges = []
    for offset in itertools.product((-1, 0, 1), repeat=3):
        if offset <= (0, 0, 0):
            continue
        other = grid + np.array(offset)
        keep = np.all((other >= 0) & (other < np.array(shape)), axis=1)
        first = np.flatnonzero(keep)
        second = np.ravel_multi_index(other[keep].T, shape)
        edges.append(np.column_stack((first, second)))
    edges = np.concatenate(edges)
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    lengths = np.linalg.norm(delta, axis=1)
    directions = delta / lengths[:, None]
    # Each row computes one extension, independently of Newton's 3x3 assembly.
    dofs = (3 * edges[:, :, None] + np.arange(3)).reshape(-1, 6)
    data = np.concatenate((-directions, directions), axis=1)
    extension = sp.csr_matrix(
        (data.ravel(), (np.repeat(np.arange(len(edges)), 6), dofs.ravel())),
        shape=(len(edges), 3 * len(points)),
    )
    stiffness = (extension.T @ sp.diags(SPRING / lengths) @ extension).tocsc()
    stiffness.eliminate_zeros()
    retained = np.arange(stiffness.shape[0])
    if boundary == "fixed":
        retained = np.flatnonzero(np.repeat((grid[:, 0] > 0) & (grid[:, 0] < shape[0] - 1), 3))
        stiffness = stiffness[retained, :][:, retained].tocsc()
    mass = sp.eye(len(retained), format="csc") * NODE_MASS
    return stiffness, mass, points, retained


def upstream(path, download=False):
    """Fetch one Apache-2.0 source file, never datasets or mesh assets."""
    path = Path(path)
    if download and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(SOURCE_URL, timeout=60).read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Newton modal_sparse.py does not match the pinned commit")
    spec = importlib.util.spec_from_file_location("newton_ffrf_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_pairs(stiffness, mass, values, vectors, reference, rigid=None):
    """Original equation, complete lowest spectrum and invariant space."""
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    ref_values, ref_vectors = reference
    av, bv = stiffness @ vectors, mass @ vectors
    residual = np.linalg.norm(av - bv * values, axis=0) / (
        np.linalg.norm(av, axis=0) + abs(values) * np.linalg.norm(bv, axis=0)
    )
    orth = float(np.max(abs(vectors.T @ bv - np.eye(len(values)))))
    spectrum = float(np.max(abs(values / ref_values - 1)))
    remainder = vectors - ref_vectors @ (ref_vectors.T @ bv)
    space = float(np.sqrt(max(0, np.trace(remainder.T @ (mass @ remainder)))))
    overlap = 0.0 if rigid is None else float(np.max(abs(rigid.T @ bv)))
    return dict(
        accepted=bool(
            residual.max() < 1e-8
            and orth < 1e-8
            and spectrum < 1e-8
            and space < 1e-6
            and overlap < 1e-8
            and values.min() > 0
        ),
        relative_residual=float(residual.max()),
        relative_eigenvalue_error=spectrum,
        mass_orthogonality=orth,
        requested_subspace_error=space,
        rigid_overlap=overlap,
    )


def cpu_trial(module, stiffness, mass, rigid, count, sigma):
    """Actual upstream helper, including its second/Sturm factorization."""
    start = time.perf_counter()
    factor = module.ShiftedFactorization(stiffness, mass, sigma)
    setup = time.perf_counter() - start
    threshold = max(1e-12, 8 * module.max_row_nonzeros(stiffness) * np.finfo(float).eps)
    threshold *= module.generalized_eigenvalue_bound(stiffness, mass)
    sturm_seconds = []
    original_count = module.count_eigenvalues_below

    def counted(*args):
        start = time.perf_counter()
        result = original_count(*args)
        sturm_seconds.append(time.perf_counter() - start)
        return result

    module.count_eigenvalues_below = counted
    try:
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values, vectors = module.lowest_generalized_eigenpairs(
                stiffness,
                mass,
                count,
                sigma=sigma,
                zero_threshold=threshold,
                deflation=rigid,
                factorization=factor,
                tol=0.0,
                margin=2,
            )
        total_solve = time.perf_counter() - start
    finally:
        module.count_eigenvalues_below = original_count
    row = dict(
        setup_seconds=setup,
        solve_seconds=total_solve - sum(sturm_seconds),
        completeness_check_seconds=sum(sturm_seconds),
        total_seconds=setup + total_solve,
        warnings=[str(w.message) for w in caught],
        sturm_factorizations=len(sturm_seconds),
        guard_eigenvalue=float(values[count]),
        eigenvalues=values[:count].tolist(),
    )
    return row, (values[:count], vectors[:, :count])


@wp.kernel
def recover(
    source_values: wp.array[wp.float64],
    source_vectors: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    vectors: wp.array2d[wp.float64],
    sigma: wp.float64,
    inverse: bool,
):
    j, i = wp.tid()
    vectors[j, i] = source_vectors[j, i] / wp.sqrt(wp.float64(NODE_MASS))
    if i == 0:
        value = source_values[j]
        if inverse:
            value = sigma + wp.float64(1.0) / value
        values[j] = value * wp.float64(EIGENVALUE_SCALE)


def gpu_solver(
    stiffness, points, boundary, backend, count, sigma, *, steps=200, inner=25, cycles=2
):
    """All eigen iterations and output recovery use Warp (+ optional cuDSS)."""
    a = ws.bsr_scale(matrix(stiffness), SPACING / SPRING)
    constraints = None
    if boundary == "free":
        constraints = rigid_body_basis(
            wp.array(points, dtype=wp.vec3d), wp.ones(len(points), dtype=wp.float64)
        )
    inverse = None
    start = time.perf_counter()
    normalized_shift = sigma / EIGENVALUE_SCALE
    if backend == "cudss":
        shifted = ws.bsr_copy(a)
        ws.bsr_axpy(ws.bsr_diag(wp.float64(-normalized_shift), a.shape[0]), shifted)
        inverse = CuDSSInverse(shifted, 1)
        wp.synchronize()
        factor_seconds = time.perf_counter() - start
        op = inverse if constraints is None else OrthogonalComplementOperator(inverse, constraints)
        solver = KrylovSchur(op, count, ncv=48, which="LA", tol=1e-12)
    else:
        # This is a preconditioning shift, not a changed physical eigenproblem.
        shifted = ws.bsr_copy(a)
        ws.bsr_axpy(ws.bsr_diag(wp.float64(0.01), a.shape[0]), shifted)
        fsai = linear.FSAI(shifted, max_row_size=32, max_step_size=3, factor_dtype=wp.float32)
        wp.synchronize()
        factor_seconds = time.perf_counter() - start
        width = max(16, count + 4)
        pre = BatchedCGPreconditioner(
            shifted,
            width,
            tol=1e-3,
            atol=0.0,
            maxiter=inner,
            preconditioner=fsai,
            working_dtype=wp.float32,
            matrix_layout="columns",
        )
        solver = BlockLOBPCG(
            BatchedSparseOperator(a),
            count,
            width,
            preconditioner=pre,
            constraints=constraints,
            tol=1e-11,
            reuse_operator_images=True,
            search_bucket_size=4,
            preconditioner_bucket_size=4,
        )
    values = wp.empty(count, dtype=wp.float64)
    vectors = wp.empty((count, a.shape[0]), dtype=wp.float64)

    def finish():
        wp.launch(
            recover,
            vectors.shape,
            [
                solver.eigenvalues,
                solver.eigenvectors,
                values,
                vectors,
                normalized_shift,
                backend == "cudss",
            ],
        )

    graph = solver.capture(cycles if backend == "cudss" else steps, finalizer=finish)
    return solver, graph, values, vectors, inverse, factor_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, default=(40, 20, 12))
    parser.add_argument("--boundary", choices=("free", "fixed"), default="free")
    parser.add_argument("--backend", choices=("newton", "cudss", "warp"), required=True)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--inner", type=int, default=25)
    parser.add_argument("--source", default="build/newton-ffrf-source/modal_sparse.py")
    parser.add_argument("--download-source", action="store_true")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.modes, args.repeats, args.steps, args.inner, args.cycles) < 1:
        parser.error("positive mode count, repeats and iteration budgets required")
    start = time.perf_counter()
    stiffness, mass, points, _ = lattice(tuple(args.shape), args.boundary)
    rigid = rigid_modes(points, mass) if args.boundary == "free" else None
    sigma = -1e-6 * float(abs(stiffness).sum(axis=0).max()) / NODE_MASS
    if args.boundary == "fixed":
        sigma = 0.0
    report = dict(
        shape=args.shape,
        boundary=args.boundary,
        n=stiffness.shape[0],
        nnz=stiffness.nnz,
        modes=args.modes,
        sigma=sigma,
        backend=args.backend,
        assembly_seconds=time.perf_counter() - start,
        rows=[],
        newton_commit=NEWTON_COMMIT,
        newton_source_sha256=SOURCE_SHA256,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matrix_sha256=hashlib.sha256(
            stiffness.data.tobytes() + stiffness.indices.tobytes() + stiffness.indptr.tobytes()
        ).hexdigest(),
        versions=versions(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        threads={
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    if args.backend == "newton":
        module = upstream(args.source, args.download_source)
        reference = None
        for _ in range(args.repeats):
            row, pairs = cpu_trial(module, stiffness, mass, rigid, args.modes, sigma)
            if reference is None:
                reference = pairs
                if row["guard_eigenvalue"] / reference[0][-1] - 1 < 1e-6:
                    raise ValueError("requested count splits a cluster; choose another count")
                args.reference.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    args.reference,
                    values=reference[0],
                    vectors=reference[1],
                    matrix_sha256=report["matrix_sha256"],
                )
            row["quality"] = check_pairs(stiffness, mass, *pairs, reference, rigid)
            row["quality"]["completeness_certified"] = bool(
                row["sturm_factorizations"] and not row["warnings"]
            )
            row["quality"]["accepted"] &= row["quality"]["completeness_certified"]
            report["rows"].append(row)
            print(row, flush=True)
    else:
        with np.load(args.reference) as cached:
            if str(cached["matrix_sha256"]) != report["matrix_sha256"]:
                raise ValueError("reference matrix mismatch")
            reference = cached["values"], cached["vectors"]
        wp.init()
        report["gpu"] = wp.get_device().name
        wp.synchronize()
        start = time.perf_counter()
        solver, graph, values, vectors, inverse, factor_seconds = gpu_solver(
            stiffness,
            points,
            args.boundary,
            args.backend,
            args.modes,
            sigma,
            steps=args.steps,
            inner=args.inner,
            cycles=args.cycles,
        )
        wp.synchronize()
        report["setup_seconds"] = time.perf_counter() - start
        report["factor_or_preconditioner_seconds"] = factor_seconds
        dot = args.out.with_suffix(".dot")
        dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(graph, str(dot))
        report["graph"] = graph_summary(dot)
        if report["graph"]["node_types_including_conditional_body"].get("HOST", 0):
            raise RuntimeError("Unexpected host callback in solver graph")
        dot.unlink()
        for index in range(args.repeats + 1):
            wp.synchronize()
            start = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize()
            seconds = time.perf_counter() - start
            quality = check_pairs(
                stiffness, mass, values.numpy(), vectors.numpy().T, reference, rigid
            )
            quality.update(
                status=int(solver.status.numpy()[0]), converged=int(solver.converged.numpy()[0])
            )
            quality["accepted"] &= quality["status"] == 0 and quality["converged"] == args.modes
            row = dict(
                solve_seconds=seconds, quality=quality, iterations=int(solver.iterations.numpy()[0])
            )
            if index == 0:
                report["warmup"] = row
            else:
                report["rows"].append(row)
            print(row, flush=True)
        solver.graph = None
        del graph, solver
        if inverse is not None:
            inverse.close()
    report["accepted"] = all(r["quality"]["accepted"] for r in report["rows"])
    report["accepted"] &= report.get("warmup", {"quality": {"accepted": True}})["quality"][
        "accepted"
    ]
    report["settings"] = vars(args).copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
