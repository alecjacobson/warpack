"""Reference-only Anvil feasibility screen: CuPy LOBPCG + upstream Warp FSAI.

This is not a production solver or a graph-captured timing comparison. It uses
the existing independent Anvil reference; it never launches a CPU factorization.
"""

import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cs
import cupyx.scipy.sparse.linalg as cl
import numpy as np
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear

from benchmarks.anvil import validate
from benchmarks.applications.compare import other_gpu_processes
from examples.problems.elasticity import assemble_rest, mass_normalized, read_mesh, rigid_body_basis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shift", type=float, default=10000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--block-size", type=int, default=20)
    p.add_argument("--reference", default="build/anvil_reference.npz")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.block_size < 20 or args.steps < 1 or args.width < 1 or not args.shift > 0:
        p.error("Require block-size >= 20 and positive steps, FSAI width and shift")
    if not Path(args.reference).is_file():
        p.error("An existing Anvil reference is required; see benchmarks/anvil.py")
    other = other_gpu_processes()
    x, t = read_mesh("anvil for export-ftetwild.mesh")
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, _, mass = assemble_rest(x, t)
    a = mass_normalized(h, mass)
    a.nnz_sync()
    sa = sp.bsr_matrix(
        (a.values.numpy()[: a.nnz], a.columns.numpy()[: a.nnz], a.offsets.numpy()), shape=a.shape
    ).tocsr()
    rigid = rigid_body_basis(wp.array(x, dtype=wp.vec3d), mass)
    shifted = mass_normalized(h, mass, shift=args.shift)
    shifted.nnz_sync()
    wp.synchronize()
    begin = time.perf_counter()
    pre = linear.FSAI(
        shifted, max_row_size=args.width, max_step_size=3, apply_lanes=8, factor_dtype=wp.float32
    )
    wp.synchronize()
    setup = time.perf_counter() - begin
    calls = [0]
    stream = cp.cuda.Stream.from_external(wp.get_stream())
    with stream:
        A = cs.csr_matrix(sa)
        X = cp.asarray(np.random.default_rng(42).normal(size=(a.shape[0], args.block_size)))
        Y = cp.asarray(rigid).T

        def apply(v):
            single = v.ndim == 1
            rows = cp.ascontiguousarray(v.reshape(-1, 1).T if single else v.T)
            out = cp.empty_like(rows)
            w = wp.from_dlpack(rows)
            o = wp.from_dlpack(out)
            for j in range(len(rows)):
                pre.matvec(w[j], o[j], o[j], 1.0, 0.0)
                calls[0] += 1
            return out[0] if single else out.T

        P = cl.LinearOperator(A.shape, matvec=apply, matmat=apply, dtype=np.float64)
        stream.synchronize()
        begin = time.perf_counter()
        vals, vec, hist = cl.lobpcg(
            A,
            X,
            M=P,
            Y=Y,
            largest=False,
            tol=1e-5,
            maxiter=args.steps,
            retResidualNormsHistory=True,
        )
        stream.synchronize()
        seconds = time.perf_counter() - begin
        vectors = np.concatenate([rigid.numpy(), vec.get().T[:20]])
        ref = np.load(args.reference)
        quality, _ = validate(sa, vectors, (ref["vectors"], ref["values"]))
        other_after = other_gpu_processes()
        report = dict(
            args=vars(args),
            setup_seconds=setup,
            seconds=seconds,
            other_gpu_processes=other,
            other_gpu_processes_after=other_after,
            timing_eligible=False,
            timing_note="Single cold feasibility screen, not a matched warm performance measurement",
            gpu_uncontended=not (other or other_after),
            preconditioner_applications=calls[0],
            residual_history_length=len(hist),
            last_absolute_residuals=hist[-1].get().tolist(),
            quality=quality,
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            versions={
                name: importlib.metadata.version(name)
                for name in ("warp-lang", "numpy", "scipy", "cupy-cuda12x")
            },
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
