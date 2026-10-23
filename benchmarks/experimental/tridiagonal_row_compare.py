"""Tridiagonal component: isolated FP32-basis/FP64-recovery versus global projected Jacobi controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp

from benchmarks.anvil_inputs import array_identity
from benchmarks.applications.compare import timed
from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.tridiagonal_ql_rows import RowTridiagonalQL
from benchmarks.source_provenance import source_manifest
from warpack.projected import GlobalJacobi


@wp.kernel
def extract(h: wp.array2d[wp.float64], d: wp.array[wp.float64], e: wp.array[wp.float64]):
    i = wp.tid()
    d[i] = h[i, i]
    if i + 1 < h.shape[0]:
        e[i] = h[i, i + 1]


@wp.kernel
def store_values(h: wp.array2d[wp.float64], d: wp.array[wp.float64]):
    i, j = wp.tid()
    h[i, j] = wp.float64(0.0)
    if i == j:
        h[i, j] = d[i]


class RowComponentQL:
    def __init__(self, h, v, cs, ss, partners, temp, vtemp):
        self.h, self.v, self.n = h, v, h.shape[0]
        self.diagonal = wp.empty(self.n, dtype=wp.float64, device=h.device)
        self.off_diagonal = wp.empty(self.n - 1, dtype=wp.float64, device=h.device)
        self.solver = RowTridiagonalQL(self.diagonal, self.off_diagonal)
        self.sweeps = self.solver.state[:, 2]
        self.status = self.solver.state[:, 1]

    def apply(self):
        wp.launch(extract, self.n, [self.h, self.diagonal, self.off_diagonal], device=self.h.device)
        self.solver.apply()
        wp.launch(store_values, self.h.shape, [self.h, self.solver.values], device=self.h.device)
        wp.copy(self.v, self.solver.vectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="48,72,104,128,208,312,512")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sizes = list(map(int, args.sizes.split(",")))
    if args.repeats < 1 or any(n < 2 or n > 512 or n % 2 for n in sizes):
        parser.error("positive repeats and even dimensions 2..512 required")
    if args.out.exists():
        parser.error("refusing to overwrite an existing report")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in sizes:
        random = np.random.default_rng(159).normal(size=(n, n))
        original = (
            np.diag(np.diag(random))
            + np.diag(np.diag(random, 1), 1)
            + np.diag(np.diag(random, 1), -1)
        )
        reference = la.eigvalsh(original)
        source = wp.array(original, dtype=wp.float64)
        for backend, constructor in (("global", GlobalJacobi), ("ql", RowComponentQL)):
            h, v, temp, vtemp = (wp.empty_like(source) for _ in range(4))
            cs, ss = (wp.empty(n, dtype=wp.float64) for _ in range(2))
            partners = wp.empty(n, dtype=wp.int32)
            solver = constructor(h, v, cs, ss, partners, temp, vtemp)

            def run():
                wp.copy(h, source)
                solver.apply()

            run()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                run()
            samples, checks = [], []
            for trial in range(args.repeats + 1):
                seconds, _ = timed(lambda: wp.capture_launch(capture.graph))
                values, vectors = np.diag(h.numpy()), v.numpy()
                norm = max(1, np.linalg.norm(original))
                check = dict(
                    residual=float(np.linalg.norm(original @ vectors - vectors * values) / norm),
                    orthogonality=float(np.linalg.norm(vectors.T @ vectors - np.eye(n))),
                    spectrum_error=float(np.max(abs(np.sort(values) - reference)) / norm),
                    sweeps=int(solver.sweeps.numpy()[0]),
                    status=int(solver.status.numpy()[0]) if backend == "ql" else 0,
                )
                check["accepted"] = bool(
                    all(
                        np.isfinite(check[k]) and check[k] < 1e-11
                        for k in ("residual", "orthogonality", "spectrum_error")
                    )
                    and 0 <= check["sweeps"] <= 30 * n
                    and check["status"] == 0
                )
                if trial:
                    samples.append(seconds)
                    checks.append(check)
                else:
                    warmup = check
            dot = args.out.with_suffix(f".{n}-{backend}.dot")
            wp.capture_debug_dot_print(capture.graph, str(dot))
            graph = graph_summary(dot)
            dot.unlink()
            assert not graph["node_types_including_conditional_body"].get("HOST", 0)
            assert all("DtoD" in kind for kind in graph["memcpy_kinds"])
            row = dict(
                n=n,
                backend=backend,
                input=array_identity(original),
                seconds=samples,
                median_seconds=float(np.median(samples)),
                warmup=warmup,
                checks=checks,
                graph=graph,
                accepted=all(c["accepted"] for c in [warmup, *checks]),
            )
            rows.append(row)
            print(n, backend, row["median_seconds"], row["accepted"], flush=True)
    report = dict(
        arguments=vars(args) | dict(out=str(args.out)),
        rows=rows,
        accepted=all(r["accepted"] for r in rows),
        source_sha256=source_manifest(__file__),
        dependency=warp_dependencies(),
        protocol="Isolated even-dimensional indefinite symmetric projected eigensolves, including device input reset. Original equation, full orthogonality and independent SciPy spectrum checked after one warmup and every measured replay. Same matrix per dimension/backend, with exact identity. Setup/capture excluded; no complete FEM speed claim. External occupancy required. FP64 tridiagonal component only: excludes dense reduction and back-transformation. QL counts total implicit iterations, global counts Jacobi sweeps. Production unchanged.",
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit("Rejected projected eigensolve; evidence retained")


if __name__ == "__main__":
    main()
