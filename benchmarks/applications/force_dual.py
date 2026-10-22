"""Force-aware ROM via low-rank covariance, outside the installed solver core.

For Sigma_F=F F.T, solve G=H^-1 F and diagonalize G.T M G. This avoids both
inverting a singular force covariance and forming an n-by-n dense covariance.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import warp as wp
from warp.optim import linear

from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.source_provenance import source_manifest
from benchmarks.upstream.adapters import matrix
from warpack import (
    BatchedCGPreconditioner,
    BatchedSparseOperator,
    CuDSSInverse,
    KrylovSchur,
    SparseOperator,
)
from warpack.linalg_tiles import gram_tiled, rotate_tiled

from .fem_hierarchy import beam_problem, matrix_hash
from .fem_shapes import elapsed
from .weighted_snapshot_reference import weighted_reference


def localized_forces(side, columns):
    shape = (3 * side + 1, side + 1, side + 1)
    xyz = np.array(list(np.ndindex(shape)), float)[(side + 1) ** 2 :] / side
    candidates = np.flatnonzero(np.repeat((xyz[:, 0] > 2) | (xyz[:, 2] == 0), 3))
    if columns > len(candidates):
        raise ValueError("too many localized forces for this mesh")
    chosen = np.random.default_rng(135).choice(candidates, columns, replace=False)
    f = np.zeros((len(xyz) * 3, columns))
    f[chosen, np.arange(columns)] = np.linspace(0.5, 1.5, columns)
    return f


def reference(h, mass, forces, k, method="gram"):
    if method not in ("gram", "svd"):
        raise ValueError("reference method must be gram or svd")
    snapshots = sla.splu(h.tocsc()).solve(forces)
    if method == "svd":
        return weighted_reference(snapshots, mass, k) | dict(snapshots=snapshots)
    gram = snapshots.T @ (mass @ snapshots)
    values, vectors = la.eigh(gram)
    order = np.argsort(-values)
    values, vectors = values[order], vectors[:, order]
    positive = values > max(0.0, values[0]) * 1e-12
    basis = None
    if positive[:k].all():
        basis = (snapshots @ vectors[:, :k]) / np.sqrt(values[:k])
    return dict(
        snapshots=snapshots,
        values=values[:k],
        basis=basis,
        energy=float(np.trace(gram)),
        rank=int(positive.sum()),
    )


@wp.kernel
def normalize_lift(
    values: wp.array[wp.float64], vectors: wp.array2d[wp.float64], valid: wp.array[wp.int32]
):
    row, i = wp.tid()
    lam = values[row]
    ok = wp.isfinite(lam) and lam > wp.float64(1e-12) * wp.max(values[0], wp.float64(0))
    if ok:
        vectors[row, i] /= wp.sqrt(lam)
        if i == 0:
            wp.atomic_add(valid, 0, 1)
    else:
        vectors[row, i] = wp.float64(0)


class SnapshotROM:
    """Two captured stages: sparse response/Gram construction, then small eigensolve.

    H and M are fixed; force-buffer values may change on replay. Numerical
    rank uses an explicit relative cutoff 1e-12 on covariance eigenvalues.
    The Gram formulation squares snapshot conditioning; this is a benchmark
    application with independent validation, not a replacement for PartialSVD.
    """

    def __init__(self, h, mass, forces, k, backend="cudss"):
        self.h, self.mass, self.forces = h, BatchedSparseOperator(mass), forces
        self.r, self.n = forces.shape
        if not 1 <= k < self.r or self.n != h.shape[0]:
            raise ValueError("require 1<=k<number of force columns and matching matrix dimensions")
        if (
            forces.dtype != wp.float64
            or forces.device != h.device
            or mass.device != h.device
            or mass.shape != h.shape
        ):
            raise ValueError("FP64 forces and matching matrices must share one device")
        if backend not in ("cudss", "cg"):
            raise ValueError("backend must be cudss or cg")
        self.backend, self.k = backend, k
        self.factor = None
        if backend == "cudss":
            self.inverse = CuDSSInverse(h, self.r, mtype="spd")
        else:
            self.factor = linear.FSAI(h, max_row_size=32, max_step_size=3, factor_dtype=wp.float32)
            self.inverse = BatchedCGPreconditioner(
                h,
                self.r,
                tol=1e-13,
                maxiter=1000,
                preconditioner=self.factor,
                working_dtype=wp.float64,
                fsai_layout="columns",
                matrix_layout="columns",
            )
        self.g, self.mg = wp.empty_like(forces), wp.empty_like(forces)
        self.gram = wp.empty((self.r, self.r), dtype=wp.float64, device=h.device)
        with wp.ScopedDevice(h.device):
            self.small = matrix(np.ones((self.r, self.r)))
        self.vectors = wp.empty((k, self.n), dtype=wp.float64, device=h.device)
        self.positive = wp.zeros(1, dtype=wp.int32, device=h.device)
        self.solver = KrylovSchur(SparseOperator(self.small), k, ncv=self.r, which="LA", tol=1e-12)
        self.response_graph = self.mode_graph = None

    def response(self):
        if self.backend == "cg":
            self.inverse.reset_stats()
        self.inverse.apply(self.forces, self.g)
        self.mass.apply(self.g, self.mg)
        self.gram.zero_()
        wp.launch_tiled(
            gram_tiled,
            dim=((self.r + 7) // 8, (self.r + 7) // 8, (self.n + 255) // 256),
            inputs=[self.g, self.mg, self.gram],
            block_dim=128,
            device=self.h.device,
        )
        wp.copy(self.small.values[: self.r * self.r], self.gram.flatten())

    def lift(self):
        wp.launch_tiled(
            rotate_tiled,
            dim=((self.k + 3) // 4, (self.n + 31) // 32),
            inputs=[self.g, self.solver.eigenvectors, self.vectors],
            block_dim=128,
            device=self.h.device,
        )
        self.positive.zero_()
        wp.launch(
            normalize_lift,
            self.vectors.shape,
            [self.solver.eigenvalues, self.vectors, self.positive],
            device=self.h.device,
        )

    def capture(self):
        self.response()
        wp.synchronize_device(self.h.device)
        with wp.ScopedCapture(device=self.h.device) as cap:
            self.response()
        self.response_graph = cap.graph
        self.mode_graph = self.solver.capture(10, finalizer=self.lift)

    def quality(self, h, mass, forces, ref):
        values, u, g = self.solver.eigenvalues.numpy(), self.vectors.numpy().T, self.g.numpy().T
        source_residual = float(
            np.max(
                np.linalg.norm(h @ g - forces, axis=0)
                / np.maximum(1.0, np.linalg.norm(forces, axis=0))
            )
        )
        mu = mass @ u
        au = mass @ (ref["snapshots"] @ (ref["snapshots"].T @ mu))
        residual = float(
            np.max(
                np.linalg.norm(au - mu * values, axis=0)
                / np.maximum(
                    1, np.linalg.norm(au, axis=0) + abs(values) * np.linalg.norm(mu, axis=0)
                )
            )
        )
        orth = float(abs(u.T @ mu - np.eye(self.k)).max())
        spectrum = float(np.max(abs(values - ref["values"]) / np.maximum(1, abs(ref["values"]))))
        subspace = None
        if ref["basis"] is not None:
            remainder = u - ref["basis"] @ (ref["basis"].T @ mu)
            subspace = float(np.sqrt(max(0, np.sum(remainder * (mass @ remainder)))))
        actual_energy = float(np.sum((u.T @ (mass @ ref["snapshots"])) ** 2))
        energy_error = abs(actual_energy - float(sum(ref["values"]))) / max(1, ref["energy"])
        status, count = int(self.solver.status.numpy()[0]), int(self.solver.converged.numpy()[0])
        positive = int(self.positive.numpy()[0])
        cg_stats = self.inverse.stats.numpy().tolist() if self.backend == "cg" else None
        accepted = (
            status == 0
            and count == self.k
            and positive == self.k
            and subspace is not None
            and (cg_stats is None or cg_stats[2] == 0)
            and all(
                np.isfinite(x) and 0 <= x < 1e-8
                for x in (source_residual, residual, orth, spectrum, subspace, energy_error)
            )
        )
        return dict(
            accepted=bool(accepted),
            status=status,
            converged=count,
            positive=positive,
            cg_stats=cg_stats,
            source_residual=source_residual,
            residual=residual,
            orthogonality=orth,
            spectrum_error=spectrum,
            subspace_error=subspace,
            energy_error=energy_error,
            retained_energy=actual_energy / ref["energy"] if ref["energy"] else None,
        )

    def close(self):
        self.response_graph = self.mode_graph = None
        if self.backend == "cudss":
            self.inverse.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--mass", choices=("consistent", "lumped"), default="consistent")
    parser.add_argument("--backend", choices=("cudss", "cg"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-method", choices=("gram", "svd"), default="gram")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.side, args.modes, args.repeats) < 1 or args.columns <= args.modes:
        parser.error("positive sizes and columns>modes required")
    begin = time.perf_counter()
    p = beam_problem(args.side, args.modes, args.mass == "consistent")
    forces = localized_forces(args.side, args.columns)
    assembly_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    ref = reference(p.a, p.b, forces, args.modes, args.reference_method)
    reference_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    driver = SnapshotROM(
        matrix(p.a),
        matrix(p.b),
        wp.array(forces.T.copy(), dtype=wp.float64),
        args.modes,
        args.backend,
    )
    wp.synchronize()
    setup_seconds = time.perf_counter() - begin
    capture_seconds = elapsed(driver.capture)
    rows = []
    for trial in range(args.repeats + 1):
        response_seconds = elapsed(lambda: wp.capture_launch(driver.response_graph))
        modes_seconds = elapsed(lambda: wp.capture_launch(driver.mode_graph))
        q = driver.quality(p.a, p.b, forces, ref)
        row = dict(
            response_seconds=response_seconds,
            modes_seconds=modes_seconds,
            total_seconds=response_seconds + modes_seconds,
            quality=q,
        )
        print(trial, row, flush=True)
        if trial:
            rows.append(row)
        else:
            warmup = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    graphs = {}
    for name, graph in (("response", driver.response_graph), ("modes", driver.mode_graph)):
        path = args.out.with_suffix(f".{name}.dot")
        wp.capture_debug_dot_print(graph, str(path))
        graphs[name] = graph_summary(path)
        path.unlink()
        assert "HOST" not in graphs[name]["node_types_including_conditional_body"]
        if name == "modes" or args.backend == "cg":
            assert not any(k.startswith(("HtoD", "DtoH")) for k in graphs[name]["memcpy_kinds"])
    report = dict(
        dependency=warp_dependencies(),
        arguments=vars(args) | {"out": str(args.out)},
        n=p.n,
        assembly_seconds=assembly_seconds,
        reference_seconds=reference_seconds,
        setup_seconds=setup_seconds,
        capture_seconds=capture_seconds,
        warmup=warmup,
        rows=rows,
        graphs=graphs,
        matrix_sha256=matrix_hash(p),
        forces_sha256=hashlib.sha256(forces.tobytes()).hexdigest(),
        source_sha256=source_manifest(__file__),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        accepted=all(r["quality"]["accepted"] for r in [warmup] + rows),
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    driver.close()
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
