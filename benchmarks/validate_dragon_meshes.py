"""Full-mesh eigenvector agreement across Warp, CuPy, ARPACK, and Spectra.

CPU references share an independent SuperLU factorization; GPU methods share
cuDSS factors. CPU/GPU transfers and callbacks occur only in this validator.
"""

import argparse
import ctypes
import json
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as csl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp

from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator
from warpack.fem import assemble_rest, mass_normalized, read_mesh


def mode_quality(a, values, vectors):
    # vectors are ROWS and mass-normalized; their Euclidean inner products equal
    # the mass inner products of the physical displacement eigenvectors.
    av = (a @ vectors.T).T
    rayleigh = np.einsum("ij,ij->i", av, vectors) / np.einsum("ij,ij->i", vectors, vectors)
    order = np.argsort(rayleigh)
    vectors, rayleigh, av = vectors[order], rayleigh[order], av[order]
    residual = np.linalg.norm(av - rayleigh[:, None] * vectors, axis=1) / np.maximum(
        1, abs(rayleigh)
    )
    gram = vectors @ vectors.T
    return (
        {
            "eigenvalues": rayleigh.tolist(),
            "frequencies_hz": (np.sqrt(np.maximum(rayleigh[6:], 0)) / (2 * np.pi)).tolist(),
            "original_residuals": residual.tolist(),
            "max_original_residual": float(residual.max()),
            "max_elastic_residual": float(residual[6:].max()),
            "orthogonality_max_error": float(np.max(abs(gram - np.eye(len(values))))),
        },
        vectors,
        rayleigh,
    )


def agreement(reference, other, ref_values, other_values):
    # Report sign-invariant MAC per elastic mode and principal angles for the
    # six-dimensional rigid subspace, which has an arbitrary eigenvector basis.
    mac = np.abs(np.sum(reference[6:] * other[6:], axis=1)) ** 2
    qa = np.linalg.qr(reference[:6].T, mode="reduced")[0]
    qb = np.linalg.qr(other[:6].T, mode="reduced")[0]
    rigid = np.linalg.svd(qa.T @ qb, compute_uv=False)
    qa = np.linalg.qr(reference[6:].T, mode="reduced")[0]
    qb = np.linalg.qr(other[6:].T, mode="reduced")[0]
    elastic = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return {
        "elastic_relative_eigenvalue_error": (
            abs(other_values[6:] - ref_values[6:]) / abs(ref_values[6:])
        ).tolist(),
        "elastic_mac": mac.tolist(),
        "minimum_elastic_mac": float(mac.min()),
        "rigid_subspace_min_cosine": float(rigid.min()),
        "elastic_subspace_min_cosine": float(elastic.min()),
    }


def validate(mesh, out):
    x, t = read_mesh(mesh)
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, _, mass = assemble_rest(x, t)
    a = mass_normalized(h, mass)
    a.nnz_sync()
    n = a.shape[0]
    sa = sp.bsr_matrix(
        (a.values.numpy()[: a.nnz], a.columns.numpy()[: a.nnz], a.offsets.numpy()), shape=(n, n)
    ).tocsr()
    shifted = mass_normalized(h, mass, shift=1.0)
    inverse = CuDSSInverse(shifted, 1)
    solver = KrylovSchur(inverse, 26, ncv=64, which="LA", tol=1e-12)
    evaluate = EigenpairEvaluation(SparseOperator(a), solver.eigenvectors, tol=1e-7)
    for cycles in [1, 2, 3, 5]:
        graph = solver.capture(cycles, adaptive=False, finalizer=evaluate.run)
        wp.capture_launch(graph)
        if evaluate.residuals.numpy().max() < 1e-7:
            break
    else:
        raise RuntimeError("Warp did not meet original-problem accuracy")
    warp_vectors = solver.eigenvectors.numpy()
    report = {
        "mesh": str(mesh),
        "n": n,
        "k": 26,
        "ncv": 64,
        "material": {"young": 1e5, "poisson": 0.3, "density": 1000, "longest_extent_m": 1},
        "purpose": "accuracy validation; callback/transfer times are not native GPU performance benchmarks",
        "methods": {},
        "agreement_with_warp": {},
    }
    quality, warp_vectors, warp_values = mode_quality(sa, evaluate.values.numpy(), warp_vectors)
    quality["cycles"] = cycles
    report["methods"]["Warp + cuDSS"] = quality
    solver.initialize()
    initial = solver.views[0].numpy().ravel()
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(report, indent=2) + "\n")

    def compare(name, values, vectors):
        quality, vectors, values = mode_quality(sa, values, vectors)
        comparison = agreement(warp_vectors, vectors, warp_values, values)
        report["methods"][name] = quality
        report["agreement_with_warp"][name] = comparison
        save()
        print(
            name,
            "residual",
            quality["max_original_residual"],
            "min MAC",
            comparison["minimum_elastic_mac"],
            flush=True,
        )
        assert quality["max_original_residual"] < 1e-7
        assert quality["orthogonality_max_error"] < 1e-8
        assert comparison["minimum_elastic_mac"] > 0.99999
        assert max(comparison["elastic_relative_eigenvalue_error"]) < 1e-7
        assert comparison["rigid_subspace_min_cosine"] > 0.99999

    save()
    stream = cp.cuda.Stream.from_external(wp.get_stream())
    with stream:
        calls = [0]

        def mv(v, backend=inverse):
            calls[0] += 1
            backend.apply(wp.from_dlpack(v.reshape(1, n)), backend.x)
            return cp.asarray(backend.x).reshape(n).copy()

        op = csl.LinearOperator((n, n), matvec=mv, dtype=np.float64)
        values, vectors = csl.eigsh(
            op, k=26, ncv=64, which="LA", tol=1e-12, v0=cp.asarray(initial).copy()
        )
        compare("CuPy + cuDSS", cp.asnumpy(1 / values - 1), cp.asnumpy(vectors).T)
        report["methods"]["CuPy + cuDSS"]["inverse_applications"] = calls[0]
    del graph, solver, evaluate
    inverse.close()
    del inverse, h, mass, a, shifted
    wp.synchronize()
    print("Factoring independent CPU matrix with SuperLU:", mesh, flush=True)
    start = time.perf_counter()
    lu = sla.splu((sa + sp.eye(n, format="csr")).tocsc())
    report["cpu_factor_seconds"] = time.perf_counter() - start
    print("CPU factor seconds", report["cpu_factor_seconds"], flush=True)
    op = sla.LinearOperator((n, n), matvec=lu.solve, dtype=np.float64)
    values, vectors = sla.eigsh(op, k=26, ncv=64, which="LA", tol=1e-12, v0=initial.copy())
    compare("SciPy ARPACK + SuperLU", 1 / values - 1, vectors.T)
    del vectors
    ptr = ctypes.POINTER(ctypes.c_double)
    callback_type = ctypes.CFUNCTYPE(None, ptr, ptr)

    @callback_type
    def apply(src, dst):
        np.ctypeslib.as_array(dst, shape=(n,))[:] = lu.solve(np.ctypeslib.as_array(src, shape=(n,)))

    library = ctypes.CDLL(str(Path("build/spectra_callback.so").resolve()))
    function = library.spectra_inverse
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr, callback_type, ptr, ptr]
    function.restype = ctypes.c_int
    values = np.empty(26)
    vectors = np.empty((26, n))
    count = function(
        n,
        26,
        64,
        initial.ctypes.data_as(ptr),
        apply,
        values.ctypes.data_as(ptr),
        vectors.ctypes.data_as(ptr),
    )
    assert count == 26, count
    compare("Spectra + shared SuperLU", 1 / values - 1, vectors)
    report["passed"] = True
    save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    validate(args.mesh, args.out)


if __name__ == "__main__":
    main()
