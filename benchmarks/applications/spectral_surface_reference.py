"""Immutable, fingerprinted CPU references; never starting vectors for GPU solves."""

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as sla

from .spectral_surface import surface_pencil


def array_hash(array):
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def identity(pencil, count):
    n = pencil.a.shape[0]
    if not 0 < count < n - 1:
        raise ValueError("reference requires 0<count<n-1 and a spectral guard")
    dense = n <= 1024
    requested = n if dense else min(n - 1, count + 4)
    matrices = {
        name: {key: array_hash(getattr(matrix, key)) for key in ("indptr", "indices", "data")}
        for name, matrix in (("a", pencil.a), ("b", pencil.b))
    }
    return dict(
        schema=1,
        n=n,
        matrices=matrices,
        method="dense-generalized-eigh" if dense else "arpack-largest-algebraic",
        pairs=requested,
        ncv=None if dense else min(n, max(64, 4 * requested)),
        tol=None if dense else 1e-13,
        seed=None if dense else 136,
        numpy=np.__version__,
        scipy=scipy.__version__,
        sources={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("spectral_surface.py"),
            )
        },
    )


def compute(pencil, specification):
    if specification["method"] == "dense-generalized-eigh":
        values, vectors = la.eigh(pencil.a.toarray(), pencil.b.toarray())
    else:
        values, vectors = sla.eigsh(
            pencil.a,
            M=pencil.b,
            k=specification["pairs"],
            which="LA",
            ncv=specification["ncv"],
            tol=specification["tol"],
            v0=np.random.default_rng(specification["seed"]).normal(size=specification["n"]),
        )
    order = np.argsort(-values)
    return values[order].copy(), vectors[:, order].copy()


def validate(pencil, specification, values, vectors):
    if (
        values.dtype != np.float64
        or vectors.dtype != np.float64
        or values.shape != (specification["pairs"],)
        or vectors.shape != (specification["n"], specification["pairs"])
        or not np.isfinite(values).all()
        or not np.isfinite(vectors).all()
        or np.any(np.diff(values) > 0)
    ):
        raise ValueError("invalid reference arrays")
    av, bv = pencil.a @ vectors, pencil.b @ vectors
    residual = np.linalg.norm(av - bv * values, axis=0) / np.maximum(
        1, np.linalg.norm(av, axis=0) + np.abs(values) * np.linalg.norm(bv, axis=0)
    )
    if residual.max() >= 1e-8 or np.max(np.abs(vectors.T @ bv - np.eye(len(values)))) >= 1e-8:
        raise ValueError("reference fails original-pencil accuracy")


def reference(pencil, count, cache=None):
    specification = identity(pencil, count)
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    target = Path(cache) / (key + ".npz") if cache is not None else None
    if target is not None and target.exists():
        with np.load(target, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            values, vectors = data["values"], data["vectors"]
        if metadata["identity"] != specification or any(
            metadata[name] != array_hash(array)
            for name, array in (("values_sha256", values), ("vectors_sha256", vectors))
        ):
            raise ValueError("reference cache identity/checksum mismatch")
        validate(pencil, specification, values, vectors)
        return values, vectors, dict(identity=specification, cache_hit=True, key=key)
    values, vectors = compute(pencil, specification)
    validate(pencil, specification, values, vectors)
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / ("." + uuid.uuid4().hex + ".tmp")
        metadata = dict(
            identity=specification,
            values_sha256=array_hash(values),
            vectors_sha256=array_hash(vectors),
        )
        try:
            with temporary.open("xb") as output:
                np.savez(
                    output,
                    values=values,
                    vectors=vectors,
                    metadata=json.dumps(metadata, sort_keys=True),
                )
            try:
                os.link(
                    temporary, target
                )  # Publish a complete file without overwriting an existing fixture.
            except FileExistsError:
                return reference(pencil, count, cache)
        finally:
            temporary.unlink(missing_ok=True)
    return values, vectors, dict(identity=specification, cache_hit=False, key=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--modes", type=int, default=1, help="modes per independent copy")
    parser.add_argument("--alpha", type=float, default=100)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--anisotropy", type=float, default=10)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    pencil = surface_pencil(
        args.side, alpha=args.alpha, gamma=args.gamma, anisotropy=args.anisotropy
    )
    _, _, metadata = reference(pencil, args.modes, args.cache)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
