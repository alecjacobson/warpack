"""Explicit verified CPU reference reuse for isolated FEM comparison workers.

Only the independent reference function is wrapped. Assembly, boundary-cluster
selection, known nullspace construction and every backend setup remain fresh.
The context is for a single sequential benchmark worker, not concurrent calls
that mutate the FEM fixture module.
"""

import hashlib
import inspect
import time
from pathlib import Path

import numpy as np
import scipy

from benchmarks import reference_cache
from benchmarks.anvil_inputs import matrix_identity
from benchmarks.applications import fem_cases, problems


def validate_reference(a, b, count, arrays):
    if set(arrays) != {"values", "vectors", "method_utf8"}:
        raise ValueError("incomplete independent reference")
    w, v, label = arrays["values"], arrays["vectors"], arrays["method_utf8"]
    expected = a.shape[0] if a.shape[0] <= 768 else min(count, a.shape[0] - 2)
    if (
        w.shape != (expected,)
        or v.shape != (a.shape[0], expected)
        or w.dtype != np.float64
        or v.dtype != np.float64
    ):
        raise ValueError("reference shape, dtype or guard capacity mismatch")
    if label.ndim != 1 or label.dtype != np.uint8 or len(label) > 4096:
        raise ValueError("invalid reference method label")
    label.tobytes().decode("utf-8")
    if np.any(np.diff(w) < 0):
        raise ValueError("reference spectrum is not sorted")
    av, bv = a @ v, b @ v
    anorm = float(np.max(np.asarray(abs(a).sum(axis=1))))
    bnorm = float(np.max(np.asarray(abs(b).sum(axis=1))))
    denominator = (anorm + abs(w) * bnorm) * np.linalg.norm(v, axis=0)
    residual = np.linalg.norm(av - bv * w, axis=0) / np.maximum(denominator, 1e-300)
    orthogonality = float(np.linalg.norm(v.T @ bv - np.eye(expected)))
    if (
        not np.all(np.isfinite(residual))
        or np.max(residual) > 1e-10
        or not np.isfinite(orthogonality)
        or orthogonality > 1e-8
    ):
        raise ValueError("independent reference fails original equations or mass orthogonality")


class FEMReferenceCache:
    def __init__(self, root):
        self.root = Path(root)
        self.records = []

    def __enter__(self):
        if isinstance(getattr(fem_cases.reference, "__self__", None), FEMReferenceCache):
            raise RuntimeError("FEM reference cache contexts cannot be nested")
        self.original = fem_cases.reference
        self.original_source = inspect.getsource(self.original)
        fem_cases.reference = self.reference
        return self

    def __exit__(self, *_):
        fem_cases.reference = self.original

    def reference(self, a, b, count):
        start = time.perf_counter()
        request = dict(
            schema=1,
            operators={"a": matrix_identity(a), "b": matrix_identity(b)},
            count=int(count),
            # Full original function text preserves every fixed ARPACK/LAPACK
            # option, seed and basis-size expression without duplicating them.
            reference_function_source=self.original_source,
            sources={
                name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for name, module in [
                    ("fem_cases", fem_cases),
                    ("problems", problems),
                    ("reference_cache", reference_cache),
                ]
            }
            | {"cached_fem_reference": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            versions={"numpy": np.__version__, "scipy": scipy.__version__},
            guard_policy="Fresh FEM fixture applies current full boundary-cluster and positive-guard policy after loading full independent reference arrays.",
        )

        def compute():
            w, v, label = self.original(a, b, count)
            return dict(
                values=w,
                vectors=v,
                method_utf8=np.frombuffer(label.encode("utf-8"), dtype=np.uint8),
            )

        arrays, evidence = reference_cache.cached_reference(
            self.root, request, compute, lambda data: validate_reference(a, b, count, data)
        )
        evidence["preparation_seconds"] = time.perf_counter() - start
        evidence["scope"] = (
            "Independent CPU reference only; original equations and mass orthogonality checked on every use. Assembly, cluster/nullity policy and backend factors remain fresh. Entire cache preparation is outside backend setup/solve timing."
        )
        self.records.append(evidence)
        return arrays["values"], arrays["vectors"], arrays["method_utf8"].tobytes().decode("utf-8")
