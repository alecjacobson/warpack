"""Run unmodified upstream pytest assertions with optional Warpack substitution."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

records = []


def pytest_configure(config):
    from cupy import testing

    # Expand dtype loops into individually reported tests, so unsupported types
    # cannot skip a whole loop and accidentally hide its supported FP64 cases.
    def for_dtypes(dtypes, name="dtype"):
        return pytest.mark.parametrize(name, [np.dtype(d).type for d in dtypes])

    testing.for_dtypes = for_dtypes
    if os.environ.get("WARPACK_UPSTREAM_BACKEND", "warp") == "warp":
        import cupyx.scipy.sparse.linalg as linalg

        from .adapters import eigsh, lobpcg, svds

        linalg.eigsh = eigsh
        linalg.svds = svds
        linalg.lobpcg = lobpcg
    elif os.environ.get("WARPACK_UPSTREAM_BACKEND") == "cupy-main-overlay":
        import importlib.util
        import sys

        import cupyx.scipy.sparse.linalg as linalg

        name = "cupyx.scipy.sparse.linalg._eigen"
        spec = importlib.util.spec_from_file_location(name, "build/cupy_upstream_eigen.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
        linalg._eigen = module
        linalg.eigsh = module.eigsh
        linalg.svds = module.svds


def pytest_collection_modifyitems(config, items):
    selected = []
    excluded = []
    for item in items:
        name = item.cls.__name__ if item.cls else ""
        if any(x in name for x in ("Eigsh", "Svds", "LOBPCG")) or name == "TestRng":
            selected.append(item)
            if os.environ.get("WARPACK_UPSTREAM_BACKEND", "warp") == "warp":
                reason = None
                spec = getattr(item, "callspec", None)
                params = spec.params if spec else {}
                dtype = params.get("dtype")
                case_params = params.get("_cupy_testing_param", {})
                if dtype is not None and np.dtype(dtype).char in "fF":
                    reason = "Warpack supports FP64/complex128 only"
                method = item.originalname or item.name
                is_complex = dtype is not None and np.dtype(dtype).char == "D"
                if is_complex and (name.startswith("TestSvds") or method.startswith("test_svds")):
                    reason = "PartialSVD supports real FP64 only"
                if (
                    is_complex
                    and name == "TestEigshSigma"
                    and case_params.get("which", "LM") != "LM"
                ):
                    reason = "Complex shift-invert supports nearest eigenvalues only"
                if name == "TestLOBPCG":
                    if method in (
                        "test_small_generate_input_for_elastic_rod",
                        "test_small_generate_input_for_mikota_pair",
                    ):
                        reason = "BlockLOBPCG has no dense full-space fallback (k=n)"
                    elif method in ("test_diagonal", "test_random_initial_float32"):
                        reason = "BlockLOBPCG requires FP64 starts/matrices; fixture uses FP32"
                if "LOBPCG" in name:
                    for key in ("A_dtype", "X_dtype", "Y_dtype", "preconditioner_dtype"):
                        value = case_params.get(key)
                        if value is not None and np.dtype(value) != np.dtype("float64"):
                            reason = "BlockLOBPCG supports FP64 matrices/starts/constraints only"
                            break
                if name == "TestRng":
                    reason = "CuPy RNG API/internal tests; no equivalent Warpack interface"
                if reason:
                    item.add_marker(pytest.mark.skip(reason=reason))
        else:
            excluded.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=excluded)


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        records.append(
            dict(
                nodeid=report.nodeid,
                outcome=report.outcome,
                expected_failure=getattr(report, "wasxfail", None),
                seconds=report.duration,
                detail=str(report.longrepr) if report.longrepr else None,
            )
        )


def pytest_sessionfinish(session, exitstatus):
    path = Path(os.environ.get("WARPACK_UPSTREAM_REPORT", "results/upstream_cupy.json"))
    path.write_text(
        json.dumps(
            dict(
                backend=os.environ.get("WARPACK_UPSTREAM_BACKEND", "warp"),
                exitstatus=exitstatus,
                tests=records,
            ),
            indent=2,
        )
        + "\n"
    )
