"""Explicit host-side cuDSS factor metadata queries, never part of replay."""

import ctypes
from dataclasses import dataclass

import warp as wp


@dataclass(frozen=True)
class InertiaResult:
    """Numerical inertia snapshot; ``available`` excludes suspect factorizations.

    Counts are positive then negative. Zero pivots can be perturbed by cuDSS,
    so no zero count is inferred. A nonzero pivot counter is conservatively
    inconclusive even if its returned inertia happens to be correct.
    """

    status: str
    positive: int | None = None
    negative: int | None = None
    info: int | None = None
    pivots: int | None = None
    reason: str = ""


def _integers(solver, key, count):
    # cuDSS 0.8 cudssDataGet; the pinned wrapper uses int32 CSR indices.
    prototype = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    )
    get = prototype(("cudssDataGet", solver._lib))
    data = (ctypes.c_int * count)(*([-1] * count))
    written = ctypes.c_size_t()
    code = get(solver._handle, solver._data, key, data, ctypes.sizeof(data), ctypes.byref(written))
    if code == 4:  # CUDSS_STATUS_NOT_SUPPORTED
        return None
    from warp_cudss._bindings import _check

    _check(code, "cudssDataGet")
    if written.value != ctypes.sizeof(data):
        return None
    return tuple(data)


def query_inertia(inverse):
    """Synchronize and query a live factor outside CUDA capture.

    cuDSS documents INERTIA only for symmetric/Hermitian non-SPD types, and
    only in host memory. This query is optional; ordinary solve/capture never
    calls it. General LU and SPD declarations are not inertia certificates.
    """
    if wp.get_stream(inverse.device).is_capturing:
        raise RuntimeError("inertia is a host metadata query and cannot run during capture")
    solver = inverse.solver
    if solver._closed:
        raise RuntimeError("cannot query a closed cuDSS factor")
    if inverse.mtype not in ("symmetric", "hermitian"):
        return InertiaResult(
            "unavailable", reason="requires a symmetric/Hermitian indefinite factor"
        )
    if inverse.reordering in ("colamd", "btf_colamd"):
        return InertiaResult(
            "unavailable", reason="global LU reordering is not qualified for inertia"
        )
    wp.synchronize_device(inverse.device)
    info, pivots, counts = (
        _integers(solver, key, count) for key, count in ((0, 1), (2, 1), (3, 2))
    )
    if info is None or pivots is None or counts is None:
        return InertiaResult("unavailable", reason="cuDSS did not provide the required metadata")
    positive, negative = counts
    valid = info[0] == 0 and pivots[0] == 0 and min(counts) >= 0 and sum(counts) == inverse.n
    return InertiaResult(
        "available" if valid else "inconclusive",
        positive,
        negative,
        info[0],
        pivots[0],
        "" if valid else "factor error, nonzero pivot counter, or incomplete inertia",
    )
