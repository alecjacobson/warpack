"""The public matching flag must have identical semantics across cuDSS runtimes."""

import ctypes as ct
from types import SimpleNamespace

import pytest
import warp as wp

from warpack import CuDSSInverse


@pytest.mark.parametrize(
    "mtype,matching,expected,corrections",
    [
        ("spd", False, 0, 0),
        ("symmetric", False, 0, 1),
        ("general", False, 0, 1),
        ("general", True, 5, 0),
    ],
)
@pytest.mark.parametrize("native_status", [0, 4])
def test_explicit_matching_policy_precedes_analysis(
    monkeypatch, mtype, matching, expected, corrections, native_status
):
    import warp_cudss
    from warp_cudss._bindings import CUDSS_CONFIG_MATCHING_ALG, CudssError

    import warpack.cudss_memory

    events = []

    class Native:
        def cudssConfigSet(self, config, option, pointer, size):
            assert size == ct.sizeof(ct.c_int)
            events.append(("config", option, ct.cast(pointer, ct.POINTER(ct.c_int))[0]))
            return native_status

    class FakeSolver:
        def __init__(self, **kwargs):
            self._lib = Native()
            self._config = None

        def setup(self, *args, **kwargs):
            events.append(("analysis",))

        def release(self):
            pass

    monkeypatch.setattr(warp_cudss, "CudssSolver", FakeSolver)
    monkeypatch.setattr(warpack.cudss_memory, "Workspace", lambda _: SimpleNamespace(error=None))
    monkeypatch.setattr(wp, "empty", lambda *args, **kwargs: object())
    monkeypatch.setattr(wp, "empty_like", lambda *args, **kwargs: object())
    matrix = SimpleNamespace(
        shape=(8, 8), scalar_type=wp.float64, device="cpu", nnz_sync=lambda: None
    )
    if native_status:
        with pytest.raises(CudssError, match="MATCHING_ALG"):
            CuDSSInverse(matrix, 1, mtype=mtype, matching=matching)
    else:
        inverse = CuDSSInverse(matrix, 1, mtype=mtype, matching=matching)
        assert inverse.matching == matching
        assert inverse.refinement_steps == corrections
    assert events == [("config", CUDSS_CONFIG_MATCHING_ALG, expected)] + (
        [] if native_status else [("analysis",)]
    )
