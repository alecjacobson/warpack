"""Allocator installation must retain each native object through failure cleanup."""

import ctypes as ct
from types import SimpleNamespace

import pytest

import warpack.cudss_memory as memory


class NativeResources:
    def __init__(self, failure, partial, cleanup_error=False):
        self.failure = failure
        self.partial = partial
        self.cleanup_error = cleanup_error
        self.calls = []
        self.live = {}
        self.next_id = 1
        self.handler = None
        self.callback_checks = 0

    def __getattr__(self, name):
        def call(*args):
            self.calls.append(name)
            fail = (
                (self.failure == "destroy" and name == "cudssDataDestroy" and self.calls.count(name) == 1)
                or (self.failure == "handler" and name == "cudssSetDeviceMemHandler")
                or (self.failure == "create" and name == "cudssDataCreate" and self.calls.count(name) == 2)
            )
            if name in ("cudssCreate", "cudssConfigCreate", "cudssDataCreate"):
                if not fail or self.partial:
                    pointer = args[1] if name == "cudssDataCreate" else args[0]
                    ct.cast(pointer, ct.POINTER(ct.c_void_p))[0] = self.next_id
                    self.live[self.next_id] = name
                    self.next_id += 1
            if name == "cudssSetDeviceMemHandler":
                self.handler = memory.Handler()
                ct.memmove(ct.byref(self.handler), args[1], ct.sizeof(self.handler))
            if name.endswith("Destroy"):
                pointer = args[-1].value
                assert pointer in self.live, "duplicate or unowned native destruction"
                del self.live[pointer]
                if name == "cudssDataDestroy" and self.handler is not None:
                    # Native destruction may call the registered allocator. Its
                    # Python owner must survive until every native object is gone.
                    assert self.handler.device_free(None, None, 0, 1) == 0
                    self.callback_checks += 1
            return 4 if fail or (self.cleanup_error and name == "cudssConfigDestroy") else 0
        return call


@pytest.mark.parametrize("failure", ["destroy", "handler", "create"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("cleanup_error", [False, True])
def test_workspace_construction_failure_unwinds_all_owners(
    monkeypatch, failure, partial, cleanup_error
):
    import warp_cudss.solver as implementation

    resources = NativeResources(failure, partial, cleanup_error)
    monkeypatch.setattr(implementation, "get_library", lambda: resources)
    monkeypatch.setattr(implementation.wp, "get_device", lambda _: SimpleNamespace(is_cuda=True))
    monkeypatch.setattr(implementation.wp, "get_stream", lambda _: SimpleNamespace(cuda_stream=1))
    solver = implementation.CudssSolver()
    message = {"destroy": "cudssDataDestroy", "handler": "allocator setup", "create": "cudssDataCreate"}[failure]
    with pytest.raises(RuntimeError, match=message):
        memory.Workspace(solver)
    assert solver._closed
    assert not solver._data
    assert not resources.live
    if failure == "create" and partial:
        assert resources.callback_checks == 1
    calls = resources.calls.copy()
    solver.release()
    assert resources.calls == calls


def test_successful_workspace_keeps_replacement_until_release(monkeypatch):
    import warp_cudss.solver as implementation

    resources = NativeResources(None, False)
    monkeypatch.setattr(implementation, "get_library", lambda: resources)
    monkeypatch.setattr(implementation.wp, "get_device", lambda _: SimpleNamespace(is_cuda=True))
    monkeypatch.setattr(implementation.wp, "get_stream", lambda _: SimpleNamespace(cuda_stream=1))
    solver = implementation.CudssSolver()
    original = solver._data.value
    workspace = memory.Workspace(solver)
    assert original not in resources.live
    assert solver._data.value in resources.live
    assert not solver._closed
    assert len(resources.live) == 3
    solver.release()
    assert not resources.live
    assert resources.callback_checks == 1
    assert workspace.error is None
