"""Standalone CUDA graph setup shared by solvers and projected workspaces.

This module is setup-only. Numerical methods record into an existing capture;
replaying a graph never calls Python here.
"""

from contextlib import contextmanager

import warp as wp


def require_capture_setup(device):
    """Reject unsupported setup before warmup can mutate caller-owned state."""
    device = wp.get_device(device)
    if not device.is_cuda:
        raise ValueError("graph capture requires a CUDA device")
    if wp.get_stream(device).is_capturing:
        raise RuntimeError(
            "stream is already capturing; record solve() or initialize()/step()/finalize() "
            "in the existing capture instead of calling capture()"
        )


@contextmanager
def capture_scope(device):
    """Finish pending setup writes, then record one standalone graph.

    Keep the device synchronization: pending eager writes before graph upload
    reproduce a failure on driver 570 (#61). This is not an iteration barrier.
    Call require_capture_setup() before any warmup as well as entering here.
    """
    require_capture_setup(device)
    wp.synchronize_device(device)
    with wp.ScopedCapture(device=device) as capture:
        yield capture
