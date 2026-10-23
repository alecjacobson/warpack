"""Fixed scratch views for mutually exclusive internal solver branches."""

import math

import warp as wp


def scratch(storage, name, shape, dtype, device, *, zero=False):
    """Allocate scratch, or take a contiguous prefix of supplied backing storage.

    Sharing is internal and requires serialized use. In particular, reshaping
    a flat prefix keeps column-layout kernels contiguous at each branch width.
    This function runs only during construction, never during graph replay.
    """
    if storage is None:
        constructor = wp.zeros if zero else wp.empty
        return constructor(shape, dtype=dtype, device=device)
    source = storage[name]
    count = math.prod(shape) if isinstance(shape, tuple) else shape
    if (
        source.dtype != dtype
        or source.device != device
        or not source.is_contiguous
        or source.size < count
    ):
        raise ValueError(f"incompatible internal scratch storage: {name}")
    result = source.flatten()[:count].reshape(shape)
    if zero:
        result.zero_()
    return result
