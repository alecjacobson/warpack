"""Batched application of upstream FSAI factors with adjacent RHS storage.

Only the application layout changes. Factor construction and CG remain
upstream Warp. Transposes keep CG's independent contiguous RHS batches intact.
"""

from typing import Any

import warp as wp

from ._scratch import scratch


@wp.func
def copy_value(value: Any):
    return value


@wp.kernel(module="unique")
def transpose_rows(source: wp.array2d(dtype=Any), target: wp.array2d(dtype=Any)):
    i, j = wp.tid()
    # A padded shared stride avoids the worst transpose bank conflicts.
    tile = wp.tile_load(source, shape=(32, 33), offset=(i * 32, j * 32), storage="shared")
    square = wp.tile_view(tile, shape=(32, 32), offset=(0, 0))
    result = wp.tile_map(copy_value, wp.tile_transpose(square))
    wp.tile_store(target, result, offset=(j * 32, i * 32))


@wp.kernel(module="unique")
def csr_eight_rhs(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array(dtype=Any),
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
):
    group, row, lane = wp.tid()
    rhs, slot = group * 8 + lane % 8, lane // 8
    value = x.dtype(0.0)
    if rhs < x.shape[1]:
        for e in range(offsets[row] + slot, offsets[row + 1], 4):
            value += x.dtype(values[e]) * x[columns[e], rhs]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=0)
    if rhs < x.shape[1] and slot == 0:
        y[row, rhs] = wp.tile_extract(totals, lane % 8)


@wp.func
def multiply(a: Any, b: Any):
    return a * b


@wp.kernel(module="unique")
def csr_eight_rhs_padded(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array(dtype=Any),
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
):
    group, row, lane = wp.tid()
    rhs, slot = group * 8 + lane % 8, lane // 8
    value = x.dtype(0.0)
    if rhs < x.shape[1]:
        for e in range(offsets[row] + slot, offsets[row] + counts[row], 4):
            value += x.dtype(values[e]) * x[columns[e], rhs]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=0)
    if rhs < x.shape[1] and slot == 0:
        y[row, rhs] = wp.tile_extract(totals, lane % 8)


@wp.kernel(module="unique")
def transpose_axpby(
    source: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    z: wp.array2d(dtype=Any),
    alpha: wp.float64,
    beta: wp.float64,
):
    i, j = wp.tid()
    tile = wp.tile_load(source, shape=(32, 33), offset=(i * 32, j * 32), storage="shared")
    square = wp.tile_view(tile, shape=(32, 32), offset=(0, 0))
    result = wp.tile_map(copy_value, wp.tile_transpose(square))
    result = wp.tile_map(
        multiply,
        result,
        wp.tile_full(shape=(32, 32), value=source.dtype(alpha), dtype=source.dtype),
    )
    if beta != wp.float64(0.0):
        previous = wp.tile_load(y, shape=(32, 32), offset=(j * 32, i * 32))
        result += wp.tile_map(
            multiply,
            previous,
            wp.tile_full(shape=(32, 32), value=source.dtype(beta), dtype=source.dtype),
        )
    wp.tile_store(z, result, offset=(j * 32, i * 32))


class CoalescedFactorApplication:
    def __init__(self, preconditioner, width, n, device, dtype, _storage=None):
        self.preconditioner, self.width, self.n, self.device = preconditioner, width, n, device
        for factor in (preconditioner.G, preconditioner.GT):
            if factor.block_shape != (1, 1) or factor.shape != (n, n) or factor.device != device:
                raise ValueError("require scalar FSAI factors on the working device")
        self.first, self.second = (
            scratch(_storage, name, (n, width), dtype, device)
            for name in ("factor_first", "factor_second")
        )

    def matvec(self, x, y, z, alpha, beta):
        rows = x.reshape((self.width, self.n))
        wp.launch_tiled(
            transpose_rows,
            dim=((self.width + 31) // 32, (self.n + 31) // 32),
            inputs=[rows, self.first],
            block_dim=128,
            device=self.device,
        )
        for factor, source, result in (
            (self.preconditioner.G, self.first, self.second),
            (self.preconditioner.GT, self.second, self.first),
        ):
            if factor.row_counts is not None:
                wp.launch_tiled(
                    csr_eight_rhs_padded,
                    dim=((self.width + 7) // 8, self.n),
                    inputs=[
                        factor.offsets,
                        factor.row_counts,
                        factor.columns,
                        factor.values,
                        source,
                        result,
                    ],
                    block_dim=32,
                    device=self.device,
                )
                continue
            wp.launch_tiled(
                csr_eight_rhs,
                dim=((self.width + 7) // 8, self.n),
                inputs=[factor.offsets, factor.columns, factor.values, source, result],
                block_dim=32,
                device=self.device,
            )
        wp.launch_tiled(
            transpose_axpby,
            dim=((self.n + 31) // 32, (self.width + 31) // 32),
            inputs=[
                self.first,
                y.reshape((self.width, self.n)),
                z.reshape((self.width, self.n)),
                wp.float64(alpha),
                wp.float64(beta),
            ],
            block_dim=128,
            device=self.device,
        )
