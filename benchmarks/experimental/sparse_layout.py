"""Experimental RHS/row lane allocation for existing sparse products."""

import functools
from typing import Any

import warp as wp


@functools.lru_cache(None)
def csr_kernel(lanes):
    row_lanes = wp.constant(lanes)
    rhs_count = wp.constant(32 // lanes)

    @wp.kernel(module="unique")
    def compute(
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array(dtype=Any),
        x: wp.array2d(dtype=Any),
        y: wp.array2d(dtype=Any),
    ):
        group, row, lane = wp.tid()
        rhs, slot = group * rhs_count + lane // row_lanes, lane % row_lanes
        value = x.dtype(0.0)
        if rhs < x.shape[0]:
            for e in range(offsets[row] + slot, offsets[row + 1], row_lanes):
                value += x.dtype(values[e]) * x[rhs, columns[e]]
        totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(rhs_count, row_lanes)), axis=1)
        if rhs < x.shape[0] and slot == 0:
            y[rhs, row] = wp.tile_extract(totals, lane // row_lanes)

    return compute


@functools.lru_cache(None)
def bsr_kernel(dtype, lanes):
    row_lanes = wp.constant(lanes)
    rhs_count = wp.constant(32 // lanes)
    vector = wp.types.vector(length=3, dtype=dtype)
    matrix = wp.types.matrix(shape=(3, 3), dtype=dtype)

    @wp.kernel(module="unique")
    def compute(
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array(dtype=matrix),
        x: wp.array2d(dtype=dtype),
        y: wp.array2d(dtype=dtype),
    ):
        group, row, lane = wp.tid()
        rhs, slot = group * rhs_count + lane // row_lanes, lane % row_lanes
        value = vector(dtype(0.0))
        if rhs < x.shape[0]:
            for e in range(offsets[row] + slot, offsets[row + 1], row_lanes):
                col = columns[e] * 3
                value += values[e] * vector(x[rhs, col], x[rhs, col + 1], x[rhs, col + 2])
        a = wp.tile_sum(wp.tile_reshape(wp.tile(value[0]), shape=(rhs_count, row_lanes)), axis=1)
        b = wp.tile_sum(wp.tile_reshape(wp.tile(value[1]), shape=(rhs_count, row_lanes)), axis=1)
        c = wp.tile_sum(wp.tile_reshape(wp.tile(value[2]), shape=(rhs_count, row_lanes)), axis=1)
        if rhs < x.shape[0] and slot == 0:
            y[rhs, row * 3] = wp.tile_extract(a, lane // row_lanes)
            y[rhs, row * 3 + 1] = wp.tile_extract(b, lane // row_lanes)
            y[rhs, row * 3 + 2] = wp.tile_extract(c, lane // row_lanes)

    return compute


def product(matrix, x, y, lanes=8):
    if lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("lanes must be a power of two from 1 to 32")
    if matrix.block_shape == (1, 1):
        kernel = csr_kernel(lanes)
    elif matrix.block_shape == (3, 3):
        kernel = bsr_kernel(matrix.scalar_type, lanes)
    else:
        raise ValueError("expected scalar CSR or 3x3 BSR")
    rhs_count = 32 // lanes
    wp.launch_tiled(
        kernel,
        dim=((x.shape[0] + rhs_count - 1) // rhs_count, matrix.nrow),
        inputs=[matrix.offsets, matrix.columns, matrix.values, x, y],
        block_dim=32,
        device=matrix.device,
    )
