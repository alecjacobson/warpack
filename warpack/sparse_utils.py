"""Shared fixed-topology sparse value mappings, computed and refreshed in Warp."""

import warp as wp
import warp.sparse as ws


@wp.kernel
def transpose_indices(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    transposed_offsets: wp.array[wp.int32],
    transposed_columns: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    rows: int,
):
    e = wp.tid()
    if e < offsets[rows]:
        lo, hi = int(0), rows
        while lo < hi:
            mid = (lo + hi) // 2
            if offsets[mid + 1] <= e:
                lo = mid + 1
            else:
                hi = mid
        row, col = lo, columns[e]
        lo, hi = int(transposed_offsets[col]), int(transposed_offsets[col + 1])
        while lo < hi:
            mid = (lo + hi) // 2
            if transposed_columns[mid] < row:
                lo = mid + 1
            else:
                hi = mid
        indices[e] = lo


@wp.kernel
def refresh_transpose(
    offsets: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    values: wp.array[wp.float64],
    transposed_values: wp.array[wp.float64],
    rows: int,
):
    e = wp.tid()
    if e < offsets[rows]:
        transposed_values[indices[e]] = values[e]


class _Transpose:
    """Fixed-topology transpose with a device-only value refresh for replay."""

    def __init__(self, source):
        self.source = source
        self.matrix = ws.bsr_transposed(source)
        self.indices = wp.empty(source.values.shape[0], dtype=wp.int32, device=source.device)
        wp.launch(
            transpose_indices,
            self.indices.size,
            [
                source.offsets,
                source.columns,
                self.matrix.offsets,
                self.matrix.columns,
                self.indices,
                source.nrow,
            ],
            device=source.device,
        )

    def refresh(self):
        wp.launch(
            refresh_transpose,
            self.indices.size,
            [
                self.source.offsets,
                self.indices,
                self.source.values,
                self.matrix.values,
                self.source.nrow,
            ],
            device=self.source.device,
        )
