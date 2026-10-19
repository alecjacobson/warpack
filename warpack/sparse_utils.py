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
        wp.atomic_add(transposed_values, indices[e], values[e])


@wp.kernel
def transpose_owners(
    offsets: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    owners: wp.array[wp.int32],
    rows: int,
):
    e = wp.tid()
    if e < offsets[rows]:
        wp.atomic_min(owners, indices[e], e)


class _Transpose:
    """Fixed-topology scalar transpose with device-only numeric refresh.

    Duplicate source coordinates sum into one destination slot; unused duplicate
    slots are zeroed. Source rows need not be sorted. Topology, logical row
    counts and array addresses must stay fixed between captured replays.
    """

    def __init__(self, source, *, with_source_owners=False):
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

        self.owners = None
        if with_source_owners:
            # One source entry owns each logical coefficient. Constraint norms
            # use this to count a coalesced coefficient exactly once per row.
            self.owners = wp.full(
                self.matrix.values.size, 2147483647, dtype=wp.int32, device=source.device
            )
            wp.launch(
                transpose_owners,
                self.indices.size,
                [source.offsets, self.indices, self.owners, source.nrow],
                device=source.device,
            )

    def refresh(self):
        # Transpose construction may preserve duplicate slots. Clear all of
        # them, then sum each source coordinate into its first destination.
        self.matrix.values.zero_()
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
