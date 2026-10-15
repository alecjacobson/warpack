"""Weight-space matrices from Fast Complementary Dynamics via Skinning Eigenmodes."""

import warp as wp
import warp.sparse as ws

from . import kernels as K


@wp.kernel
def _trace_blocks(blocks: wp.array[wp.mat33d], values: wp.array[wp.float64]):
    p = wp.tid()
    values[p] = wp.trace(blocks[p])


def skinning_stiffness(hessian):
    """Sum P_i.T H P_i for an interleaved xyz elastic Hessian (3x3 FP64 BSR).

    Returns scalar n-by-n BSR. This is a setup operation. The physical mass
    accompanying it is the scalar vertex mass, not a 3n displacement mass.
    """
    if hessian.values.dtype != wp.mat33d or hessian.nrow != hessian.ncol:
        raise ValueError("expected square 3x3 FP64 BSR Hessian")
    if hessian.row_counts is not None:
        raise ValueError("compress the Hessian first")
    result = ws.bsr_zeros(hessian.nrow, hessian.ncol, wp.float64, device=hessian.device)
    result.nnz = hessian.nnz
    result.columns = wp.clone(hessian.columns)
    wp.copy(result.offsets, hessian.offsets)
    result.values = wp.empty(hessian.values.size, dtype=wp.float64, device=hessian.device)
    wp.launch(
        _trace_blocks, result.values.size, [hessian.values, result.values], device=hessian.device
    )
    # Invalidate the empty matrix's cached nnz readback after replacing topology.
    # Some Warp versions otherwise restore nnz=0 during later KKT assembly.
    result.notify_nnz_changed(nnz=hessian.nnz)
    return result


@wp.kernel
def _rig_rows(
    positions: wp.array[wp.vec3d], jacobian: wp.array3d[wp.float64], rows: wp.array2d[wp.float64]
):
    row, vertex = wp.tid()
    parameter, axis, coordinate = row // 12, (row % 12) // 4, row % 4
    value = jacobian[parameter, vertex, axis]
    if coordinate < 3:
        value *= positions[vertex][coordinate]
    rows[row, vertex] = value


def skinning_constraint_rows(positions, effective_jacobian):
    """Appendix B.1 coordinate-weighted rows, before removal of dependencies.

    positions: n vec3d entries. effective_jacobian: FP64 (p,n,3), storing the
    columns of the EFFECTIVE rig Jacobian (e.g. D M J from paper section 8.1).
    Mass/leakage weighting must already be included; no bare-J shortcut.
    Output (12*p,n) rows ordered parameter, displacement axis, [x,y,z,1].
    Repeated/zero rows are expected: remove dependencies before KKT factors.
    This assembly is pure Warp; it does not choose a numerical rank or rig.
    """
    if positions.dtype != wp.vec3d or positions.ndim != 1:
        raise ValueError("positions must be a vec3d array")
    if effective_jacobian.dtype != wp.float64 or effective_jacobian.ndim != 3:
        raise ValueError("effective_jacobian must be an FP64 (p,n,3) array")
    if effective_jacobian.shape[1:] != (positions.size, 3) or effective_jacobian.shape[0] < 1:
        raise ValueError("effective_jacobian must have shape (p,n,3), p > 0")
    if positions.device != effective_jacobian.device:
        raise ValueError("positions and Jacobian must be on the same device")
    rows = wp.empty(
        (12 * effective_jacobian.shape[0], positions.size),
        dtype=wp.float64,
        device=positions.device,
    )
    wp.launch(_rig_rows, rows.shape, [positions, effective_jacobian, rows], device=positions.device)
    return rows


@wp.kernel
def _row_norms(rows: wp.array2d[wp.float64], norms: wp.array2d[wp.float64]):
    j, chunk = wp.tid()
    v = wp.tile_load(rows[j], shape=256, offset=chunk * 256)
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(wp.mul, v, v)), offset=0)


@wp.kernel
def _normalize_rows(
    rows: wp.array2d[wp.float64], norms: wp.array2d[wp.float64], info: wp.array[wp.int32]
):
    j, i = wp.tid()
    norm = wp.sqrt(norms[j, 0])
    if not wp.isfinite(norm):
        wp.atomic_max(info, 1, 1)
    if norm > wp.float64(0.0):
        rows[j, i] /= norm


@wp.kernel
def _pivot(
    norms: wp.array2d[wp.float64],
    selected: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    j: int,
    tolerance: wp.float64,
):
    best, index = wp.float64(0.0), int(-1)
    for i in range(norms.shape[0]):
        if norms[i, 0] > best:
            best = norms[i, 0]
            index = i
    if best > tolerance * tolerance:
        rank[0] = j + 1
        selected[0] = index
    else:
        selected[0] = -1


@wp.kernel
def _pivot_row(
    work: wp.array2d[wp.float64],
    norms: wp.array2d[wp.float64],
    selected: wp.array[wp.int32],
    basis: wp.array2d[wp.float64],
    j: int,
):
    i = wp.tid()
    p = selected[0]
    value = wp.float64(0.0)
    if p >= 0:
        value = work[p, i] / wp.sqrt(norms[p, 0])
    basis[j, i] = value


@wp.kernel
def _subtract_rows(
    work: wp.array2d[wp.float64], row: wp.array2d[wp.float64], dots: wp.array2d[wp.float64]
):
    j, i = wp.tid()
    work[j, i] -= dots[j, 0] * row[0, i]


@wp.kernel
def _dense_triplets(
    basis: wp.array2d[wp.float64],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    j, i = wp.tid()
    p = j * basis.shape[1] + i
    rows[p] = j
    cols[p] = i
    vals[p] = basis[j, i]


def independent_constraints(rows, tolerance=1e-10):
    """Pure-Warp pivoted, twice-reorthogonalized row QR for small dense rig spaces.

    Scale nonzero input rows to unit norm, then drop residual directions of norm
    <= tolerance. Return scalar BSR with independent orthonormal rows spanning
    the retained constraint space. Does not mutate the input. Cost O(p^2*n);
    intended for small rig Jacobians, not arbitrary large constraint systems.

    This is SETUP: only integer rank/validity metadata is read back to allocate
    the CSR shape; all rank decisions and numerical linear algebra run on device. It
    is not graph capturable. A changed rig requires rebuilding the constraints
    and KKT factors. Verify dropped directions against the original rig rows.
    """
    import math

    if rows.dtype != wp.float64 or rows.ndim != 2 or min(rows.shape) < 1:
        raise ValueError("rows must be a nonempty FP64 matrix")
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must be finite and between zero and one")
    p, n = rows.shape
    device = rows.device
    work = wp.clone(rows)
    basis = wp.zeros((min(p, n), n), dtype=wp.float64, device=device)
    norms = wp.zeros((p, 1), dtype=wp.float64, device=device)
    dots = wp.zeros_like(norms)
    pivot = wp.zeros(1, dtype=wp.int32, device=device)
    rank = wp.zeros(2, dtype=wp.int32, device=device)

    def row_norms():
        norms.zero_()
        wp.launch_tiled(
            _row_norms,
            dim=(p, (n + 255) // 256),
            inputs=[work, norms],
            block_dim=128,
            device=device,
        )

    row_norms()
    wp.launch(_normalize_rows, work.shape, [work, norms, rank], device=device)
    for j in range(min(p, n)):
        row_norms()
        wp.launch(_pivot, 1, [norms, pivot, rank, j, tolerance], device=device)
        wp.launch(_pivot_row, n, [work, norms, pivot, basis, j], device=device)
        for _ in range(2):
            dots.zero_()
            wp.launch_tiled(
                K.gram_partial,
                dim=(p, 1, (n + 255) // 256),
                inputs=[work, basis[j : j + 1], dots],
                block_dim=128,
                device=device,
            )
            wp.launch(_subtract_rows, work.shape, [work, basis[j : j + 1], dots], device=device)
    info = rank.numpy()
    if info[1]:
        raise ValueError("constraint rows must have finite norms")
    count = int(info[0])
    if not 0 < count < n:
        raise ValueError("constraints must have rank between 1 and n-1")
    rr = wp.empty(count * n, dtype=wp.int32, device=device)
    cc = wp.empty_like(rr)
    vv = wp.empty(count * n, dtype=wp.float64, device=device)
    wp.launch(_dense_triplets, (count, n), [basis, rr, cc, vv], device=device)
    return ws.bsr_from_triplets(count, n, rr, cc, vv)
