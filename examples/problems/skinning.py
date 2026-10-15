"""Weight-space matrices from Fast Complementary Dynamics via Skinning Eigenmodes."""

import warp as wp
import warp.sparse as ws


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
