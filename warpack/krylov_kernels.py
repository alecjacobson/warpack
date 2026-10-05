"""Two-stage reductions for FP64 Krylov orthogonalization."""

import warp as wp


@wp.kernel
def dot_partials(q: wp.array2d[wp.float64], parts: wp.array2d[wp.float64], j: int):
    r, chunk = wp.tid()
    a = wp.tile_load(q[r], shape=1024, offset=chunk * 1024)
    b = wp.tile_load(q[j], shape=1024, offset=chunk * 1024)
    wp.tile_store(parts[r], wp.tile_sum(wp.tile_map(wp.mul, a, b)), offset=chunk)


@wp.kernel
def finish_dots(
    parts: wp.array2d[wp.float64],
    dots: wp.array2d[wp.float64],
    source: wp.array2d[wp.float64],
    g: wp.array2d[wp.float64],
    j: int,
    project: bool,
):
    r = wp.tid()
    value = wp.float64(0.0)
    for block in range((parts.shape[1] + 1023) // 1024):
        part = wp.tile_load(parts[r], shape=1024, offset=block * 1024)
        value += wp.tile_extract(wp.tile_sum(part), 0)
    wp.tile_store(dots[r], wp.tile_from_thread(shape=1, value=value, thread_idx=0))
    if r == j:
        wp.tile_store(source[0], wp.tile_from_thread(shape=1, value=value, thread_idx=0))
    elif project:
        scalar = wp.tile_from_thread(shape=1, value=value, thread_idx=0)
        wp.tile_store(g[r], scalar, offset=j - 1)
        wp.tile_store(g[j - 1], scalar, offset=r)


@wp.kernel
def subtract(
    q: wp.array2d[wp.float64],
    dots: wp.array2d[wp.float64],
    norms: wp.array[wp.float64],
    j: int,
    norm: bool,
):
    chunk = wp.tid()
    value = wp.tile_load(q[j], shape=1024, offset=chunk * 1024)
    for r in range(j):
        value -= wp.tile_load(q[r], shape=1024, offset=chunk * 1024) * dots[r, 0]
    wp.tile_store(q[j], value, offset=chunk * 1024)
    if norm:
        wp.tile_store(norms, wp.tile_sum(wp.tile_map(wp.mul, value, value)), offset=chunk)


@wp.kernel
def finish_norm(
    parts: wp.array[wp.float64],
    norm: wp.array2d[wp.float64],
    source: wp.array2d[wp.float64],
    flag: wp.array[wp.int32],
    recoveries: wp.array[wp.int32],
    check: bool,
):
    value = wp.float64(0.0)
    for block in range((parts.shape[0] + 1023) // 1024):
        part = wp.tile_load(parts, shape=1024, offset=block * 1024)
        value += wp.tile_extract(wp.tile_sum(part), 0)
    wp.tile_store(norm[0], wp.tile_from_thread(shape=1, value=value, thread_idx=0))
    if check:
        result = int(value <= wp.float64(1.0e-26) * source[0, 0])
        wp.tile_store(flag, wp.tile_from_thread(shape=1, value=result, thread_idx=0))
        count = recoveries[0] + result
        wp.tile_store(recoveries, wp.tile_from_thread(shape=1, value=count, thread_idx=0))


@wp.kernel
def last_projection(
    q: wp.array2d[wp.float64], aq: wp.array2d[wp.float64], parts: wp.array2d[wp.float64]
):
    r, chunk = wp.tid()
    a = wp.tile_load(q[r], shape=1024, offset=chunk * 1024)
    b = wp.tile_load(aq[aq.shape[0] - 1], shape=1024, offset=chunk * 1024)
    wp.tile_store(parts[r], wp.tile_sum(wp.tile_map(wp.mul, a, b)), offset=chunk)


@wp.kernel
def finish_projection(parts: wp.array2d[wp.float64], g: wp.array2d[wp.float64]):
    r = wp.tid()
    value = wp.float64(0.0)
    for block in range((parts.shape[1] + 1023) // 1024):
        part = wp.tile_load(parts[r], shape=1024, offset=block * 1024)
        value += wp.tile_extract(wp.tile_sum(part), 0)
    column = g.shape[0] - 1
    scalar = wp.tile_from_thread(shape=1, value=value, thread_idx=0)
    wp.tile_store(g[r], scalar, offset=column)
    wp.tile_store(g[column], scalar, offset=r)


@wp.kernel
def restart_projection(
    g: wp.array2d[wp.float64], values: wp.array[wp.float64], order: wp.array[wp.int32], keep: int
):
    i, j = wp.tid()
    value = wp.float64(0.0)
    if i == j and i < keep:
        value = values[order[i]]
    g[i, j] = value


@wp.kernel
def loop_update(
    converged: wp.array[wp.int32],
    status: wp.array[wp.int32],
    iterations: wp.array[wp.int32],
    running: wp.array[wp.int32],
    recoveries: wp.array[wp.int32],
    k: int,
    maxiter: int,
):
    iterations[0] += 1
    # After invariant-subspace breakdown, residuals alone cannot certify that
    # all requested multiplicities were explored. Allow at least k chains.
    explore = recoveries[0] > 0 and recoveries[0] + 1 < k
    running[0] = int((converged[0] < k or explore) and iterations[0] < maxiter and status[0] == 0)
