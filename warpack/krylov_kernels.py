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
    g: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    order: wp.array[wp.int32],
    keep: int,
    seed: wp.array[wp.int32],
):
    i, j = wp.tid()
    value = wp.float64(0.0)
    if i == j and i < keep:
        source = i
        if i == seed[0]:
            source = keep - 1
        elif i == keep - 1:
            source = seed[0]
        value = values[order[source]]
    g[i, j] = value


@wp.kernel
def restart_residuals(
    q: wp.array2d[wp.float64],
    z: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    order: wp.array[wp.int32],
    scores: wp.array2d[wp.float64],
):
    row, chunk = wp.tid()
    x = wp.tile_load(q[row], shape=256, offset=chunk * 256)
    ax = wp.tile_load(z[row], shape=256, offset=chunk * 256)
    value = values[order[row]]
    residual = ax - value * x
    wp.tile_atomic_add(scores[row], wp.tile_sum(wp.tile_map(wp.mul, residual, residual)), offset=0)
    wp.tile_atomic_add(
        scores[row],
        wp.tile_sum(wp.tile_map(wp.mul, x, x)) * wp.max(wp.float64(1.0), value * value),
        offset=1,
    )


@wp.kernel
def restart_seed(scores: wp.array2d[wp.float64], seed: wp.array[wp.int32]):
    best = wp.float64(-1.0)
    index = int(0)
    for j in range(scores.shape[0]):
        score = scores[j, 0] / wp.max(scores[j, 1], wp.float64(1.0e-300))
        if score > best:
            best = score
            index = j
    seed[0] = index


@wp.kernel
def restart_swap(
    q: wp.array2d[wp.float64], z: wp.array2d[wp.float64], seed: wp.array[wp.int32], keep: int
):
    i = wp.tid()
    j, last = seed[0], keep - 1
    x = q[j, i]
    ax = z[j, i]
    q[j, i] = q[last, i]
    z[j, i] = z[last, i]
    q[last, i] = x
    z[last, i] = ax


@wp.kernel
def loop_update(
    converged: wp.array[wp.int32],
    status: wp.array[wp.int32],
    iterations: wp.array[wp.int32],
    running: wp.array[wp.int32],
    recoveries: wp.array[wp.int32],
    k: int,
    maxiter: int,
    full_space: bool,
):
    iterations[0] += 1
    # This guard also applies before the FIRST breakdown: a single chain can
    # converge to distinct eigenvalues while missing their multiplicities.
    # A full-space projection already spans every eigenspace. Otherwise require
    # at least k independently seeded chains as a conservative exploration guard.
    explore = not full_space and recoveries[0] + 1 < k
    running[0] = int((converged[0] < k or explore) and iterations[0] < maxiter and status[0] == 0)
