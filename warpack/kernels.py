"""FP64 kernels. All reductions and projected eigensolves stay on device."""

import warp as wp


@wp.kernel
def random_block(q: wp.array2d[wp.float64], seed: int):
    j, i = wp.tid()
    state = wp.rand_init(seed, i * q.shape[0] + j)
    q[j, i] = wp.float64(wp.randn(state))


@wp.kernel
def gram_partial(x: wp.array2d[wp.float64], y: wp.array2d[wp.float64], g: wp.array2d[wp.float64]):
    a, b, chunk = wp.tid()
    tx = wp.tile_load(x[a], shape=256, offset=chunk * 256)
    ty = wp.tile_load(y[b], shape=256, offset=chunk * 256)
    s = wp.tile_sum(wp.tile_map(wp.mul, tx, ty))
    wp.tile_atomic_add(g[a], s, offset=b)


@wp.kernel
def cholesky(g: wp.array2d[wp.float64], l: wp.array2d[wp.float64], status: wp.array[wp.int32]):
    m = g.shape[0]
    for i in range(m):
        for j in range(i + 1):
            v = g[i, j]
            for k in range(j):
                v -= l[i, k] * l[j, k]
            if i == j:
                if v <= wp.float64(0.0):
                    wp.atomic_max(status, 0, 1)
                l[i, j] = wp.sqrt(wp.max(v, wp.float64(1.0e-30)))
            else:
                l[i, j] = v / l[j, j]


@wp.kernel
def triangular(q: wp.array2d[wp.float64], l: wp.array2d[wp.float64]):
    i = wp.tid()
    for j in range(q.shape[0]):
        v = q[j, i]
        for k in range(j):
            v -= l[j, k] * q[k, i]
        q[j, i] = v / l[j, j]


@wp.kernel
def jacobi(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    order: wp.array[wp.int32],
    which: int,
):
    m = h.shape[0]
    for i in range(m):
        for j in range(m):
            v[i, j] = wp.float64(0.0)
            if i == j:
                v[i, j] = wp.float64(1.0)
            if i < j:
                a = (h[i, j] + h[j, i]) * wp.float64(0.5)
                h[i, j] = a
                h[j, i] = a
    for sweep in range(20):
        for p in range(m - 1):
            for q in range(p + 1, m):
                apq = h[p, q]
                if wp.abs(apq) > wp.float64(1.0e-16) * (wp.abs(h[p, p]) + wp.abs(h[q, q])):
                    tau = (h[q, q] - h[p, p]) / (wp.float64(2.0) * apq)
                    t = wp.float64(1.0) / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                    if tau < wp.float64(0.0):
                        t = -t
                    c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
                    s = t * c
                    h[p, p] -= t * apq
                    h[q, q] += t * apq
                    h[p, q] = wp.float64(0.0)
                    h[q, p] = wp.float64(0.0)
                    for r in range(m):
                        if r != p and r != q:
                            a = h[r, p]
                            b = h[r, q]
                            h[r, p] = c * a - s * b
                            h[p, r] = h[r, p]
                            h[r, q] = s * a + c * b
                            h[q, r] = h[r, q]
                        a = v[r, p]
                        b = v[r, q]
                        v[r, p] = c * a - s * b
                        v[r, q] = s * a + c * b
    for i in range(m):
        values[i] = h[i, i]
        order[i] = i
    for i in range(m):
        best = i
        for j in range(i + 1, m):
            a = values[order[j]]
            b = values[order[best]]
            if which == 0 and a < b:
                best = j
            if which == 1 and a > b:
                best = j
            if which == 2 and wp.abs(a) > wp.abs(b):
                best = j
            if which == 3 and wp.abs(a) < wp.abs(b):
                best = j
        tmp = order[i]
        order[i] = order[best]
        order[best] = tmp


@wp.kernel
def rotate(
    q: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    order: wp.array[wp.int32],
    out: wp.array2d[wp.float64],
):
    j, i = wp.tid()
    val = wp.float64(0.0)
    for r in range(q.shape[0]):
        val += q[r, i] * v[r, order[j]]
    out[j, i] = val


@wp.kernel
def csr_mm(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    a: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
    shift: wp.float64,
    sign: wp.float64,
):
    j, i = wp.tid()
    val = wp.float64(0.0)
    for e in range(offsets[i], offsets[i + 1]):
        val += a[e] * x[j, columns[e]]
    y[j, i] = wp.float64(sign) * val + wp.float64(shift) * x[j, i]


@wp.kernel
def bsr_mm(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    a: wp.array[wp.mat33d],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
    shift: wp.float64,
    sign: wp.float64,
):
    j, i = wp.tid()
    row = i // 3
    d = i % 3
    val = wp.float64(0.0)
    for e in range(offsets[row], offsets[row + 1]):
        col = columns[e] * 3
        for c in range(3):
            val += a[e][d, c] * x[j, col + c]
    y[j, i] = wp.float64(sign) * val + wp.float64(shift) * x[j, i]


@wp.kernel
def residual(
    x: wp.array2d[wp.float64],
    ax: wp.array2d[wp.float64],
    vals: wp.array[wp.float64],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
):
    j, chunk = wp.tid()
    a = wp.tile_load(ax[j], shape=256, offset=chunk * 256)
    b = wp.tile_load(x[j], shape=256, offset=chunk * 256)
    r = a - b * vals[order[j]]
    s = wp.tile_sum(wp.tile_map(wp.mul, r, r))
    wp.tile_atomic_add(norms[j], s)


@wp.kernel
def output_values(
    vals: wp.array[wp.float64],
    order: wp.array[wp.int32],
    out: wp.array[wp.float64],
    norms: wp.array2d[wp.float64],
    rel: wp.array[wp.float64],
    tol: float,
    converged: wp.array[wp.int32],
):
    j = wp.tid()
    lam = vals[order[j]]
    out[j] = lam
    err = wp.sqrt(norms[j, 0]) / wp.max(wp.abs(lam), wp.float64(1.0))
    rel[j] = err
    if err <= wp.float64(tol):
        wp.atomic_add(converged, 0, 1)


@wp.kernel
def basis_dots(q: wp.array2d[wp.float64], dots: wp.array2d[wp.float64], j: int):
    r, chunk = wp.tid()
    a = wp.tile_load(q[r], shape=256, offset=chunk * 256)
    b = wp.tile_load(q[j], shape=256, offset=chunk * 256)
    s = wp.tile_sum(wp.tile_map(wp.mul, a, b))
    wp.tile_atomic_add(dots[r], s)


@wp.kernel
def basis_subtract(q: wp.array2d[wp.float64], dots: wp.array2d[wp.float64], j: int):
    i = wp.tid()
    val = q[j, i]
    for r in range(j):
        val -= q[r, i] * dots[r, 0]
    q[j, i] = val


@wp.kernel
def basis_norm(q: wp.array2d[wp.float64], norm: wp.array2d[wp.float64], j: int):
    chunk = wp.tid()
    a = wp.tile_load(q[j], shape=256, offset=chunk * 256)
    wp.tile_atomic_add(norm[0], wp.tile_sum(wp.tile_map(wp.mul, a, a)))


@wp.kernel
def basis_scale(q: wp.array2d[wp.float64], norm: wp.array2d[wp.float64], j: int):
    i = wp.tid()
    q[j, i] = q[j, i] / wp.sqrt(wp.max(norm[0, 0], wp.float64(1.0e-300)))


@wp.kernel
def jacobi_init(h: wp.array2d[wp.float64], v: wp.array2d[wp.float64]):
    i, j = wp.tid()
    v[i, j] = wp.float64(0.0)
    if i == j:
        v[i, j] = wp.float64(1.0)
    if i < j:
        a = (h[i, j] + h[j, i]) * wp.float64(0.5)
        h[i, j] = a
        h[j, i] = a


@wp.kernel
def jacobi_pairs(
    h: wp.array2d[wp.float64],
    cs: wp.array[wp.float64],
    ss: wp.array[wp.float64],
    partners: wp.array[wp.int32],
    round: int,
):
    i = wp.tid()
    n = h.shape[0]
    p = (round + i) % (n - 1)
    q = (round - i + n - 1) % (n - 1)
    if i == 0:
        q = n - 1
    apq = h[p, q]
    c = wp.float64(1.0)
    s = wp.float64(0.0)
    if wp.abs(apq) > wp.float64(1.0e-16) * (wp.abs(h[p, p]) + wp.abs(h[q, q])):
        tau = (h[q, q] - h[p, p]) / (wp.float64(2.0) * apq)
        t = wp.float64(1.0) / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
        if tau < wp.float64(0.0):
            t = -t
        c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
        s = t * c
    cs[p] = c
    cs[q] = c
    ss[p] = s
    ss[q] = -s
    partners[p] = q
    partners[q] = p


@wp.kernel
def jacobi_right(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    tmp: wp.array2d[wp.float64],
    vt: wp.array2d[wp.float64],
    c: wp.array[wp.float64],
    s: wp.array[wp.float64],
    p: wp.array[wp.int32],
):
    i, j = wp.tid()
    tmp[i, j] = c[j] * h[i, j] - s[j] * h[i, p[j]]
    vt[i, j] = c[j] * v[i, j] - s[j] * v[i, p[j]]


@wp.kernel
def jacobi_left(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    tmp: wp.array2d[wp.float64],
    vt: wp.array2d[wp.float64],
    c: wp.array[wp.float64],
    s: wp.array[wp.float64],
    p: wp.array[wp.int32],
):
    i, j = wp.tid()
    h[i, j] = c[i] * tmp[i, j] - s[i] * tmp[p[i], j]
    v[i, j] = vt[i, j]


@wp.kernel
def sort_diagonal(
    h: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    order: wp.array[wp.int32],
    which: int,
):
    m = h.shape[0]
    for i in range(m):
        values[i] = h[i, i]
        order[i] = i
    for i in range(m):
        best = i
        for j in range(i + 1, m):
            a = values[order[j]]
            b = values[order[best]]
            if which == 0 and a < b:
                best = j
            if which == 1 and a > b:
                best = j
            if which == 2 and wp.abs(a) > wp.abs(b):
                best = j
            if which == 3 and wp.abs(a) < wp.abs(b):
                best = j
        tmp = order[i]
        order[i] = order[best]
        order[best] = tmp


@wp.kernel
def loop_update(
    converged: wp.array[wp.int32],
    status: wp.array[wp.int32],
    iterations: wp.array[wp.int32],
    running: wp.array[wp.int32],
    k: int,
    maxiter: int,
):
    iterations[0] += 1
    running[0] = 0
    if converged[0] < k and iterations[0] < maxiter and status[0] == 0:
        running[0] = 1


@wp.kernel
def rayleigh(
    x: wp.array2d[wp.float64],
    ax: wp.array2d[wp.float64],
    values: wp.array2d[wp.float64],
):
    j, chunk = wp.tid()
    a = wp.tile_load(x[j], shape=256, offset=chunk * 256)
    b = wp.tile_load(ax[j], shape=256, offset=chunk * 256)
    wp.tile_atomic_add(values[j], wp.tile_sum(wp.tile_map(wp.mul, a, b)))


@wp.kernel
def identity_order(order: wp.array[wp.int32]):
    i = wp.tid()
    order[i] = i


@wp.kernel
def rotation_coefficients(
    v: wp.array2d[wp.float64], order: wp.array[wp.int32], coeff: wp.array2d[wp.float64]
):
    j, r = wp.tid()
    coeff[j, r] = v[r, order[j]]


@wp.kernel
def mark_breakdown(
    norm: wp.array2d[wp.float64],
    reference: wp.array2d[wp.float64],
    flag: wp.array[wp.int32],
):
    flag[0] = 0
    if norm[0, 0] <= wp.float64(1.0e-26) * reference[0, 0]:
        flag[0] = 1


@wp.kernel
def fallback_seed(q: wp.array2d[wp.float64], flag: wp.array[wp.int32], j: int):
    i = wp.tid()
    if flag[0] != 0:
        state = wp.rand_init(971 + j, i)
        q[j, i] = wp.float64(wp.randn(state))


@wp.kernel
def fallback_dots(
    q: wp.array2d[wp.float64],
    dots: wp.array2d[wp.float64],
    flag: wp.array[wp.int32],
    j: int,
):
    r, chunk = wp.tid()
    if flag[0] != 0:
        a = wp.tile_load(q[r], shape=256, offset=chunk * 256)
        b = wp.tile_load(q[j], shape=256, offset=chunk * 256)
        wp.tile_atomic_add(dots[r], wp.tile_sum(wp.tile_map(wp.mul, a, b)))


@wp.kernel
def fallback_subtract(
    q: wp.array2d[wp.float64],
    dots: wp.array2d[wp.float64],
    flag: wp.array[wp.int32],
    j: int,
):
    i = wp.tid()
    if flag[0] != 0:
        val = q[j, i]
        for r in range(j):
            val -= q[r, i] * dots[r, 0]
        q[j, i] = val


@wp.kernel
def csr_mv_cooperative(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    a: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
    shift: wp.float64,
    sign: wp.float64,
):
    j, i, lane = wp.tid()
    val = wp.float64(0.0)
    for e in range(offsets[i] + lane, offsets[i + 1], 32):
        val += a[e] * x[j, columns[e]]
    total = wp.tile_sum(wp.tile(val))
    value = wp.tile_extract(total, 0)
    if lane == 0:
        y[j, i] = wp.float64(sign) * value + wp.float64(shift) * x[j, i]


@wp.kernel
def sort_extra(
    values: wp.array[wp.float64],
    work: wp.array[wp.int32],
    order: wp.array[wp.int32],
    which: int,
    target: wp.float64,
):
    n = values.shape[0]
    for i in range(n):
        work[i] = i
    for i in range(n):
        best = i
        for j in range(i + 1, n):
            a = values[work[j]]
            b = values[work[best]]
            if which == 4 and a > b:
                best = j
            if which == 5 and wp.abs(a - target) < wp.abs(b - target):
                best = j
        temp = work[i]
        work[i] = work[best]
        work[best] = temp
    for i in range(n):
        idx = i
        if which == 4:
            idx = i // 2
            if i % 2 == 1:
                idx = n - 1 - i // 2
        order[i] = work[idx]
