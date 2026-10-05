"""Device complex Hessenberg reduction and shifted QR for small projected systems.

This FP64 reference implementation favors coverage and diagnostics over speed;
large sparse operators are handled by the outer Arnoldi iteration.
"""

import warp as wp

from .hermitian import cdot, cmul, norm2


@wp.func
def conj(a: wp.vec2d):
    return wp.vec2d(a[0], -a[1])


@wp.func
def cabs(a: wp.vec2d):
    return wp.sqrt(wp.dot(a, a))


@wp.func
def cdiv(a: wp.vec2d, b: wp.vec2d):
    return cmul(a, conj(b)) / wp.max(wp.dot(b, b), wp.float64(1.0e-300))


@wp.func
def csqrt(a: wp.vec2d):
    r = cabs(a)
    x = wp.sqrt(wp.max(wp.float64(0.0), (r + a[0]) * wp.float64(0.5)))
    y = wp.sqrt(wp.max(wp.float64(0.0), (r - a[0]) * wp.float64(0.5)))
    if a[1] < wp.float64(0.0):
        y = -y
    return wp.vec2d(x, y)


@wp.kernel
def schur(
    h: wp.array2d[wp.vec2d],
    q: wp.array2d[wp.vec2d],
    vectors: wp.array2d[wp.vec2d],
    values: wp.array[wp.vec2d],
    u: wp.array[wp.vec2d],
    cs: wp.array[wp.float64],
    ss: wp.array[wp.vec2d],
    status: wp.array[wp.int32],
    maxiter: int,
):
    n = h.shape[0]
    for i in range(n):
        for j in range(n):
            q[i, j] = wp.vec2d(0.0)
            if i == j:
                q[i, j] = wp.vec2d(1.0, 0.0)
    # Unitary Householder similarity to upper Hessenberg form.
    for k in range(n - 2):
        r = wp.float64(0.0)
        for i in range(k + 1, n):
            r += norm2(h[i, k])
        r = wp.sqrt(r)
        if r > wp.float64(1.0e-150):
            a = h[k + 1, k]
            phase = wp.vec2d(1.0, 0.0)
            if cabs(a) > wp.float64(1.0e-150):
                phase = a / cabs(a)
            for i in range(n):
                u[i] = wp.vec2d(0.0)
            for i in range(k + 1, n):
                u[i] = h[i, k]
            u[k + 1] += phase * r
            r = wp.float64(0.0)
            for i in range(k + 1, n):
                r += norm2(u[i])
            r = wp.sqrt(r)
            for i in range(k + 1, n):
                u[i] = u[i] / r
            for j in range(k, n):
                val = wp.vec2d(0.0)
                for i in range(k + 1, n):
                    val += cdot(u[i], h[i, j])
                for i in range(k + 1, n):
                    h[i, j] -= wp.float64(2.0) * cmul(u[i], val)
            for i in range(n):
                val = wp.vec2d(0.0)
                vq = wp.vec2d(0.0)
                for j in range(k + 1, n):
                    val += cmul(h[i, j], u[j])
                    vq += cmul(q[i, j], u[j])
                for j in range(k + 1, n):
                    h[i, j] -= wp.float64(2.0) * cmul(val, conj(u[j]))
                    q[i, j] -= wp.float64(2.0) * cmul(vq, conj(u[j]))
            for i in range(k + 2, n):
                h[i, k] = wp.vec2d(0.0)
    end = n - 1
    iteration = int(0)
    while end > 0 and iteration < maxiter:
        threshold = wp.float64(2.0e-15) * (
            cabs(h[end - 1, end - 1]) + cabs(h[end, end])
        ) + wp.float64(1.0e-300)
        if cabs(h[end, end - 1]) <= threshold:
            h[end, end - 1] = wp.vec2d(0.0)
            end -= 1
        else:
            a = h[end - 1, end - 1]
            b = h[end - 1, end]
            c = h[end, end - 1]
            d = h[end, end]
            delta = (a - d) * wp.float64(0.5)
            root = csqrt(cmul(delta, delta) + cmul(b, c))
            mu = (a + d) * wp.float64(0.5) + root
            other = (a + d) * wp.float64(0.5) - root
            if cabs(other - d) < cabs(mu - d):
                mu = other
            if iteration % 40 == 39:
                mu = d + wp.vec2d(cabs(c) * wp.float64(0.75), 0.0)
            for i in range(end + 1):
                h[i, i] -= mu
            # QR factorization with complex Givens rotations.
            for i in range(end):
                a = h[i, i]
                b = h[i + 1, i]
                aa = cabs(a)
                bb = cabs(b)
                r = wp.sqrt(aa * aa + bb * bb)
                co = wp.float64(1.0)
                si = wp.vec2d(0.0)
                if r > wp.float64(1.0e-150):
                    co = aa / r
                    if aa > wp.float64(1.0e-150):
                        si = cmul(a / aa, conj(b)) / r
                    else:
                        si = conj(b) / bb
                cs[i] = co
                ss[i] = si
                for j in range(i, n):
                    x = h[i, j]
                    y = h[i + 1, j]
                    h[i, j] = co * x + cmul(si, y)
                    h[i + 1, j] = -cmul(conj(si), x) + co * y
                h[i + 1, i] = wp.vec2d(0.0)
            # RQ similarity, and accumulation of Schur vectors.
            for j in range(end):
                co = cs[j]
                si = ss[j]
                for i in range(n):
                    x = h[i, j]
                    y = h[i, j + 1]
                    h[i, j] = co * x + cmul(conj(si), y)
                    h[i, j + 1] = -cmul(si, x) + co * y
                    x = q[i, j]
                    y = q[i, j + 1]
                    q[i, j] = co * x + cmul(conj(si), y)
                    q[i, j + 1] = -cmul(si, x) + co * y
            for i in range(end + 1):
                h[i, i] += mu
            iteration += 1
    status[0] = 0
    if end > 0:
        status[0] = 2
    scale = wp.float64(0.0)
    for i in range(n):
        values[i] = h[i, i]
        scale = wp.max(scale, cabs(h[i, i]))
    # Eigenvectors of triangular Schur form, then transform by accumulated Q.
    for j in range(n):
        for i in range(n):
            u[i] = wp.vec2d(0.0)
        u[j] = wp.vec2d(1.0, 0.0)
        for step in range(j):
            i = j - 1 - step
            val = wp.vec2d(0.0)
            for k in range(i + 1, j + 1):
                val += cmul(h[i, k], u[k])
            den = h[i, i] - values[j]
            if cabs(den) < wp.float64(1.0e-14) * scale:
                den = wp.vec2d(wp.max(wp.float64(1.0e-14) * scale, wp.float64(1.0e-300)), 0.0)
            u[i] = -cdiv(val, den)
        r = wp.float64(0.0)
        for i in range(n):
            val = wp.vec2d(0.0)
            for k in range(n):
                val += cmul(q[i, k], u[k])
            vectors[i, j] = val
            r += norm2(val)
        r = wp.sqrt(wp.max(r, wp.float64(1.0e-300)))
        for i in range(n):
            vectors[i, j] = vectors[i, j] / r
