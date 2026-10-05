import numpy as np
import warp as wp

from warpack import SparseOperator
from warpack.fem import assemble_rest, mass_normalized


def dense(a):
    n = a.shape[0]
    x = wp.array(np.eye(n), dtype=wp.float64, device=a.device)
    y = wp.empty_like(x)
    SparseOperator(a).apply(x, y)
    return y.numpy().T


def test_rest_hessian_and_mass():
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    h, m, mass = assemble_rest(x, np.array([[0, 1, 2, 3]]), young=10.0, density=3.0)
    a = dense(h)
    np.testing.assert_allclose(a, a.T, atol=1.0e-14)
    eig = np.linalg.eigvalsh(a)
    np.testing.assert_allclose(eig[:6], 0, atol=1.0e-12)
    assert eig[6] > 0.1
    np.testing.assert_allclose(mass.numpy(), 0.125)
    np.testing.assert_allclose(dense(m), np.eye(12) * 0.125)
    np.testing.assert_allclose(dense(mass_normalized(h, mass)), a / 0.125)
    # Independent energy finite-difference Hessian, including mixed entries.
    mu = 10 / 2.6
    lam = 10 * 0.3 / (1.3 * 0.4)
    ls = lam + mu
    alpha = 1 + mu / ls

    def energy(u):
        p = x + u.reshape(4, 3)
        f = (p[1:] - p[0]).T
        return (mu / 2 * (np.sum(f * f) - 3) + ls / 2 * (np.linalg.det(f) - alpha) ** 2) / 6

    eps = 2.0e-4
    eye = np.eye(12) * eps
    fd = np.empty((12, 12))
    for i in range(12):
        for j in range(12):
            fd[i, j] = (
                energy(eye[i] + eye[j])
                - energy(eye[i] - eye[j])
                - energy(-eye[i] + eye[j])
                + energy(-eye[i] - eye[j])
            ) / (4 * eps * eps)
    np.testing.assert_allclose(a, fd, atol=2.0e-6)


def test_assembly_graph_replay():
    from warpack.fem import RestElasticity

    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    model = RestElasticity(x, np.array([[0, 1, 2, 3]]))
    h = dense(model.hessian)
    m = model.mass.numpy()
    model.update()
    with wp.ScopedCapture(device="cuda:0") as cap:
        model.update()
    wp.copy(model.positions, wp.array(x * 2, dtype=wp.vec3d))
    for _ in range(2):
        wp.capture_launch(cap.graph)
        np.testing.assert_allclose(dense(model.hessian), h * 2, atol=1.0e-10)
        np.testing.assert_allclose(model.mass.numpy(), m * 8, atol=1.0e-10)
        assert model.invalid.numpy()[0] == 0
