import numpy as np
import pytest
import warp as wp
import warp.sparse as ws

from warpack import BucklingEigensolver, CayleyEigensolver, CuDSSInverse, SparseOperator


def diagonal(values):
    ids = wp.array(np.arange(len(values)), dtype=wp.int32)
    return ws.bsr_from_triplets(
        len(values), len(values), ids, ids, wp.array(values, dtype=wp.float64)
    )


@pytest.mark.parametrize("mode", ["buckling", "cayley"])
def test_generalized_transform(mode):
    a = np.arange(1.0, 31.0)
    b = np.ones(30)
    sigma = 7.3
    if mode == "buckling":
        b[1::2] = -1
        sigma = 2.3
    aa, bb = diagonal(a), diagonal(b)
    inv = CuDSSInverse(diagonal(a - sigma * b), 20, mtype="symmetric")
    cls = BucklingEigensolver if mode == "buckling" else CayleyEigensolver
    s = cls(SparseOperator(aa), SparseOperator(bb), inv, 4, sigma, ncv=20, tol=1.0e-10)
    wp.capture_launch(s.capture(60))
    eigen = a / b
    transformed = (
        eigen / (eigen - sigma) if mode == "buckling" else (eigen + sigma) / (eigen - sigma)
    )
    ref = eigen[np.argsort(-abs(transformed))[:4]]
    np.testing.assert_allclose(s.eigenvalues.numpy(), ref, atol=1.0e-9)
    assert s.converged.numpy()[0] == 4, s.residuals.numpy()
    v = s.eigenvectors.numpy().T
    assert (
        np.linalg.norm(a[:, None] * v - b[:, None] * v * s.eigenvalues.numpy(), axis=0).max()
        < 1.0e-9
    )
    inv.close()
