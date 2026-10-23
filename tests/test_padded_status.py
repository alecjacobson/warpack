"""A failed normalization must not be mistaken for legitimate exact padding."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from tests.test_first_whitening import AllSizePaddedWhitening
from warpack import SparseOperator
from warpack.block import Identity


@pytest.mark.parametrize("failure", ["negative_mass", "prior_failure"])
def test_failed_scaling_routes_to_canonical_and_recovers_on_same_graph(failure):
    n = 32
    diagonal = np.ones(n)
    diagonal[0] = -1 if failure == "negative_mass" else 1
    mass = matrix(sp.diags(diagonal, format="csr"))
    solver = AllSizePaddedWhitening(Identity(n, wp.get_device()), 2, 4, mass=SparseOperator(mass))
    source = np.eye(n)[:8].copy()
    injected_status = wp.zeros(1, dtype=wp.int32)
    solver.su.assign(source)
    solver.whiten(solver.su, solver.u1, normalize=True)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.copy(solver.status, injected_status)
        solver._first_whitening_usage.zero_()
        solver.whiten(solver.su, solver.u1, normalize=True)
    records = []
    for negative in [True, False]:
        diagonal[0] = -1 if negative and failure == "negative_mass" else 1
        injected_status.fill_(7 if negative and failure == "prior_failure" else 0)
        mass.values.assign(diagonal)
        solver.su.assign(source)
        wp.capture_launch(capture.graph)
        records.append(
            dict(
                status=int(solver.status.numpy()[0]),
                usage=solver._first_whitening_usage.numpy().tolist(),
                rank=int(solver.rank.numpy()[0]),
            )
        )
        if not negative:
            np.testing.assert_array_equal(solver.u1.numpy(), source)
            assert np.all(solver.active_u.numpy() == 1)
    print(records)
    assert [r["status"] for r in records] == [1 if failure == "negative_mass" else 7, 0]
    assert [r["rank"] for r in records] == [7 if failure == "negative_mass" else 8, 8]
    assert records[1]["usage"] == [1, 0, 0]
    assert records[0]["usage"] == [0, 1, 1]
