"""Device postprocessing follows the complete block loop once per replay."""

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import BlockLOBPCG, SparseOperator


@wp.kernel
def read_final_block(
    values: wp.array[wp.float64],
    converged: wp.array[wp.int32],
    status: wp.array[wp.int32],
    output: wp.array[wp.float64],
    state: wp.array[wp.int32],
    calls: wp.array[wp.int32],
):
    i = wp.tid()
    output[i] = -values[i]
    if i == 0:
        state[0] = converged[0]
        state[1] = status[0]
        calls[0] += 1


@pytest.mark.parametrize("adaptive", [False, True])
def test_block_finalizer_sees_final_pairs_once_per_replay(adaptive, tmp_path):
    a = matrix(np.diag(np.arange(1.0, 31.0)))
    original = wp.clone(a.values)
    solver = BlockLOBPCG(SparseOperator(a), 3, width=6, tol=1e-10)
    output = wp.empty(3, dtype=wp.float64)
    state = wp.empty(2, dtype=wp.int32)
    calls = wp.zeros(1, dtype=wp.int32)
    recording_calls = []

    def finalizer():
        recording_calls.append(True)
        wp.launch(
            read_final_block,
            3,
            [solver.eigenvalues, solver.converged, solver.status, output, state, calls],
        )

    graph = solver.capture(100, adaptive=adaptive, finalizer=finalizer)
    assert len(recording_calls) == 2  # Warmup and recording, never replay.
    calls.zero_()
    for count in (1, 2, 3):
        wp.capture_launch(graph)
        np.testing.assert_allclose(output.numpy(), [-1.0, -2.0, -3.0], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(state.numpy(), [3, 0])
        assert calls.numpy()[0] == count
        assert len(recording_calls) == 2
    # Finalization must also run on failure and leave its status visible.
    a.values.fill_(float("nan"))
    wp.capture_launch(graph)
    observed = state.numpy()
    assert observed[0] < 3 and observed[1] != 0
    np.testing.assert_array_equal(observed, [solver.converged.numpy()[0], solver.status.numpy()[0]])
    assert calls.numpy()[0] == 4
    wp.copy(a.values, original)
    wp.capture_launch(graph)
    np.testing.assert_array_equal(state.numpy(), [3, 0])
    np.testing.assert_allclose(output.numpy(), [-1.0, -2.0, -3.0], atol=1e-9, rtol=0)
    assert calls.numpy()[0] == 5 and len(recording_calls) == 2
    dot = tmp_path / "block-finalizer.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    assert not any(token in dot.read_text() for token in ("HOST", "HtoD", "DtoH"))
