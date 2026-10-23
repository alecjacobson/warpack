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
@pytest.mark.parametrize("search", [False, True])
def test_block_finalizer_sees_final_pairs_once_per_replay(adaptive, tmp_path, search):
    a = matrix(np.diag(np.arange(1.0, 31.0)))
    original = wp.clone(a.values)
    solver = BlockLOBPCG(
        SparseOperator(a),
        3,
        width=6,
        tol=1e-10,
        reuse_operator_images=search,
        search_bucket_size=2 if search else 0,
    )
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


@wp.kernel
def prepare_block(
    source: wp.array2d[wp.float64], target: wp.array2d[wp.float64], calls: wp.array[wp.int32]
):
    row, i = wp.tid()
    target[row, i] = source[row, i]
    if row == 0 and i == 0:
        calls[0] += 1


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("search", [False, True])
def test_initializer_precedes_normalization_and_finalizer(adaptive, tmp_path, search):
    a = matrix(np.diag(np.arange(1.0, 31.0)))
    source = wp.array(np.eye(30)[:3].copy(), dtype=wp.float64)
    starts = wp.zeros_like(source)
    calls, final_calls = wp.zeros(1, dtype=wp.int32), wp.zeros(1, dtype=wp.int32)
    solver = BlockLOBPCG(
        SparseOperator(a),
        3,
        width=3,
        initial_vectors=starts,
        tol=1e-10,
        reuse_operator_images=search,
        search_bucket_size=1 if search else 0,
    )
    output, state = wp.empty(3, dtype=wp.float64), wp.empty(2, dtype=wp.int32)
    recording = []

    def initializer():
        recording.append("before")
        wp.launch(prepare_block, source.shape, [source, starts, calls])

    def finalizer():
        recording.append("after")
        wp.launch(
            read_final_block,
            3,
            [solver.eigenvalues, solver.converged, solver.status, output, state, final_calls],
        )

    with pytest.raises(ValueError, match="positive"):
        solver.capture(0, initializer=initializer)
    assert recording == []
    graph = solver.capture(1, adaptive=adaptive, initializer=initializer, finalizer=finalizer)
    assert recording == ["before", "after", "before", "after"]
    calls.zero_()
    final_calls.zero_()
    for count, order in enumerate((np.arange(30), np.roll(np.arange(30), 7)), 1):
        values = np.empty(30)
        values[order] = np.arange(1.0, 31.0)
        a.values.assign(values)
        # Nonunit inputs ensure preparation precedes solver orthonormalization.
        source.assign((np.eye(30)[order[:3]] * np.array([2, 3, 4])[:, None]).copy())
        starts.zero_()
        wp.capture_launch(graph)
        np.testing.assert_array_equal(state.numpy(), [3, 0])
        np.testing.assert_allclose(output.numpy(), [-1, -2, -3], atol=1e-12)
        assert calls.numpy()[0] == count and final_calls.numpy()[0] == count
        assert len(recording) == 4
    dot = tmp_path / "prepare-solve-finalize.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    assert not any(token in dot.read_text() for token in ("HOST", "HtoD", "DtoH"))
