"""Wider-precision diagnostics must evaluate the original assembled pencil."""

import numpy as np
import pytest

from benchmarks.applications.damped_pencil import consistent_chain
from benchmarks.experimental.pencil_refinement import extended_residual


@pytest.mark.parametrize("kind", ["nonproportional", "gyroscopic"])
def test_extended_original_residual_and_block_split(kind):
    case = consistent_chain(12, 6, kind)
    for noise in [0.0, 1e-5]:
        vectors = case.right.copy()
        vectors[0, 0] += noise * (1.0 + 0.5j)
        ax, bx = case.a @ vectors, case.b @ vectors
        residual = ax - bx * case.values
        denominator = np.linalg.norm(ax, axis=0) + np.maximum(1, abs(case.values)) * np.linalg.norm(
            bx, axis=0
        )
        result = extended_residual(case, case.values, vectors)
        half = len(vectors) // 2
        for name, block in [
            ("residual", residual),
            ("first_block", residual[:half]),
            ("second_block", residual[half:]),
        ]:
            expected = np.max(np.linalg.norm(block, axis=0) / denominator)
            np.testing.assert_allclose(result[name], expected, atol=2e-15, rtol=1e-10)
        assert result["mantissa_bits"] >= 52
        assert (result["residual"] < 1e-12) == (noise == 0)
