"""Installed register recurrence: dispatch boundaries and physical graph replays."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from tests import test_certified_whitening, test_register_triangular, test_triangular_rotate
from warpack import BlockLOBPCG, SparseOperator
from warpack import kernels as K
from warpack._triangular import register_triangular


class CanonicalBlock(BlockLOBPCG):
    _register_triangular_enabled = False


@pytest.mark.parametrize("width", [95, 96, 97, 255, 256, 257])
@pytest.mark.parametrize("arch", [80, 89, 90, None])
@pytest.mark.parametrize("enabled", [False, True])
def test_dispatch_guards(monkeypatch, width, arch, enabled):
    owner = SimpleNamespace(
        device=SimpleNamespace(is_cuda=arch is not None, arch=arch),
        _register_triangular_enabled=enabled,
    )
    launches = []
    monkeypatch.setattr(wp, "launch", lambda *args, **kwargs: launches.append((args, kwargs)))
    inputs = [SimpleNamespace(shape=(width, 1031)), object()]
    BlockLOBPCG.launch(owner, K.triangular, 1031, inputs)
    assert len(launches) == 1
    args, kwargs = launches[0]
    selected = enabled and arch == 89 and 96 <= width <= 256
    assert args[0] is (register_triangular if selected else K.triangular)
    assert args[1] == 1031 and args[2] is inputs and kwargs["device"] is owner.device
    if selected:
        assert kwargs["block_dim"] == 256


@pytest.mark.parametrize("width", [7, 17, 52, 95, 96, 97, 104, 120, 204, 255, 256, 257])
def test_installed_exact_recurrence(monkeypatch, width):
    monkeypatch.setattr(test_register_triangular, "register_triangular", register_triangular)
    test_register_triangular.test_exact_recurrence_strides_poisoned_upper_and_changed_replay(width)


def test_installed_large_even_replay(monkeypatch):
    monkeypatch.setattr(test_register_triangular, "register_triangular", register_triangular)
    test_register_triangular.test_large_even_identity_replay()


@pytest.mark.parametrize("width", [96, 97, 120, 204, 256])
@pytest.mark.parametrize("cached", [False, True])
def test_installed_complete_stage_against_canonical(monkeypatch, width, cached):
    monkeypatch.setattr(test_triangular_rotate, "BlockLOBPCG", CanonicalBlock)
    test_triangular_rotate.test_orthogonalization_with_changed_mass_scaling_and_cached_images(
        width, cached, BlockLOBPCG
    )


@pytest.mark.parametrize("history", [False, True])
def test_installed_wide_coupled_constraints_clusters(tmp_path, history):
    test_certified_whitening.test_coupled_mass_constraints_clusters_and_changed_problem(
        tmp_path, "installed", history, problem_shape=(336, 20, 104)
    )


@pytest.mark.parametrize("width", [96, 204])
def test_invalid_qr_status_and_same_graph_recovery(width):
    n = 3 * width + 7
    rng = np.random.default_rng(173)
    source = rng.normal(size=(width, n))
    mass = matrix(sp.eye(n, format="csr"))
    solver = BlockLOBPCG(SparseOperator(mass), 4, width, mass=SparseOperator(mass))
    solver.sx.assign(source)
    solver.orthogonalize_x()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.status.zero_()
        solver.orthogonalize_x()
    for kind in ["negative_mass", "healthy", "nonfinite", "healthy", "zero", "healthy"]:
        mass.values.fill_(-1.0 if kind == "negative_mass" else 1.0)
        current = source.copy()
        if kind == "nonfinite":
            current[0, 0] = np.nan
        elif kind == "zero":
            current[:] = 0
        solver.sx.assign(current)
        wp.capture_launch(capture.graph)
        assert (solver.status.numpy()[0] == 0) == (kind == "healthy")
        if kind == "healthy":
            result = solver.sx.numpy()
            assert np.linalg.norm(result @ result.T - np.eye(width)) < 1e-10
