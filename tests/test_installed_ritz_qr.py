"""Installed QR reuse guards, numerical fallback and changing physical pencils."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import warp as wp

from tests import (
    test_certified_whitening,
    test_first_whitening,
    test_metric_qr_reuse,
    test_triangular_rotate,
)
from warpack import BlockLOBPCG, _ritz_qr
from warpack.block import Identity


class QualifiedBlock(BlockLOBPCG):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **(
                {"reuse_operator_images": True, "symmetric_products": True}
                | kwargs
                | {"gram_chunk": 4096}
            ),
        )
        # Existing numerical regressions also check explicit usage counters.
        self.metric_reuse_usage = self._ritz_qr_usage


@pytest.mark.parametrize(
    "override,selected",
    [
        ({}, True),
        ({"b": 24}, False),
        ({"b": 95}, False),
        ({"b": 96}, True),
        ({"b": 256}, True),
        ({"b": 257}, False),
        ({"gram_chunk": 256}, False),
        ({"tol": 1e-12}, False),
        ({"tol": 1e-11}, True),
        ({"reuse_operator_images": False}, False),
        ({"symmetric_products": False}, False),
        ({"_ritz_qr_enabled": False}, False),
        ({"device": SimpleNamespace(is_cuda=True, arch=90)}, False),
        ({"device": SimpleNamespace(is_cuda=True, arch=80)}, False),
        ({"device": SimpleNamespace(is_cuda=False)}, False),
    ],
)
def test_metadata_guards(monkeypatch, override, selected):
    monkeypatch.setattr(wp, "is_conditional_graph_supported", lambda: True)
    metadata = dict(
        b=104,
        device=SimpleNamespace(is_cuda=True, arch=89),
        gram_chunk=4096,
        tol=1e-8,
        reuse_operator_images=True,
        symmetric_products=True,
        mass=Identity(400, None),
    )
    assert bool(_ritz_qr.qualified(SimpleNamespace(**(metadata | override)))) == selected
    monkeypatch.setattr(wp, "is_conditional_graph_supported", lambda: False)
    assert not _ritz_qr.qualified(SimpleNamespace(**metadata))


def test_kernel_arithmetic_matches_qualified_prototype():
    root = Path(__file__).resolve().parents[1]
    trees = [
        ast.parse((root / name).read_text())
        for name in ["warpack/_ritz_qr.py", "benchmarks/experimental/metric_qr_reuse.py"]
    ]
    for name in ["metric_defect", "choose_qr"]:
        functions = [
            next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)
            for t in trees
        ]
        assert ast.dump(functions[0]) == ast.dump(functions[1])


@pytest.mark.parametrize("width", [24, 95, 96, 256, 257])
def test_no_control_arrays_outside_qualified_range(width):
    solver = QualifiedBlock(Identity(3 * width + 7, wp.get_device()), 4, width)
    selected = wp.get_device().arch == 89 and 96 <= width <= 256
    assert (solver._ritz_qr_reuse is not None) == selected
    assert (solver._ritz_qr_usage is not None) == selected


@pytest.mark.parametrize("wrapped", [False, True])
def test_custom_mass_no_controls(wrapped):
    class Custom(Identity):
        pass

    mass = (Identity if wrapped else Custom)(400, wp.get_device())
    if wrapped:
        original = mass.apply
        mass.apply = lambda x, y: original(x, y)
    solver = QualifiedBlock(Identity(400, wp.get_device()), 4, 104, mass=mass)
    assert solver._ritz_qr_reuse is None and solver._ritz_qr_usage is None


def test_installed_reuse_fallback_invalid_and_recovery(monkeypatch):
    if wp.get_device().arch != 89:
        pytest.skip("installed reuse qualified on Ada")
    monkeypatch.setattr(test_metric_qr_reuse, "MetricQRReuseBlock", QualifiedBlock)
    test_metric_qr_reuse.test_reuse_fallback_and_failure_recovery_on_one_graph(1e-13, width=104)


@pytest.mark.parametrize("width", [96, 104, 204, 256])
@pytest.mark.parametrize("cached", [False, True])
def test_complete_coupled_scaled_stage(width, cached):
    test_triangular_rotate.test_orthogonalization_with_changed_mass_scaling_and_cached_images(
        width, cached, QualifiedBlock
    )


@pytest.mark.parametrize("history", [False, True])
def test_installed_changed_coupled_constraints_clusters(monkeypatch, tmp_path, history):
    monkeypatch.setattr(test_first_whitening, "AllSizeInstalledWhitening", QualifiedBlock)
    test_certified_whitening.test_coupled_mass_constraints_clusters_and_changed_problem(
        tmp_path, "installed", history, problem_shape=(336, 20, 104)
    )


@pytest.mark.parametrize("history", [False, True])
def test_branch_controls_are_shared_and_owned(history):
    solver = QualifiedBlock(
        Identity(400, wp.get_device()), 4, 104, search_bucket_size=4, keep_history=history
    )
    if wp.get_device().arch != 89:
        assert solver._ritz_qr_usage is None
        return
    for branch in [solver, *solver._search.buckets]:
        helper = branch._ritz_qr_reuse
        assert helper.owner is branch and helper.usage.ptr == solver._ritz_qr_usage.ptr
        assert helper.bound.ptr == solver._ritz_qr_reuse.bound.ptr
        assert helper.invalid.ptr == solver._ritz_qr_reuse.invalid.ptr
        assert helper.fallback.ptr == solver._ritz_qr_reuse.fallback.ptr
