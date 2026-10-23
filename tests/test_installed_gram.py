"""Installed Gram dispatch, original products and changed generalized replays."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import (
    test_certified_whitening,
    test_first_whitening,
    test_gram_tiles,
    test_installed_register_triangular,
    test_triangular_rotate,
)
from warpack import BlockLOBPCG, _gram, block


class WideBlock(BlockLOBPCG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **(kwargs | {"gram_chunk": 4096}))


class CanonicalBlock(WideBlock):
    _wide_gram_enabled = False


@pytest.mark.parametrize("width", [23, 24, 25, 95, 96, 255, 256, 257])
@pytest.mark.parametrize("arch", [None, 80, 89, 90])
@pytest.mark.parametrize("chunk", [256, 4096])
@pytest.mark.parametrize("enabled", [False, True])
def test_dispatch_guards(monkeypatch, width, arch, chunk, enabled):
    owner = SimpleNamespace(
        b=width,
        device=SimpleNamespace(is_cuda=arch is not None, arch=arch),
        gram_chunk=chunk,
        symmetric_products=True,
        _wide_gram_enabled=enabled,
    )
    calls = []
    monkeypatch.setattr(
        block, "gram", lambda *args, **kwargs: calls.append(("canonical", args, kwargs))
    )
    monkeypatch.setattr(_gram, "gram", lambda *args, **kwargs: calls.append(("wide", args, kwargs)))
    inputs = [object(), object(), object()]
    BlockLOBPCG.gram(owner, *inputs, symmetric=True)
    selected = enabled and arch == 89 and chunk == 4096 and 24 <= width <= 256
    expected = dict(symmetric=True) if selected else dict(symmetric=True, chunk=chunk)
    assert calls == [("wide" if selected else "canonical", tuple(inputs), expected)]
    owner.symmetric_products = False
    BlockLOBPCG.gram(owner, *inputs, symmetric=True)
    assert calls[-1][2]["symmetric"] is False


def test_numerical_kernel_matches_qualified_prototype():
    root = Path(__file__).resolve().parents[1]
    trees = [
        ast.parse((root / name).read_text())
        for name in ["warpack/_gram.py", "benchmarks/experimental/gram_tiles.py"]
    ]
    kernels = [
        next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == "compute")
        for t in trees
    ]
    assert ast.dump(kernels[0]) == ast.dump(kernels[1])


@pytest.mark.parametrize("symmetric", [False, True])
def test_strided_changed_products(monkeypatch, symmetric):
    def installed(x, y, out, **kwargs):
        _gram.gram(x, y, out, symmetric=kwargs["symmetric"])

    monkeypatch.setattr(test_gram_tiles, "gram", installed)
    test_gram_tiles.test_strided_rectangular_scaled_changed_replay(16, 128, symmetric)


@pytest.mark.parametrize("width", [24, 52, 96, 204, 256])
@pytest.mark.parametrize("cached", [False, True])
def test_full_two_pass_stage(monkeypatch, width, cached):
    monkeypatch.setattr(test_triangular_rotate, "BlockLOBPCG", CanonicalBlock)
    test_triangular_rotate.test_orthogonalization_with_changed_mass_scaling_and_cached_images(
        width, cached, WideBlock
    )


@pytest.mark.parametrize("history", [False, True])
def test_wide_changed_coupled_pencil(monkeypatch, tmp_path, history):
    monkeypatch.setattr(test_first_whitening, "AllSizeInstalledWhitening", WideBlock)
    test_certified_whitening.test_coupled_mass_constraints_clusters_and_changed_problem(
        tmp_path, "installed", history, problem_shape=(336, 20, 104)
    )


@pytest.mark.parametrize("width", [24, 96, 204])
def test_invalid_qr_and_same_graph_recovery(monkeypatch, width):
    monkeypatch.setattr(test_installed_register_triangular, "BlockLOBPCG", WideBlock)
    test_installed_register_triangular.test_invalid_qr_status_and_same_graph_recovery(width)
