"""Installed eight-row rotation preserves both products and full captured solves."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import warp as wp

from warpack import BlockLOBPCG
from warpack.block_products import _use_wide_paired_rotation, wide_paired_ritz_rotation


@pytest.mark.parametrize(
    "n,width,arch,cuda,enabled,expected",
    [
        (32768, 24, 89, True, True, True),
        (32768, 256, 89, True, True, True),
        (100000, 104, 89, True, True, True),
        (32767, 104, 89, True, True, False),
        (32768, 23, 89, True, True, False),
        (32768, 257, 89, True, True, False),
        (32768, 104, 80, True, True, False),
        (32768, 104, 90, True, True, False),
        (32768, 104, 89, False, True, False),
        (32768, 104, 89, True, False, False),
    ],
)
def test_static_scope_guard(n, width, arch, cuda, enabled, expected):
    config = SimpleNamespace(
        n=n,
        b=width,
        device=SimpleNamespace(arch=arch, is_cuda=cuda),
        _wide_paired_rotation_enabled=enabled,
    )
    assert bool(_use_wide_paired_rotation(config)) == expected


def test_installed_kernel_matches_qualified_prototype_arithmetic():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "benchmarks/experimental/paired_rotation_tiles.py").read_text())
    prototype = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compute"
    )

    class Constants(ast.NodeTransformer):
        def visit_Name(self, node):
            values = {"nr": 8, "nc": 32, "nk": 8}
            return ast.Constant(values[node.id]) if node.id in values else node

    prototype = Constants().visit(prototype)
    tree = ast.parse((root / "warpack/block_products.py").read_text())
    installed = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "wide_paired_ritz_rotation"
    )
    assert [ast.dump(node) for node in prototype.body] == [
        ast.dump(node) for node in installed.body
    ]


@pytest.mark.parametrize(
    "width,search,n", [(7, 3, 37), (13, 0, 63), (24, 48, 257), (104, 8, 1031), (204, 408, 257)]
)
def test_installed_products_strides_changed_scales_and_padding(monkeypatch, width, search, n):
    from tests import test_paired_rotation_tiles as fixture

    def installed(sx, su, cx, cu, x, p, **kwargs):
        wp.launch_tiled(
            wide_paired_ritz_rotation,
            dim=((x.shape[0] + 7) // 8, (x.shape[1] + 31) // 32),
            inputs=[sx, su, cx, cu, x, p],
            block_dim=128,
            device=x.device,
        )

    monkeypatch.setattr(fixture.candidate, "rotation", installed)
    fixture.test_split_strided_partial_tiles_changed_replay_and_sentinels(
        (8, 32, 8, 128), width, search, n
    )


@pytest.mark.parametrize("history,compact", [(False, False), (True, False), (True, True)])
def test_installed_coupled_constraints_clusters_changed_replay(
    monkeypatch, tmp_path, history, compact
):
    from tests import test_certified_whitening, test_first_whitening

    if wp.get_device().arch != 89:
        pytest.skip("installed dispatch is qualified on Ada")

    class Installed(BlockLOBPCG):
        _wide_paired_rotation_minimum = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, compact_projected_graphs=compact)

    monkeypatch.setattr(test_first_whitening, "AllSizeInstalledWhitening", Installed)
    test_certified_whitening.test_coupled_mass_constraints_clusters_and_changed_problem(
        tmp_path, "installed", history, problem_shape=(336, 20, 104)
    )
    assert "wide_paired_ritz_rotation" in (tmp_path / "certified.dot").read_text()


@pytest.mark.parametrize("active", [False, True])
def test_profiler_accounts_for_installed_rotation(monkeypatch, active):
    from tests import test_legacy_block_profile as fixture

    if wp.get_device().arch != 89:
        pytest.skip("installed dispatch is qualified on Ada")

    class Installed(BlockLOBPCG):
        _wide_paired_rotation_minimum = 1

    monkeypatch.setattr(fixture, "BlockLOBPCG", Installed)
    fixture.test_static_and_active_steps_preserve_complete_solves_and_wrappers(104, active)
