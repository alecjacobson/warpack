"""Changing geometry must update mass, CG and refitted factor values on replay."""

import numpy as np
import pytest
import warp as wp
from warp.optim import linear

from benchmarks.applications.fem_shapes import ShapeReplay, shape_family


def test_inner_work_reports_all_buckets_after_changed_replay():
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    states = shape_family(2, 6, True)
    driver = ShapeReplay(states, 12, refit=True)
    for _, p, _ in states:
        driver.update(p)
        driver.solve()
        assert driver.quality(p)["accepted"]
        wrapper = driver.solver.preconditioner
        assert wrapper is not driver.pre
        assert np.count_nonzero(wrapper.usage.numpy()[:-1]) > 1
        reported = driver.cg_statistics()
        assert reported == wrapper.stats.numpy().tolist()
        assert reported[0] > driver.pre.stats.numpy()[0]
        assert reported[3] > driver.pre.stats.numpy()[3]


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("perturbation", [0.0, 1e-3])
def test_shape_mass_and_factor_updates_on_one_graph(consistent, refit, perturbation, tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    states = shape_family(2, 6, consistent)
    p0, p1 = states[0][1], states[1][1]
    assert np.linalg.norm((p0.a - p1.a).data) > 0.1
    assert np.linalg.norm((p0.b - p1.b).data) > 0.001
    driver = ShapeReplay(states, 12, refit=refit, perturbation=perturbation)
    pointers = tuple(
        a.ptr
        for a in (
            driver.a.values,
            driver.mass.values,
            driver.shifted.values,
            driver.factor.G.values,
            driver.factor.GT.values,
        )
    )
    original_factor = driver.factor.G.values.numpy().copy()
    for repeat in range(2):
        for i, (_, p, _) in enumerate(states):
            driver.update(p)
            driver.solve(warm=i > 0)
            quality = driver.quality(p)
            assert quality["accepted"], quality
            assert driver.solver.graph is driver.graph
            assert pointers == tuple(
                a.ptr
                for a in (
                    driver.a.values,
                    driver.mass.values,
                    driver.shifted.values,
                    driver.factor.G.values,
                    driver.factor.GT.values,
                )
            )
            current_factor = driver.factor.G.values.numpy()
            if i == 1:
                assert bool(np.linalg.norm(current_factor - original_factor) > 1e-5) == refit
            np.testing.assert_allclose(
                driver.mass.values.numpy()[: len(driver.mrows)],
                np.asarray(p.b[driver.mrows, driver.mcols]).ravel(),
                atol=0,
            )
    path = tmp_path / "shape.dot"
    wp.capture_debug_dot_print(driver.graph, str(path))
    dot = path.read_text()
    assert not any(x in dot for x in ("HOST", "HtoD", "DtoH"))


@wp.kernel
def clear_second_member(vectors: wp.array2d[wp.float64], split: int):
    j, i = wp.tid()
    if i >= split:
        vectors[j, i] = wp.float64(0)


def crossing_members(n=24, k=4):
    """Two independent clamped P1 bars exchange lengths; analytic FE spectra."""
    import scipy.linalg as la
    import scipy.sparse as sp

    from benchmarks.applications.problems import Problem, bar

    a, b, w = bar(n, consistent=True)
    sine = np.sin(np.pi * np.arange(1, n + 1)[:, None] * np.arange(1, n + 1)[None, :] / (n + 1))
    sine /= np.sqrt(np.sum(sine * (b @ sine), axis=0))
    states = []
    for lengths in ((10.0, 1.0), (1.0, 10.0), (5.5, 5.5)):
        stiffness = sp.block_diag([a / length for length in lengths], format="csr")
        mass = sp.block_diag([b * length for length in lengths], format="csr")
        values = np.concatenate([w / length**2 for length in lengths])
        basis = la.block_diag(*[sine / np.sqrt(length) for length in lengths])
        order = np.argsort(values)
        p = Problem(
            "crossing_members",
            "A",
            stiffness,
            mass,
            k,
            -0.01,
            expected=values[order[:k]],
            requested_basis=basis[:, order[:k]],
        )
        states.append((lengths[0], p, float(values[order[k]])))
    return states


@pytest.mark.parametrize("perturbation", [0.0, 1e-3])
def test_exact_invariant_warm_start_after_physical_mode_crossing(perturbation):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    from warpack import LowestModeCertificate

    states = crossing_members()
    # No random padding rows: exercise a full invariant starting block.
    driver = ShapeReplay(states, 4, perturbation=perturbation)
    driver.solve()
    assert driver.quality(states[0][1])["accepted"]
    # Independent members have exactly zero motion in the other member.
    # Remove roundoff leakage from the computed basis, not reference vectors.
    wp.launch(
        clear_second_member,
        driver.solver.eigenvectors.shape,
        [driver.solver.eigenvectors, driver.n // 2],
    )
    assert driver.quality(states[0][1])["accepted"]
    _, p, guard = states[1]
    driver.update(p)
    driver.solve(warm=True)
    quality = driver.quality(p)
    certificate = LowestModeCertificate(
        driver.a,
        driver.solver.eigenvectors,
        driver.solver.eigenvalues,
        mass=driver.mass,
        upper_bound=0.5 * (p.expected[-1] + guard),
    )
    certificate.run()
    if perturbation:
        assert quality["accepted"], quality
        assert certificate.certified.numpy()[0] == 1
    else:
        # Documented limitation: native residual convergence is not completeness.
        assert quality["status"] == 0 and quality["converged"] == p.k
        assert quality["residual"] < 1e-10
        assert quality["spectrum_error"] > 1.0
        assert not quality["accepted"]
        assert certificate.certified.numpy()[0] == 0
        assert certificate.flags.numpy()[0] & certificate.GAP_UNRESOLVED


@wp.kernel
def rotate_pairs(source: wp.array2d[wp.float64], target: wp.array2d[wp.float64]):
    row, i = wp.tid()
    first = row - row % 2
    c, s = wp.cos(wp.float64(0.71)), wp.sin(wp.float64(0.71))
    if row % 2 == 0:
        target[row, i] = c * source[first, i] + s * source[first + 1, i]
    else:
        target[row, i] = -s * source[first, i] + c * source[first + 1, i]


def test_complete_repeated_cluster_accepts_rotated_computed_basis():
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    states = crossing_members()
    driver = ShapeReplay(states, 4, perturbation=1e-3)
    driver.solve()
    assert driver.quality(states[0][1])["accepted"]
    p = states[2][1]  # Equal lengths: two complete double eigenvalues.
    np.testing.assert_allclose(p.expected[::2], p.expected[1::2], atol=1e-13)
    driver.update(p)
    driver.solve(warm=True)
    assert driver.quality(p)["accepted"]
    source = wp.clone(driver.solver.eigenvectors)
    wp.launch(rotate_pairs, source.shape, [source, driver.solver.eigenvectors])
    assert driver.quality(p)["accepted"]  # Projector, not componentwise vector identity.
    driver.solve(warm=True)
    assert driver.quality(p)["accepted"]
