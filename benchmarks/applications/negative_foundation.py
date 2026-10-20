"""Analytic Q1 FEM negative-curvature fixtures; host benchmark/reference code only."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .problems import Problem


@dataclass
class NegativeFoundation:
    problem: Problem
    full_spectrum: np.ndarray
    foundation: float
    selection: str
    mass: str
    largest_magnitude_is_largest_algebraic: bool


def make_negative_foundation(side=6, modes=6, selection="SA", consistent=True):
    """Two repeated polarizations of anisotropic diffusion minus a foundation.

    Trilinear Q1 FEM on a unit cube with homogeneous Dirichlet boundary conditions.
    This is a negative-foundation curvature surrogate, not geometric prestress or
    a buckling load-factor pencil. Tensor-product sine eigenpairs give an
    independent complete spectrum and requested subspaces without factorization.
    """
    if side < 3 or modes < 1 or modes >= 2 * side**3 - 2:
        raise ValueError("require side>=3 and 1<=modes<2*side**3-2")
    if selection not in ("SA", "SM", "LM"):
        raise ValueError("selection must be SA, SM or LM")
    h = 1.0 / (side + 1)
    one = np.ones(side)
    stiffness = sp.diags([-one[:-1], 2 * one, -one[:-1]], [-1, 0, 1], format="csr") / h
    mass = (
        sp.diags([one[:-1], 4 * one, one[:-1]], [-1, 0, 1], format="csr") * (h / 6)
        if consistent
        else sp.eye(side, format="csr") * h
    )
    theta = np.arange(1, side + 1) * np.pi / (side + 1)
    mass_eigenvalues = h * ((2 + np.cos(theta)) / 3 if consistent else np.ones(side))
    values_1d = (2 - 2 * np.cos(theta)) / (h * mass_eigenvalues)
    sine = np.sin(np.arange(1, side + 1)[:, None] * theta)
    sine /= np.sqrt((side + 1) / 2 * mass_eigenvalues)[None, :]
    indices = np.array(list(np.ndindex((side,) * 3)))
    coefficients = np.array([1.0, 1.7, 2.3])
    scalar_values = values_1d[indices] @ coefficients
    order = np.argsort(scalar_values, kind="stable")
    # Place zero between low eigenvalues, with enough modes on both sides to
    # distinguish low algebraic modes from modes nearest zero. No pole at zero.
    cut = min(max(2, (modes + 1) // 2), len(order) - 2)
    while cut < len(order) - 1 and abs(
        scalar_values[order[cut]] - scalar_values[order[cut - 1]]
    ) < 1e-10 * max(1, scalar_values[order[cut]]):
        cut += 1
    foundation = float((scalar_values[order[cut - 1]] + scalar_values[order[cut]]) / 2)
    scalar_values -= foundation
    all_values = np.repeat(scalar_values, 2)
    ranking = np.argsort(
        all_values
        if selection == "SA"
        else (abs(all_values) if selection == "SM" else -abs(all_values)),
        kind="stable",
    )
    measure = all_values if selection == "SA" else abs(all_values)
    k = modes
    while k < len(ranking) and abs(measure[ranking[k]] - measure[ranking[k - 1]]) < 1e-10 * max(
        1, abs(measure[ranking[k]])
    ):
        k += 1
    if k >= len(ranking):
        raise ValueError("requested boundary cluster consumes entire spectrum")
    selected = ranking[:k]
    basis = np.zeros((len(all_values), k))
    for j, entry in enumerate(selected):
        i0, i1, i2 = indices[entry // 2]
        vector = np.einsum("i,j,k->ijk", sine[:, i0], sine[:, i1], sine[:, i2]).ravel()
        basis[entry % 2 :: 2, j] = vector
    m0 = sp.kron(sp.kron(mass, mass), mass, format="csr")
    k0 = sum(
        coefficients[axis]
        * sp.kron(
            sp.kron(stiffness if axis == 0 else mass, stiffness if axis == 1 else mass),
            stiffness if axis == 2 else mass,
            format="csr",
        )
        for axis in range(3)
    )
    identity = sp.eye(2, format="csr")
    a = sp.kron(k0 - foundation * m0, identity, format="csr")
    b = sp.kron(m0, identity, format="csr")
    a.eliminate_zeros()
    b.eliminate_zeros()
    largest = np.argsort(-all_values, kind="stable")[:k]
    magnitude = np.argsort(-abs(all_values), kind="stable")[:k]
    largest_equivalent = np.array_equal(
        np.sort(all_values[magnitude]), np.sort(all_values[largest])
    )
    if selection == "SA":
        # K - sigma M reduces exactly to the original positive stiffness plus M.
        target, mode, api = -foundation - 1.0, "shift-invert", "C"
    elif selection == "SM":
        target, mode, api = 0.0, "shift-invert", "B"
    else:
        target, mode, api = 0.0, "regular", "C"
    problem = Problem(
        name=f"negative_foundation_{selection}_{'consistent' if consistent else 'lumped'}",
        api_class=api,
        a=a,
        b=b,
        k=k,
        target=target,
        mode=mode,
        expected=all_values[selected],
        requested_basis=basis,
        block_size=2,
        notes="3D Q1 negative-foundation surrogate, two repeated polarizations; analytic tensor-product sine spectrum; complete requested boundary clusters; no reference vectors used as starts",
    )
    return NegativeFoundation(
        problem,
        np.sort(all_values),
        foundation,
        selection,
        "consistent" if consistent else "lumped",
        largest_equivalent,
    )
