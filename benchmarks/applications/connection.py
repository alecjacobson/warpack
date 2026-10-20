"""Manufactured U(1) connection pencils on periodic P1 triangular grids.

Fixtures/references only. Uniform holonomy gives an independent Fourier spectrum;
vertex gauge/amplitude congruence makes the coefficients complex/nonuniform.
Consistent mass transports local P1 values into a common triangle frame.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .problems import Problem


@dataclass
class ConnectionCase:
    problem: Problem
    side: int
    copies: int
    mass_kind: str
    angles: tuple
    gauge: np.ndarray
    mass_symbol: np.ndarray


def make_connection_case(side=8, modes=6, mass_kind="consistent", copies=1, gauge_seed=11):
    if side < 3 or copies < 1 or not 1 <= modes < side * side * copies:
        raise ValueError("require side>=3, positive copies and 1<=modes<n")
    if mass_kind not in ("identity", "lumped", "consistent"):
        raise ValueError("mass_kind must be identity, lumped or consistent")
    n, h = side * side, 1 / side
    angles = (0.37, 0.91)
    alpha = np.asarray(angles) / side
    grid = np.indices((side, side)).reshape(2, -1).T
    ids = np.arange(n)

    def stencil(diagonal, edges):
        rows, columns, values = [ids], [ids], [np.full(n, diagonal, dtype=complex)]
        for delta, weight in edges:
            offset = np.asarray(delta)
            dest = (grid + offset) % side
            rows.append(ids)
            columns.append(dest[:, 0] * side + dest[:, 1])
            values.append(np.full(n, weight * np.exp(1j * np.dot(alpha, offset))))
        result = sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(n, n)
        )
        result.sort_indices()
        return result

    axial = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    stiffness_scale = 1 / h**2 if mass_kind == "identity" else 1.0
    k0 = stiffness_scale * stencil(4.0, [(d, -1.0) for d in axial])
    frequencies = np.indices((side, side)).reshape(2, -1).T
    phase = (2 * np.pi * frequencies + np.asarray(angles)) / side
    symbol_k = stiffness_scale * (4 - 2 * np.cos(phase[:, 0]) - 2 * np.cos(phase[:, 1]))
    if mass_kind == "identity":
        m0 = sp.eye(n, dtype=complex, format="csr")
        symbol_m = np.ones(n)
    elif mass_kind == "lumped":
        m0 = h**2 * sp.eye(n, dtype=complex, format="csr")
        symbol_m = np.full(n, h**2)
    else:
        directions = axial + [(1, 1), (-1, -1)]
        m0 = stencil(h**2 / 2, [(d, h**2 / 12) for d in directions])
        symbol_m = h**2 * (
            0.5 + (np.cos(phase[:, 0]) + np.cos(phase[:, 1]) + np.cos(phase.sum(axis=1))) / 6
        )
    eigenvalues = symbol_k / symbol_m
    total_values = np.tile(eigenvalues, copies)
    expected = np.sort(total_values)[:modes]
    boundary = expected[-1] + 1e-10 * max(1, abs(expected[-1]))
    selected = np.flatnonzero(total_values <= boundary)
    wanted = np.zeros((n * copies, len(selected)), dtype=complex)
    gauges, stiffness, mass = [], [], []
    for component in range(copies):
        x, y = grid.T / side
        seed = gauge_seed + component * 7
        phi = 0.4 * np.sin(2 * np.pi * x + seed) + 0.3 * np.cos(2 * np.pi * y - seed)
        amplitude = np.ones(n)
        if mass_kind != "identity":
            amplitude += 0.17 * np.sin(2 * np.pi * (x + y) + seed)
        gauge = amplitude * np.exp(1j * phi)
        gauges.append(gauge)
        d = sp.diags(gauge, format="csr")
        stiffness.append((d.conj().T @ k0 @ d).tocsr())
        mass.append((d.conj().T @ m0 @ d).tocsr())
        for column, selected_index in enumerate(selected):
            if selected_index // n != component:
                continue
            j = selected_index % n
            fourier = np.exp(2j * np.pi * (grid @ frequencies[j]) / side)
            wanted[component * n : (component + 1) * n, column] = (
                fourier / gauge / np.sqrt(n * symbol_m[j])
            )
    a = sp.block_diag(stiffness, format="csr")
    b = sp.block_diag(mass, format="csr")
    a.sort_indices()
    b.sort_indices()
    problem = Problem(
        f"connection_{mass_kind}_{copies}copies",
        "E",
        a,
        None if mass_kind == "identity" else b,
        modes,
        -0.1,
        mode="shift-invert",
        expected=expected,
        requested_basis=wanted,
        notes="Periodic triangular U(1) connection with prescribed nontrivial holonomy, gauge congruence, independent Fourier spectrum and full boundary eigenspaces. Identity case is mass-normalized; consistent mass has complex off-diagonal entries. Copies give exact repeated complex modes.",
    )
    return ConnectionCase(
        problem, side, copies, mass_kind, angles, np.concatenate(gauges), symbol_m
    )


def triangle_energy(case, vectors):
    """Independent element gradient/mass evaluation, no assembled matrix products."""
    side, h = case.side, 1 / case.side
    alpha = np.asarray(case.angles) / side
    values = np.asarray(vectors, dtype=complex)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != case.problem.n:
        raise ValueError("vector size must match the fixture")
    stiffness = np.zeros(values.shape[1])
    mass = np.zeros_like(stiffness)
    local_triangles = [np.array([[0, 0], [1, 0], [1, 1]]), np.array([[0, 0], [1, 1], [0, 1]])]
    for component in range(case.copies):
        for i in range(side):
            for j in range(side):
                for corners in local_triangles:
                    grid = (corners + [i, j]) % side
                    indices = component * side**2 + grid[:, 0] * side + grid[:, 1]
                    transported = (
                        values[indices]
                        * (case.gauge[indices] * np.exp(1j * (corners @ alpha)))[:, None]
                    )
                    xy = h * corners
                    interpolation = np.column_stack([np.ones(3), xy])
                    coefficients = np.linalg.solve(interpolation, transported)
                    area = h**2 / 2
                    stiffness += area * np.sum(abs(coefficients[1:]) ** 2, axis=0)
                    if case.mass_kind == "consistent":
                        mass += (
                            area
                            / 12
                            * (
                                np.sum(abs(transported) ** 2, axis=0)
                                + abs(transported.sum(axis=0)) ** 2
                            )
                        )
                    else:
                        mass += area / 3 * np.sum(abs(transported) ** 2, axis=0)
    if case.mass_kind == "identity":
        stiffness /= h**2
        mass = np.sum(abs(values) ** 2, axis=0)
    return stiffness, mass
