"""Coupled elastic pencils with independent nearest-frequency-band references."""

from dataclasses import dataclass, replace

import numpy as np

from .fem_cases import make_fem_case

NAMES = (
    "clamped_consistent",
    "clamped_lumped",
    "heterogeneous",
    "disconnected",
    "repeated_bending",
)


@dataclass
class InteriorCase:
    problem: object
    spectrum: np.ndarray
    vectors: np.ndarray
    selected_indices: np.ndarray
    reference: str
    band_gap: tuple
    selection_margin: float
    base_nullity: int


def make_interior_case(name, side=2, modes=6, band_index=6, fraction=0.37):
    """Target a gap after the given number of distinct positive eigenvalues.

    The physical shift is an eigenvalue (squared angular frequency). Reference
    eigenvectors are used only for acceptance, never for solver initialization.
    Extend k for a complete distance-boundary cluster instead of splitting it.
    """
    if name not in NAMES or modes < 1 or band_index < 1:
        raise ValueError("Unknown fixture or nonpositive mode/band count")
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("The target must lie strictly inside the reference gap")
    base = make_fem_case(name, side, modes=2 * band_index + 2 * modes + 4)
    values = base.spectrum
    distinct = []
    for i in range(base.problem.nullity, len(values)):
        if not distinct or abs(values[i] - values[distinct[-1]]) > 1e-8 * max(1, abs(values[i])):
            distinct.append(i)
    if band_index >= len(distinct):
        raise ValueError("Insufficient reference capacity for the requested interior band")
    left, right = values[distinct[band_index - 1]], values[distinct[band_index]]
    target = float(left + fraction * (right - left))
    distances = abs(values - target)
    order = np.argsort(distances, kind="stable")
    k = modes
    while k < len(values) and abs(distances[order[k]] - distances[order[k - 1]]) <= 1e-8 * max(
        1, distances[order[k]]
    ):
        k += 1
    if k >= len(values) or values[-1] - target <= distances[order[k - 1]]:
        raise ValueError("Reference window does not enclose the entire nearest set")
    selected = np.sort(order[:k])
    expected = values[selected]
    # Interior selection excludes the free-body rigid block. It is not a low-mode
    # nullity estimate, and the accepted basis is the complete nearest subspace.
    if np.any(selected < base.problem.nullity):
        raise ValueError("Interior band overlaps the known rigid block")
    margin = float(distances[order[k]] - distances[order[k - 1]])
    p = replace(
        base.problem,
        name="interior_" + name,
        api_class="B",
        k=k,
        target=target,
        expected=expected,
        nullity=0,
        null_basis=None,
        requested_basis=base.vectors[:, selected],
        notes=f"{base.reference}; nearest physical eigenvalues to sigma={target:.17g}; full distance-boundary clusters; shift is squared angular frequency; shifted pencil is indefinite",
    )
    return InteriorCase(
        p,
        values,
        base.vectors,
        selected,
        base.reference,
        (float(left), float(right)),
        margin,
        base.problem.nullity,
    )
